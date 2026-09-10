"""EP-local CUTLASS fused-MoE control benchmark for the SM120 MegaMoE case.

This intentionally measures only the work that a destination EP rank performs
after dispatch.  Each rank sees the four source microbatches and retains only
its 32 local experts through CUTLASS's native EP mask.  Consequently the
routing is identical on every rank and contains exactly ``tokens * top_k``
routed rows per destination rank.  No all-to-all is timed here.

Run with four processes, one per EP rank.  The default case is the production
EP4 M=48,360 (6,045 tokens/rank, top-k 8) comparison point.
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import cutlass_fused_moe_workspace_size
from flashinfer.testing.utils import bench_gpu_time


FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-per-rank", type=int, default=6045)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--intermediate-size", type=int, default=2304)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def _quantize_weights(
    w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Use the same NVFP4 packing/scales as the unified CUTLASS benchmark."""
    experts, w1_rows, hidden = w1.shape
    intermediate = w2.shape[2]
    w1_q = torch.empty((experts, w1_rows, hidden // 2), dtype=torch.uint8, device="cuda")
    w2_q = torch.empty(
        (experts, hidden, intermediate // 2), dtype=torch.uint8, device="cuda"
    )
    w1_sf = torch.empty(
        (experts, _round_up(w1_rows, 128), _round_up(hidden // 16, 4)),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    w2_sf = torch.empty(
        (experts, _round_up(hidden, 128), _round_up(intermediate // 16, 4)),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    w1_gs = torch.empty(experts, dtype=torch.float32, device="cuda")
    w2_gs = torch.empty(experts, dtype=torch.float32, device="cuda")
    for expert in range(experts):
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1[expert].abs().max()
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2[expert].abs().max()
        w1_q[expert], w1_sf[expert] = fp4_quantize(w1[expert], w1_gs[expert])
        w2_q[expert], w2_sf[expert] = fp4_quantize(w2[expert], w2_gs[expert])

    one = torch.ones((), dtype=torch.float32, device="cuda")
    return w1_q, w2_q, [
        one,
        w1_sf.view(torch.int32),
        1.0 / w1_gs,
        one,
        w2_sf.view(torch.int32),
        1.0 / w2_gs,
    ]


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    if args.num_experts % args.ep_size:
        raise ValueError("num-experts must divide evenly across EP ranks")
    if dist.get_world_size() != args.ep_size:
        raise ValueError(f"expected {args.ep_size} torchrun ranks")

    device = torch.device("cuda", local_rank)
    local_experts = args.num_experts // args.ep_size
    aggregate_tokens = args.tokens_per_rank * args.ep_size
    local_rows = args.tokens_per_rank * args.top_k

    # This is the concatenation of the four source batches.  Round-robin global
    # ids give every expert either floor(M/E) or ceil(M/E) rows, precisely as
    # the MegaMoE benchmark's balanced routing contract specifies.
    source_rows = args.tokens_per_rank * args.top_k
    source_flat = torch.arange(source_rows, device=device)
    source_offsets = torch.arange(args.ep_size, device=device) * local_experts
    selected_experts = (
        (source_flat.unsqueeze(0) + source_offsets.unsqueeze(1)) % args.num_experts
    ).reshape(aggregate_tokens, args.top_k)
    selected_experts_i32 = selected_experts.to(torch.int)
    routing_weights = torch.full(
        (aggregate_tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32, device=device
    )
    local_count = (selected_experts // local_experts == rank).sum()
    if int(local_count) != local_rows:
        raise RuntimeError(f"rank {rank}: got {int(local_count)} local rows, expected {local_rows}")

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    x = torch.empty((aggregate_tokens, args.hidden_size), dtype=torch.bfloat16, device=device)
    x.normal_(mean=0.0, std=0.05, generator=generator)
    w1 = torch.randn(
        local_experts, 2 * args.intermediate_size, args.hidden_size,
        dtype=torch.bfloat16, device=device, generator=generator,
    ).div_(10)
    w2 = torch.randn(
        local_experts, args.hidden_size, args.intermediate_size,
        dtype=torch.bfloat16, device=device, generator=generator,
    ).div_(10)
    w1_q, w2_q, quant_scales = _quantize_weights(w1, w2)
    del w1, w2
    a_gs = torch.ones((), dtype=torch.float32, device=device)
    x_q, x_sf = fp4_quantize(x, a_gs)
    output = torch.empty_like(x)
    workspace_bytes = cutlass_fused_moe_workspace_size(
        aggregate_tokens, args.hidden_size, args.intermediate_size, args.num_experts,
        args.top_k, x_dtype=x_q.dtype, weight_dtype=w1_q.view(torch.long).dtype,
        output_dtype=torch.bfloat16, ep_size=args.ep_size, ep_rank=rank, device=device,
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)

    def run() -> torch.Tensor:
        return cutlass_fused_moe(
            x_q, selected_experts_i32, routing_weights,
            w1_q.contiguous().view(torch.long), w2_q.contiguous().view(torch.long),
            torch.bfloat16, quant_scales=quant_scales, input_sf=x_sf, output=output,
            ep_size=args.ep_size, ep_rank=rank, enable_pdl=None,
            workspace_buffer=workspace,
        )

    # Compile/autotune outside the timing interval, then prove the masked EP
    # result is usable before graph capture/replay timing.
    run()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError(f"rank {rank}: CUTLASS output contains non-finite values")
    dist.barrier()
    times_ms = bench_gpu_time(
        run, dry_run_iters=args.warmup, repeat_iters=args.iters,
        use_cuda_graph=True, cold_l2_cache=False, input_args=(),
    )
    median_ms = float(np.median(times_ms))
    # SwiGLU FC1 [H,2I] plus FC2 [I,H]: 6*H*I FLOPs per routed row.
    effective_tflops = (6 * local_rows * args.hidden_size * args.intermediate_size) / (median_ms * 1e9)
    if rank == 0:
        per_expert = local_rows // local_experts
        tail_experts = local_rows % local_experts
        print(
            "CUTLASS_EP4_LOCAL_COMPUTE "
            f"critical_path_median_ms={median_ms:.6f} "
            f"effective_tflops_per_rank={effective_tflops:.3f} "
            f"aggregate_tokens={aggregate_tokens} local_rows={local_rows} "
            f"local_experts={local_experts} rows_per_expert={per_expert}/{per_expert + 1} "
            f"tail_experts={tail_experts} workspace_gib={workspace_bytes / 2**30:.3f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
