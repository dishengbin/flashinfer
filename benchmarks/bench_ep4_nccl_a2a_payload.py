"""NCCL all-to-all reference for the SM120 NVFP4 MegaMoE EP4 workload.

It models the wire payload, not a CUTLASS kernel: dispatch transfers one
NVFP4 activation plus one E4M3 scale per routed row; combine transfers the
BF16 FC2 result per routed row.  Calls are deliberately sequential, so the
result is a no-overlap communication baseline for a conventional EP pipeline.
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-per-rank", type=int, default=6045)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if dist.get_world_size() != args.ep_size:
        raise ValueError(f"expected {args.ep_size} ranks")
    rows = args.tokens_per_rank * args.top_k
    if rows % args.ep_size:
        raise ValueError("routed rows must divide evenly across EP ranks")
    rows_per_peer = rows // args.ep_size
    # NVFP4 = hidden/2 bytes and E4M3 scales = hidden/16 bytes per row.
    dispatch_row_bytes = args.hidden_size // 2 + args.hidden_size // 16
    dispatch_send = torch.empty((rows, dispatch_row_bytes), dtype=torch.uint8, device="cuda")
    dispatch_recv = torch.empty_like(dispatch_send)
    combine_send = torch.empty((rows, args.hidden_size), dtype=torch.bfloat16, device="cuda")
    combine_recv = torch.empty_like(combine_send)
    splits = [rows_per_peer] * args.ep_size

    def dispatch() -> None:
        dist.all_to_all_single(dispatch_recv, dispatch_send, splits, splits)

    def combine() -> None:
        dist.all_to_all_single(combine_recv, combine_send, splits, splits)

    for _ in range(args.warmup):
        dispatch()
        combine()
    torch.cuda.synchronize()
    dist.barrier()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dispatch_ms, combine_ms, total_ms = [], [], []
    for _ in range(args.iters):
        dist.barrier()
        start.record(); dispatch(); stop.record(); stop.synchronize()
        d_ms = start.elapsed_time(stop)
        start.record(); combine(); stop.record(); stop.synchronize()
        c_ms = start.elapsed_time(stop)
        samples = [d_ms, c_ms, d_ms + c_ms]
        gathered = [None] * args.ep_size
        dist.all_gather_object(gathered, samples)
        d_ms, c_ms, t_ms = (max(x[i] for x in gathered) for i in range(3))
        dispatch_ms.append(d_ms); combine_ms.append(c_ms); total_ms.append(t_ms)
    if rank == 0:
        print(
            "NCCL_EP4_A2A_NO_OVERLAP "
            f"dispatch_median_ms={np.median(dispatch_ms):.6f} "
            f"combine_median_ms={np.median(combine_ms):.6f} "
            f"total_median_ms={np.median(total_ms):.6f} "
            f"rows={rows} rows_per_peer={rows_per_peer} "
            f"dispatch_payload_mib={rows * dispatch_row_bytes / 2**20:.3f} "
            f"combine_payload_mib={rows * args.hidden_size * 2 / 2**20:.3f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
