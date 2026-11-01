"""Compare Blackwell sparse MLA schedules on Rubin (D576, FP8, BF16 output).

Example::

    python benchmarks/bench_dsa_rubin.py --cases 1,1,64 4,8,32 1,8192,24 \
        --json /tmp/dsa-rubin-baseline.json

Cases are B,Sq,H. Inputs are native E4M3 with random storage-row indices
(duplicates permitted by the API), a 128k-row pool, and top-k 2048. CUPTI
measures the entire prepared attention call including its split reducer, using
CUDA graphs and explicit L2 flushing. Planning, compilation,
input generation, and reference checks are outside timing. JSON is checkpointed
after each case. Use --blackwell-policy for the unretuned scheduling baseline.
"""

import argparse
import dataclasses
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cutlass
import torch

from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.testing import bench_gpu_time, bench_gpu_time_with_cudagraph


def check_reference(q, kv, indices, out, lse):
    """FP64 oracle plus the existing E4M3 probability rounding error bound."""
    flat = q.reshape(-1, q.shape[-2], 576)
    sample = torch.linspace(0, flat.shape[0] - 1, min(4, flat.shape[0])).long()
    sample = sample.to(q.device)
    query = flat[sample].double()
    values = kv[indices[sample].long()].double()
    logits = torch.bmm(query, values.transpose(1, 2)) * 576**-0.5
    weights = logits.softmax(-1)
    expected = torch.bmm(weights, values[..., :512])
    sensitivity = torch.bmm(weights, values[..., :512].abs())
    underflow = weights.amax(-1)[..., None] * values[..., :512].abs().sum(1)[:, None]
    bound = (2**-4 + 3 * 2**-8 + 1e-5) * sensitivity + 2**-10 / 448 * underflow + 1e-6
    error = (out.reshape(-1, q.shape[-2], 512)[sample].double() - expected).abs()
    torch.testing.assert_close(
        lse.reshape(-1, q.shape[-2])[sample].double(),
        logits.logsumexp(-1),
        atol=2e-4,
        rtol=1e-4,
    )
    assert (error <= bound).all(), f"FP8 bound exceeded: {error.max().item()}"
    assert torch.isfinite(out).all()
    return {
        "max_abs_error": error.max().item(),
        "max_error_over_bound": (error / bound).max().item(),
    }


def packed_cache(pool):
    """Valid D576 NVFP4 records and their effective E4M3 MMA values."""
    cache = torch.randint(256, (pool, 352), device="cuda", dtype=torch.uint8)
    sf = (torch.rand(pool, 32, device="cuda") * 0.12 + 0.04).to(torch.float8_e4m3fn)
    rope = (torch.randn(pool, 64, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    cache[:, 256:320] = rope.view(torch.uint8)
    cache[:, 320:] = (
        sf.view(torch.uint8).unflatten(-1, (8, 4)).transpose(-1, -2).flatten(-2)
    )
    packed = cache[:, :256]
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    vals = lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    nope = (vals.unflatten(-1, (32, 16)) * sf.float()[..., None]).flatten(-2)
    reference = torch.cat((nope.to(torch.float8_e4m3fn).float(), rope.float()), -1).to(
        torch.float8_e4m3fn
    )
    return cache, reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases", nargs="+", default=["1,1,64", "4,8,32", "64,4,48", "64,8,96"]
    )
    p.add_argument("--pool", type=int, default=131072)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--timer", choices=("cupti", "rotated"), default="cupti")
    p.add_argument("--fp8-mma-k", type=int, choices=(32, 64), default=32)
    p.add_argument(
        "--cache-format", choices=("native", "nvfp4.fp8rope"), default="native"
    )
    p.add_argument(
        "--split-kv",
        type=int,
        help="Override native D576 reload split count (research only)",
    )
    p.add_argument("--json", type=Path, required=True)
    p.add_argument(
        "--force-family", choices=("keep", "swap"), help="Research schedule override"
    )
    p.add_argument("--gather-warps", type=int, choices=(4, 8), default=4)
    p.add_argument("--v-split", type=int, choices=(1, 2, 4), default=1)
    p.add_argument(
        "--reuse-kv", action="store_true", help="Retain gathered KV in forced profiles"
    )
    p.add_argument(
        "--kv-stages",
        type=int,
        help="Override retained KV stage count in the automatic profile",
    )
    p.add_argument(
        "--blackwell-policy",
        action="store_true",
        help="Disable the qualified Rubin policy refinements",
    )
    a = p.parse_args()
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )

    original_config = ThroughputLatencyMlaDecodeTs._make_config
    selected_configs = []

    def make_config(kernel):
        cfg = original_config(kernel)
        full_datapath = (
            cfg.kernel_variant == "swaps_mma_ab"
            and cfg.tile_size_kv >= 128
            and cfg.head_dim_per_stage_v >= 128
            and not cfg.ws_qk
            and not cfg.ws_pv
        )
        step = a.fp8_mma_k if full_datapath else 32
        if step == 64:
            assert torch.cuda.get_device_capability() == (10, 7)
        selected_configs.append(
            dict(
                fp8_mma_k=step,
                tile_q=cfg.tile_size_q,
                family=cfg.kernel_variant,
                kv_stages=cfg.kv_stages,
                v_splits=cfg.num_ctas_per_head_dim,
                reuse_kv=cfg.sparse_reuse_kv,
                clc=bool(cfg.use_clc_dynamic_persistent_scheduler),
            )
        )
        return dataclasses.replace(cfg, fp8_mma_k=step)

    ThroughputLatencyMlaDecodeTs._make_config = make_config
    if a.split_kv is not None:
        from flashinfer.attention.prims_ts.kernels.mla_decode import sparse_policy

        sparse_policy.select_sparse_mla_d576_split = lambda *args: (
            a.split_kv,
            a.v_split,
        )
        sparse_policy.select_sparse_mla_rubin_split = (
            sparse_policy.select_sparse_mla_d576_split
        )
    if a.blackwell_policy:
        from flashinfer.attention.prims_ts import mla_decode
        from flashinfer.attention.prims_ts.kernels.mla_decode import sparse_policy

        mla_decode.refine_sparse_mla_rubin_profile = lambda profile, **kwargs: profile
        sparse_policy.select_sparse_mla_rubin_split = (
            sparse_policy.select_sparse_mla_d576_split
        )
    if a.force_family:
        from flashinfer.attention.prims_ts import mla_decode
        from flashinfer.attention.prims_ts.kernels.mla_decode.sparse_policy import (
            SparseMlaProfile,
        )

        mla_decode.prefer_sparse_d576_reload = lambda **kwargs: False
        mla_decode.select_sparse_mla_profile = lambda **kwargs: SparseMlaProfile(
            family=a.force_family,
            tile_size_q=64 if a.force_family == "keep" else 16,
            split_kv=a.split_kv or 1,
            gather_issue_warps=a.gather_warps,
            reuse_kv=a.reuse_kv,
            reuse_kv_stages=10 if a.reuse_kv else 0,
            uniform_offset_cache=a.reuse_kv and a.force_family == "keep",
            kv_pipeline_stages=10,
        )
    if a.kv_stages is not None:
        from flashinfer.attention.prims_ts import mla_decode

        mla_decode.refine_sparse_mla_rubin_profile = lambda profile, **kwargs: profile

        original_select = mla_decode.select_sparse_mla_profile

        def select_profile(**kwargs):
            profile = original_select(**kwargs)
            if profile.reuse_kv:
                profile = dataclasses.replace(profile, reuse_kv_stages=a.kv_stages)
            return profile

        mla_decode.select_sparse_mla_profile = select_profile
    torch.cuda.set_device(0)
    torch.manual_seed(7984)
    props = torch.cuda.get_device_properties(0)
    result = dict(
        commit=subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "HEAD"], text=True
        ).strip(),
        gpu=str(props),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cutlass=cutlass.__version__,
        timer=(
            "CUPTI, CUDA graph, explicit cold L2"
            if a.timer == "cupti"
            else "CUDA graph/events, rotated inputs (allocated footprint)"
        ),
        pool=a.pool,
        cache_format=a.cache_format,
        topk=a.topk,
        fp8_mma_k=a.fp8_mma_k,
        split_kv_override=a.split_kv,
        forced_family=a.force_family,
        gather_warps=a.gather_warps,
        v_split_override=a.v_split if a.split_kv is not None else None,
        reuse_kv_override=a.reuse_kv if a.force_family else None,
        retained_stages_override=a.kv_stages,
        blackwell_policy=a.blackwell_policy,
        results=[],
    )
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    for case in a.cases:
        selected_configs.clear()
        batch, sq, heads = map(int, case.split(","))
        rows = batch * sq
        q = (torch.randn(batch, sq, heads, 576, device="cuda") * 0.5).to(
            torch.float8_e4m3fn
        )
        kv = (torch.randn(a.pool, 576, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        reference_kv = kv
        if a.cache_format != "native":
            kv, reference_kv = packed_cache(a.pool)
        indices = torch.randint(
            a.pool, (rows, a.topk), device="cuda", dtype=torch.int32
        )
        lengths = torch.full((rows,), a.topk, device="cuda", dtype=torch.int32)
        out = torch.empty(batch, sq, heads, 512, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(batch, sq, heads, device="cuda")
        w = BatchSparseMLADecodePagedTSWrapper()
        begin = time.monotonic()
        w.plan(
            "cuda",
            batch,
            heads,
            max_q_len=sq,
            max_topk=a.topk,
            q_dtype=q.dtype,
            qk_rope_head_dim=64,
            return_lse=True,
            kv_cache_format=a.cache_format,
        )

        def run(q, kv, indices, lengths, out, lse):
            w.run(
                q,
                kv,
                indices,
                lengths,
                out=out,
                lse=lse,
                softmax_scale=576**-0.5,
                validate=False,
            )

        run(q, kv, indices, lengths, out, lse)
        torch.cuda.synchronize()
        checked = check_reference(q, reference_kv, indices, out, lse)
        timer_kwargs = dict(
            input_args=(q, kv, indices, lengths, out, lse),
            cold_l2_cache=True,
            dry_run_time_ms=100,
            repeat_iters=a.reps,
        )
        if a.timer == "cupti":
            from importlib import import_module

            import_module("cupti.cupti")  # Require CUPTI for comparable results.

            times = bench_gpu_time(
                run, enable_cupti=True, use_cuda_graph=True, **timer_kwargs
            )
        else:
            times = bench_gpu_time_with_cudagraph(
                run, num_iters_within_graph=4, **timer_kwargs
            )
        profile = {
            "splits": w._state["requested_splits"],
            "reload_d576": w._state["reload_d576"],
        }
        profile["mma"] = selected_configs[-1] if selected_configs else "2cta/k32"
        row = dict(
            case=case,
            arm=f"k{a.fp8_mma_k}",
            median_us=statistics.median(times) * 1000,
            min_us=min(times) * 1000,
            max_us=max(times) * 1000,
            samples_ms=times,
            workspace_bytes=w.workspace_size_bytes,
            setup_seconds=time.monotonic() - begin,
            correctness=checked,
            profile=profile,
        )
        result["results"].append(row)
        print(json.dumps(row), flush=True)
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
