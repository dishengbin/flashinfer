"""SM120 NVFP4 split-MegaMoE benchmark for the customer EP2/EP4 cases.

The default geometry is fixed to the requested workload:

* hidden = 6144
* post-SwiGLU intermediate = 2304 (W13 stores 2 * intermediate rows)
* top-k = 8
* total experts = 128 (64 local experts at EP2, 32 at EP4)
* grouped-GEMM M/rank = 48360, 95160, 111600, 219600, 327600,
  370800, 392400, 414000, 651600

``grouped_gemm_m`` is the routed-row count processed by one destination rank.
Balanced routing makes ``tokens_per_rank = grouped_gemm_m / top_k`` and gives
every destination rank exactly ``grouped_gemm_m`` rows.

Each graph point owns an exact-capacity workspace (capacity equals its live
token count), so the sweep measures nine independent shape specializations.
At the largest default point, rough live allocations before compiler/runtime
overhead are about 18 GiB/rank for EP2 and 24 GiB/rank for EP4.  The configured
16 GiB NVSHMEM heap may reserve additional memory; use at least 32 GiB for EP2
and 48 GiB for EP4.  The reported ``torch_peak_*`` fields exclude allocations
owned by NVSHMEM.

Two targets are available:

* ``graph`` measures the production compute graph after staging once.  The
  timed region contains reset + concurrent K1/K2 + K3, but excludes BF16 input
  quantization, JIT compilation, graph capture, and output copies.  K2 is
  producer-coupled to K1 and cannot be timed faithfully as a standalone call.
  This is the default because it is the target that includes EP communication.
* ``k3`` measures ``kernel_combine_reduce.py`` directly with the production
  ``topk_score=None`` specialization.  This is a local single-kernel bandwidth
  benchmark; torchrun is retained so EP2/EP4 reports have identical rank-load
  and critical-rank statistics.

Use ``--combine-dtype both`` for a sequential BF16/NVFP4 A/B.  NVFP4 stores
two E2M1 values per byte plus one BF16 amax per 16 hidden values.  Consequently
the combine plane is 0.625 bytes/value instead of BF16's 2 bytes/value.  K3
bandwidth accounts for both the packed data and scale planes.

Run on two or four physical, peer-accessible SM120 GPUs from one NUMA domain:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1 \\
    NVSHMEM_HEAP_KIND=VIDMEM NVSHMEM_SYMMETRIC_SIZE=16G \\
    torchrun --standalone --nproc_per_node=2 \\
      benchmarks/bench_moe_ep_sm120_nvfp4_mega.py

For a one-GPU functional/performance diagnostic (which does not measure EP
communication), launch with ``--nproc-per-node=1 --allow-single-rank``.

    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    NVSHMEM_HEAP_KIND=VIDMEM NVSHMEM_SYMMETRIC_SIZE=16G \\
    torchrun --standalone --nproc_per_node=4 \\
      benchmarks/bench_moe_ep_sm120_nvfp4_mega.py

The script emits one ``BENCH_CSV`` row per target and problem size.  Use
``--target graph`` for the production headline number.  Use Nsight Systems on
that run to inspect ``fc2_combine_kernel_impl`` without destroying its overlap
with K1.  ``--nvtx`` labels each measured interval for profiler runs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence, TextIO


_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]

# These must be set before NVSHMEM is imported.  setdefault preserves explicit
# cluster settings while making the documented same-NUMA configuration usable
# without extra boilerplate.
os.environ.setdefault("NVSHMEM_HEAP_KIND", "VIDMEM")
os.environ.setdefault("NVSHMEM_SYMMETRIC_SIZE", "16G")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


GROUPED_GEMM_M_VALUES = (
    48360,
    95160,
    111600,
    219600,
    327600,
    370800,
    392400,
    414000,
    651600,
)

DEFAULT_HIDDEN = 6144
DEFAULT_INTERMEDIATE = 2304
DEFAULT_TOP_K = 8
DEFAULT_NUM_EXPERTS = 128
KERNEL_NAME = "sm120_nvfp4_nvfp4_bf16_cutedsl"

# A failed rank must not enter NVSHMEM collective frees while its peers are
# still unwinding a different code path.  Failure cleanup detaches the
# workspace from MoEEpMegaLayer.__del__ and deliberately keeps it alive until
# torchrun terminates the process.
_ABANDONED_GRAPH_WORKSPACES: list[object] = []

CSV_FIELDS = (
    "record",
    "kernel",
    "target",
    "combine_dtype",
    "status",
    "gpu_name",
    "world_size",
    "total_experts",
    "local_experts",
    "hidden",
    "intermediate_post_swiglu",
    "topk",
    "grouped_gemm_m_per_rank",
    "tokens_per_rank",
    "expected_rows_per_expert",
    "combine_plane_gib_per_rank",
    "warmup",
    "iters",
    "inner_repeats",
    "critical_min_us",
    "critical_p50_us",
    "critical_p90_us",
    "critical_p99_us",
    "critical_max_us",
    "critical_mean_us",
    "critical_std_us",
    "rank_p50_min_us",
    "rank_p50_max_us",
    "rank_p50_imbalance",
    "global_tokens_per_s",
    "routed_rows_per_rank_per_s",
    "fc1_flops_per_rank",
    "fc2_flops_per_rank",
    "total_gemm_flops_per_rank",
    "effective_gemm_tflops_per_rank",
    "k3_io_bytes_per_rank",
    "effective_k3_gbytes_per_s",
    "setup_max_s",
    "torch_peak_allocated_gib",
    "torch_peak_reserved_gib",
    "knobs_json",
)


@dataclass(frozen=True)
class TimingStats:
    critical_min_us: float
    critical_p50_us: float
    critical_p90_us: float
    critical_p99_us: float
    critical_max_us: float
    critical_mean_us: float
    critical_std_us: float
    rank_p50_min_us: float
    rank_p50_max_us: float

    @property
    def rank_p50_imbalance(self) -> float:
        return self.rank_p50_max_us / self.rank_p50_min_us


@dataclass(frozen=True)
class PointResult:
    timing: TimingStats
    setup_max_s: float
    peak_allocated_gib: float
    peak_reserved_gib: float


class CsvSink:
    def __init__(self, output_path: str | None) -> None:
        self._file: TextIO | None = None
        self._writers = [csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)]
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", newline="", encoding="utf-8")
            self._writers.append(csv.DictWriter(self._file, fieldnames=CSV_FIELDS))

    def write_header(self) -> None:
        for writer in self._writers:
            writer.writeheader()
        self.flush()

    def write_row(self, row: dict[str, object]) -> None:
        for writer in self._writers:
            writer.writerow(row)
        self.flush()

    def flush(self) -> None:
        sys.stdout.flush()
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--grouped-gemm-m-values",
        "--m-values",
        default=",".join(map(str, GROUPED_GEMM_M_VALUES)),
        help="comma-separated routed grouped-GEMM M values per destination rank",
    )
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument(
        "--intermediate",
        type=int,
        default=DEFAULT_INTERMEDIATE,
        help="post-SwiGLU intermediate size; W13 uses twice this value",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument(
        "--target",
        choices=("graph", "k3", "both"),
        default="graph",
        help="production K1/K2/K3 graph (default), standalone K3, or both",
    )
    parser.add_argument(
        "--combine-dtype",
        choices=("bf16", "nvfp4", "both"),
        default="bf16",
        help="combine wire format to benchmark; 'both' runs a sequential A/B",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--k3-inner-repeats",
        type=int,
        default=10,
        help="K3 launches per CUDA-event interval; reported time is per launch",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--knobs-json",
        default=None,
        help="optional JSON object passed to the SM120 production heuristic overrides",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="also write BENCH_CSV records to this file (rank 0 only)",
    )
    parser.add_argument(
        "--nvtx", action="store_true", help="wrap measured intervals in NVTX ranges"
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print the M-to-token mapping for EP2/EP4 and exit without CUDA",
    )
    parser.add_argument(
        "--allow-single-rank",
        action="store_true",
        help="allow world_size=1 for diagnostic graph/K3 measurements",
    )
    return parser.parse_args()


def _parse_m_values(raw: str, top_k: int) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--grouped-gemm-m-values must contain integers") from exc
    if not values:
        raise ValueError("at least one grouped-GEMM M value is required")
    if any(value <= 0 for value in values):
        raise ValueError("grouped-GEMM M values must be positive")
    not_divisible = [value for value in values if value % top_k]
    if not_divisible:
        raise ValueError(
            f"grouped-GEMM M values must be divisible by top-k={top_k}: {not_divisible}"
        )
    return values


def _parse_knobs(raw: str | None) -> dict | None:
    if raw is None:
        return None
    knobs = json.loads(raw)
    if not isinstance(knobs, dict):
        raise ValueError("--knobs-json must decode to a JSON object")
    return knobs


def _validate_static_args(args: argparse.Namespace, m_values: Sequence[int]) -> None:
    positive = {
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "top-k": args.top_k,
        "num-experts": args.num_experts,
        "iters": args.iters,
        "k3-inner-repeats": args.k3_inner_repeats,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"benchmark arguments must be positive: {invalid}")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.hidden % 32 or args.intermediate % 32:
        raise ValueError("SM120 NVFP4 requires hidden and intermediate multiples of 32")
    if args.top_k > args.num_experts:
        raise ValueError("top-k cannot exceed num-experts")
    if any(value // args.top_k <= 0 for value in m_values):
        raise ValueError(
            "every grouped-GEMM M value must map to a positive token count"
        )


def _validate_graph_dependencies(*, single_rank: bool) -> None:
    missing: list[tuple[str, str]] = []
    required = [("cuda.core", "cuda-core")]
    if not single_rank:
        required.append(("nvshmem.core", "nvshmem4py-cu13"))
    for module_name, package_name in required:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append((module_name, package_name))
    if missing:
        details = ", ".join(
            f"{module_name} (package {package_name})"
            for module_name, package_name in missing
        )
        raise RuntimeError(
            "--target graph/both requires the full MoE-EP runtime; missing "
            f"{details}. Use the repository's FlashInfer-EP container setup."
        )


def _combine_plane_bytes(grouped_gemm_m: int, hidden: int, combine_dtype: str) -> int:
    if combine_dtype == "bf16":
        return grouped_gemm_m * hidden * 2
    if combine_dtype == "nvfp4":
        return grouped_gemm_m * (hidden // 2 + (hidden // 16) * 2)
    raise ValueError(f"unsupported combine dtype {combine_dtype!r}")


def _combine_plane_gib(grouped_gemm_m: int, hidden: int, combine_dtype: str) -> float:
    return _combine_plane_bytes(grouped_gemm_m, hidden, combine_dtype) / 2**30


def _selected_combine_dtypes(args: argparse.Namespace) -> tuple[str, ...]:
    return ("bf16", "nvfp4") if args.combine_dtype == "both" else (args.combine_dtype,)


def _print_cases(args: argparse.Namespace, m_values: Sequence[int]) -> None:
    print(
        "world_size,local_experts,grouped_gemm_m_per_rank,tokens_per_rank,"
        "combine_dtype,expected_rows_per_expert,combine_plane_gib_per_rank"
    )
    for world_size in (1, 2, 4) if args.allow_single_rank else (2, 4):
        if args.num_experts % world_size:
            continue
        for combine_dtype in _selected_combine_dtypes(args):
            for grouped_m in m_values:
                tokens = grouped_m // args.top_k
                rows = grouped_m * world_size / args.num_experts
                print(
                    f"{world_size},{args.num_experts // world_size},{grouped_m},"
                    f"{tokens},{combine_dtype},{rows:.6f},"
                    f"{_combine_plane_gib(grouped_m, args.hidden, combine_dtype):.6f}"
                )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - fraction)
        + float(sorted_values[upper]) * fraction
    )


def _aggregate_samples(local_samples: list[float], dist) -> TimingStats:
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_samples)
    samples_by_rank = [samples for samples in gathered if samples is not None]
    if len(samples_by_rank) != dist.get_world_size():
        raise RuntimeError("failed to gather timing samples from every EP rank")
    lengths = {len(samples) for samples in samples_by_rank}
    if lengths != {len(local_samples)}:
        raise RuntimeError(f"ranks reported inconsistent sample counts: {lengths}")

    critical = sorted(
        max(rank_values) for rank_values in zip(*samples_by_rank, strict=True)
    )
    rank_p50 = [statistics.median(values) for values in samples_by_rank]
    return TimingStats(
        critical_min_us=min(critical),
        critical_p50_us=statistics.median(critical),
        critical_p90_us=_percentile(critical, 0.90),
        critical_p99_us=_percentile(critical, 0.99),
        critical_max_us=max(critical),
        critical_mean_us=statistics.fmean(critical),
        critical_std_us=statistics.pstdev(critical),
        rank_p50_min_us=min(rank_p50),
        rank_p50_max_us=max(rank_p50),
    )


def _run_with_nvtx(call: Callable[[], object], label: str, enabled: bool) -> None:
    if not enabled:
        call()
        return
    import torch

    torch.cuda.nvtx.range_push(label)
    try:
        call()
    finally:
        torch.cuda.nvtx.range_pop()


def _time_calls(
    call: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    inner_repeats: int,
    label: str,
    nvtx: bool,
    dist,
) -> list[float]:
    import torch

    # CUDA events are intentional here: every sample must start after a
    # cross-rank barrier and remain attributable to one rank so rank-wise
    # samples can be reduced into a critical-path distribution.  The generic
    # CUPTI helper does not provide that multi-rank sample correspondence.
    def launch_batch() -> None:
        for _ in range(inner_repeats):
            call()

    for _ in range(warmup):
        launch_batch()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        start.record()
        _run_with_nvtx(launch_batch, label, nvtx)
        stop.record()
        stop.synchronize()
        elapsed_us = start.elapsed_time(stop) * 1e3 / inner_repeats
        if not math.isfinite(elapsed_us) or elapsed_us <= 0:
            raise RuntimeError(
                f"invalid CUDA-event measurement {elapsed_us} us; under "
                "Confidential Computing use a CUPTI/nsys timing run"
            )
        samples.append(elapsed_us)
    return samples


def _all_reduce_max(value: float, dist, device) -> float:
    import torch

    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _peak_memory_gib(dist, device) -> tuple[float, float]:
    import torch

    allocated = _all_reduce_max(torch.cuda.max_memory_allocated(), dist, device)
    reserved = _all_reduce_max(torch.cuda.max_memory_reserved(), dist, device)
    return allocated / 2**30, reserved / 2**30


def _collective_assert(local_ok: bool, message: str, dist, device) -> None:
    import torch

    passed = torch.tensor(int(local_ok), dtype=torch.int32, device=device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if not bool(passed.item()):
        raise RuntimeError(f"{message} failed on at least one EP rank")


def _probe_token_rows(tokens: int, device):
    import torch

    return torch.tensor(
        sorted({0, tokens // 2, tokens - 1}),
        dtype=torch.int64,
        device=device,
    )


def _packed_e2m1(shape: tuple[int, ...], generator) -> object:
    import torch

    return torch.randint(
        0,
        256,
        shape,
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    ).view(torch.float4_e2m1fn_x2)


def _make_transformed_weights(args, *, local_experts: int, rank: int):
    import torch

    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_sm120_nvfp4_cutedsl_mega_weights,
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed + rank)
    weights = MoEWeightPack(
        w13=_packed_e2m1(
            (local_experts, 2 * args.intermediate, args.hidden // 2), generator
        ),
        w2=_packed_e2m1(
            (local_experts, args.hidden, args.intermediate // 2), generator
        ),
        w13_scale=torch.ones(
            (local_experts, 2 * args.intermediate, args.hidden // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        w2_scale=torch.ones(
            (local_experts, args.hidden, args.intermediate // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
    )
    transformed = preprocess_sm120_nvfp4_cutedsl_mega_weights(
        weights,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
    )
    torch.cuda.synchronize()
    del weights
    gc.collect()
    torch.cuda.empty_cache()
    return transformed


def _make_inputs(args, grouped_m: int, *, rank: int, world_size: int, dist):
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    tokens = grouped_m // args.top_k
    local_experts = args.num_experts // world_size
    generator = torch.Generator(device="cuda").manual_seed(
        args.seed + 1009 * rank + grouped_m
    )
    hidden_states = torch.empty(
        (tokens, args.hidden), dtype=torch.bfloat16, device="cuda"
    )
    hidden_states.normal_(mean=0.0, std=0.05, generator=generator)

    flat = torch.arange(grouped_m, dtype=torch.int64, device="cuda")
    topk_ids = ((flat + rank * local_experts) % args.num_experts).view(
        tokens, args.top_k
    )
    topk_weights = torch.full(
        (tokens, args.top_k),
        1.0 / args.top_k,
        dtype=torch.float32,
        device="cuda",
    )

    # Verify the benchmark contract once, outside the timed region: after
    # combining all source ranks, every destination rank receives exactly M
    # routed rows.  Per-expert counts can differ by at most one row.
    destination_counts = torch.bincount(
        (topk_ids.flatten() // local_experts), minlength=world_size
    ).to(torch.int64)
    dist.all_reduce(destination_counts, op=dist.ReduceOp.SUM)
    expected = torch.full_like(destination_counts, grouped_m)
    if not torch.equal(destination_counts, expected):
        raise RuntimeError(
            "balanced routing did not produce the requested per-rank grouped M: "
            f"actual={destination_counts.tolist()}, expected={expected.tolist()}"
        )

    return MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


def _detach_layer_workspace(layer):
    if layer is None:
        return None
    workspace = layer._workspace
    layer._workspace = None
    return workspace


def _cleanup_layer(layer) -> None:
    import torch

    workspace = _detach_layer_workspace(layer)
    torch.cuda.synchronize()
    if workspace is not None:
        # Detach first so a teardown exception cannot make __del__ retry a
        # collective free from only this rank.
        layer._kernel.destroy(workspace)
    gc.collect()
    torch.cuda.empty_cache()


def _abandon_layer_after_failure(layer) -> None:
    workspace = _detach_layer_workspace(layer)
    if workspace is not None:
        _ABANDONED_GRAPH_WORKSPACES.append(workspace)


def _run_graph_point(
    args,
    grouped_m: int,
    *,
    combine_dtype: str,
    rank: int,
    world_size: int,
    bootstrap,
    transformed_weights,
    knobs: dict | None,
    dist,
    device,
) -> PointResult:
    import torch

    from flashinfer.moe_ep import (
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    tokens = grouped_m // args.top_k
    layer = None
    inputs = None
    completed = False
    torch.cuda.reset_peak_memory_stats()
    setup_start = perf_counter()
    try:
        inputs = _make_inputs(
            args, grouped_m, rank=rank, world_size=world_size, dist=dist
        )
        layer = MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=FleetParams(
                num_experts=args.num_experts,
                max_tokens_per_rank=tokens,
                token_hidden_size=args.hidden,
            ),
            weights=None,
            backend=MegaConfig(
                megakernel=Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                    intermediate_size=args.intermediate,
                    top_k=args.top_k,
                    combine_dtype=combine_dtype,
                    knobs=knobs,
                ),
                quantize_input=True,
                preprocess_weights=False,
                transformed_weights=transformed_weights,
            ),
        )

        # Collective one-time path: quantize/stage, allocate symmetric buffers,
        # JIT compile, capture the native Green Context graph, and launch once.
        # Capacity equals live tokens, which is essential because >256-token
        # compile hints otherwise fall back to the workspace capacity.
        layer.stage_inputs(inputs, compile_tokens_per_rank=tokens)
        first_output = layer.compute_staged(output=None)
        torch.cuda.synchronize()
        probe_rows = _probe_token_rows(tokens, device)
        first_probe = first_output.index_select(0, probe_rows).clone()
        replay_output = layer.compute_staged(output=None)
        torch.cuda.synchronize()
        replay_probe = replay_output.index_select(0, probe_rows)
        local_ok = bool(torch.isfinite(replay_probe).all().item()) and torch.equal(
            first_probe,
            replay_probe,
        )
        _collective_assert(
            local_ok,
            f"graph finite/deterministic replay check for M={grouped_m}",
            dist,
            device,
        )
        del first_output, replay_output, probe_rows, first_probe, replay_probe
        setup_max_s = _all_reduce_max(perf_counter() - setup_start, dist, device)

        # compute_staged consumes only the staged workspace. Drop our Python
        # reference to the original batch; the fused-stager descriptor retains
        # it until layer teardown, which the memory guidance above includes.
        del inputs
        inputs = None
        gc.collect()
        torch.cuda.empty_cache()

        local_samples = _time_calls(
            lambda: layer.compute_staged(output=None),
            warmup=args.warmup,
            iters=args.iters,
            inner_repeats=1,
            label=f"sm120_{combine_dtype}_compute_graph_m{grouped_m}",
            nvtx=args.nvtx,
            dist=dist,
        )
        timing = _aggregate_samples(local_samples, dist)
        peak_allocated, peak_reserved = _peak_memory_gib(dist, device)
        completed = True
        return PointResult(
            timing=timing,
            setup_max_s=setup_max_s,
            peak_allocated_gib=peak_allocated,
            peak_reserved_gib=peak_reserved,
        )
    except BaseException:
        # Do not let MoEEpMegaLayer.__del__ invoke a rank-collective NVSHMEM
        # teardown while torchrun is propagating a one-rank failure.
        _abandon_layer_after_failure(layer)
        raise
    finally:
        del inputs
        # The successful path is rank-aligned by the stats all-reduces above,
        # so collective workspace destruction is safe.
        if completed:
            _cleanup_layer(layer)


def _run_k3_point(
    args,
    grouped_m: int,
    *,
    combine_dtype: str,
    dist,
    device,
) -> PointResult:
    import torch
    from cuda.bindings import driver as cuda

    # Importing the shim establishes the raw drop's module search path.
    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        bootstrap_paths,
    )

    bootstrap_paths()
    from moe_sm120_nvfp4_split.api import compile_combine_reduce

    tokens = grouped_m // args.top_k
    combine_output = None
    combine_sf = None
    reduced_output = None
    compiled = None
    executor = None
    combine_cute = None
    combine_sf_cute = None
    reduced_cute = None
    score_cute = None
    k3_stream = None
    runtime = None
    torch.cuda.reset_peak_memory_stats()
    setup_start = perf_counter()
    try:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + grouped_m)
        combine_format = "bf16"
        if combine_dtype == "bf16":
            combine_output = torch.empty(
                (tokens, args.top_k, args.hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            combine_output.normal_(mean=0.0, std=0.05, generator=generator)
        else:
            combine_format = "16e2m1xbf16"
            combine_output = torch.randint(
                0,
                256,
                (tokens, args.top_k, args.hidden // 2),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ).view(torch.float4_e2m1fn_x2)
            combine_sf = torch.ones(
                (tokens, args.top_k, args.hidden // 16),
                dtype=torch.bfloat16,
                device=device,
            )
        reduced_output = torch.empty(
            (tokens, args.hidden), dtype=torch.bfloat16, device=device
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        (
            compiled,
            combine_cute,
            combine_sf_cute,
            reduced_cute,
            score_cute,
            k3_stream,
        ) = compile_combine_reduce(
            combine_output,
            reduced_output,
            None,
            combine_sf=combine_sf,
            combine_format=combine_format,
            stream=stream,
        )
        executor = compiled.to(None)
        runtime = dict(
            combine_cute=combine_cute,
            combine_sf_cute=combine_sf_cute,
            reduced_cute=reduced_cute,
            topk_score_cute=score_cute,
            stream=k3_stream,
        )
        executor(**runtime)
        torch.cuda.synchronize()
        probe_rows = _probe_token_rows(tokens, device)
        actual = reduced_output.index_select(0, probe_rows)
        expected = torch.zeros_like(actual, dtype=torch.float32)
        if combine_dtype == "bf16":
            combine_probe = combine_output.index_select(0, probe_rows)
            for topk_idx in range(args.top_k):
                expected.add_(combine_probe[:, topk_idx].float())
        else:
            packed_probe = combine_output.view(torch.uint8).index_select(0, probe_rows)
            sf_probe = combine_sf.index_select(0, probe_rows).float()
            decode_table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
                device=device,
            )
            low = decode_table[(packed_probe & 0x0F).long()]
            high = decode_table[(packed_probe >> 4).long()]
            decoded = torch.stack((low, high), dim=-1).flatten(-2)
            decoded.mul_(sf_probe.repeat_interleave(16, dim=-1) / 6.0)
            for topk_idx in range(args.top_k):
                expected.add_(decoded[:, topk_idx])
        expected = expected.to(torch.bfloat16)
        local_ok = bool(torch.isfinite(actual).all().item()) and torch.equal(
            actual,
            expected,
        )
        _collective_assert(
            local_ok,
            f"standalone {combine_dtype} K3 reference check for M={grouped_m}",
            dist,
            device,
        )
        del probe_rows, actual, expected
        if combine_dtype == "bf16":
            del combine_probe
        else:
            del packed_probe, sf_probe, decode_table, low, high, decoded
        setup_max_s = _all_reduce_max(perf_counter() - setup_start, dist, device)

        local_samples = _time_calls(
            lambda: executor(**runtime),
            warmup=args.warmup,
            iters=args.iters,
            inner_repeats=args.k3_inner_repeats,
            label=f"sm120_{combine_dtype}_k3_reduce_m{grouped_m}",
            nvtx=args.nvtx,
            dist=dist,
        )
        timing = _aggregate_samples(local_samples, dist)
        peak_allocated, peak_reserved = _peak_memory_gib(dist, device)
        return PointResult(
            timing=timing,
            setup_max_s=setup_max_s,
            peak_allocated_gib=peak_allocated,
            peak_reserved_gib=peak_reserved,
        )
    finally:
        executor = None
        compiled = None
        runtime = None
        combine_cute = None
        combine_sf_cute = None
        reduced_cute = None
        score_cute = None
        k3_stream = None
        combine_output = None
        combine_sf = None
        reduced_output = None
        gc.collect()
        torch.cuda.empty_cache()


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def _result_row(
    args,
    result: PointResult,
    *,
    target: str,
    combine_dtype: str,
    grouped_m: int,
    world_size: int,
    gpu_name: str,
    knobs: dict | None,
) -> dict[str, object]:
    tokens = grouped_m // args.top_k
    p50_us = result.timing.critical_p50_us
    fc1_flops = 4 * grouped_m * args.hidden * args.intermediate
    fc2_flops = 2 * grouped_m * args.hidden * args.intermediate
    total_flops = fc1_flops + fc2_flops
    combine_bytes = _combine_plane_bytes(grouped_m, args.hidden, combine_dtype)
    k3_io_bytes = combine_bytes + tokens * args.hidden * 2

    is_graph = target == "compute_graph"
    is_k3 = target == "k3_standalone"
    return {
        "record": "BENCH_CSV",
        "kernel": KERNEL_NAME,
        "target": target,
        "combine_dtype": combine_dtype,
        "status": "pass",
        "gpu_name": gpu_name,
        "world_size": world_size,
        "total_experts": args.num_experts,
        "local_experts": args.num_experts // world_size,
        "hidden": args.hidden,
        "intermediate_post_swiglu": args.intermediate,
        "topk": args.top_k,
        "grouped_gemm_m_per_rank": grouped_m,
        "tokens_per_rank": tokens,
        "expected_rows_per_expert": _format_metric(
            grouped_m * world_size / args.num_experts
        ),
        "combine_plane_gib_per_rank": _format_metric(
            _combine_plane_gib(grouped_m, args.hidden, combine_dtype)
        ),
        "warmup": args.warmup,
        "iters": args.iters,
        "inner_repeats": 1 if is_graph else args.k3_inner_repeats,
        "critical_min_us": _format_metric(result.timing.critical_min_us),
        "critical_p50_us": _format_metric(result.timing.critical_p50_us),
        "critical_p90_us": _format_metric(result.timing.critical_p90_us),
        "critical_p99_us": _format_metric(result.timing.critical_p99_us),
        "critical_max_us": _format_metric(result.timing.critical_max_us),
        "critical_mean_us": _format_metric(result.timing.critical_mean_us),
        "critical_std_us": _format_metric(result.timing.critical_std_us),
        "rank_p50_min_us": _format_metric(result.timing.rank_p50_min_us),
        "rank_p50_max_us": _format_metric(result.timing.rank_p50_max_us),
        "rank_p50_imbalance": _format_metric(result.timing.rank_p50_imbalance),
        "global_tokens_per_s": _format_metric(tokens * world_size * 1e6 / p50_us),
        "routed_rows_per_rank_per_s": _format_metric(grouped_m * 1e6 / p50_us),
        "fc1_flops_per_rank": fc1_flops if is_graph else "",
        "fc2_flops_per_rank": fc2_flops if is_graph else "",
        "total_gemm_flops_per_rank": total_flops if is_graph else "",
        "effective_gemm_tflops_per_rank": (
            _format_metric(total_flops / p50_us / 1e6) if is_graph else ""
        ),
        "k3_io_bytes_per_rank": k3_io_bytes if is_k3 else "",
        "effective_k3_gbytes_per_s": (
            _format_metric(k3_io_bytes / p50_us / 1e3) if is_k3 else ""
        ),
        "setup_max_s": _format_metric(result.setup_max_s),
        "torch_peak_allocated_gib": _format_metric(result.peak_allocated_gib),
        "torch_peak_reserved_gib": _format_metric(result.peak_reserved_gib),
        "knobs_json": json.dumps(knobs, sort_keys=True) if knobs else "",
    }


def _validate_same_numa_topology(rank: int, local_rank: int, dist) -> None:
    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        bootstrap_paths,
    )

    bootstrap_paths()
    from moe_sm120_nvfp4_split.runtime.gpu_topology import (
        derive_ep_transport_topology,
        discover_gpu_topology,
        format_ep_topology,
    )

    try:
        local = discover_gpu_topology(local_rank, rank)
        local_error = ""
    except Exception as exc:  # noqa: BLE001 - report the failure on every rank
        local = None
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"

    gathered: list[tuple[object | None, str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (local, local_error))
    discovery_errors = [item[1] for item in gathered if item is not None and item[1]]
    if discovery_errors:
        raise RuntimeError(
            "cannot verify the required same-NUMA topology: "
            + "; ".join(discovery_errors)
        )

    topology = derive_ep_transport_topology(
        [item[0] for item in gathered if item is not None],
        ep_rank=rank,
    )
    if topology.cross_numa_peer_count:
        raise RuntimeError(
            "this benchmark supports only one-NUMA EP groups, but discovered "
            f"{format_ep_topology(topology)}"
        )
    if rank == 0:
        print(
            f"# verified topology: {format_ep_topology(topology)}",
            file=sys.stderr,
            flush=True,
        )


def _validate_runtime_topology(args, torch, dist) -> tuple[int, int, int, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    missing = [name for name in required_env if name not in os.environ]
    if missing:
        raise RuntimeError(
            "launch this benchmark with torchrun; missing environment variables "
            f"{missing}"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    single_rank = args.allow_single_rank and world_size == 1
    if world_size not in (2, 4) and not single_rank:
        raise RuntimeError(
            "this benchmark requires EP world size 2 or 4; "
            f"got {world_size} (use --allow-single-rank for diagnostics)"
        )
    if local_world_size != world_size:
        raise RuntimeError(
            "all EP ranks must be on one node/NUMA domain: "
            f"LOCAL_WORLD_SIZE={local_world_size}, WORLD_SIZE={world_size}"
        )
    if args.num_experts % world_size:
        raise RuntimeError(
            f"num-experts={args.num_experts} is not divisible by EP{world_size}"
        )
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"EP{world_size} performance requires {world_size} physical visible GPUs; "
            f"only {torch.cuda.device_count()} are visible"
        )
    if torch.cuda.get_device_capability(local_rank)[0] != 12:
        raise RuntimeError(
            "the target kernels require SM120/SM121; current capability is "
            f"{torch.cuda.get_device_capability(local_rank)}"
        )
    if not single_rank:
        inaccessible = [
            peer
            for peer in range(world_size)
            if peer != local_rank
            and not torch.cuda.can_device_access_peer(local_rank, peer)
        ]
        if inaccessible:
            raise RuntimeError(
                f"GPU {local_rank} lacks peer access to visible GPUs {inaccessible}; "
                "select same-NUMA peer-accessible GPUs with CUDA_VISIBLE_DEVICES"
            )
        _validate_same_numa_topology(rank, local_rank, dist)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def main() -> int:
    args = _parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    m_values = _parse_m_values(args.grouped_gemm_m_values, args.top_k)
    _validate_static_args(args, m_values)
    knobs = _parse_knobs(args.knobs_json)
    if args.list_cases:
        _print_cases(args, m_values)
        return 0
    requested_world_size = int(os.environ.get("WORLD_SIZE", "0"))
    single_rank = args.allow_single_rank and requested_world_size == 1
    if single_rank:
        # The production single-rank runtime deliberately bypasses NVSHMEM.
        # Set this before importing flashinfer.moe_ep.
        os.environ["MEGA_NO_DIST"] = "1"
    elif args.target in ("graph", "both") and requested_world_size > 1:
        try:
            no_dist = bool(int(os.environ.get("MEGA_NO_DIST", "0")))
        except ValueError as exc:
            raise ValueError("MEGA_NO_DIST must be an integer flag") from exc
        if no_dist:
            raise RuntimeError(
                "MEGA_NO_DIST=1 is invalid for an EP2/EP4 graph benchmark; "
                "unset it so NVSHMEM peer communication is measured"
            )
    if args.target in ("graph", "both"):
        _validate_graph_dependencies(single_rank=single_rank)

    import torch
    import torch.distributed as dist

    rank, world_size, _local_rank, device = _validate_runtime_topology(
        args, torch, dist
    )
    gpu_name = torch.cuda.get_device_name(device)
    sink = CsvSink(args.output_csv) if rank == 0 else None
    runtime = None
    transformed_weights = None
    run_succeeded = False
    try:
        if sink is not None:
            sink.write_header()
            print(
                f"# {KERNEL_NAME}: EP{world_size}, "
                f"{args.num_experts // world_size} local experts/rank, "
                f"GPU={gpu_name!r}",
                file=sys.stderr,
                flush=True,
            )
            if (
                args.target in ("graph", "both")
                and args.hidden == DEFAULT_HIDDEN
                and args.intermediate == DEFAULT_INTERMEDIATE
                and args.top_k == DEFAULT_TOP_K
                and args.num_experts == DEFAULT_NUM_EXPERTS
                and max(m_values) >= max(GROUPED_GEMM_M_VALUES)
            ):
                rough_live_gib = 18 if world_size == 2 else 24
                print(
                    f"# max-case memory guide: roughly {rough_live_gib} GiB/rank "
                    "of live buffers before JIT/runtime overhead; the 16 GiB "
                    "NVSHMEM heap may reserve additional memory, and torch_peak_* "
                    "does not include NVSHMEM allocations",
                    file=sys.stderr,
                    flush=True,
                )

        if args.target in ("graph", "both"):
            from flashinfer.moe_ep import (
                BootstrapConfig,
                bootstrap_moe_ep_runtime,
                ensure_moe_ep_cuda_device,
                finalize_moe_ep_runtime,
            )
            from flashinfer.moe_ep.core.runtime import (
                nvfp4_cutedsl_runtime_requirements,
            )

            weight_start = perf_counter()
            transformed_weights = _make_transformed_weights(
                args,
                local_experts=args.num_experts // world_size,
                rank=rank,
            )
            weight_setup_s = _all_reduce_max(
                perf_counter() - weight_start, dist, device
            )
            if rank == 0:
                print(
                    f"# transformed NVFP4 weights ready in {weight_setup_s:.3f} s "
                    "(max rank)",
                    file=sys.stderr,
                    flush=True,
                )

            bootstrap = BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
                device=device.index,
            )
            ensure_moe_ep_cuda_device(bootstrap)
            runtime = bootstrap_moe_ep_runtime(
                bootstrap, nvfp4_cutedsl_runtime_requirements(bootstrap)
            )
            for combine_dtype in _selected_combine_dtypes(args):
                for grouped_m in m_values:
                    if rank == 0:
                        print(
                            f"# [graph/{combine_dtype}] M/rank={grouped_m}, "
                            f"tokens/rank={grouped_m // args.top_k}, "
                            "combine="
                            f"{_combine_plane_gib(grouped_m, args.hidden, combine_dtype):.3f} GiB",
                            file=sys.stderr,
                            flush=True,
                        )
                    result = _run_graph_point(
                        args,
                        grouped_m,
                        combine_dtype=combine_dtype,
                        rank=rank,
                        world_size=world_size,
                        bootstrap=bootstrap,
                        transformed_weights=transformed_weights,
                        knobs=knobs,
                        dist=dist,
                        device=device,
                    )
                    if sink is not None:
                        sink.write_row(
                            _result_row(
                                args,
                                result,
                                target="compute_graph",
                                combine_dtype=combine_dtype,
                                grouped_m=grouped_m,
                                world_size=world_size,
                                gpu_name=gpu_name,
                                knobs=knobs,
                            )
                        )

            dist.barrier()
            finalize_moe_ep_runtime(runtime)
            runtime = None
            dist.barrier()
            del transformed_weights
            transformed_weights = None
            gc.collect()
            torch.cuda.empty_cache()

        if args.target in ("k3", "both"):
            for combine_dtype in _selected_combine_dtypes(args):
                for grouped_m in m_values:
                    if rank == 0:
                        print(
                            f"# [k3/{combine_dtype}] M/rank={grouped_m}, "
                            f"tokens/rank={grouped_m // args.top_k}, "
                            "input="
                            f"{_combine_plane_gib(grouped_m, args.hidden, combine_dtype):.3f} GiB",
                            file=sys.stderr,
                            flush=True,
                        )
                    result = _run_k3_point(
                        args,
                        grouped_m,
                        combine_dtype=combine_dtype,
                        dist=dist,
                        device=device,
                    )
                    if sink is not None:
                        sink.write_row(
                            _result_row(
                                args,
                                result,
                                target="k3_standalone",
                                combine_dtype=combine_dtype,
                                grouped_m=grouped_m,
                                world_size=world_size,
                                gpu_name=gpu_name,
                                knobs=knobs,
                            )
                        )
        run_succeeded = True
    finally:
        if runtime is not None and run_succeeded:
            from flashinfer.moe_ep import finalize_moe_ep_runtime

            with contextlib.suppress(Exception):
                finalize_moe_ep_runtime(runtime)
        del transformed_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if sink is not None:
            sink.close()
        if dist.is_initialized() and run_succeeded:
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
