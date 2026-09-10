# SM120 NVFP4 x NVFP4 Split MegaMoE

This package implements the SM120 split MegaMoE path with packed E2M1
NVFP4 weights and activations, per-16 E4M3 scale factors, FP32 accumulation,
and BF16 output.

The execution pipeline has three phases:

1. `kernel_dispatch_fc1.py` dispatches routed NVFP4 activation rows, executes
   FC1, applies `fc1_alpha`, SwiGLU and the selected top-k weight, then
   requantizes the result to block-16 NVFP4.
2. `kernel_fc2_combine.py` consumes FC1 ready bundles, executes FC2, applies
   `fc2_alpha`, rounds to BF16, and writes each routed partial output back to
   its source rank as either BF16 or block-scaled NVFP4.
3. `kernel_combine_reduce.py` decodes the selected wire format and reduces the
   top-k partial outputs to BF16.

K1 and K2 execute concurrently in disjoint Green Contexts. K1 publishes FC1
output ready state bundle-by-bundle, so K2 can start before K1 finishes the
whole expert pool. Same-NUMA EP uses direct P2P activation pull and direct
peer-store combine. Cross-NUMA EP keeps local peers on P2P and sends only
cross-NUMA traffic through staged NVSHMEM IBGDA transport.

K1's epilogue processes a complete token tile in two passes. The first keeps
SwiGLU values in consumed accumulator registers and stores each token's four
warp-local maxima in a dense shared layout. The second combines warp-pair
maxima, quantizes, and packs disjoint token rows. Two whole-tile barriers
replace the two barriers per 16-token group; the existing 512-float scratch
allocation accommodates every supported token tile (16/32/64/128). Numerical
operation order and the workspace ABI are unchanged.

The FlashInfer frontend has an experimental BF16 `rank_local_combine` knob,
disabled by default. With this knob, dispatch publishes each source token's
contributor count to its expert ranks and records the local route rows. K2
stores BF16 route results locally. For aligned hidden dimensions of at least
2048, it partitions each row into four segments (1536 elements each at
H=6144). Smaller or unaligned rows use one segment. A GPU completion counter
joins the hidden tiles within each segment; only its last hidden tile updates
the segment's per-token contributor counters. Each lane joins one token's
contributor counter, then a warp ballot selects the last contributors for
reduction. Each warp assigns two winners to separate 16-lane groups, with an
inactive second group for an odd final pair. A warp barrier transfers each
winner's GPU acquire before the cooperative route loads. Local BF16 route
stores omit the peer-store shuffle packing by default; an explicit JIT store
override remains available.
The last contributor reduces the local routes in FP32, rounds the segment
to BF16, and posts it to the source rank. Each lane reduces 16 adjacent BF16
values, and the final
256-element iteration is masked at the segment boundary. This also handles
640-element segments at H=2560 without overlapping the neighboring segment.
This spreads row reduction across K2 tasks and shortens the publication tail.
Route loads are issued in pairs while retaining the original slot
accumulation order; cached word offsets use Int32 when the
complete address range fits, otherwise Int64.
For route pools whose element count reaches the signed 32-bit indexing
limit, K2 widens the local BF16 output row coordinate to Int64 before the
layout multiplies it by the hidden stride. Widening only the resulting
pointer would be too late. This prevents address wraparound when the live
routed rows exceed roughly 349,525 at H=6144, as observed in the larger
EP2/EP4 prefill cases. Smaller pools retain their existing indexing path.

K3 (`kernel_rank_local_combine.py`) reduces the live rank partials with
consecutive scalar BF16 accesses across lanes. This adds one BF16 rounding
compared with the ordinary route-level combine.

All completion counters reside on the local GPU. CTA/warp barriers and
GPU acquire-release counters carry the route writes to each segment's
publisher. After peer stores, a system acquire-release RMW chain on a local
counter joins the completed segments. Its last participant publishes the
ready word with a system-release store; K3 acquires that word before reading
the full rank partial. This does not require PCIe peer atomics. See the PTX
[memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).
There is no standalone aggregation kernel in this graph and no polling in
the K2 epilogue. The standalone route-map aggregator is retained as a
diagnostic reference. Extra workspace holds the BF16 route rows, route map,
and counters. Segment counters add `4 * world_size * max_tokens_per_rank`
Int32 words over the single-segment layout (386,880 bytes at EP4/T=6045);
cache ABI 8 separates this workspace layout from older compiled kernels.
The opt-in path requires direct P2P, token-strided dispatch,
epi-warp BF16 combine, and `world_size <= top_k`.

For the measured 110-SM, same-NUMA EP4 prefill geometry (H=6144,
post-SwiGLU intermediate=2304, 128 experts, top-k=8, DP=TP=1,
2560–11895 tokens/rank), enabling `rank_local_combine` also selects a deeper
native pipeline: K1 stage=5, K2 stage=4, one active dispatch warp, and
ready-queue bundle=12. This preset requires rank caching, (64,128,128) K1/K2
tiles, eight K2 warps, and the 72/38 SM partition. Explicit tuning overrides
take precedence. Other shapes retain their existing selection.

At 6045 tokens/rank (M/rank=48360), three alternating CUDA-event measurements
with 10 warmups and 100 iterations per run gave a median critical-rank graph
latency of 9.893 ms, versus 11.980 ms for the previous tuned configuration
(17.42% lower latency, 21.09% higher throughput). The graph includes reset,
concurrent K1/K2, and K3; input staging and JIT compilation are excluded.
A subsequent K2 reduction change processes two tokens per warp with wider
per-lane vectors. Three interleaved before/after CUDA-event runs (20 warmups,
200 iterations each) reduced median critical-rank graph latency from 9.896 ms
to 9.652 ms: 2.46% lower latency and 2.52% higher throughput, reaching
2.505 million global tokens/s and 425.54 effective GEMM TFLOP/s per rank.
The pipeline stages and other tuning knobs were held fixed. Across
M/rank=20480, 48360, and 95160, all 503,808,000 BF16 output elements on four
ranks were bitwise identical to the preceding rank-local implementation.

The current K1 implementation reuses the otherwise idle FC1 auxiliary warp
for the single dispatcher, reducing its CTA from twelve to eight warps.
This applies to direct-P2P rank-local combine with `dispatch_warps=1` and
the (64,128,128) K1 tile. The four compute warps keep the same MMA order,
but preload the next K128 tile's first K64 fragment before finishing the
current tile's second MMA. A warp barrier precedes shared-stage release;
the final tile is peeled to avoid a reload branch in every iteration.
The compact CTA uses static register allocation for its mixed-role warp
group. In the measured toolchain, K1 uses 216 registers/thread and no
stack frame; the original twelve-warp prefetch prototype spilled and was
rejected. Both production kernels still run one CTA per SM.

With K1/K2 stages 5/4, a controlled native comparison against the previous
5/3 implementation at 6045 tokens/rank reduced EP2 latency from 9.773 to
9.430 ms (3.50%) and EP4 from 9.672 to 9.361 ms (3.22%). Measurements use
20 warmups and 200 iterations; three preceding alternating EP4 prototype
pairs also exceeded 3% by their median P50. Across 2560, 6045, 40950, and
81450 tokens/rank, the final EP2/EP4 comparison improved latency by
2.39–3.74%; large EP4 cases improved about 2.6%. These are complete compute
graph measurements under the devices' existing 350 W power limits, not
clock-normalized or model end-to-end gains. All 24 final rank outputs
(4,829,368,320 BF16 elements) were bitwise equal to the preceding source.
Globaltimer builds execute the same new register pipeline and retain
sampled TMA wait counters; diagnostic timings are excluded from gains.
EP2 and shapes outside the existing EP4 preset require explicit 5/4 knobs
to reproduce this configuration; the preset's scope is unchanged.

No globaltimer or profiler instrumentation is needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  benchmarks/bench_moe_ep_sm120_nvfp4_mega.py \
  --grouped-gemm-m-values 48360 --combine-dtype bf16 --target graph \
  --knobs-json '{"rank_local_combine": true}' --warmup 20 --iters 200
```

The `k1_stages` knob independently overrides the K1 AB pipeline depth and is
included in the compile cache key. `None` uses the heuristic/kernel default;
the kernel checks the requested depth against its shared-memory capacity.
Explicit K1 depths above two require `rank_local_combine` for multi-rank
execution: the ordinary route-level combine path failed replay consistency
with the deeper pipeline and is excluded from this optimization.

## Data contract

- Activation and FC1/FC2 weights: packed `torch.float4_e2m1fn_x2` (two logical
  E2M1 elements per byte).
- Activation and weight scales: one `torch.float8_e4m3fn` value per 16 K
  elements.
- FC1 handoff: packed E2M1 data plus per-16 E4M3 scales.
- Accumulator: FP32; reduced output: BF16.
- Combine wire format: BF16 by default, or packed E2M1 plus one BF16 amax per
  16 hidden values (`16e2m1xbf16`). The NVFP4 plane occupies 0.625 bytes per
  hidden value versus 2 bytes for BF16.
- `hidden` must be divisible by 32 and `intermediate` by 64.

The numerical order is:

```text
K1: FP32 accumulator -> fc1_alpha -> SwiGLU -> top-k weight
    -> per-16 E4M3 scale + packed E2M1
K2 BF16: FP32 accumulator -> fc2_alpha -> BF16 -> source-rank output
K2 NVFP4: FP32 accumulator -> fc2_alpha -> BF16 -> per-16 BF16 amax
           -> E2M1(value / (amax / 6)) -> source-rank data + scale planes
K3: optional E2M1 dequantization by (amax / 6) -> FP32 top-k reduction -> BF16
```

`fc1_alpha`, `fc2_alpha`, and `fc1_norm_const` are explicit per-expert inputs.
They are part of the NVFP4 numerical contract and must not be folded into a
different stage without updating the reference implementation.

## Validated environment

- Python 3.12
- `nvidia-cutlass-dsl==4.6.0`
- CUDA Toolkit 13.3
- NVSHMEM 3.7.0 and NVSHMEM4Py 0.3.1
- SM120 RTX Pro 5000

The same-NUMA EP4 path requires CUDA peer access and an NVSHMEM symmetric GPU
heap. The cross-NUMA hybrid path additionally requires the matching NVSHMEM
3.7.0 headers/device bitcode and an IBGDA-capable NIC/GDR stack; its
compatibility guard rejects an unknown NVSHMEM device ABI.

## Run

From the repository root, run the validated DSV4-flash EP4 correctness case:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVSHMEM_HEAP_KIND=VIDMEM
export NVSHMEM_SYMMETRIC_SIZE=16G
export MEGA_STRICT_TORCH_REF=1

torchrun --standalone --nproc_per_node=4 \
  -m moe_sm120_nvfp4_split.mega_runner \
  --num_tokens_per_rank 2048 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --data_parallel_size 1 \
  --tensor_parallel_size 1 \
  --route_distribution balanced \
  --enable_static_expert_shape \
  --comm_backend p2p_direct \
  --split_launch green_graph
```

Run the same specialization as a CUDA-event benchmark:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m moe_sm120_nvfp4_split.mega_runner \
  --num_tokens_per_rank 2048 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --data_parallel_size 1 \
  --tensor_parallel_size 1 \
  --route_distribution balanced \
  --enable_static_expert_shape \
  --comm_backend p2p_direct \
  --split_launch green_graph \
  --perf_run --skip_ref_check --use_cuda_events \
  --perf_warmup 50 --perf_iters 200
```

`run_mega_tests.sh` covers EP4 P2P reference cases. `run_hybrid_tests.sh`
covers EP8 route distributions plus dynamic/static 100-replay transport
checks.

## Production API and cache

Framework integrations should import `api.py`, not `mega_runner.py`:

1. Construct `MegaMoEProblemSpec`.
2. Call `select_compile_spec(...)` with topology and SM properties.
3. Cache by `MegaMoECompileSpec.cache_key`.
4. Select `SplitKernelBuildOptions.combine_format` as `"bf16"` or
   `"16e2m1xbf16"`.
5. Build with `build_split_kernels(spec)` and allocate the returned local and
   symmetric workspace sizes.

The NVFP4 combine implementation currently supports direct P2P token-back with
`token_back_mode="epi_warps"`; IBGDA and dispatch-warp token-back remain BF16.

The cache ABI includes the NVFP4 dtype/layout specialization. Do not reuse a
W4A8 or W8A8 compiled-kernel cache entry.

`green_graph` is the default production launch mode. `sequential` remains only
for bring-up and debugging.
