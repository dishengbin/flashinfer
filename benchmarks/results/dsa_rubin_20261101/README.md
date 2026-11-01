# DSA sparse MLA on Rubin — 2026-11-01

Reference: [FlashInfer PR #7984](https://github.com/flashinfer-ai/flashinfer/pull/7984)
(`335e25d0b8d95bf42709d6851c569dddc26577d6`). The baseline is the current
Blackwell implementation at checkout `64c816a749996c80ceff3cf6d92e3670890e8b4d`,
including later D576 reload improvements, with the pre-existing SM107 eligibility
change. This is not a Blackwell-versus-Rubin hardware comparison.

## Scope and environment

- DSA **attention**, given selected indices; indexer/top-k generation, input
  quantization and query RoPE are outside timing.
- D576 QK / D512 V, top-k 2048, 131072 physical KV rows. Native E4M3 Q/KV or
  NVFP4 NoPE + E4M3 RoPE (352 bytes/record); BF16 output and natural-log LSE.
- The native FP8 path is non-block-scaled, equivalent to per-tensor FP8 with
  unit input descales: Q/KV are directly cast to E4M3, the attention scale is
  `1/sqrt(576)`, and the output multiplier is one. Dynamic quantization, amax
  and scale generation are not measured. MXFP8 block scaling is not tested.
- Seed 7984, iid inputs, random physical indices with duplicates allowed by the
  API. This synthetic workload does not replace production routing distributions.
- Rubin SM107, 212 SMs, 126 MiB L2, 285227 MiB usable memory reported by PyTorch;
  driver 620.62. Four GPUs were available. Each JSON includes the physical GPU
  UUID, and each reported comparison uses the same GPU.
- Native ARM Python 3.12, PyTorch 2.11.0+cu130, CuTe DSL 4.8.1, native CUDA
  13.4.92 compiler/NVRTC, CUPTI Python 13.4.0. Runtime/driver/compiler versions
  intentionally differ; the exact environment is in `dsa-rubin-packages.txt`.
- CUPTI GPU span of the complete prepared CUDA graph, including any split
  reducer, 30 samples, explicit 2x-L2 flush before each iteration. Compilation,
  allocation and FP64 reference checks are excluded. Clocks were not locked.
- Initial `rotated` runs are retained separately for diagnostics. Rotation based
  on allocated KV-pool size can under-flush sparse working sets at small batch
  sizes; final comparisons use the explicit-flush CUPTI runs only.
- Python source was mirrored to `/tmp/dsa-rubin-source` to avoid NFS import
  latency. Kernel source content is identical to the working checkout.

## Reproduction

The supplied environment script selects the isolated tools installed under
`/tmp/dsa-rubin-venv` on this node. It is a session-local environment, not a
portable installation script. From the repository root:

```bash
source /tmp/dsa-rubin-env.sh
python benchmarks/bench_dsa_rubin.py --cases 1,1,64 4,8,32 1,8192,16 --blackwell-policy --json /tmp/baseline.json
python benchmarks/bench_dsa_rubin.py --cases 1,1,64 1,8192,24 1,8192,64 --json /tmp/rubin.json
python benchmarks/bench_dsa_rubin.py --cases 1,1,16 1,8192,16 --fp8-mma-k 64 --json /tmp/k64.json
python benchmarks/bench_dsa_rubin.py --cache-format nvfp4.fp8rope --cases 1,1,16 64,4,48 1,8192,24 1,8192,96 --json /tmp/nvfp4.json
python benchmarks/bench_dsa_rubin.py --cases 1,1,64 1,8,64 --split-kv 16 --v-split 4 --json /tmp/split.json
pytest -q tests/attention/test_rubin_sparse_mla.py
```

Cases are `B,Sq,H`. Research overrides are confined to the benchmark. A K64
request retains K32 when the selected schedule is ineligible; the actual MMA
mode is recorded in each row's `profile`, and must be checked before calling
that row a K32/K64 comparison.

## Constraints and validation

Rubin FP8 K64 is not supported on half-datapath MMA shapes (M64/1CTA,
M128/2CTA) or `.ws`. An initial M64 K64 probe raised an illegal instruction;
that process was terminated. Subsequent experiments use full-datapath Swap-AB
shapes, and the descriptor builder rejects M64 before GPU execution. The
existing Keep-AB/2CTA/WS schedules retain K32.

The implementation doubles FP8 K steps and adjusts QK SMEM descriptor offsets,
PV SMEM and TMEM increments and descriptor K-dimension encoding together.
K32 remains the default. Numerical validation uses an independent FP64 oracle,
the existing E4M3 probability-rounding error bound, and natural-log LSE checks.
Twelve graph tests passed, including holes, ragged and empty rows, live route
updates and padded page strides; an additional test rejects half-datapath K64.

## Measurements

All times below are median microseconds. Qualification compares the exact
candidate configurations before automatic dispatch was added. The final H64
rows were measured again on the final code, with `--blackwell-policy` versus
automatic Rubin dispatch, 100 samples each. Every comparison uses the same GPU;
all raw JSON files preserve the UUID, sample distribution and correctness result.

### Qualified and integrated policies

| B,Sq,H | Blackwell policy (us) | Rubin policy (us) | Speedup | Latency reduction |
|---|---:|---:|---:|---:|
| 1,1,64 | 8.464 | 7.729 | 1.095x | 8.7% |
| 2,1,64 | 8.608 | 7.936 | 1.085x | 7.8% |
| 1,1,128 | 8.960 | 8.160 | 1.098x | 8.9% |
| 1,8192,24 | 924.079 | 855.870 | 1.080x | 7.4% |
| 1,8192,48 | 974.801 | 903.487 | 1.079x | 7.3% |
| 1,8192,64 | 1000.945 | 931.711 | 1.074x | 6.9% |

Tiny decode switches from 8x4 to 16x4 KV/V partitions only for one or two
64-head work tiles, top-k 2048, and 212-SM SM107. This doubles split scratch:
H64 B1 uses 1,052,928 bytes versus 526,592 bytes. H64 B2 and H128 B1 also
benefit. Four work tiles regress (9.153 -> 11.680 us), so they retain the
baseline. The A/B/A control returned to 8.416/8.608/8.896 us for H64 B1/B2
and H128 B1, confirming repeatability.

Sustained native D576/FP8 Keep-AB CLC grids with eight gather warps use eleven
retained stages instead of ten. One extra D128 slice can prefetch beyond two
complete D576 tiles. The H64 A/B/A control was 999.280 -> 930.032 -> 999.136 us.
The global workspace is unchanged; per-CTA shared memory increases by one stage.
Selection is limited to the qualified 212-SM, capacity-2048, >=8-SM-wave region,
with native FP8, no extra pool, no sinks and no valid-prefix specialization.

### FP8 K32 versus legal K64

| Format | B,Sq,H | K32 (us) | K64 (us) | Speedup |
|---|---|---:|---:|---:|
| native | 1,1,16 | 9.216 | 9.121 | 1.010x |
| native | 1,8192,16 | 788.237 | 775.948 | 1.016x |
| nvfp4.fp8rope | 1,1,16 | 10.528 | 10.496 | 1.003x |

These are 0.3%-1.6% measured changes rather than an operator-level 2x gain.
K64 remains an explicit benchmark experiment. M64/1CTA, M128/2CTA and WS
schedules are ineligible; requested-but-ineligible K64 runs record their actual
K32 MMA mode. For example H32 B4 Sq8 remains K32 and is not evidence of a
K64 speedup. The results suggest that gather/pipeline overlap is a more useful
optimization target for these shapes than increasing MMA K alone.

### Blackwell implementation baseline on Rubin

These measurements use the checkout's unretuned scheduling policy with CUPTI
and explicit cold L2; they do not imply Blackwell hardware measurements.

| Format | B,Sq,H | Baseline (us) |
|---|---|---:|
| native | 1,1,16 | 9.248 |
| native | 1,1,64 | 8.416 |
| native | 1,8,64 | 11.057 |
| native | 4,8,32 | 16.416 |
| native | 64,4,48 | 57.185 |
| native | 64,8,96 | 70.993 |
| native | 1,8192,16 | 792.205 |
| native | 1,8192,24 | 925.487 |
| native | 1,8192,64 | 1000.720 |
| native | 1,8192,96 | 1085.138 |
| nvfp4.fp8rope | 1,1,16 | 10.528 |
| nvfp4.fp8rope | 64,4,48 | 73.729 |
| nvfp4.fp8rope | 1,8192,24 | 1266.037 |
| nvfp4.fp8rope | 1,8192,96 | 1683.915 |

### Other trials

- Raising KV splits to 2/4/8/16 with one V owner did not improve the tested
  grids. At 96 H64 rows the default was 26.128 us; 2/4/8/16 splits measured
  26.672/34.193/49.184/69.345 us.
- Sixteen splits with two V owners measured 8.768 us at H64 B1, versus the
  8.416-8.448 us baseline; this was not selected.
- A forced nonpersistent eight-loader retained-KV schedule regressed H64
  96-row decode and 8192-row prefill to 32.017 and 1101.874 us. It was not
  selected; the eleven-stage optimization preserves the original CLC schedule.
- The initial input-rotation measurements and rejected M64/K64 probe are
  diagnostic runs only, not the final comparison methodology.

## Validation and review

- 12 K32/K64 numerical and graph cases passed on Rubin.
- The half-datapath descriptor rejection check passed.
- All five final automatic-dispatch graph cases passed, including H64 B1/B2,
  H128 B1, H64 B4 fallback, and a 2048-query retained prefill with holes,
  empty rows and changed routes after capture.
- An existing BF16 H64/160-row/top-k 513 test passed with holes, ragged
  lengths and graph replay, covering unchanged BF16 address stepping.
- Ruff and `git diff --check` passed for the edited sources.

The self-review checked descriptor byte steps, TMEM increments, K64 geometry,
SM107-only dispatch, split workspace sizing, stage capacity and graph-lifetime
semantics. The full 18-case new test file is runnable as shown above; verification
was performed in targeted batches. No outstanding correctness findings remain.
Blackwell GPU hardware was unavailable on this node; its policy is preserved
behind architecture gates and by leaving the original selection functions intact.

[CSV comparison](comparison.csv), [raw CUPTI results](raw/),
[initial diagnostic runs](diagnostics/), [environment](dsa-rubin-env.sh),
[package versions](dsa-rubin-packages.txt),
[K32/K64 tests](dsa-rubin-correctness.log),
[automatic-dispatch tests](dsa-rubin-final-correctness.log),
[BF16 regression](dsa-rubin-bf16-regression.log).

## Architecture reference provenance

The optimization round consulted SM107 DSL implementations and the primitive
examples, including the Rubin-421 FD summary in
`cutlass_ir/compiler/python/examples/CuTeDSL/experimental/primitives/tcgen05/sm107_mma_mixed_fp8_fp4_k64.py`.
The subsequent documentation audit located these additional paths relative to
the user's dynamic-kernel-generator checkout:

- `skills/cutlass-ir/cudeepy/docs/rubin/01_architecture_delta.md`: architecture
  changes summarized from Rubin training slides and transcripts.
- `skills/cutlass-ir/cudeepy/docs/cute_sm107/01_tcgen05_mma_delta.md`: DSL MMA
  operation and collector interfaces.
- `DkgDSL/skills/hw-spec/assets/arch/GR100.yaml`: resource data sourced from dlsim.
- `DkgDSL/skills/hw-spec/assets/mma/105.yaml`: the GR100 instruction-throughput
  table, including the FP8 K64 `2xFP8` variant; the hw-spec package identifies
  KTM as the MMA data source.

These are curated documentation and specification data, not a claim that the
original FD or training slide decks were read in full. This round does not
systematically evaluate all Rubin features, such as the larger shared-memory
carveout, packed MUFU or collector reuse. Qualification uses the measured
212-SM node rather than projected counts in older overview documents.
