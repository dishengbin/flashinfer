#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Compare K64 and opt-in SM103 K96 FC2 on real fused-MoE FC1 outputs.

Uses the existing FC2 benchmark's operands and CUPTI kernel timing. The
reported all-kernel sum is distinct from whole-MoE elapsed latency. No SF
repacking or extra conversion kernels are needed by the K96 path.

    python benchmarks/bench_cute_dsl_moe_fc2_sm103.py --jsonl /tmp/fc2_k96.jsonl
"""

import argparse
import importlib.metadata
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from bench_cute_dsl_moe_fc2_epilogue import make_activations, make_weights, time_us
from flashinfer import cute_dsl_fused_moe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[9000, 18000, 36000, 54000]
    )
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-stages", nargs="+", type=int, choices=[1, 2], default=[2])
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (10, 3):
        parser.error("requires an SM103 GPU")
    if args.rounds <= 0 or args.iters <= 0:
        parser.error("rounds and iters must be positive")
    configs = [
        (stages, warps, swizzle, True)
        for stages in args.c_stages
        for warps, swizzle in [(2, 2), (4, 2), (4, 4), (2, 4)]
    ]
    choices = [(enabled, cfg) for enabled in [False, True] for cfg in configs]
    weights = make_weights(384, 4096, 768)
    rng = random.Random(args.seed)
    environment = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        experts=384,
        hidden=4096,
        inter=768,
        top_k=20,
        timer="CUPTI via torch.profiler; all_kernel_sum_us is not elapsed latency",
        rounds=args.rounds,
        iters=args.iters,
        seed=args.seed,
        configs=configs,
    )
    print(json.dumps(environment), flush=True)
    output = args.jsonl.open("x") if args.jsonl else None
    if args.jsonl:
        args.jsonl.with_suffix(".env.json").write_text(
            json.dumps(environment, indent=2)
        )
    try:
        for tokens in args.tokens:
            inputs = dict(weights, **make_activations(tokens, 384, 4096, 20))

            def run(enabled, config):
                return cute_dsl_fused_moe(
                    **inputs,
                    num_experts=384,
                    top_k=20,
                    quant_mode="w4a4",
                    tactic=(
                        256,
                        ((256, 256), (2, 1), False),
                        ((256, 256), (2, 1), True, (*config, enabled)),
                    ),
                )

            reference = run(False, configs[0]).float().clone()

            def check(enabled, config):
                actual = run(enabled, config).float()
                torch.cuda.synchronize()
                relative = ((actual - reference).norm() / reference.norm()).item()
                if not torch.isfinite(actual).all().item() or relative >= 0.02:
                    raise AssertionError((tokens, enabled, config, relative))
                return relative

            before = {choice: check(*choice) for choice in choices}
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                for choice in choices:
                    run(*choice)
                torch.cuda.synchronize()
            samples = {choice: [] for choice in choices}
            for _ in range(args.rounds):
                order = choices.copy()
                rng.shuffle(order)
                for choice in order:
                    values = time_us(lambda: run(*choice), args.iters)
                    if not all(math.isfinite(x) and x > 0 for x in values):
                        raise AssertionError(values)
                    samples[choice].append(values)
            for enabled, config in choices:
                values = samples[enabled, config]
                record = dict(
                    tokens=tokens,
                    variant="k96" if enabled else "k64",
                    epilogue=config,
                    fc2_us=statistics.median(v[0] for v in values),
                    all_kernel_sum_us=statistics.median(v[1] for v in values),
                    samples=values,
                    relative_l2_before=before[enabled, config],
                    relative_l2_after=check(enabled, config),
                )
                text = json.dumps(record)
                print(text, flush=True)
                if output:
                    output.write(text + "\n")
                    output.flush()
            reference = None
            inputs = None
            torch.cuda.empty_cache()
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
