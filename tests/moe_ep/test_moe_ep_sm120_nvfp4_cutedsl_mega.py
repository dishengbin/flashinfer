"""SM120 NVFP4 x NVFP4 FlashInfer MegaMoE integration tests."""

from __future__ import annotations

import os

import pytest
import torch


def _packed_e2m1(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randint(
        0,
        256,
        shape,
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    ).view(torch.float4_e2m1fn_x2)


def _problem(
    rank: int,
    world_size: int,
    *,
    tokens: int,
    capacity: int,
):
    from flashinfer.moe_ep import MoEEpTensors, MoEWeightPack

    hidden = 1024
    intermediate = 1024
    experts = 8
    top_k = 2
    local_experts = experts // world_size
    generator = torch.Generator(device="cuda").manual_seed(91 + rank)
    weights = MoEWeightPack(
        _packed_e2m1((local_experts, 2 * intermediate, hidden // 2), generator),
        _packed_e2m1((local_experts, hidden, intermediate // 2), generator),
        torch.ones(
            (local_experts, 2 * intermediate, hidden // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        torch.ones(
            (local_experts, hidden, intermediate // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
    )
    hidden_states = (
        torch.randn(
            (tokens, hidden),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    rows = torch.arange(tokens, device="cuda")
    topk_ids = torch.stack(
        ((rows * 3 + rank) % experts, (rows * 5 + rank + 1) % experts), 1
    ).long()
    inputs = MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=torch.full(
            (tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        ),
    )
    return {
        "capacity": capacity,
        "experts": experts,
        "hidden": hidden,
        "intermediate": intermediate,
        "top_k": top_k,
        "weights": weights,
        "inputs": inputs,
    }


def _make_layer(
    rank: int,
    world_size: int,
    problem: dict,
    *,
    combine_dtype: str = "bf16",
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=problem["experts"],
            max_tokens_per_rank=problem["capacity"],
            token_hidden_size=problem["hidden"],
        ),
        weights=problem["weights"],
        backend=MegaConfig(
            megakernel=Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["top_k"],
                combine_dtype=combine_dtype,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


@pytest.mark.arch_sm120
def test_sm120_nvfp4_single_rank_replay_and_outer_cuda_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    problem = _problem(0, 1, tokens=16, capacity=16)
    outputs = {}
    for combine_dtype in ("bf16", "nvfp4"):
        layer = _make_layer(0, 1, problem, combine_dtype=combine_dtype)
        try:
            layer.warmup(problem["inputs"])
            eager0 = layer(problem["inputs"]).clone()
            eager1 = layer(problem["inputs"]).clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer(problem["inputs"])
            graph.replay()
            replay0 = captured.clone()
            graph.replay()
            replay1 = captured.clone()
            torch.cuda.synchronize()
            assert torch.isfinite(eager0).all()
            torch.testing.assert_close(eager0, eager1, atol=0.0, rtol=0.0)
            torch.testing.assert_close(eager0, replay0, atol=0.0, rtol=0.0)
            torch.testing.assert_close(replay0, replay1, atol=0.0, rtol=0.0)
            outputs[combine_dtype] = eager0
        finally:
            layer.destroy()

    bf16 = outputs["bf16"].float()
    nvfp4 = outputs["nvfp4"].float()
    rel_l2 = torch.linalg.vector_norm(nvfp4 - bf16) / torch.linalg.vector_norm(bf16)
    assert rel_l2.item() < 0.15


@pytest.mark.arch_sm120
def test_sm120_nvfp4_combine_reduce_above_grid_y_limit() -> None:
    """K3 must cover token 65536+ without using CUDA gridDim.y."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    from cuda.bindings import driver as cuda

    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        bootstrap_paths,
    )

    bootstrap_paths()
    from moe_sm120_nvfp4_split.api import compile_combine_reduce

    tokens = 81450
    top_k = 8
    threads = 32
    # Two hidden blocks with a short tail, matching the customer's two-block
    # H=6144 launch while keeping the regression test's allocation modest.
    hidden = 8 * (threads + 1)
    combine_output = torch.zeros(
        (tokens, top_k, hidden), dtype=torch.bfloat16, device="cuda"
    )
    reduced_output = torch.full(
        (tokens, hidden), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    probe_tokens = torch.tensor(
        (0, 65535, 65536, tokens - 1), dtype=torch.int64, device="cuda"
    )
    probe_values = torch.arange(1, 5, dtype=torch.bfloat16, device="cuda")
    combine_output[probe_tokens] = probe_values[:, None, None]

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    (
        compiled,
        combine_cute,
        combine_sf_cute,
        reduced_cute,
        score_cute,
        stream,
    ) = compile_combine_reduce(
        combine_output,
        reduced_output,
        None,
        threads=threads,
        stream=stream,
    )
    executor = compiled.to(None)
    executor(
        combine_cute=combine_cute,
        combine_sf_cute=combine_sf_cute,
        reduced_cute=reduced_cute,
        topk_score_cute=score_cute,
        stream=stream,
    )
    torch.cuda.synchronize()

    actual = reduced_output[probe_tokens]
    expected = (probe_values * top_k)[:, None].expand_as(actual)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.arch_sm120
def test_sm120_nvfp4_combine_reduce_matches_decode_reference() -> None:
    """NVFP4 K3 must decode E2M1 with its per-16 BF16 amax scale."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    from cuda.bindings import driver as cuda

    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        bootstrap_paths,
    )

    bootstrap_paths()
    from moe_sm120_nvfp4_split.api import compile_combine_reduce

    tokens, top_k, hidden = 3, 2, 32
    packed = torch.arange(
        tokens * top_k * (hidden // 2), dtype=torch.uint8, device="cuda"
    ).reshape(tokens, top_k, hidden // 2)
    combine_output = packed.view(torch.float4_e2m1fn_x2)
    combine_sf = (
        torch.arange(
            1,
            tokens * top_k * (hidden // 16) + 1,
            dtype=torch.float32,
            device="cuda",
        )
        .reshape(tokens, top_k, hidden // 16)
        .to(torch.bfloat16)
    )
    reduced_output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    (
        compiled,
        combine_cute,
        combine_sf_cute,
        reduced_cute,
        score_cute,
        stream,
    ) = compile_combine_reduce(
        combine_output,
        reduced_output,
        None,
        combine_sf=combine_sf,
        combine_format="16e2m1xbf16",
        stream=stream,
    )
    compiled.to(None)(
        combine_cute=combine_cute,
        combine_sf_cute=combine_sf_cute,
        reduced_cute=reduced_cute,
        topk_score_cute=score_cute,
        stream=stream,
    )
    torch.cuda.synchronize()

    decode_table = torch.tensor(
        (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6),
        dtype=torch.float32,
        device="cuda",
    )
    low = decode_table[(packed & 0xF).long()]
    high = decode_table[(packed >> 4).long()]
    decoded = torch.stack((low, high), dim=-1).flatten(-2)
    expected = (
        (decoded * combine_sf.float().repeat_interleave(16, dim=-1) * (1.0 / 6.0))
        .sum(dim=1)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(reduced_output, expected, atol=0.0, rtol=0.0)


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_sm120_nvfp4_four_rank_imbalanced_second_epoch() -> None:
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    tokens_by_rank = (17, 16, 7, 1)
    problem = _problem(
        rank,
        world_size,
        tokens=tokens_by_rank[rank],
        capacity=32,
    )
    combine_outputs = {}
    for combine_dtype in ("bf16", "nvfp4"):
        layer = _make_layer(
            rank,
            world_size,
            problem,
            combine_dtype=combine_dtype,
        )
        try:
            outputs = []
            for _ in range(3):
                layer.stage_inputs(
                    problem["inputs"], compile_tokens_per_rank=max(tokens_by_rank)
                )
                outputs.append(layer.compute_staged(output=None).clone())
            torch.cuda.synchronize()
            dist.barrier()
            assert torch.isfinite(outputs[0]).all()
            for output in outputs[1:]:
                torch.testing.assert_close(outputs[0], output, atol=0.0, rtol=0.0)
            combine_outputs[combine_dtype] = outputs[0]
        finally:
            layer.destroy()

    bf16 = combine_outputs["bf16"].float()
    nvfp4 = combine_outputs["nvfp4"].float()
    rel_l2 = torch.linalg.vector_norm(nvfp4 - bf16) / torch.linalg.vector_norm(bf16)
    assert rel_l2.item() < 0.15
