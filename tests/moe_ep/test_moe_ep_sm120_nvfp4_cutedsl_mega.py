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
    hidden: int = 1024,
    intermediate: int = 1024,
    experts: int = 8,
    top_k: int = 2,
):
    from flashinfer.moe_ep import MoEEpTensors, MoEWeightPack

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
        tuple((rows * (2 * slot + 3) + rank + slot) % experts for slot in range(top_k)),
        1,
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
    knobs: dict | None = None,
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
                knobs=knobs,
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
@pytest.mark.parametrize(
    ("production_shape", "hidden", "compact_k1"),
    [
        (False, 1024, False),
        (False, 2048, False),
        (False, 2304, False),
        # Four 640-element segments need a masked final 256-element reduction.
        (False, 2560, False),
        (True, 6144, False),
        (True, 6144, True),
    ],
)
def test_sm120_nvfp4_rank_local_combine_routing_and_replay(
    production_shape, hidden, compact_k1
) -> None:
    """Rank reduction must handle missing ranks and changing duplicate routes."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, 2, 4):
        pytest.skip("requires one, two, or four ranks")
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    tokens_by_rank = (6045, 6044, 3001, 1) if production_shape else (17, 16, 7, 1)
    tokens = tokens_by_rank[rank]
    problem = _problem(
        rank,
        world_size,
        tokens=tokens,
        capacity=6045 if production_shape else 32,
        hidden=hidden,
        intermediate=2304 if production_shape else 1024,
        experts=128 if production_shape else 8,
        top_k=8,
    )
    inputs = problem["inputs"]
    rows = torch.arange(tokens, device="cuda")[:, None]
    slots = torch.arange(8, device="cuda")[None, :]
    inputs.topk_weights = torch.full(
        (tokens, 8), 0.125, dtype=torch.float32, device="cuda"
    )
    experts = problem["experts"]
    sparse = (rows + slots // 4 + rank + 3) % experts
    sparse = torch.where((rows % 7 == 0) | (slots == 7), -1, sparse)
    routes = [
        ((rows + rank) % experts).expand(tokens, 8).contiguous(),
        (rows + slots * (experts // 8) + rank) % experts,
        sparse,
    ]
    reference = []
    for enabled in (False, True):
        knobs = {"rank_local_combine": enabled}
        if enabled and compact_k1:
            # Exercise the compact CTA and register pipeline on EP2 as well
            # as EP4; EP2 does not select the EP4 prefill preset automatically.
            knobs.update(
                dispatch_rank_cache=True,
                dispatch_warps=1,
                k1_stages=5,
                k2_stages=4,
                ready_queue_bundle=12,
            )
        layer = _make_layer(rank, world_size, problem, knobs=knobs)
        try:
            for epoch, routing in enumerate(routes):
                # The route-level baseline does not clear skipped live slots
                # between epochs. Compute their zero-weight equivalents so
                # stale baseline payloads cannot enter the reference sum.
                inputs.topk_ids = (routing if enabled else routing.clamp_min(0)).long()
                inputs.topk_weights = torch.where(routing >= 0, 0.125, 0.0).float()
                layer.stage_inputs(inputs, compile_tokens_per_rank=max(tokens_by_rank))
                actual = layer.compute_staged(output=None).clone()
                replay = layer.compute_staged(output=None).clone()
                torch.cuda.synchronize()
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, replay, atol=0, rtol=0)
                if enabled:
                    diff = actual.float() - reference[epoch].float()
                    rel_l2 = diff.norm() / reference[epoch].float().norm().clamp_min(1)
                    # One additional BF16 rounding of each rank partial.
                    assert rel_l2.item() < 0.006, (rank, epoch, rel_l2.item())
                else:
                    reference.append(actual)
        finally:
            layer.destroy()


@pytest.mark.arch_sm120
def test_sm120_nvfp4_rank_local_standalone_route_map() -> None:
    """The repaired aggregate interface also handles absent rows and a K3 tail."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    from cuda.bindings import driver as cuda
    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        bootstrap_paths,
    )

    bootstrap_paths()
    from moe_sm120_nvfp4_split.kernel_rank_local_combine import (
        compile_rank_local_combine,
    )
    from src.sym_buffer import SymBufferHost

    tokens, top_k, hidden = 3, 4, 4100
    routes = torch.arange(7, dtype=torch.bfloat16, device="cuda")[:, None, None]
    routes = routes.expand(7, 1, hidden).contiguous()
    route_map = torch.tensor(
        [[2, -1, 0, 4], [-1, -1, -1, -1], [3, 5, 6, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    combine = torch.full(
        (tokens, top_k, hidden), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    ready = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    aggregate, aggregate_args, reduce, reduce_args = compile_rank_local_combine(
        routes,
        route_map,
        combine,
        ready,
        output,
        SymBufferHost(base_addr=0, offsets=(0,), rank_idx=0, num_max_ranks=1),
        world_size=1,
        local_rank=0,
        stream=stream,
    )
    for epoch in range(2):
        if epoch:
            route_map.copy_(
                torch.tensor(
                    [[-1, -1, -1, -1], [1, 3, -1, -1], [0, 2, 4, 6]],
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            ready.zero_()
        aggregate.to(None)(**aggregate_args)
        reduce.to(None)(**reduce_args)
        expected = (
            torch.where(route_map >= 0, route_map, 0).sum(dim=1).to(torch.bfloat16)
        )
        torch.testing.assert_close(
            output, expected[:, None].expand_as(output), atol=0, rtol=0
        )


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
