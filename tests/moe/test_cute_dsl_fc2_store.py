# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Fused FC2 store stages across route tails, empty experts and graph replay."""

import importlib

import pytest
import torch

from .utils import create_moe_tensors


@pytest.mark.parametrize(
    "tokens,hidden,inter,enable_pdl",
    [
        (17, 256, 256, False),
        (257, 384, 256, True),
        (17, 256, 768, True),
        (17, 256, 1536, False),
    ],
)
@pytest.mark.parametrize(
    "epilogue",
    [
        (2, 2, 4),
        (2, 4, 4),
        (2, 2, 2),
        (2, 2, 2, True),
        (2, 2, 4, True),
        (2, 4, 4, True),
        (2, 4, 2, True),
        (1, 2, 4),
        (1, 8, 4),
        (1, 16, 4),
        (2, 8, 2, True),
        (2, 16, 4),
        (2, 16, 2, True),
    ],
)
@pytest.mark.parametrize("fused", [False, True])
def test_fc2_store_routing_and_replay(
    tokens, hidden, inter, enable_pdl, epilogue, fused
):
    if len(epilogue) > 3 and epilogue[3] and not fused:
        pytest.skip("L2 output hints require the fused BF16 reduction")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    from flashinfer import cute_dsl_fused_moe
    from flashinfer.fused_moe.cute_dsl.fused_moe import _moe_core_impl

    tensors = create_moe_tensors(tokens, hidden, inter, 8, 8, 2)
    names = (
        "x",
        "x_sf",
        "token_selected_experts",
        "token_final_scales",
        "w1_weight",
        "w1_weight_sf",
        "w1_alpha",
        "fc2_input_scale",
        "w2_weight",
        "w2_weight_sf",
        "w2_alpha",
    )
    inputs = {name: tensors[name] for name in names}
    inputs["token_selected_experts"][:] = torch.tensor([0, 1], device="cuda")
    inputs["w2_alpha"].copy_(torch.linspace(0.7, 1.3, 8, device="cuda"))
    common = dict(num_experts=8, num_local_experts=8, top_k=2, enable_pdl=enable_pdl)
    gemm = ((256, 256), (2, 1), False)
    actual = cute_dsl_fused_moe(
        **inputs,
        **common,
        use_fused_finalize=fused,
        tactic=(256, gemm, ((256, 256), (2, 1), True, epilogue)),
    )

    def run(**kwargs):
        return _moe_core_impl(
            **inputs,
            **common,
            tile_size=256,
            gemm1_mma_tiler_mn=(256, 256),
            gemm1_cluster_shape_mn=(2, 1),
            gemm2_mma_tiler_mn=(256, 256),
            gemm2_cluster_shape_mn=(2, 1),
            use_async_memset=False,
            **kwargs,
        )

    def check(output):
        reference = run(use_fused_finalize=False)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        if fused:
            relative = (
                output.float() - reference.float()
            ).norm() / reference.float().norm()
            assert relative < 0.02
        else:
            assert torch.equal(output.view(torch.int16), reference.view(torch.int16))

    check(actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(
            use_fused_finalize=fused,
            gemm2_raster_along_m=True,
            gemm2_epilogue=epilogue,
        )
    inputs["token_selected_experts"].add_(2)
    inputs["token_final_scales"].mul_(0.7)
    inputs["w2_alpha"].mul_(1.1)
    graph.replay()
    check(captured)


@pytest.mark.parametrize("stages,warps", [(1, 8), (2, 2), (2, 16)])
def test_fc2_store_empty_schedule(monkeypatch, stages, warps):
    """A sentinel-only store stream must terminate without touching output."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    core = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")
    fc2 = core.blockscaled_contiguous_grouped_gemm_finalize_fusion
    tensors = create_moe_tensors(17, 384, 768, 8, 8, 2)
    names = (
        "x",
        "x_sf",
        "token_selected_experts",
        "token_final_scales",
        "w1_weight",
        "w1_weight_sf",
        "w1_alpha",
        "fc2_input_scale",
        "w2_weight",
        "w2_weight_sf",
        "w2_alpha",
    )
    operands = {}

    def capture_fc2(**kwargs):
        operands.update(kwargs)
        return kwargs["out"]

    with monkeypatch.context() as patch:
        patch.setattr(
            core, "blockscaled_contiguous_grouped_gemm_finalize_fusion", capture_fc2
        )
        core._moe_core_impl(
            **{name: tensors[name] for name in names},
            num_experts=8,
            num_local_experts=8,
            top_k=2,
            tile_size=256,
            gemm1_mma_tiler_mn=(256, 256),
            gemm1_cluster_shape_mn=(2, 1),
            gemm2_mma_tiler_mn=(256, 256),
            gemm2_cluster_shape_mn=(2, 1),
            gemm2_epilogue=(stages, warps, 4, False),
            use_async_memset=False,
        )

    operands["num_non_exiting_tiles"].zero_()
    operands["out"].fill_(1)
    fc2(**operands)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fc2(**operands)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(operands["out"] == 1)


@pytest.mark.parametrize(
    "fused,output_dtype",
    [(False, torch.bfloat16), (True, torch.float16)],
)
def test_fc2_l2_hints_reject_unsupported_output(fused, output_dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    from flashinfer.fused_moe.cute_dsl.fused_moe import _moe_core_impl

    tensors = create_moe_tensors(17, 256, 256, 8, 8, 2)
    names = (
        "x",
        "x_sf",
        "token_selected_experts",
        "token_final_scales",
        "w1_weight",
        "w1_weight_sf",
        "w1_alpha",
        "fc2_input_scale",
        "w2_weight",
        "w2_weight_sf",
        "w2_alpha",
    )
    with pytest.raises(
        ValueError,
        match="use_l2_hints requires use_fused_finalize and a bfloat16 output",
    ):
        _moe_core_impl(
            **{name: tensors[name] for name in names},
            num_experts=8,
            num_local_experts=8,
            top_k=2,
            tile_size=256,
            gemm1_mma_tiler_mn=(256, 256),
            gemm1_cluster_shape_mn=(2, 1),
            gemm2_mma_tiler_mn=(256, 256),
            gemm2_cluster_shape_mn=(2, 1),
            gemm2_raster_along_m=True,
            gemm2_epilogue=(2, 2, 2, True),
            use_fused_finalize=fused,
            output_dtype=output_dtype,
            use_async_memset=False,
        )
