# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Native K96 FC2 with real FC1 operands and the existing block-16 SF layout."""

import importlib

import pytest
import torch

from .utils import create_moe_tensors


@pytest.fixture(autouse=True)
def require_sm103():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires native SM103 K96 MMA")
    helpers = pytest.importorskip("cutlass.utils.blackwell_helpers")
    if not hasattr(helpers, "sm103_make_blockscaled_trivial_tiled_mma"):
        pytest.skip("requires CuTe DSL K96 support")


def make_inputs(tokens, hidden, inter):
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
    # Empty experts, partial tiles, repeated routes and non-unit expert alpha.
    inputs["token_selected_experts"][:] = torch.tensor([0, 1], device="cuda")
    inputs["token_selected_experts"][0] = 0
    inputs["w2_alpha"].copy_(torch.linspace(0.7, 1.3, 8, device="cuda"))
    return inputs


@pytest.mark.parametrize(
    "tokens,hidden,inter,fused,epilogue",
    [
        (17, 256, 768, True, (2, 2, 2, True)),
        (257, 384, 768, False, (2, 4, 4, False)),
        (257, 512, 1536, True, (1, 0, 1, False)),
        (17, 256, 1536, True, (2, 4, 2, True)),
        (17, 256, 768, True, (2, 8, 2, True)),
        (257, 384, 768, False, (1, 16, 4, False)),
        (257, 384, 768, True, (2, 16, 4, False)),
    ],
)
def test_fc2_k96_routing_and_replay(tokens, hidden, inter, fused, epilogue):
    from flashinfer import cute_dsl_fused_moe

    inputs = make_inputs(tokens, hidden, inter)
    gemm1 = ((256, 256), (2, 1), False)

    def run(enabled):
        return cute_dsl_fused_moe(
            **inputs,
            num_experts=8,
            num_local_experts=8,
            top_k=2,
            use_fused_finalize=fused,
            tactic=(256, gemm1, ((256, 256), (2, 1), True, (*epilogue, enabled))),
        )

    def check(actual):
        reference = run(False).float()
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        relative = (actual.float() - reference).norm() / reference.norm()
        assert relative < 0.02

    actual = run(True)
    check(actual)
    if not fused:
        assert torch.equal(actual, run(True))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    inputs["token_selected_experts"].add_(2)
    inputs["token_final_scales"].mul_(0.7)
    inputs["w2_alpha"].mul_(1.1)
    graph.replay()
    check(captured)


@pytest.mark.parametrize(
    "inter,tile,capability,message",
    [
        (256, (256, 256), (10, 3), "K divisible by 768"),
        (768, (256, 128), (10, 3), "a \\(256, 256\\) MMA tile"),
        (768, (256, 256), (10, 0), "requires an SM103 GPU"),
    ],
)
def test_fc2_k96_rejects_unsupported_config(
    monkeypatch, inter, tile, capability, message
):
    from flashinfer.fused_moe.cute_dsl.fused_moe import _moe_core_impl

    api = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion"
    )
    monkeypatch.setattr(api, "get_compute_capability", lambda _: capability)
    with pytest.raises(ValueError, match=message):
        _moe_core_impl(
            **make_inputs(17, 256, inter),
            num_experts=8,
            num_local_experts=8,
            top_k=2,
            tile_size=256,
            gemm1_mma_tiler_mn=(256, 256),
            gemm1_cluster_shape_mn=(2, 1),
            gemm2_mma_tiler_mn=tile,
            gemm2_cluster_shape_mn=(2, 1),
            gemm2_epilogue=(2, 2, 2, True, True),
            use_async_memset=False,
        )


@pytest.mark.parametrize(
    "output_dtype,epilogue",
    [
        (torch.float16, (1, 2, 2, False)),
        (torch.float32, (1, 2, 2, False)),
        (torch.float16, (2, 8, 2, False)),
        (torch.float32, (1, 16, 2, False)),
    ],
)
def test_fc2_k96_output_dtype(monkeypatch, output_dtype, epilogue):
    from flashinfer.fused_moe.cute_dsl.fused_moe import _moe_core_impl

    inputs = make_inputs(17, 256, 768)
    core = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")
    fc2 = core.blockscaled_contiguous_grouped_gemm_finalize_fusion
    operands = {}

    def capture_fc2(**kwargs):
        operands.update(kwargs)
        return kwargs["out"]

    # Retain actual FC1 outputs, then exercise the lower-level FC2 API. The
    # public MoE API only supports FP16/BF16 intermediates and output buffers.
    monkeypatch.setattr(
        core, "blockscaled_contiguous_grouped_gemm_finalize_fusion", capture_fc2
    )
    _moe_core_impl(
        **inputs,
        num_experts=8,
        num_local_experts=8,
        top_k=2,
        tile_size=256,
        gemm1_mma_tiler_mn=(256, 256),
        gemm1_cluster_shape_mn=(2, 1),
        gemm2_mma_tiler_mn=(256, 256),
        gemm2_cluster_shape_mn=(2, 1),
        gemm2_epilogue=epilogue,
        use_async_memset=False,
    )
    operands.update(
        out=None,
        out_dtype="float32" if output_dtype == torch.float32 else "float16",
    )
    operands.pop("enable_3x_mma")

    def run(enabled):
        return fc2(**operands, enable_3x_mma=enabled)

    reference = run(False).float()
    actual = run(True)
    torch.cuda.synchronize()
    assert actual.dtype == output_dtype
    assert torch.isfinite(actual).all()
    relative = (actual.float() - reference).norm() / reference.norm()
    assert relative < (2e-5 if output_dtype == torch.float32 else 0.02)


@pytest.mark.parametrize(
    "stages,warps,native", [(1, 8, True), (2, 2, False), (2, 16, True)]
)
def test_fc2_store_empty_schedule(monkeypatch, stages, warps, native):
    """A sentinel-only store stream must terminate without touching output."""
    core = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")
    fc2 = core.blockscaled_contiguous_grouped_gemm_finalize_fusion
    operands = {}

    def capture_fc2(**kwargs):
        operands.update(kwargs)
        return kwargs["out"]

    with monkeypatch.context() as patch:
        patch.setattr(
            core, "blockscaled_contiguous_grouped_gemm_finalize_fusion", capture_fc2
        )
        core._moe_core_impl(
            **make_inputs(17, 384, 768),
            num_experts=8,
            num_local_experts=8,
            top_k=2,
            tile_size=256,
            gemm1_mma_tiler_mn=(256, 256),
            gemm1_cluster_shape_mn=(2, 1),
            gemm2_mma_tiler_mn=(256, 256),
            gemm2_cluster_shape_mn=(2, 1),
            gemm2_epilogue=(stages, warps, 4, False, native),
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
