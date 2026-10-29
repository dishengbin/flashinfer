# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Complete two-stage FC2, including non-unit alpha and graph routing changes."""

import pytest
import torch

from .utils import create_moe_tensors


@pytest.mark.parametrize("tokens", [1, 17, 257])
@pytest.mark.parametrize("config", [(4, 0, False, 1), (4, 1, True, 2)])
def test_tma_fc2_matches_deterministic_and_replays(tokens, config):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    from flashinfer.fused_moe.cute_dsl.fused_moe import _moe_core_impl

    tensors = create_moe_tensors(tokens, 512, 256, 8, 8, 2)
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
    inputs.update(
        num_experts=8,
        top_k=2,
        num_local_experts=8,
        tile_size=256,
        gemm1_mma_tiler_mn=(256, 256),
        gemm1_cluster_shape_mn=(2, 1),
        gemm2_mma_tiler_mn=(256, 256),
        gemm2_cluster_shape_mn=(2, 1),
        use_async_memset=False,
    )

    def run(**kw):
        return _moe_core_impl(**inputs, **kw)

    expected = run(use_fused_finalize=False)
    actual = run(fc2_tma_config=config)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    fused = run()
    rel = ((fused.float() - actual.float()).norm() / actual.float().norm()).item()
    assert rel < 0.02

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(fc2_tma_config=config)
    inputs["token_selected_experts"].add_(1).remainder_(8)
    inputs["token_final_scales"].mul_(0.7)
    inputs["w2_alpha"].mul_(1.1)
    expected = run(use_fused_finalize=False)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
