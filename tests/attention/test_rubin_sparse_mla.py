# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Rubin FP8 K64 correctness, masking and captured-route lifetime."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.8.0")

from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
    ThroughputLatencyMlaDecodeTs,
)
from tests.attention.test_attention_ts_sparse_mla import (
    _D576_SCALE,
    _check_d576,
    _d576_inputs,
    test_sparse_d576_graph as run_graph_case,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Requires Rubin SM107",
)


@pytest.mark.parametrize("mma_k", (32, 64))
@pytest.mark.parametrize(
    "rows,heads,topk,holes,layout",
    [
        (1, 16, 2048, 0.3, "flat"),
        (4, 16, 2048, 0.1, "nhd4"),
        (3, 16, 66, 0.1, "flat"),
        (3, 16, 513, 0.2, "padded3"),
        (32, 16, 513, 0.2, "flat"),
        (160, 16, 2048, 0.0, "flat"),
    ],
)
def test_fp8_mma_graph(monkeypatch, mma_k, rows, heads, topk, holes, layout):
    original = ThroughputLatencyMlaDecodeTs._make_config

    def make_config(kernel):
        cfg = original(kernel)
        assert not cfg.ws_qk and not cfg.ws_pv
        return replace(cfg, fp8_mma_k=mma_k)

    monkeypatch.setattr(ThroughputLatencyMlaDecodeTs, "_make_config", make_config)
    run_graph_case(rows, heads, topk, holes, 0.0, layout)


def test_fp8_k64_rejects_half_datapath():
    from flashinfer.attention.prims_ts.kernels.mla_decode.helpers.math import (
        qkv_instr_desc,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        MlaConfig,
    )

    cfg = MlaConfig(qkv_dtype="e4m3", fp8_mma_k=64)
    with pytest.raises(ValueError, match="full datapath"):
        qkv_instr_desc(cfg, m_dim=64, n_dim=128)


@pytest.mark.parametrize("rows,heads", [(1, 64), (2, 64), (1, 128), (4, 64)])
def test_tiny_decode_graph(rows, heads):
    run_graph_case(rows, heads, 2048, 0.2, 0.0, "flat")


def test_retained_prefill_graph():
    from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

    rows, heads = 2048, 64
    q, kv, indices, lengths = _d576_inputs(rows, heads, 2048, 0.15)
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        "cuda",
        1,
        heads,
        max_q_len=rows,
        max_topk=2048,
        q_dtype=q.dtype,
        qk_rope_head_dim=64,
        return_lse=True,
    )
    out = torch.empty(1, rows, heads, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(1, rows, heads, device="cuda")

    def run():
        wrapper.run(
            q.reshape(1, rows, heads, 576),
            kv,
            indices,
            lengths,
            out=out,
            lse=lse,
            softmax_scale=_D576_SCALE,
            validate=False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    indices.copy_(indices.flip(-1))
    indices[0, ::2] = -1
    lengths[-1] = 0
    graph.replay()
    sample = torch.tensor([0, 1, rows // 2, rows - 1], device="cuda")
    _check_d576(
        q[sample],
        kv,
        indices[sample],
        lengths[sample],
        out.reshape(rows, 1, heads, 512)[sample],
        lse.reshape(rows, 1, heads)[sample],
    )
    assert torch.isfinite(out).all()
