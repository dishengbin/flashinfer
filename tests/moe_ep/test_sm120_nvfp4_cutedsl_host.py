"""Host-side contracts for the SM120 NVFP4 x NVFP4 MegaMoE backend."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep import BootstrapConfig, FleetParams
from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cutedsl import (
    Sm120Nvfp4Nvfp4CutedslMegaKernelBackend,
    Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
    select_graph_compile_bucket,
)
from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel.shim.weights import (
    interleave_gate_up_16,
    scale_storage_size,
)


def test_config_uses_post_swiglu_intermediate() -> None:
    config = Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=2048,
        top_k=6,
    )
    assert config.intermediate_size == 2048
    assert config.kernel_name == "sm120_nvfp4_nvfp4_bf16_cutedsl"
    assert config.combine_dtype == "bf16"


def test_config_rejects_unknown_combine_dtype() -> None:
    with pytest.raises(ValueError, match="combine_dtype"):
        Sm120Nvfp4Nvfp4CutedslMegaKernelBackend(
            Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=2048,
                top_k=6,
                combine_dtype="fp8",  # type: ignore[arg-type]
            )
        )


def test_rank_local_combine_remains_opt_in() -> None:
    # Importing api activates architecture-specific top-level modules. Keep
    # this specialization check isolated from the other backends' host tests.
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import bootstrap_paths
            bootstrap_paths()
            from moe_sm120_nvfp4_split.api import MegaMoEProblemSpec, select_compile_spec
            from moe_sm120_nvfp4_split.heuristic import MegaMoEHeuristicOverrides
            problem = MegaMoEProblemSpec(
                tokens_per_rank=6045, num_topk=8, num_total_experts=128,
                hidden=6144, intermediate=4608, expert_parallel_size=4,
                expert_parallel_rank=0,
            )
            options = dict(
                problem=problem, ep_same_numa_peer_count=3, ep_cross_numa_peer_count=0,
                num_sms=148, sm_min_partition=4, sm_partition_alignment=4,
            )
            default = select_compile_spec(**options)
            assert default.kernel.dispatch_rank_cache
            assert not default.kernel.rank_local_combine
            explicit = select_compile_spec(
                **options, overrides=MegaMoEHeuristicOverrides(rank_local_combine=True)
            )
            assert explicit.kernel.rank_local_combine
            import pytest
            from moe_sm120_nvfp4_split.api import SplitKernelBuildOptions
            with pytest.raises(ValueError, match="requires BF16 combine"):
                select_compile_spec(
                    **options,
                    overrides=MegaMoEHeuristicOverrides(rank_local_combine=True),
                    build=SplitKernelBuildOptions(combine_format="16e2m1xbf16"),
                )
        """),
        ],
        check=True,
    )


def test_k1_pipeline_stages_are_independent_and_cacheable() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import pytest
            from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import bootstrap_paths
            bootstrap_paths()
            from moe_sm120_nvfp4_split.api import MegaMoEProblemSpec, select_compile_spec
            from moe_sm120_nvfp4_split.heuristic import MegaMoEHeuristicOverrides
            from moe_sm120_nvfp4_split.jit_config import Sm120JitConfig
            problem = MegaMoEProblemSpec(
                tokens_per_rank=6045, num_topk=8, num_total_experts=128,
                hidden=6144, intermediate=4608, expert_parallel_size=4,
                expert_parallel_rank=0,
            )
            options = dict(
                problem=problem, ep_same_numa_peer_count=3, ep_cross_numa_peer_count=0,
                num_sms=110, sm_min_partition=4, sm_partition_alignment=4,
            )
            automatic = select_compile_spec(**options)
            assert automatic.kernel.k1_stages is None
            explicit = select_compile_spec(
                **options,
                overrides=MegaMoEHeuristicOverrides(rank_local_combine=True, k1_stages=5, k2_stages=3),
            )
            assert explicit.cache_key != automatic.cache_key
            assert explicit.jit == Sm120JitConfig()
            assert explicit.kernel.k1_stages == 5
            assert explicit.kernel.k2_stages == 3
            assert not explicit.jit.enable_globaltimer
            assert not explicit.jit.enable_k2_tile_trace
            shallow_k2 = select_compile_spec(
                **options, overrides=MegaMoEHeuristicOverrides(rank_local_combine=True, k1_stages=5, k2_stages=2)
            )
            assert shallow_k2.kernel.k2_stages == automatic.kernel.k2_stages
            assert shallow_k2.cache_key != explicit.cache_key
            with pytest.raises(ValueError, match="requires rank_local_combine"):
                select_compile_spec(
                    **options, overrides=MegaMoEHeuristicOverrides(k1_stages=5)
                )
            from moe_sm120_nvfp4_split.api import SplitKernelBuildOptions
            fp4_build = SplitKernelBuildOptions(combine_format="16e2m1xbf16")
            compact = select_compile_spec(
                **options, build=fp4_build,
                overrides=MegaMoEHeuristicOverrides(dispatch_warps=1, k1_stages=5),
            )
            assert compact.cache_key != automatic.cache_key
            assert not compact.kernel.rank_local_combine
            for overrides in (
                MegaMoEHeuristicOverrides(dispatch_warps=4, k1_stages=5),
                MegaMoEHeuristicOverrides(dispatch_warps=1, k1_stages=3),
                MegaMoEHeuristicOverrides(dispatch_warps=1, k1_stages=5, k1_tile=(64,64,128)),
            ):
                with pytest.raises(ValueError, match="FP4 compact K1"):
                    select_compile_spec(**options, build=fp4_build, overrides=overrides)
            tuned = select_compile_spec(
                **options, overrides=MegaMoEHeuristicOverrides(rank_local_combine=True)
            )
            assert (tuned.kernel.k1_stages, tuned.kernel.k2_stages) == (5, 4)
            assert (tuned.kernel.dispatch_warps, tuned.kernel.ready_queue_bundle) == (1, 12)
            pinned = select_compile_spec(
                **options,
                overrides=MegaMoEHeuristicOverrides(
                    rank_local_combine=True, k1_stages=2, k2_stages=2,
                    dispatch_warps=4, ready_queue_bundle=4,
                ),
            )
            assert (pinned.kernel.k1_stages, pinned.kernel.k2_stages) == (2, 2)
            assert (pinned.kernel.dispatch_warps, pinned.kernel.ready_queue_bundle) == (4, 4)
            from dataclasses import replace
            for tokens in (2559, 11896):
                outside = select_compile_spec(
                    **dict(options, problem=replace(problem, tokens_per_rank=tokens)),
                    overrides=MegaMoEHeuristicOverrides(rank_local_combine=True),
                )
                assert outside.kernel.k1_stages is None
            outside = select_compile_spec(
                **dict(options, num_sms=148),
                overrides=MegaMoEHeuristicOverrides(rank_local_combine=True),
            )
            assert outside.kernel.k1_stages is None
            for stages in (0, -1):
                with pytest.raises(ValueError, match="k1_stages must be positive"):
                    select_compile_spec(
                        **options, overrides=MegaMoEHeuristicOverrides(k1_stages=stages)
                    )
        """),
        ],
        check=True,
    )


def test_gate_up_interleave_is_grouped_in_sixteen_rows() -> None:
    rows = torch.arange(64, dtype=torch.int64).view(1, 64, 1)
    actual = interleave_gate_up_16(rows, full_width=64).flatten().tolist()
    assert actual == (
        list(range(0, 16))
        + list(range(32, 48))
        + list(range(16, 32))
        + list(range(48, 64))
    )


def test_scale_storage_uses_block16_and_atom_padding() -> None:
    assert scale_storage_size(4096, 4096) == 4096 * 256
    assert scale_storage_size(33, 65) == 128 * 8


def test_decode_graph_compile_bucket_selection() -> None:
    capacity = 8192
    expected = {
        1: 7,
        7: 7,
        8: 16,
        16: 16,
        17: 32,
        127: 128,
        129: 168,
        168: 168,
        169: 256,
        256: 256,
        257: capacity,
    }
    for requested, bucket in expected.items():
        assert select_graph_compile_bucket(requested, capacity) == bucket


def test_combine_reduce_flattens_large_token_grid() -> None:
    # The raw SM120 modules share names with SM100; isolate this import so
    # the workspace-teardown contract below can load its own staging tree.
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import bootstrap_paths
            bootstrap_paths()
            from moe_sm120_nvfp4_split.kernel_combine_reduce import _topk_reduce_launch_geometry
            hidden_blocks, grid = _topk_reduce_launch_geometry(
                tokens=81450, hidden=6144, threads=512,
            )
            assert hidden_blocks == 2
            assert grid == [162900, 1, 1]
        """),
        ],
        check=True,
    )


def test_workspace_pool_key_covers_nvfp4_contract(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    fleet = FleetParams(
        num_experts=256,
        max_tokens_per_rank=8192,
        token_hidden_size=4096,
    )

    def key(*, norm_const: float = 1.0, combine_dtype: str = "bf16"):
        backend = Sm120Nvfp4Nvfp4CutedslMegaKernelBackend(
            Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=4096,
                top_k=6,
                input_norm_const=norm_const,
                combine_dtype=combine_dtype,
            )
        )
        backend.bind_ep_bootstrap(
            BootstrapConfig(world_size=4, rank=1, auto_bootstrap=False)
        )
        return backend._workspace_pool_key(fleet)

    assert key() == key()
    assert key(norm_const=2.0) != key()
    assert key(combine_dtype="nvfp4") != key()


def test_workspace_teardown_forgets_fused_stage_descriptors(monkeypatch) -> None:
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import quant_stage

    backend = Sm120Nvfp4Nvfp4CutedslMegaKernelBackend(
        Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=2304,
            top_k=8,
        )
    )
    topk_ids = object()
    forgotten: list[object] = []
    monkeypatch.setattr(
        quant_stage,
        "forget_staged_tokens",
        forgotten.append,
    )

    backend._forget_workspace_state(SimpleNamespace(topk_ids=topk_ids))

    assert forgotten == [topk_ids]


def test_production_modules_do_not_import_mega_runner() -> None:
    package = Path(__file__).parents[2] / "flashinfer" / "moe_ep"
    roots = (
        package / "backends" / "mega" / "kernel" / "sm120" / "nvfp4_nvfp4_bf16_cutedsl",
        package / "kernel_src" / "sm120" / "nvfp4_split_cutedsl_megakernel" / "shim",
    )
    offenders: list[str] = []
    for root in roots:
        for source in root.rglob("*.py"):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any("mega_runner" in name for name in names):
                    offenders.append(str(source.relative_to(package)))
    assert not offenders


@pytest.mark.parametrize("option", ["k2_register_prefetch", "k2_fused_quant_pack"])
def test_k2_optimization_is_explicit_and_cacheable(option: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
            import pytest
            from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import bootstrap_paths
            bootstrap_paths()
            from moe_sm120_nvfp4_split.api import MegaMoEProblemSpec, SplitKernelBuildOptions, select_compile_spec
            from moe_sm120_nvfp4_split.heuristic import MegaMoEHeuristicOverrides

            problem = MegaMoEProblemSpec(tokens_per_rank=6045, num_topk=8, num_total_experts=128, hidden=6144,
                                        intermediate=4608, expert_parallel_size=4, expert_parallel_rank=0)
            options = dict(problem=problem, ep_same_numa_peer_count=3, ep_cross_numa_peer_count=0,
                           num_sms=110, sm_min_partition=4, sm_partition_alignment=4)
            base = select_compile_spec(**options)
            assert not base.kernel.k2_register_prefetch
            fp4 = SplitKernelBuildOptions(combine_format='16e2m1xbf16')
            kwargs = dict(k1_stages=5, dispatch_warps=1, k2_stages=3, k1_sms=66, k2_sms=44, tx_sms=0, rx_sms=0)
            off = select_compile_spec(**options, build=fp4, overrides=MegaMoEHeuristicOverrides(**kwargs))
            on = select_compile_spec(**options, build=fp4, overrides=MegaMoEHeuristicOverrides(**kwargs, k2_register_prefetch=True))
            assert on.kernel.k2_register_prefetch and on.cache_key != off.cache_key
            assert (on.kernel.k1_sms,on.kernel.k2_sms)==(66,44)
            with pytest.raises(ValueError,match='requires EP2/EP4'):
                select_compile_spec(**options, overrides=MegaMoEHeuristicOverrides(k2_register_prefetch=True))
            with pytest.raises(ValueError,match='requires EP2/EP4'):
                select_compile_spec(**options, build=fp4, overrides=MegaMoEHeuristicOverrides(k2_register_prefetch=True,k2_tile=(64,64,128)))
        """.replace("k2_register_prefetch", option)
            ),
        ],
        check=True,
    )


def test_green_context_honors_requested_sm_partition() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            from types import SimpleNamespace
            import pytest
            from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import bootstrap_paths
            bootstrap_paths()
            from moe_sm120_nvfp4_split.runtime.green_context import _split_k1_k2_sm_resources
            def resource(n):
                return SimpleNamespace(sm=SimpleNamespace(smCount=n))
            calls=[]
            def split(groups, source, flags, requested):
                calls.append((groups, flags, requested))
                n=72 if flags==0 else requested
                return (0,[resource(n)],1,resource(110-n))
            fake=SimpleNamespace(cuDevSmResourceSplitByCount=split)
            a,b=_split_k1_k2_sm_resources(fake,resource(110),72)
            assert (a.sm.smCount,b.sm.smCount)==(72,38) and calls==[(1,0,72)]
            calls.clear()
            a,b=_split_k1_k2_sm_resources(fake,resource(110),66)
            assert (a.sm.smCount,b.sm.smCount)==(66,44) and calls==[(1,0,66),(1,1,66)]
            fake.cuDevSmResourceSplitByCount=lambda *args: (0,[],0,resource(110))
            with pytest.raises(RuntimeError,match='one K1 SM partition'):
                _split_k1_k2_sm_resources(fake,resource(110),66)
        """),
        ],
        check=True,
    )
