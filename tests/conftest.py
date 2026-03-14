
# tests/conftest.py

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import pytest

from pimnode_dse.workload import build_attention_dag

from pimnode_dse.mapping.fusion_gene import (
    FusionGene,
    OpFusionGroup,
    FusionStyle,
)

from pimnode_dse.mapping.tilling_gene import (
    TilingGene,
    GroupTilingSpec,
    MemTileSpec,
)

from pimnode_dse.placement.templates import build_core_templates

from pimnode_dse.mapping.mapping_builder import (
    MappingBuilder,
    BuildConfig,
)


@pytest.fixture
def prefill_dag():
    return build_attention_dag(
        name="attn_prefill",
        batch_size=1,
        num_heads=8,
        seq_len=128,
        d_model=512,
        phase="prefill",
        variant="MHA",
    )


@pytest.fixture
def decode_dag():
    return build_attention_dag(
        name="attn_decode",
        batch_size=1,
        num_heads=8,
        seq_len=128,
        d_model=512,
        phase="decode",
        variant="decode",
    )


@pytest.fixture
def single_group_fusion():
    return FusionGene(
        gene_id="fusion_single",
        groups=[
            OpFusionGroup(
                group_id="g0",
                op_names=["Op_QK", "Op_Softmax", "Op_AV", "Op_O"],
                fusion_style=FusionStyle.SEQUENTIAL,
                phase="prefill",
            )
        ],
    )


@pytest.fixture
def two_group_fusion():
    return FusionGene(
        gene_id="fusion_two",
        groups=[
            OpFusionGroup(
                group_id="g0",
                op_names=["Op_QK", "Op_Softmax"],
                fusion_style=FusionStyle.SEQUENTIAL,
                phase="prefill",
            ),
            OpFusionGroup(
                group_id="g1",
                op_names=["Op_AV", "Op_O"],
                fusion_style=FusionStyle.SEQUENTIAL,
                phase="prefill",
            ),
        ],
    )


@pytest.fixture
def basic_tiling():
    tg = TilingGene()
    tg.add_group_spec(
        GroupTilingSpec(
            group_id="g0",
            tiles={
                "DRAM": MemTileSpec(
                    mem_level="DRAM",
                    fixed_tile_size={"Sq": 128, "Skv": 128, "Dh": 64},
                    fixed_loop_order=["Sq", "Skv", "Dh"],
                    is_spatial=False,
                ),
                "SRAM": MemTileSpec(
                    mem_level="SRAM",
                    fixed_tile_size={"Sq": 64, "Skv": 64, "Dh": 64},
                    fixed_loop_order=["Sq", "Skv", "Dh"],
                    is_spatial=False,
                ),
            },
            phase="prefill",
        )
    )
    return tg


@pytest.fixture
def two_group_tiling():
    tg = TilingGene()
    for gid in ("g0", "g1"):
        tg.add_group_spec(
            GroupTilingSpec(
                group_id=gid,
                tiles={
                    "DRAM": MemTileSpec(
                        mem_level="DRAM",
                        fixed_tile_size={"Sq": 128, "Skv": 128, "Dh": 64},
                        fixed_loop_order=["Sq", "Skv", "Dh"],
                    ),
                    "SRAM": MemTileSpec(
                        mem_level="SRAM",
                        fixed_tile_size={"Sq": 64, "Skv": 64, "Dh": 64},
                        fixed_loop_order=["Sq", "Skv", "Dh"],
                    ),
                },
                phase="prefill",
            )
        )
    return tg


@pytest.fixture
def placement_templates():
    return build_core_templates()


@pytest.fixture
def builder_config():
    return BuildConfig(
        outer_mem_level="DRAM",
        inner_mem_level="SRAM",
        default_template_name="Balanced",
        include_storage_nodes=True,
        include_load_for_resident=False,
        apply_hard_rules_before_attach=True,
    )
