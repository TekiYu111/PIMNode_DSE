from pimnode_dse.mapping.mapping_builder import MappingBuilder
from tests.helpers import first_group_scope, first_tile_under, collect_storage, collect_actions, collect_loops


def test_mapping_builder_tree_structure(
    prefill_dag,
    single_group_fusion,
    basic_tiling,
    placement_templates,
    builder_config,
):
    builder = MappingBuilder(
        dag=prefill_dag,
        fusion=single_group_fusion,
        tiling=basic_tiling,
        placement_plan=placement_templates,
        selected_templates={"g0": "Balanced"},
        config=builder_config,
    )
    result = builder.build(phase="prefill")
    tree = result.tree

    # 直接打印树
    tree.display()

    group = first_group_scope(tree)
    dram_tile = first_tile_under(group, "DRAM")
    sram_tile = first_tile_under(dram_tile, "SRAM")

    assert dram_tile.tile_size
    assert sram_tile.tile_size
    assert sram_tile.loop_order
    assert sram_tile.placement_state is not None

    assert len(collect_storage(tree)) > 0
    assert len(collect_actions(tree)) > 0
    assert len(collect_loops(tree)) > 0
