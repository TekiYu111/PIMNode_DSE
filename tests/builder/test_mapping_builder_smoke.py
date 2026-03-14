from pimnode_dse.mapping.mapping_builder import MappingBuilder
from tests.helpers import collect_tiles, collect_ops


def test_mapping_builder_smoke(prefill_dag, single_group_fusion, basic_tiling, placement_templates, builder_config):
    builder = MappingBuilder(
        dag=prefill_dag,
        fusion=single_group_fusion,
        tiling=basic_tiling,
        placement_plan=placement_templates,
        selected_templates={"g0": "Balanced"},
        config=builder_config,
    )
    result = builder.build(phase="prefill")

    assert result.tree is not None
    assert result.report is not None
    assert len(collect_tiles(result.tree)) >= 2
    assert len(collect_ops(result.tree)) >= 4
