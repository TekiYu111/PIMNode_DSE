from tests.helpers import collect_nodes
from pimnode_dse.mapping.mapping_tree import ScopeNode
from pimnode_dse.mapping.mapping_builder import MappingBuilder


def test_mapping_builder_multigroup(prefill_dag, two_group_fusion, two_group_tiling, placement_templates, builder_config):
    builder = MappingBuilder(
        dag=prefill_dag,
        fusion=two_group_fusion,
        tiling=two_group_tiling,
        placement_plan=placement_templates,
        selected_templates={"g0": "Balanced", "g1": "Balanced"},
        config=builder_config,
    )
    result = builder.build(phase="prefill")

    group_scopes = [
        n for n in result.tree.walk()
        if isinstance(n, ScopeNode) and n.name.startswith("Group(")
    ]
    assert len(group_scopes) == 2
    assert set(result.report.placement_selection.keys()) == {"g0", "g1"}
