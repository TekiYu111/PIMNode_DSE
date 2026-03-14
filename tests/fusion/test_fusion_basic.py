import pytest


def test_fusion_validate_topology_single_group(prefill_dag, single_group_fusion):
    single_group_fusion.validate_topology(prefill_dag)


def test_fusion_op_to_group_mapping(single_group_fusion):
    mapping = single_group_fusion.get_op_to_group_mapping()
    assert mapping["Op_QK"] == "g0"
    assert mapping["Op_O"] == "g0"


def test_fusion_validate_topology_two_groups(prefill_dag, two_group_fusion):
    two_group_fusion.validate_topology(prefill_dag)
