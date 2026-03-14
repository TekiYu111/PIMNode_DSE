import pytest

from pimnode_dse.workload import WorkloadDAG, TensorSpec, OpSpec, build_attention_dag


def test_build_attention_dag_prefill(prefill_dag):
    prefill_dag.validate()
    assert prefill_dag.name == "attn_prefill"
    assert len(prefill_dag.ops) >= 4
    assert "Q" in prefill_dag.tensors
    assert "K" in prefill_dag.tensors
    assert "V" in prefill_dag.tensors
    assert "O" in prefill_dag.tensors


def test_build_attention_dag_decode(decode_dag):
    decode_dag.validate()
    assert decode_dag.spec.mode == "decode"
    assert decode_dag.tensors["K"].special_role == "KV_CACHE"
    assert decode_dag.tensors["V"].special_role == "KV_CACHE"


def test_workload_get_op(prefill_dag):
    op = prefill_dag.get_op("Op_QK")
    assert op.op_type == "MatMul"


def test_workload_topo_order(prefill_dag):
    order = prefill_dag.topo_order()
    assert "Op_QK" in order
    assert "Op_Softmax" in order
    assert order.index("Op_QK") < order.index("Op_Softmax")


def test_workload_validate_undefined_tensor():
    dag = WorkloadDAG(name="bad")
    dag.add_tensor(TensorSpec("Q", (1, 8, 16, 64)))
    dag.add_op(
        OpSpec(
            op_id="bad_op",
            op_type="MatMul",
            inputs=["Q", "MISSING"],
            outputs=["OUT"],
        )
    )
    with pytest.raises(KeyError):
        dag.validate()
