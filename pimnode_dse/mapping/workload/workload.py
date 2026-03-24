"""
Construct a WorkloadDAG for the given AttentionWorkloadSpec.
Supports prefill/decode modes and MHA/GQA/MQA head configurations.

This version models:
  - decode cache update explicitly via K_CACHE_OUT / V_CACHE_OUT
  - compute views explicitly via K_CTX / V_CTX
  - head sharing explicitly via HeadBroadcast ops

Notes on semantic fields:
  - TensorSpec.role is the canonical coarse semantic role consumed by downstream
    mapping / placement code (for example: Q / K / V / SCORES / PROBS / O / STATS).
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class TensorType(str, Enum):
    """Categorises how a tensor is used in the workload."""

    ACTIVATION = "activation"
    WEIGHT = "weight"
    CACHE = "cache"
    OUTPUT = "output"


class TensorRole(str, Enum):
    """Canonical coarse semantic tensor roles used by downstream mapping code."""

    Q = "Q"
    K = "K"
    V = "V"
    SCORES = "SCORES"
    STATS = "STATS"
    PROBS = "PROBS"
    O = "O"


@dataclass(frozen=True)
class TensorSpec:
    """Describes a single tensor in the workload graph."""

    name: str
    shape: Tuple[int, ...]
    tensor_type: TensorType = TensorType.ACTIVATION
    dtype: str = "float16"
    role: Optional[TensorRole] = None

    def num_elements(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result

    def size_bytes(self) -> int:
        dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}
        return self.num_elements() * dtype_bytes.get(self.dtype, 2)


@dataclass(frozen=True)
class OpDataEdge:
    """op -> op dataflow edge."""

    src_op: str
    dst_op: str
    tensor: str
    src_dims: Tuple[str, ...] = ()
    dst_dims: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OpSpec:
    """Operator specification in the workload DAG."""

    op_id: str
    op_type: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    iter_dims: Tuple[str, ...] = ()
    tensor_dims: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    reduce_dims: Tuple[str, ...] = ()
    reduce_type: str = "full"
    attrs: Dict[str, Any] = field(default_factory=dict)
    dim_constraints: Dict[str, int] = field(default_factory=dict)


@dataclass
class AttentionWorkloadSpec:
    """High-level description of an attention workload."""

    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    mode: str
    attn_type: str
    mask_type: str = "causal"
    dtype: str = "float16"
    cache_len_before: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("prefill", "decode"):
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.attn_type not in ("mha", "gqa", "mqa"):
            raise ValueError(f"Unknown attn_type: {self.attn_type}")
        if self.batch_size <= 0 or self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("batch_size, num_heads, num_kv_heads must be > 0")
        if self.head_dim <= 0 or self.seq_len <= 0:
            raise ValueError("head_dim and seq_len must be > 0")
        if self.mode == "prefill":
            if self.cache_len_before != 0:
                raise ValueError("cache_len_before must be 0 in prefill mode")
        else:
            if self.cache_len_before < 0:
                raise ValueError("cache_len_before must be >= 0 in decode mode")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.attn_type == "mha" and self.num_kv_heads != self.num_heads:
            raise ValueError("mha requires num_kv_heads == num_heads")
        if self.attn_type == "mqa" and self.num_kv_heads != 1:
            raise ValueError("mqa requires num_kv_heads == 1")
        if self.attn_type == "gqa":
            if self.num_kv_heads == self.num_heads:
                raise ValueError("gqa requires num_kv_heads != num_heads")
            if self.num_kv_heads == 1:
                raise ValueError("gqa requires num_kv_heads != 1")

    @property
    def hq(self) -> int:
        return self.num_heads

    @property
    def hkv(self) -> int:
        return self.num_kv_heads

    @property
    def kv_head_broadcast(self) -> int:
        return self.hq // self.hkv

    def fingerprint(self) -> str:
        d = {
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "seq_len": self.seq_len,
            "mode": self.mode,
            "attn_type": self.attn_type,
            "mask_type": self.mask_type,
            "dtype": self.dtype,
            "cache_len_before": self.cache_len_before,
        }
        raw = json.dumps(d, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:8]


class WorkloadDAG:
    """Directed acyclic graph representing the data flow of an attention workload."""

    def __init__(self) -> None:
        self._tensors: Dict[str, TensorSpec] = {}
        self._ops: Dict[str, OpSpec] = {}
        self._edges: List[OpDataEdge] = []

        self._tensor_to_consumer_ops: Dict[str, List[str]] = {}
        self._tensor_to_producer_op: Dict[str, Optional[str]] = {}
        self._op_to_output_tensors: Dict[str, List[str]] = {}
        self._op_to_input_tensors: Dict[str, List[str]] = {}
        self._op_predecessors: Dict[str, List[str]] = {}
        self._op_successors: Dict[str, List[str]] = {}
        self._finalized: bool = False

    def _invalidate(self) -> None:
        self._finalized = False
        self._tensor_to_consumer_ops = {}
        self._tensor_to_producer_op = {}
        self._op_to_output_tensors = {}
        self._op_to_input_tensors = {}
        self._op_predecessors = {}
        self._op_successors = {}

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("WorkloadDAG is not finalized. Call finalize() first.")

    def add_tensor(self, spec: TensorSpec) -> None:
        if spec.name in self._tensors:
            raise ValueError(f"Tensor '{spec.name}' already exists in the DAG.")
        self._tensors[spec.name] = spec
        self._invalidate()

    def add_op(self, spec: OpSpec) -> None:
        if spec.op_id in self._ops:
            raise ValueError(f"Op '{spec.op_id}' already exists in the DAG.")
        self._ops[spec.op_id] = spec
        self._invalidate()

    def add_edge(self, edge: OpDataEdge) -> None:
        src, dst, tensor = edge.src_op, edge.dst_op, edge.tensor
        if src not in self._ops:
            raise ValueError(f"Invalid edge src '{src}': op not found.")
        if dst not in self._ops:
            raise ValueError(f"Invalid edge dst '{dst}': op not found.")
        if tensor not in self._tensors:
            raise ValueError(f"Edge tensor '{tensor}' not found in DAG.")
        if tensor not in self._ops[src].outputs:
            raise ValueError(f"Invalid edge {src} -> {dst}: tensor '{tensor}' is not an output of '{src}'.")
        if tensor not in self._ops[dst].inputs:
            raise ValueError(f"Invalid edge {src} -> {dst}: tensor '{tensor}' is not an input of '{dst}'.")

        src_dims_expected = self._ops[src].tensor_dims.get(tensor, ())
        dst_dims_expected = self._ops[dst].tensor_dims.get(tensor, ())
        if edge.src_dims and edge.src_dims != src_dims_expected:
            raise ValueError(f"Invalid edge {src} -> {dst}: src_dims {edge.src_dims} != producer tensor_dims {src_dims_expected}.")
        if edge.dst_dims and edge.dst_dims != dst_dims_expected:
            raise ValueError(f"Invalid edge {src} -> {dst}: dst_dims {edge.dst_dims} != consumer tensor_dims {dst_dims_expected}.")

        self._edges.append(
            OpDataEdge(
                src_op=src,
                dst_op=dst,
                tensor=tensor,
                src_dims=edge.src_dims or src_dims_expected,
                dst_dims=edge.dst_dims or dst_dims_expected,
            )
        )
        self._invalidate()

    def finalize(self) -> None:
        self._tensor_to_consumer_ops = {name: [] for name in self._tensors}
        self._tensor_to_producer_op = {name: None for name in self._tensors}
        self._op_to_input_tensors = {op_id: list(spec.inputs) for op_id, spec in self._ops.items()}
        self._op_to_output_tensors = {op_id: list(spec.outputs) for op_id, spec in self._ops.items()}
        self._op_predecessors = {op_id: [] for op_id in self._ops}
        self._op_successors = {op_id: [] for op_id in self._ops}

        for op_id, spec in self._ops.items():
            for t in spec.outputs:
                if t not in self._tensors:
                    raise ValueError(f"Op '{op_id}' references unknown output tensor '{t}'.")
                prev = self._tensor_to_producer_op[t]
                if prev is not None and prev != op_id:
                    raise ValueError(f"Tensor '{t}' has multiple producers: '{prev}' and '{op_id}'.")
                self._tensor_to_producer_op[t] = op_id
            for t in spec.inputs:
                if t not in self._tensors:
                    raise ValueError(f"Op '{op_id}' references unknown input tensor '{t}'.")
                if op_id not in self._tensor_to_consumer_ops[t]:
                    self._tensor_to_consumer_ops[t].append(op_id)

        seen_edges: Set[Tuple[str, str, str]] = set()
        for edge in self._edges:
            triple = (edge.src_op, edge.dst_op, edge.tensor)
            if triple in seen_edges:
                raise ValueError(f"Duplicate op->op edge: {edge.src_op} -> {edge.dst_op} [tensor={edge.tensor}].")
            seen_edges.add(triple)

            if edge.dst_op not in self._op_successors[edge.src_op]:
                self._op_successors[edge.src_op].append(edge.dst_op)
            if edge.src_op not in self._op_predecessors[edge.dst_op]:
                self._op_predecessors[edge.dst_op].append(edge.src_op)

        self._finalized = True

    def tensors(self) -> Dict[str, TensorSpec]:
        return dict(self._tensors)

    def ops(self) -> Dict[str, OpSpec]:
        return dict(self._ops)

    def edges(self) -> List[OpDataEdge]:
        return list(self._edges)

    def tensor(self, name: str) -> TensorSpec:
        return self._tensors[name]

    def op(self, op_id: str) -> OpSpec:
        return self._ops[op_id]

    def op_ids(self) -> List[str]:
        return list(self._ops.keys())

    def consumers_of(self, tensor_name: str) -> List[str]:
        self._require_finalized()
        return list(self._tensor_to_consumer_ops.get(tensor_name, []))

    def producer_of(self, tensor_name: str) -> Optional[str]:
        self._require_finalized()
        return self._tensor_to_producer_op.get(tensor_name)

    def inputs_of(self, op_id: str) -> List[str]:
        self._require_finalized()
        return list(self._op_to_input_tensors.get(op_id, []))

    def outputs_of(self, op_id: str) -> List[str]:
        self._require_finalized()
        return list(self._op_to_output_tensors.get(op_id, []))

    def predecessors(self, op_id: str) -> List[str]:
        self._require_finalized()
        return list(self._op_predecessors.get(op_id, []))

    def successors(self, op_id: str) -> List[str]:
        self._require_finalized()
        return list(self._op_successors.get(op_id, []))

    def edges_between(self, u: str, v: str) -> List[OpDataEdge]:
        return [e for e in self._edges if e.src_op == u and e.dst_op == v]

    def source_tensors(self) -> List[str]:
        self._require_finalized()
        return [n for n, p in self._tensor_to_producer_op.items() if p is None]

    def sink_tensors(self) -> List[str]:
        self._require_finalized()
        return [n for n, cs in self._tensor_to_consumer_ops.items() if not cs]

    def topological_op_order(self) -> List[str]:
        self._require_finalized()
        in_degree: Dict[str, int] = {op_id: len(self._op_predecessors[op_id]) for op_id in self._ops}
        queue = deque([op_id for op_id, d in in_degree.items() if d == 0])
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for dep in self._op_successors[node]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
        if len(order) != len(self._ops):
            raise RuntimeError("Cycle detected in WorkloadDAG.")
        return order

    def subgraph_tensors(self, op_ids: Set[str]) -> Dict[str, Set[str]]:
        self._require_finalized()
        missing = [op_id for op_id in op_ids if op_id not in self._ops]
        if missing:
            raise KeyError(f"Unknown ops in subgraph: {missing}")

        inputs: Set[str] = set()
        outputs: Set[str] = set()
        temps: Set[str] = set()

        for op_id in op_ids:
            op = self._ops[op_id]
            for t in op.inputs:
                producer = self.producer_of(t)
                if producer is None or producer not in op_ids:
                    inputs.add(t)
                else:
                    temps.add(t)

            for t in op.outputs:
                consumers = self.consumers_of(t)
                if not consumers or any(c not in op_ids for c in consumers):
                    outputs.add(t)
                else:
                    temps.add(t)

        temps = temps - inputs - outputs
        return {"inputs": inputs, "outputs": outputs, "temps": temps}

    def is_convex_subgraph(self, op_ids: Set[str]) -> bool:
        self._require_finalized()
        if not op_ids:
            return True
        missing = [op_id for op_id in op_ids if op_id not in self._ops]
        if missing:
            raise KeyError(f"Unknown ops in subgraph: {missing}")

        outside_frontier: Set[str] = set()
        queue = deque(op_ids)
        visited_in = set(op_ids)

        while queue:
            u = queue.popleft()
            for v in self._op_successors.get(u, []):
                if v in op_ids:
                    if v not in visited_in:
                        visited_in.add(v)
                        queue.append(v)
                else:
                    outside_frontier.add(v)

        rev_queue = deque(outside_frontier)
        visited_out = set(outside_frontier)
        while rev_queue:
            u = rev_queue.popleft()
            for v in self._op_successors.get(u, []):
                if v in op_ids:
                    return False
                if v not in visited_out:
                    visited_out.add(v)
                    rev_queue.append(v)
        return True

    def validate(self) -> None:
        errors: List[str] = []
        valid_reduce_types = {"full", "partial", "none"}

        for op_id, op in self._ops.items():
            if len(set(op.iter_dims)) != len(op.iter_dims):
                errors.append(f"Op '{op_id}': iter_dims contains duplicates: {op.iter_dims}.")
            for t in op.inputs:
                if t not in self._tensors:
                    errors.append(f"Op '{op_id}' references unknown input tensor '{t}'.")
            for t in op.outputs:
                if t not in self._tensors:
                    errors.append(f"Op '{op_id}' references unknown output tensor '{t}'.")
            if op.reduce_type not in valid_reduce_types:
                errors.append(f"Op '{op_id}': invalid reduce_type '{op.reduce_type}', expected one of {sorted(valid_reduce_types)}.")
            iter_dims_set = set(op.iter_dims)
            for rd in op.reduce_dims:
                if rd not in iter_dims_set:
                    errors.append(f"Op '{op_id}': reduce_dim '{rd}' is not in iter_dims {op.iter_dims}.")
            if op.reduce_type == "none" and op.reduce_dims:
                errors.append(f"Op '{op_id}': reduce_type='none' but reduce_dims={op.reduce_dims} is not empty.")

            all_dims = set(op.iter_dims)
            for t_name, t_dims in op.tensor_dims.items():
                if len(set(t_dims)) != len(t_dims):
                    errors.append(f"Op '{op_id}': tensor_dims for '{t_name}' contains duplicates: {t_dims}.")
                all_dims.update(t_dims)

            for dim_name, dim_value in op.dim_constraints.items():
                if dim_name not in all_dims:
                    errors.append(f"Op '{op_id}': dim_constraint '{dim_name}' not in iter_dims or tensor_dims.")
                if dim_value <= 0:
                    errors.append(f"Op '{op_id}': dim_constraint '{dim_name}' must be > 0, got {dim_value}.")

            for t_name in list(op.inputs) + list(op.outputs):
                if t_name not in self._tensors:
                    continue
                if t_name not in op.tensor_dims:
                    errors.append(f"Op '{op_id}': tensor_dims missing entry for tensor '{t_name}'.")
                    continue
                tensor = self._tensors[t_name]
                t_dims = op.tensor_dims[t_name]
                if len(t_dims) != len(tensor.shape):
                    errors.append(f"Op '{op_id}': tensor '{t_name}' has {len(tensor.shape)} dims but tensor_dims specifies {len(t_dims)}.")

        try:
            self.finalize()
            self.topological_op_order()
        except Exception as e:
            errors.append(str(e))

        if errors:
            self._invalidate()
            raise ValueError("WorkloadDAG validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def summary(self) -> str:
        self._require_finalized()
        lines: List[str] = ["WorkloadDAG summary:"]
        lines.append(f"  Tensors ({len(self._tensors)}):")
        for name, t in self._tensors.items():
            role = t.role.value if isinstance(t.role, Enum) else t.role
            role_part = f", role={role}" if role is not None else ""
            lines.append(f"    {name}: shape={t.shape}, type={t.tensor_type.value}, dtype={t.dtype}{role_part}")

        lines.append(f"  Ops ({len(self._ops)}):")
        for op_id, op in self._ops.items():
            lines.append(f"    {op_id}: type={op.op_type}, inputs={list(op.inputs)}, outputs={list(op.outputs)}, reduce_dims={list(op.reduce_dims)}, reduce_type={op.reduce_type}")

        lines.append(f"  Edges ({len(self._edges)}):")
        for e in self._edges:
            lines.append(f"    {e.src_op} -> {e.dst_op} [tensor={e.tensor}, src_dims={e.src_dims}, dst_dims={e.dst_dims}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"WorkloadDAG(tensors={len(self._tensors)}, ops={len(self._ops)}, edges={len(self._edges)}, finalized={self._finalized})"


def build_attention_dag(spec: AttentionWorkloadSpec) -> WorkloadDAG:
    """
    Construct a WorkloadDAG for the given AttentionWorkloadSpec.
    Supports prefill/decode modes and MHA/GQA/MQA head configurations.
    """
    dag = WorkloadDAG()

    B = spec.batch_size
    hq = spec.hq
    hkv = spec.hkv
    D = spec.head_dim
    dtype = spec.dtype

    if spec.mode == "prefill":
        q_len = spec.seq_len
        kv_len = spec.seq_len
    else:
        q_len = 1
        append_len = 1
        n_cache = spec.cache_len_before
        kv_len = n_cache + append_len

    dag.add_tensor(TensorSpec("Q", (B, hq, q_len, D), TensorType.ACTIVATION, dtype, role=TensorRole.Q))

    if spec.mode == "prefill":
        dag.add_tensor(TensorSpec("K_CTX", (B, hkv, kv_len, D), TensorType.ACTIVATION, dtype, role=TensorRole.K))
        dag.add_tensor(TensorSpec("V_CTX", (B, hkv, kv_len, D), TensorType.ACTIVATION, dtype, role=TensorRole.V))
    else:
        dag.add_tensor(TensorSpec("K_CACHE_IN", (B, hkv, n_cache, D), TensorType.CACHE, dtype, role=TensorRole.K))
        dag.add_tensor(TensorSpec("V_CACHE_IN", (B, hkv, n_cache, D), TensorType.CACHE, dtype, role=TensorRole.V))
        dag.add_tensor(TensorSpec("K_APPEND", (B, hkv, append_len, D), TensorType.ACTIVATION, dtype, role=TensorRole.K))
        dag.add_tensor(TensorSpec("V_APPEND", (B, hkv, append_len, D), TensorType.ACTIVATION, dtype, role=TensorRole.V))
        dag.add_tensor(TensorSpec("K_CACHE_OUT", (B, hkv, kv_len, D), TensorType.CACHE, dtype, role=TensorRole.K))
        dag.add_tensor(TensorSpec("V_CACHE_OUT", (B, hkv, kv_len, D), TensorType.CACHE, dtype, role=TensorRole.V))
        dag.add_tensor(TensorSpec("K_CTX", (B, hkv, kv_len, D), TensorType.ACTIVATION, dtype, role=TensorRole.K))
        dag.add_tensor(TensorSpec("V_CTX", (B, hkv, kv_len, D), TensorType.ACTIVATION, dtype, role=TensorRole.V))

    dag.add_tensor(TensorSpec("K_T", (B, hkv, D, kv_len), TensorType.ACTIVATION, dtype, role=TensorRole.K))
    dag.add_tensor(TensorSpec("K_T_ATTN", (B, hq, D, kv_len), TensorType.ACTIVATION, dtype, role=TensorRole.K))
    dag.add_tensor(TensorSpec("V_ATTN", (B, hq, kv_len, D), TensorType.ACTIVATION, dtype, role=TensorRole.V))
    dag.add_tensor(TensorSpec("SCORES", (B, hq, q_len, kv_len), TensorType.ACTIVATION, dtype, role=TensorRole.SCORES))
    dag.add_tensor(TensorSpec("STATS", (B, hq, q_len), TensorType.ACTIVATION, dtype, role=TensorRole.STATS))
    dag.add_tensor(TensorSpec("PROBS", (B, hq, q_len, kv_len), TensorType.ACTIVATION, dtype, role=TensorRole.PROBS))
    dag.add_tensor(TensorSpec("O", (B, hq, q_len, D), TensorType.OUTPUT, dtype, role=TensorRole.O))

    if spec.mode == "decode":
        dag.add_op(
            OpSpec(
                op_id="Op_K_Append",
                op_type="KVAppend",
                inputs=("K_APPEND", "K_CACHE_IN"),
                outputs=("K_CACHE_OUT",),
                iter_dims=("b", "hkv", "n_cache", "n_append", "n_total", "d"),
                tensor_dims={
                    "K_APPEND": ("b", "hkv", "n_append", "d"),
                    "K_CACHE_IN": ("b", "hkv", "n_cache", "d"),
                    "K_CACHE_OUT": ("b", "hkv", "n_total", "d"),
                },
                reduce_dims=(),
                reduce_type="none",
                attrs={"axis": 2},
                dim_constraints={"n_cache": n_cache, "n_append": append_len, "n_total": kv_len},
            )
        )
        dag.add_op(
            OpSpec(
                op_id="Op_V_Append",
                op_type="KVAppend",
                inputs=("V_APPEND", "V_CACHE_IN"),
                outputs=("V_CACHE_OUT",),
                iter_dims=("b", "hkv", "n_cache", "n_append", "n_total", "d"),
                tensor_dims={
                    "V_APPEND": ("b", "hkv", "n_append", "d"),
                    "V_CACHE_IN": ("b", "hkv", "n_cache", "d"),
                    "V_CACHE_OUT": ("b", "hkv", "n_total", "d"),
                },
                reduce_dims=(),
                reduce_type="none",
                attrs={"axis": 2},
                dim_constraints={"n_cache": n_cache, "n_append": append_len, "n_total": kv_len},
            )
        )
        dag.add_op(
            OpSpec(
                op_id="Op_K_CacheView",
                op_type="Identity",
                inputs=("K_CACHE_OUT",),
                outputs=("K_CTX",),
                iter_dims=("b", "hkv", "n", "d"),
                tensor_dims={"K_CACHE_OUT": ("b", "hkv", "n", "d"), "K_CTX": ("b", "hkv", "n", "d")},
                reduce_dims=(),
                reduce_type="none",
                attrs={"view_of_cache": True, "alias_ok": True},
            )
        )
        dag.add_op(
            OpSpec(
                op_id="Op_V_CacheView",
                op_type="Identity",
                inputs=("V_CACHE_OUT",),
                outputs=("V_CTX",),
                iter_dims=("b", "hkv", "n", "d"),
                tensor_dims={"V_CACHE_OUT": ("b", "hkv", "n", "d"), "V_CTX": ("b", "hkv", "n", "d")},
                reduce_dims=(),
                reduce_type="none",
                attrs={"view_of_cache": True, "alias_ok": True},
            )
        )

    dag.add_op(
        OpSpec(
            op_id="Op_K_T",
            op_type="Transpose",
            inputs=("K_CTX",),
            outputs=("K_T",),
            iter_dims=("b", "hkv", "n", "d"),
            tensor_dims={"K_CTX": ("b", "hkv", "n", "d"), "K_T": ("b", "hkv", "d", "n")},
            reduce_dims=(),
            reduce_type="none",
            attrs={"perm": (0, 1, 3, 2)},
        )
    )
    dag.add_op(
        OpSpec(
            op_id="Op_QK",
            op_type="MatMul",
            inputs=("Q", "K_T_ATTN"),
            outputs=("SCORES",),
            iter_dims=("b", "hq", "m", "n", "d"),
            tensor_dims={"Q": ("b", "hq", "m", "d"), "K_T_ATTN": ("b", "hq", "d", "n"), "SCORES": ("b", "hq", "m", "n")},
            reduce_dims=("d",),
            reduce_type="full",
            attrs={"transpose_a": False, "transpose_b": False, "attn_type": spec.attn_type},
        )
    )
    dag.add_op(
        OpSpec(
            op_id="Op_Softmax",
            op_type="Softmax",
            inputs=("SCORES",),
            outputs=("STATS", "PROBS"),
            iter_dims=("b", "hq", "m", "n"),
            tensor_dims={"SCORES": ("b", "hq", "m", "n"), "STATS": ("b", "hq", "m"), "PROBS": ("b", "hq", "m", "n")},
            reduce_dims=("n",),
            reduce_type="partial",
            attrs={"axis": -1, "mask_type": spec.mask_type},
        )
    )
    dag.add_op(
        OpSpec(
            op_id="Op_AV",
            op_type="MatMul",
            inputs=("STATS", "PROBS", "V_ATTN"),
            outputs=("O",),
            iter_dims=("b", "hq", "m", "n", "d"),
            tensor_dims={"STATS": ("b", "hq", "m"), "PROBS": ("b", "hq", "m", "n"), "V_ATTN": ("b", "hq", "n", "d"), "O": ("b", "hq", "m", "d")},
            reduce_dims=("n",),
            reduce_type="full",
            attrs={"transpose_a": False, "transpose_b": False, "attn_type": spec.attn_type},
        )
    )
    dag.add_op(
        OpSpec(
            op_id="Op_K_HeadBroadcast",
            op_type="HeadBroadcast",
            inputs=("K_T",),
            outputs=("K_T_ATTN",),
            iter_dims=("b", "hq", "hkv", "d", "n"),
            tensor_dims={"K_T": ("b", "hkv", "d", "n"), "K_T_ATTN": ("b", "hq", "d", "n")},
            reduce_dims=(),
            reduce_type="none",
            attrs={"broadcast_axis": "head", "from_heads": hkv, "to_heads": hq, "group_size": spec.kv_head_broadcast, "attn_type": spec.attn_type},
            dim_constraints={"hq": hq, "hkv": hkv},
        )
    )
    dag.add_op(
        OpSpec(
            op_id="Op_V_HeadBroadcast",
            op_type="HeadBroadcast",
            inputs=("V_CTX",),
            outputs=("V_ATTN",),
            iter_dims=("b", "hq", "hkv", "n", "d"),
            tensor_dims={"V_CTX": ("b", "hkv", "n", "d"), "V_ATTN": ("b", "hq", "n", "d")},
            reduce_dims=(),
            reduce_type="none",
            attrs={"broadcast_axis": "head", "from_heads": hkv, "to_heads": hq, "group_size": spec.kv_head_broadcast, "attn_type": spec.attn_type},
            dim_constraints={"hq": hq, "hkv": hkv},
        )
    )

    if spec.mode == "decode":
        dag.add_edge(OpDataEdge("Op_K_Append", "Op_K_CacheView", "K_CACHE_OUT"))
        dag.add_edge(OpDataEdge("Op_K_CacheView", "Op_K_T", "K_CTX"))
        dag.add_edge(OpDataEdge("Op_V_Append", "Op_V_CacheView", "V_CACHE_OUT"))
        dag.add_edge(OpDataEdge("Op_V_CacheView", "Op_V_HeadBroadcast", "V_CTX"))

    dag.add_edge(OpDataEdge("Op_K_T", "Op_K_HeadBroadcast", "K_T"))
    dag.add_edge(OpDataEdge("Op_K_HeadBroadcast", "Op_QK", "K_T_ATTN"))
    dag.add_edge(OpDataEdge("Op_QK", "Op_Softmax", "SCORES"))
    dag.add_edge(OpDataEdge("Op_Softmax", "Op_AV", "STATS"))
    dag.add_edge(OpDataEdge("Op_Softmax", "Op_AV", "PROBS"))
    dag.add_edge(OpDataEdge("Op_V_HeadBroadcast", "Op_AV", "V_ATTN"))

    dag.validate()
    return dag
