from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# AttentionWorkloadSpec
# ============================================================

@dataclass(frozen=True)
class AttentionWorkloadSpec:
    mode: str           # "prefill" | "decode"
    attn_type: str      # "mha" | "gqa" | "mqa"
    B: int
    Q_len: int
    KV_len: int
    Dh: int
    Hq: int
    Hkv: int
    mask_type: str = "causal"
    kv_cache_enabled: bool = False
    dtype_bytes: int = 2
    cache_len_before: Optional[int] = None
    append_len: Optional[int] = None

    @property
    def group(self) -> int:
        if self.Hkv <= 0:
            raise ValueError("Hkv must be > 0")
        return self.Hq // self.Hkv

    @property
    def KV_total(self) -> int:
        if self.mode == "decode":
            if self.cache_len_before is None or self.append_len is None:
                raise ValueError("decode mode requires cache_len_before and append_len")
            return self.cache_len_before + self.append_len
        return self.KV_len

    def validate(self) -> None:
        if self.mode not in {"prefill", "decode"}:
            raise ValueError(f"Unsupported mode: {self.mode!r}")
        if self.attn_type not in {"mha", "gqa", "mqa"}:
            raise ValueError(f"Unsupported attn_type: {self.attn_type!r}")

        if self.B <= 0 or self.Q_len <= 0 or self.KV_len <= 0 or self.Dh <= 0:
            raise ValueError("B, Q_len, KV_len, Dh must all be > 0")

        if self.Hq <= 0 or self.Hkv <= 0:
            raise ValueError("Hq and Hkv must be > 0")

        if self.Hq % self.Hkv != 0:
            raise ValueError("Hq must be divisible by Hkv")

        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be > 0")

        if self.mode == "decode":
            if self.cache_len_before is None or self.append_len is None:
                raise ValueError("decode mode requires cache_len_before and append_len")
            if self.cache_len_before < 0 or self.append_len <= 0:
                raise ValueError("cache_len_before must be >= 0 and append_len must be > 0")


# ============================================================
# TensorSpec
# ============================================================

@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    role: str = "intermediate"      # input / output / intermediate / state
    special_role: Optional[str] = None


# ============================================================
# DomainSpec
# ============================================================

@dataclass(frozen=True)
class DomainSpec:
    kind: str = "none"
    params: Dict[str, Any] = field(default_factory=dict)

    def required_vars(self) -> Set[str]:
        if self.kind == "none":
            return set()
        if self.kind in {"causal", "sliding_window", "block_sparse"}:
            return {"m", "n"}
        if self.kind == "decode_kv":
            return set(self.params.get("vars", []))
        return set(self.params.get("vars", []))


# ============================================================
# OpSpec
# ============================================================

@dataclass(frozen=True)
class OpSpec:
    """
    算子节点：
    - iter_dims: 逻辑迭代维度
    - tensor_dims: 各输入/输出张量在该算子中的逻辑索引
    - reduce_dims: 该算子的约简维度
    """
    op_id: str
    op_type: str
    inputs: List[str]
    outputs: List[str]

    iter_dims: Tuple[str, ...] = ()
    tensor_dims: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    reduce_dims: Tuple[str, ...] = ()

    phase: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# EdgeSpec
# ============================================================

@dataclass(frozen=True)
class EdgeSpec:
    """
    数据流边是一等对象。

    - src / dst: producer / consumer
    - tensor: 边上传递的张量
    - src_dims / dst_dims: 该张量在 producer / consumer 两侧的逻辑维度
    - reduce_dims: 若这条边与 reduction 直接相关，可显式记录
    """
    src: str
    dst: str
    tensor: str
    src_dims: Tuple[str, ...]
    dst_dims: Tuple[str, ...]
    reduce_dims: Tuple[str, ...] = ()


# ============================================================
# WorkloadDAG
# ============================================================

@dataclass
class WorkloadDAG:
    name: str
    spec: Optional[AttentionWorkloadSpec] = None

    ops: Dict[str, OpSpec] = field(default_factory=dict)
    tensors: Dict[str, TensorSpec] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], EdgeSpec] = field(default_factory=dict)

    # --------------------------
    # Add / register
    # --------------------------

    def add_tensor(self, tensor: TensorSpec) -> None:
        if tensor.name in self.tensors:
            raise ValueError(f"Duplicate tensor name: {tensor.name}")
        self.tensors[tensor.name] = tensor

    def add_op(self, op: OpSpec) -> None:
        if op.op_id in self.ops:
            raise ValueError(f"Duplicate op_id: {op.op_id}")

        for t in op.inputs + op.outputs:
            if t not in self.tensors:
                raise KeyError(f"Tensor {t!r} referenced by op {op.op_id!r} is not defined")

        missing_tensor_dims = [t for t in (op.inputs + op.outputs) if t not in op.tensor_dims]
        if missing_tensor_dims:
            raise ValueError(
                f"Op {op.op_id!r} 缺少 tensor_dims 定义: {missing_tensor_dims}"
            )

        unknown_reduce = set(op.reduce_dims) - set(op.iter_dims)
        if unknown_reduce:
            raise ValueError(
                f"Op {op.op_id!r} reduce_dims must be subset of iter_dims: {sorted(unknown_reduce)}"
            )

        self.ops[op.op_id] = op

    def add_edge(self, edge: EdgeSpec) -> None:
        if edge.src not in self.ops or edge.dst not in self.ops:
            raise KeyError(f"Invalid edge endpoints: ({edge.src!r} -> {edge.dst!r})")
        if edge.tensor not in self.tensors:
            raise KeyError(f"Invalid edge tensor: {edge.tensor!r}")

        src_op = self.ops[edge.src]
        dst_op = self.ops[edge.dst]

        if edge.tensor not in src_op.outputs:
            raise ValueError(
                f"Edge tensor {edge.tensor!r} is not an output of producer {edge.src!r}"
            )
        if edge.tensor not in dst_op.inputs:
            raise ValueError(
                f"Edge tensor {edge.tensor!r} is not an input of consumer {edge.dst!r}"
            )

        expected_src_dims = src_op.tensor_dims[edge.tensor]
        expected_dst_dims = dst_op.tensor_dims[edge.tensor]

        if tuple(edge.src_dims) != tuple(expected_src_dims):
            raise ValueError(
                f"Edge ({edge.src}->{edge.dst}) src_dims mismatch: "
                f"declared={edge.src_dims}, expected={expected_src_dims}"
            )
        if tuple(edge.dst_dims) != tuple(expected_dst_dims):
            raise ValueError(
                f"Edge ({edge.src}->{edge.dst}) dst_dims mismatch: "
                f"declared={edge.dst_dims}, expected={expected_dst_dims}"
            )

        self.edges[(edge.src, edge.dst)] = edge

    # --------------------------
    # Basic queries
    # --------------------------

    def op_ids(self) -> List[str]:
        return list(self.ops.keys())

    def op_names(self) -> Set[str]:
        return set(self.ops.keys())

    def get_op(self, op_id: str) -> OpSpec:
        if op_id not in self.ops:
            raise KeyError(f"Op {op_id!r} not found")
        return self.ops[op_id]

    def op(self, op_id: str) -> OpSpec:
        return self.get_op(op_id)

    def get_tensor(self, tensor_name: str) -> TensorSpec:
        if tensor_name not in self.tensors:
            raise KeyError(f"Tensor {tensor_name!r} not found")
        return self.tensors[tensor_name]

    def tensor(self, tensor_name: str) -> TensorSpec:
        return self.get_tensor(tensor_name)

    def get_edge(self, u: str, v: str) -> EdgeSpec:
        if (u, v) not in self.edges:
            raise KeyError(f"Edge ({u!r} -> {v!r}) not found")
        return self.edges[(u, v)]

    def edge(self, u: str, v: str) -> EdgeSpec:
        return self.get_edge(u, v)

    # --------------------------
    # Compatibility helpers
    # --------------------------

    def get_edge_tensor_name(self, u: str, v: str) -> str:
        return self.get_edge(u, v).tensor

    def get_edge_logical_dims(self, u: str, v: str) -> Tuple[str, ...]:
        """
        默认返回 producer 侧逻辑维度。
        给 fusion / tiling 用。
        """
        return self.get_edge(u, v).src_dims

    def get_edge_tensor_shape(self, u: str, v: str) -> Tuple[int, ...]:
        """
        返回边上传递张量的数值 shape。
        给容量估计 / analysis 用。
        """
        tname = self.get_edge(u, v).tensor
        return self.tensors[tname].shape

    def get_edge_reduction_dims(self, u: str, v: str) -> Tuple[str, ...]:
        edge = self.get_edge(u, v)
        if edge.reduce_dims:
            return edge.reduce_dims
        return self.get_op(u).reduce_dims

    # --------------------------
    # Graph traversal
    # --------------------------

    def predecessors(self, op_id: str) -> List[str]:
        return [u for (u, v) in self.edges if v == op_id]

    def successors(self, op_id: str) -> List[str]:
        return [v for (u, v) in self.edges if u == op_id]

    def incoming_edges(self, op_id: str) -> List[EdgeSpec]:
        return [edge for (u, v), edge in self.edges.items() if v == op_id]

    def outgoing_edges(self, op_id: str) -> List[EdgeSpec]:
        return [edge for (u, v), edge in self.edges.items() if u == op_id]

    def producer_of(self, tensor_name: str) -> Optional[str]:
        producers = [op_id for op_id, op in self.ops.items() if tensor_name in op.outputs]
        if not producers:
            return None
        if len(producers) > 1:
            raise ValueError(f"Tensor {tensor_name!r} has multiple producers: {producers}")
        return producers[0]

    def consumers_of(self, tensor_name: str) -> List[str]:
        return [op_id for op_id, op in self.ops.items() if tensor_name in op.inputs]

    def boundary_tensors(self, op_set: Set[str]) -> Dict[str, Set[str]]:
        """
        对任意子图返回：
        - inputs: 从子图外流入子图内的 tensor
        - outputs: 从子图内流向子图外的 tensor
        - temps: 仅在子图内部流动的 tensor
        - shared: 实际跨子图边界的 tensor
        """
        inputs: Set[str] = set()
        outputs: Set[str] = set()
        temps: Set[str] = set()
        shared: Set[str] = set()

        for (u, v), edge in self.edges.items():
            u_in = u in op_set
            v_in = v in op_set

            if u_in and v_in:
                temps.add(edge.tensor)
            elif (not u_in) and v_in:
                inputs.add(edge.tensor)
                shared.add(edge.tensor)
            elif u_in and (not v_in):
                outputs.add(edge.tensor)
                shared.add(edge.tensor)

        temps -= inputs
        temps -= outputs

        return {
            "inputs": inputs,
            "outputs": outputs,
            "temps": temps,
            "shared": shared,
        }

    def topo_order(self) -> List[str]:
        in_degree = {op_id: 0 for op_id in self.ops}
        adj: Dict[str, List[str]] = defaultdict(list)

        for (u, v) in self.edges:
            adj[u].append(v)
            in_degree[v] += 1

        q = deque([op_id for op_id, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while q:
            cur = q.popleft()
            order.append(cur)
            for nxt in adj[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    q.append(nxt)

        if len(order) != len(self.ops):
            raise ValueError("DAG contains a cycle or invalid edge references")
        return order

    # --------------------------
    # Validation
    # --------------------------

    def validate(self) -> None:
        if not self.name:
            raise ValueError("WorkloadDAG.name must be non-empty")

        if self.spec is not None:
            self.spec.validate()

        if not self.tensors:
            raise ValueError("WorkloadDAG has no tensors")
        if not self.ops:
            raise ValueError("WorkloadDAG has no ops")

        for op_id, op in self.ops.items():
            if not op_id:
                raise ValueError("Found op with empty op_id")
            if not op.op_type:
                raise ValueError(f"Op {op_id!r} has empty op_type")

            for t in op.inputs + op.outputs:
                if t not in self.tensors:
                    raise KeyError(f"Tensor {t!r} referenced by op {op_id!r} is not defined")

            missing_dims = [t for t in op.inputs + op.outputs if t not in op.tensor_dims]
            if missing_dims:
                raise ValueError(f"Op {op_id!r} missing tensor_dims for: {missing_dims}")

            unknown_reduce = set(op.reduce_dims) - set(op.iter_dims)
            if unknown_reduce:
                raise ValueError(
                    f"Op {op_id!r} reduce_dims must be subset of iter_dims: {sorted(unknown_reduce)}"
                )

        for (u, v), edge in self.edges.items():
            if u not in self.ops or v not in self.ops:
                raise KeyError(f"Invalid edge ({u!r} -> {v!r}) in WorkloadDAG")
            if edge.tensor not in self.tensors:
                raise KeyError(f"Edge tensor {edge.tensor!r} is undefined")
            if edge.src != u or edge.dst != v:
                raise ValueError(f"Edge key and payload mismatch: key=({u},{v}), payload={edge}")

        self.topo_order()


# ============================================================
# Parameterized Attention DAG builder
# ============================================================

def build_attention_dag(
    name: str,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    d_model: int,
    phase: str = "prefill",
    variant: str = "MHA",
) -> WorkloadDAG:
    """
    显式构建 attention 数据流图。
    """

    if batch_size <= 0 or num_heads <= 0 or seq_len <= 0 or d_model <= 0:
        raise ValueError("batch_size, num_heads, seq_len, d_model must all be > 0")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    phase_norm = str(phase).strip()
    if phase_norm not in {"prefill", "decode"}:
        raise ValueError(f"Unsupported phase: {phase!r}")

    variant_norm = str(variant).strip().upper()
    head_dim = d_model // num_heads

    if variant_norm == "MHA":
        attn_type = "mha"
        hq = num_heads
        hkv = num_heads
        mode = phase_norm
    elif variant_norm == "GQA":
        attn_type = "gqa"
        hq = num_heads
        hkv = max(1, num_heads // 2)
        mode = phase_norm
    elif variant_norm in {"MQA", "MULTIQUERY"}:
        attn_type = "mqa"
        hq = num_heads
        hkv = 1
        mode = phase_norm
    elif variant_norm == "DECODE":
        attn_type = "mha"
        hq = num_heads
        hkv = num_heads
        mode = "decode"
        phase_norm = "decode"
    else:
        raise ValueError(f"Unsupported variant: {variant!r}")

    spec = AttentionWorkloadSpec(
        mode=mode,
        attn_type=attn_type,
        B=batch_size,
        Q_len=1 if mode == "decode" else seq_len,
        KV_len=seq_len,
        Dh=head_dim,
        Hq=hq,
        Hkv=hkv,
        mask_type="causal",
        kv_cache_enabled=(mode == "decode"),
        cache_len_before=(seq_len - 1) if mode == "decode" else None,
        append_len=1 if mode == "decode" else None,
    )
    spec.validate()

    dag = WorkloadDAG(name=name, spec=spec)

    q_len = spec.Q_len
    kv_total = spec.KV_total if mode == "decode" else spec.KV_len

    # --------------------------
    # Tensors
    # --------------------------
    dag.add_tensor(TensorSpec("Q", (batch_size, hq, q_len, head_dim), role="input"))
    dag.add_tensor(
        TensorSpec(
            "K",
            (batch_size, hkv, kv_total, head_dim),
            role="state" if mode == "decode" else "input",
            special_role="KV_CACHE",
        )
    )
    dag.add_tensor(
        TensorSpec(
            "V",
            (batch_size, hkv, kv_total, head_dim),
            role="state" if mode == "decode" else "input",
            special_role="KV_CACHE",
        )
    )
    dag.add_tensor(
        TensorSpec(
            "SCORES",
            (batch_size, hq, q_len, kv_total),
            role="intermediate",
            special_role="SCORES",
        )
    )
    dag.add_tensor(
        TensorSpec(
            "STATS",
            (batch_size, hq, q_len),
            role="intermediate",
            special_role="STATS",
        )
    )
    dag.add_tensor(
        TensorSpec(
            "PROBS",
            (batch_size, hq, q_len, kv_total),
            role="intermediate",
            special_role="PROBS",
        )
    )
    dag.add_tensor(
        TensorSpec(
            "PARTIAL_O",
            (batch_size, hq, q_len, head_dim),
            role="intermediate",
            special_role="PARTIAL_O",
        )
    )
    dag.add_tensor(TensorSpec("O", (batch_size, hq, q_len, head_dim), role="output"))

    if mode == "decode":
        dag.add_tensor(TensorSpec("K_APPEND", (batch_size, hkv, 1, head_dim), role="input"))
        dag.add_tensor(TensorSpec("V_APPEND", (batch_size, hkv, 1, head_dim), role="input"))

    # --------------------------
    # Ops
    # --------------------------
    dag.add_op(
        OpSpec(
            op_id="Op_QK",
            op_type="MatMul",
            inputs=["Q", "K"],
            outputs=["SCORES"],
            iter_dims=("b", "h", "m", "n", "d"),
            tensor_dims={
                "Q": ("b", "h", "m", "d"),
                "K": ("b", "h", "n", "d"),
                "SCORES": ("b", "h", "m", "n"),
            },
            reduce_dims=("d",),
            phase=phase_norm,
        )
    )

    dag.add_op(
        OpSpec(
            op_id="Op_Softmax",
            op_type="Softmax",
            inputs=["SCORES"],
            outputs=["STATS", "PROBS"],
            iter_dims=("b", "h", "m", "n"),
            tensor_dims={
                "SCORES": ("b", "h", "m", "n"),
                "STATS": ("b", "h", "m"),
                "PROBS": ("b", "h", "m", "n"),
            },
            reduce_dims=("n",),
            phase=phase_norm,
        )
    )

    dag.add_op(
        OpSpec(
            op_id="Op_AV",
            op_type="MatMul",
            inputs=["PROBS", "V"],
            outputs=["PARTIAL_O"],
            iter_dims=("b", "h", "m", "n", "d"),
            tensor_dims={
                "PROBS": ("b", "h", "m", "n"),
                "V": ("b", "h", "n", "d"),
                "PARTIAL_O": ("b", "h", "m", "d"),
            },
            reduce_dims=("n",),
            phase=phase_norm,
        )
    )

    dag.add_op(
        OpSpec(
            op_id="Op_O",
            op_type="Identity",
            inputs=["PARTIAL_O"],
            outputs=["O"],
            iter_dims=("b", "h", "m", "d"),
            tensor_dims={
                "PARTIAL_O": ("b", "h", "m", "d"),
                "O": ("b", "h", "m", "d"),
            },
            reduce_dims=(),
            phase=phase_norm,
        )
    )

    if mode == "decode":
        dag.add_op(
            OpSpec(
                op_id="Op_K_Append",
                op_type="KVAppend",
                inputs=["K_APPEND", "K"],
                outputs=["K"],
                iter_dims=("b", "h", "n", "d"),
                tensor_dims={
                    "K_APPEND": ("b", "h", "n", "d"),
                    "K": ("b", "h", "n", "d"),
                },
                reduce_dims=(),
                phase=phase_norm,
            )
        )
        dag.add_op(
            OpSpec(
                op_id="Op_V_Append",
                op_type="KVAppend",
                inputs=["V_APPEND", "V"],
                outputs=["V"],
                iter_dims=("b", "h", "n", "d"),
                tensor_dims={
                    "V_APPEND": ("b", "h", "n", "d"),
                    "V": ("b", "h", "n", "d"),
                },
                reduce_dims=(),
                phase=phase_norm,
            )
        )

    # --------------------------
    # Explicit edges
    # --------------------------
    dag.add_edge(
        EdgeSpec(
            src="Op_QK",
            dst="Op_Softmax",
            tensor="SCORES",
            src_dims=("b", "h", "m", "n"),
            dst_dims=("b", "h", "m", "n"),
            reduce_dims=(),
        )
    )

    dag.add_edge(
        EdgeSpec(
            src="Op_Softmax",
            dst="Op_AV",
            tensor="PROBS",
            src_dims=("b", "h", "m", "n"),
            dst_dims=("b", "h", "m", "n"),
            reduce_dims=("n",),
        )
    )

    dag.add_edge(
        EdgeSpec(
            src="Op_AV",
            dst="Op_O",
            tensor="PARTIAL_O",
            src_dims=("b", "h", "m", "d"),
            dst_dims=("b", "h", "m", "d"),
            reduce_dims=(),
        )
    )

    dag.validate()
    return dag


__all__ = [
    "AttentionWorkloadSpec",
    "TensorSpec",
    "DomainSpec",
    "OpSpec",
    "EdgeSpec",
    "WorkloadDAG",
    "build_attention_dag",
]
