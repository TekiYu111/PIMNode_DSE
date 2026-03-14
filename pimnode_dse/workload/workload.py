from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# -------------------------------------------------
# AttentionWorkloadSpec
# -------------------------------------------------
@dataclass(frozen=True)
class AttentionWorkloadSpec:
    mode: str          # "prefill" | "decode"
    attn_type: str     # "mha" | "gqa" | "mqa"
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
        if self.Hkv == 0:
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


# -------------------------------------------------
# TensorSpec
# -------------------------------------------------
@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    role: str = "intermediate"         # input / output / intermediate / state
    special_role: Optional[str] = None # KV_CACHE / PARTIAL_O / STATS / SCORES / PROBS


# -------------------------------------------------
# DomainSpec
# -------------------------------------------------
@dataclass(frozen=True)
class DomainSpec:
    kind: str = "none"
    params: Dict[str, Any] = field(default_factory=dict)

    def required_vars(self) -> Set[str]:
        if self.kind in ("none",):
            return set()
        if self.kind in ("causal", "sliding_window", "block_sparse"):
            return {"m", "n"}
        if self.kind == "decode_kv":
            return set(self.params.get("vars", []))
        return set(self.params.get("vars", []))


# -------------------------------------------------
# OpSpec
# -------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    op_id: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    index_vars: Optional[List[str]] = None
    tensor_index: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    phase: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------
# WorkloadDAG
# -------------------------------------------------
@dataclass
class WorkloadDAG:
    name: str
    ops: List[OpSpec] = field(default_factory=list)
    tensors: Dict[str, TensorSpec] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (src_op_id, dst_op_id)
    _edge_tensor: Dict[Tuple[str, str], str] = field(default_factory=dict, repr=False)
    spec: Optional[AttentionWorkloadSpec] = None

    def add_tensor(self, tensor: TensorSpec) -> None:
        if tensor.name in self.tensors:
            raise ValueError(f"Duplicate tensor name: {tensor.name}")
        self.tensors[tensor.name] = tensor

    def add_op(self, op: OpSpec) -> None:
        if op.op_id in self.op_names():
            raise ValueError(f"Duplicate op_id: {op.op_id}")

        self.ops.append(op)

        # auto-generate edges based on producer-consumer relation
        for inp in op.inputs:
            for src_op in self.ops:
                if src_op.op_id == op.op_id:
                    continue
                if inp in src_op.outputs:
                    edge = (src_op.op_id, op.op_id)
                    if edge not in self.edges:
                        self.edges.append(edge)
                    if edge not in self._edge_tensor:
                        self._edge_tensor[edge] = inp

    def op_names(self) -> Set[str]:
        return {op.op_id for op in self.ops}

    def get_op(self, op_id: str) -> OpSpec:
        for op in self.ops:
            if op.op_id == op_id:
                return op
        raise KeyError(f"Op {op_id!r} not found")

    def op(self, op_id: str) -> OpSpec:
        return self.get_op(op_id)

    def get_tensor(self, tensor_name: str) -> TensorSpec:
        if tensor_name not in self.tensors:
            raise KeyError(f"Tensor {tensor_name!r} not found")
        return self.tensors[tensor_name]

    def tensor(self, tensor_name: str) -> TensorSpec:
        return self.get_tensor(tensor_name)

    def register_edge_tensor(self, u: str, v: str, tensor_name: str) -> None:
        if tensor_name not in self.tensors:
            raise KeyError(f"Cannot register unknown tensor {tensor_name!r} for edge ({u} -> {v})")
        self._edge_tensor[(u, v)] = tensor_name
        if (u, v) not in self.edges:
            self.edges.append((u, v))

    def get_edge_tensor_name(self, u: str, v: str) -> str:
        tname = self._edge_tensor.get((u, v))
        if tname is None:
            raise KeyError(f"No tensor registered for edge ({u} -> {v})")
        return tname

    def get_edge_tensor_dims(self, u: str, v: str) -> List[int]:
        tname = self.get_edge_tensor_name(u, v)
        return list(self.tensors[tname].shape)

    def get_edge_reduction_dim(self, u: str, v: str) -> Optional[str]:
        """
        Conservative helper for FusionGene pipeline checks.

        Current simplified policy:
        - if producer op declares attrs['reduction_dim'], use it
        - otherwise return None
        """
        op_u = self.get_op(u)
        red = op_u.attrs.get("reduction_dim")
        if red is None:
            return None
        return str(red)

    def topo_order(self) -> List[str]:
        in_degree = {op.op_id: 0 for op in self.ops}
        adj = {op.op_id: [] for op in self.ops}

        for u, v in self.edges:
            if u not in adj:
                adj[u] = []
            if v not in in_degree:
                in_degree[v] = 0
            adj[u].append(v)
            in_degree[v] += 1

        queue = [op_id for op_id, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            n = queue.pop(0)
            order.append(n)
            for nbr in adj.get(n, []):
                in_degree[nbr] -= 1
                if in_degree[nbr] == 0:
                    queue.append(nbr)

        if len(order) != len(self.ops):
            raise ValueError("DAG contains a cycle or disconnected invalid edge references")
        return order

    def validate(self) -> None:
        if not self.name:
            raise ValueError("WorkloadDAG.name must be non-empty")

        if self.spec is not None:
            self.spec.validate()

        op_ids = [op.op_id for op in self.ops]
        if len(op_ids) != len(set(op_ids)):
            raise ValueError("Duplicate op_id found in WorkloadDAG")

        tensor_names = set(self.tensors.keys())
        if not tensor_names:
            raise ValueError("WorkloadDAG has no tensors")

        for op in self.ops:
            if not op.op_id:
                raise ValueError("Found op with empty op_id")
            if not op.op_type:
                raise ValueError(f"Op {op.op_id!r} has empty op_type")

            for t in op.inputs + op.outputs:
                if t not in tensor_names:
                    raise KeyError(f"Tensor {t!r} referenced by op {op.op_id!r} is not defined")

        op_id_set = self.op_names()
        for u, v in self.edges:
            if u not in op_id_set or v not in op_id_set:
                raise KeyError(f"Invalid edge ({u!r} -> {v!r}) in WorkloadDAG")

        for (u, v), tname in self._edge_tensor.items():
            if u not in op_id_set or v not in op_id_set:
                raise KeyError(f"Invalid registered edge ({u!r} -> {v!r})")
            if tname not in tensor_names:
                raise KeyError(f"Registered edge tensor {tname!r} for ({u} -> {v}) is undefined")

        self.topo_order()


# -------------------------------------------------
# Parameterized Attention DAG builder
# -------------------------------------------------
def build_attention_dag(
    name: str,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    d_model: int,
    phase: str = "prefill",
    variant: str = "MHA",   # "MHA", "GQA", "MQA", "Decode"
) -> WorkloadDAG:
    """
    Build a compact attention-style DAG that is compatible with current
    FusionGene / MappingBuilder / placement role binding.

    Phase:
      - prefill / decode

    Variants:
      - MHA / GQA / MQA : normal attention DAG, mode depends on phase
      - Decode          : force decode-style DAG with KV cache semantics
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
    dag.add_tensor(TensorSpec("SCORES", (batch_size, hq, q_len, kv_total), role="intermediate", special_role="SCORES"))
    dag.add_tensor(TensorSpec("STATS", (batch_size, hq, q_len), role="intermediate", special_role="STATS"))
    dag.add_tensor(TensorSpec("PROBS", (batch_size, hq, q_len, kv_total), role="intermediate", special_role="PROBS"))
    dag.add_tensor(TensorSpec("PARTIAL_O", (batch_size, hq, q_len, head_dim), role="intermediate", special_role="PARTIAL_O"))
    dag.add_tensor(TensorSpec("O", (batch_size, hq, q_len, head_dim), role="output"))

    if mode == "decode":
        dag.add_tensor(TensorSpec("K_APPEND", (batch_size, hkv, 1, head_dim), role="input"))
        dag.add_tensor(TensorSpec("V_APPEND", (batch_size, hkv, 1, head_dim), role="input"))

    op_qk = OpSpec(
        op_id="Op_QK",
        op_type="MatMul",
        inputs=["Q", "K"],
        outputs=["SCORES"],
        index_vars=["b", "h", "m", "n", "d"],
        tensor_index={
            "Q": ("b", "h", "m", "d"),
            "K": ("b", "h", "n", "d"),
            "SCORES": ("b", "h", "m", "n"),
        },
        phase=phase_norm,
        attrs={"reduction_dim": "d"},
    )
    dag.add_op(op_qk)

    op_softmax = OpSpec(
        op_id="Op_Softmax",
        op_type="Softmax",
        inputs=["SCORES"],
        outputs=["STATS", "PROBS"],
        index_vars=["b", "h", "m", "n"],
        tensor_index={
            "SCORES": ("b", "h", "m", "n"),
            "STATS": ("b", "h", "m"),
            "PROBS": ("b", "h", "m", "n"),
        },
        phase=phase_norm,
        attrs={},
    )
    dag.add_op(op_softmax)

    op_av = OpSpec(
        op_id="Op_AV",
        op_type="MatMul",
        inputs=["PROBS", "V"],
        outputs=["PARTIAL_O"],
        index_vars=["b", "h", "m", "n", "d"],
        tensor_index={
            "PROBS": ("b", "h", "m", "n"),
            "V": ("b", "h", "n", "d"),
            "PARTIAL_O": ("b", "h", "m", "d"),
        },
        phase=phase_norm,
        attrs={"reduction_dim": "n"},
    )
    dag.add_op(op_av)

    op_out = OpSpec(
        op_id="Op_O",
        op_type="Identity",
        inputs=["PARTIAL_O"],
        outputs=["O"],
        index_vars=["b", "h", "m", "d"],
        tensor_index={
            "PARTIAL_O": ("b", "h", "m", "d"),
            "O": ("b", "h", "m", "d"),
        },
        phase=phase_norm,
        attrs={},
    )
    dag.add_op(op_out)

    if mode == "decode":
        op_k_append = OpSpec(
            op_id="Op_K_Append",
            op_type="KVAppend",
            inputs=["K_APPEND", "K"],
            outputs=["K"],
            index_vars=["b", "h", "n", "d"],
            tensor_index={
                "K_APPEND": ("b", "h", "n", "d"),
                "K": ("b", "h", "n", "d"),
            },
            phase=phase_norm,
            attrs={},
        )
        op_v_append = OpSpec(
            op_id="Op_V_Append",
            op_type="KVAppend",
            inputs=["V_APPEND", "V"],
            outputs=["V"],
            index_vars=["b", "h", "n", "d"],
            tensor_index={
                "V_APPEND": ("b", "h", "n", "d"),
                "V": ("b", "h", "n", "d"),
            },
            phase=phase_norm,
            attrs={},
        )
        dag.add_op(op_k_append)
        dag.add_op(op_v_append)

    dag.register_edge_tensor("Op_QK", "Op_Softmax", "SCORES")
    dag.register_edge_tensor("Op_Softmax", "Op_AV", "PROBS")
    dag.register_edge_tensor("Op_AV", "Op_O", "PARTIAL_O")

    dag.validate()
    return dag


__all__ = [
    "AttentionWorkloadSpec",
    "TensorSpec",
    "DomainSpec",
    "OpSpec",
    "WorkloadDAG",
    "build_attention_dag",
]
