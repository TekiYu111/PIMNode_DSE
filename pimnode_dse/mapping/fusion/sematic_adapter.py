from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from .graph_adapter import WorkloadFusionGraphAdapter
except Exception:
    try:
        from .graph_adapter_refactored import WorkloadFusionGraphAdapter
    except Exception:
        WorkloadFusionGraphAdapter = None  # type: ignore


@dataclass(frozen=True)
class TensorSemantic:
    name: str
    role: str
    category: str
    is_stateful: bool
    is_graph_input: bool
    is_graph_output: bool


@dataclass(frozen=True)
class OpSemantic:
    op_id: str
    kind: str
    phase: str
    stage: str
    is_decision_op: bool
    is_attachable_singleton: bool


@dataclass(frozen=True)
class AttentionFusionContext:
    workload_family: str
    workload_mode: str
    attn_type: str
    num_heads: int
    num_kv_heads: int
    head_share: int
    has_state_in: bool
    has_state_out: bool
    has_kv_append: bool
    has_cache_view: bool


class WorkloadFusionSemanticAdapter:
    """
    Stable semantic view over workload facts.

    Responsibilities:
    - tensor semantics
    - op semantics
    - workload-level attention context

    Non-responsibilities:
    - fusion-group scoring
    - placement or tiling hints
    - group-level execution contracts
    """

    _ROLE_MAP = {
        "Q": "Q",
        "K_CTX": "K_CTX",
        "V_CTX": "V_CTX",
        "K_CACHE_IN": "K_CACHE_IN",
        "V_CACHE_IN": "V_CACHE_IN",
        "K_APPEND": "K_APPEND",
        "V_APPEND": "V_APPEND",
        "K_CACHE_OUT": "K_CACHE_OUT",
        "V_CACHE_OUT": "V_CACHE_OUT",
        "K_T": "K_T",
        "K_T_ATTN": "K_T_ATTN",
        "V_ATTN": "V_ATTN",
        "SCORES": "SCORES",
        "STATS": "STATS",
        "PROBS": "PROBS",
        "O": "O",
    }

    def __init__(self, source: Any) -> None:
        if WorkloadFusionGraphAdapter is not None and isinstance(source, WorkloadFusionGraphAdapter):
            self._graph = source
        elif WorkloadFusionGraphAdapter is not None:
            self._graph = WorkloadFusionGraphAdapter(source)
        else:
            raise RuntimeError("WorkloadFusionGraphAdapter is unavailable.")

    @property
    def graph(self) -> WorkloadFusionGraphAdapter:
        return self._graph

    def workload_family(self) -> str:
        return "attention"

    def workload_mode(self) -> str:
        spec = getattr(self._graph.dag, "spec", None)
        return str(getattr(spec, "mode", "unknown"))

    def attention_context(self) -> AttentionFusionContext:
        num_heads = self._infer_num_heads()
        num_kv_heads = self._infer_num_kv_heads(default=num_heads)
        if num_heads <= 0:
            num_heads = 1
        if num_kv_heads <= 0:
            num_kv_heads = num_heads
        head_share = max(1, num_heads // max(1, num_kv_heads))

        op_ids = self._graph.topo_order()
        return AttentionFusionContext(
            workload_family=self.workload_family(),
            workload_mode=self.workload_mode(),
            attn_type=self._infer_attn_type(num_heads=num_heads, num_kv_heads=num_kv_heads),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_share=head_share,
            has_state_in=self._has_tensor_cat("state", as_input=True),
            has_state_out=self._has_tensor_cat("state", as_output=True),
            has_kv_append=any(self.op_kind(op_id) == "kv_append" for op_id in op_ids),
            has_cache_view=any(self.op_kind(op_id) == "cache_view" for op_id in op_ids),
        )

    def tensor_semantic(self, tensor: str) -> TensorSemantic:
        role = self.tensor_role(tensor)
        cat = self.tensor_category(tensor)
        return TensorSemantic(
            name=tensor,
            role=role,
            category=cat,
            is_stateful=(cat == "state"),
            is_graph_input=(self._graph.producer_of(tensor) is None),
            is_graph_output=(len(self._graph.consumers_of(tensor)) == 0),
        )

    def tensor_role(self, tensor: str) -> str:
        raw = self._tensor_attr(tensor, "special_role") or self._tensor_attr(tensor, "role")
        if raw is not None:
            raw_text = str(raw)
            if raw_text in self._ROLE_MAP:
                return self._ROLE_MAP[raw_text]
        if tensor in self._ROLE_MAP:
            return self._ROLE_MAP[tensor]
        return f"UNKNOWN::{tensor}"

    def tensor_category(self, tensor: str) -> str:
        role = self.tensor_role(tensor)
        prod = self._graph.producer_of(tensor)

        if role in {"K_CACHE_IN", "V_CACHE_IN", "K_CACHE_OUT", "V_CACHE_OUT"}:
            return "state"
        if role == "O":
            return "output"
        if role == "STATS":
            return "stat"
        if role == "Q" and prod is None:
            return "input"
        if role in {"K_APPEND", "V_APPEND", "K_CTX", "V_CTX"} and prod is None:
            return "input"
        return "act"

    def is_state_tensor(self, tensor: str) -> bool:
        return self.tensor_category(tensor) == "state"

    def is_boundary_writable_state(self, tensor: str) -> bool:
        return self.tensor_role(tensor) in {"K_CACHE_OUT", "V_CACHE_OUT"}

    def op_semantic(self, op_id: str) -> OpSemantic:
        kind = self.op_kind(op_id)
        phase = self.op_phase(op_id)
        stage = self.op_stage(op_id)
        return OpSemantic(
            op_id=op_id,
            kind=kind,
            phase=phase,
            stage=stage,
            is_decision_op=self.is_decision_op(op_id),
            is_attachable_singleton=self.is_attachable_singleton(op_id),
        )

    def op_kind(self, op_id: str) -> str:
        op = self._graph.op(op_id)
        op_type = str(getattr(op, "op_type", "unknown"))
        attrs = getattr(op, "attrs", {}) or {}
        outs = set(str(x) for x in getattr(op, "outputs", ()))
        ins = set(str(x) for x in getattr(op, "inputs", ()))

        if op_type == "KVAppend":
            return "kv_append"
        if op_type == "Identity" and attrs.get("view_of_cache"):
            return "cache_view"
        if op_type == "Transpose":
            return "transpose"
        if op_type == "HeadBroadcast":
            return "head_broadcast"
        if op_type == "Softmax":
            return "softmax"
        if op_type == "MatMul":
            if "SCORES" in outs and "Q" in ins and "K_T_ATTN" in ins:
                return "qk"
            if ("O" in outs or "PARTIAL_O" in outs) and "PROBS" in ins and "V_ATTN" in ins:
                return "av"
        return f"unknown::{op_type}"

    def op_phase(self, op_id: str) -> str:
        kind = self.op_kind(op_id)
        if kind in {"kv_append", "cache_view", "transpose", "head_broadcast"}:
            return "frontend"
        if kind in {"qk", "softmax", "av"}:
            return "core"
        return "unknown"

    def op_stage(self, op_id: str) -> str:
        kind = self.op_kind(op_id)
        if kind == "kv_append":
            return "kv_update"
        if kind == "cache_view":
            return "kv_view"
        if kind == "transpose":
            return "k_transform"
        if kind == "head_broadcast":
            op = self._graph.op(op_id)
            inputs = set(str(x) for x in getattr(op, "inputs", ()))
            if "K_T" in inputs:
                return "k_transform"
            if "V_CTX" in inputs:
                return "v_transform"
            return "transform"
        if kind == "qk":
            return "score"
        if kind == "softmax":
            return "normalize"
        if kind == "av":
            return "value"
        return "unknown"

    def is_decision_op(self, op_id: str) -> bool:
        return self.op_kind(op_id) in {"qk", "softmax", "av"}

    def is_attachable_singleton(self, op_id: str) -> bool:
        return self.attachable_class(op_id) != "none"

    def attachable_class(self, op_id: str) -> str:
        kind = self.op_kind(op_id)
        if kind == "kv_append":
            return "state_update"
        if kind == "cache_view":
            return "state_view"
        if kind in {"transpose", "head_broadcast"}:
            return "transform"
        return "none"

    def boundary_tensor_class(
        self,
        tensor: str,
        *,
        is_input: bool,
        is_output: bool,
    ) -> str:
        cat = self.tensor_category(tensor)
        role = self.tensor_role(tensor)

        if is_input:
            if cat == "state":
                return "state_input"
            if cat == "input":
                return "graph_input"
            return "forward_input"

        if is_output:
            if role == "O":
                return "final_output"
            if role == "STATS":
                return "stat_output"
            if cat == "state":
                return "state_output"
            return "forward_output"

        return "internal"

    def preferred_main_chain_order(self) -> Tuple[str, ...]:
        return ("qk", "softmax", "av")

    def _tensor_attr(self, tensor: str, name: str) -> Any:
        obj = self._graph.tensor(tensor)
        if obj is None:
            return None
        return getattr(obj, name, None)

    def _all_tensors(self) -> Tuple[str, ...]:
        names = set(self._graph.source_tensors())
        names.update(self._graph.sink_tensors())
        for op_id in self._graph.op_names():
            op = self._graph.op(op_id)
            names.update(str(x) for x in getattr(op, "inputs", ()))
            names.update(str(x) for x in getattr(op, "outputs", ()))
        return tuple(sorted(names))

    def _has_tensor_cat(self, cat: str, *, as_input: bool = False, as_output: bool = False) -> bool:
        for tensor in self._all_tensors():
            if self.tensor_category(tensor) != cat:
                continue
            if as_input and self._graph.producer_of(tensor) is not None:
                continue
            if as_output and self._graph.consumers_of(tensor):
                continue
            return True
        return False

    def _infer_num_heads(self) -> int:
        return self._infer_spec_int(("num_heads", "n_heads", "heads"), default=1)

    def _infer_num_kv_heads(self, *, default: int) -> int:
        return self._infer_spec_int(("num_kv_heads", "kv_heads", "n_kv_heads"), default=default)

    def _infer_spec_int(self, keys: Sequence[str], *, default: int) -> int:
        spec = getattr(self._graph.dag, "spec", None)
        if spec is None:
            return default
        for key in keys:
            if not hasattr(spec, key):
                continue
            value = getattr(spec, key)
            try:
                out = int(value)
            except (TypeError, ValueError):
                continue
            if out > 0:
                return out
        return default

    @staticmethod
    def _infer_attn_type(*, num_heads: int, num_kv_heads: int) -> str:
        if num_heads <= 0 or num_kv_heads <= 0:
            return "mha"
        if num_kv_heads == num_heads:
            return "mha"
        if num_kv_heads == 1:
            return "mqa"
        return "gqa"


__all__ = [
    "AttentionFusionContext",
    "OpSemantic",
    "TensorSemantic",
    "WorkloadFusionSemanticAdapter",
]
