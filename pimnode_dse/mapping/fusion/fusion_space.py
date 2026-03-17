from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .fusion_gene import FusionGene, FusionStyle, OpFusionGroup, WorkloadDAGProtocol


# ============================================================
# workload kind
# ============================================================

class WorkloadKind(str, Enum):
    MHA_PREFILL = "mha_prefill"
    DECODE = "decode"
    GEMM_LIKE = "gemm_like"
    UNKNOWN = "unknown"


def infer_workload_kind(dag: WorkloadDAGProtocol) -> WorkloadKind:
    """
    根据 workload DAG 的结构与属性推断默认 workload family。

    约定：
    1. 若存在 Softmax 且 phase/pattern 显示为 prefill，则归为 MHA_PREFILL
    2. 若存在 Softmax 且 phase/pattern 显示为 decode，则归为 DECODE
    3. 若不存在 Softmax，且主链以 MatMul / elementwise 为主，则归为 GEMM_LIKE
    4. 否则回退为 UNKNOWN，并使用较保守默认规则
    """
    op_objs = _get_all_ops(dag)
    op_types = [str(getattr(op, "op_type", "") or "") for op in op_objs]
    op_names = [str(getattr(op, "name", "") or "") for op in op_objs]
    phases = [str(getattr(op, "phase", "") or "") for op in op_objs]

    has_softmax = any(t.lower() == "softmax" or "softmax" in n.lower()
                      for t, n in zip(op_types, op_names))

    # 尝试从 workload / dag 上取 phase
    dag_phase = str(getattr(dag, "phase", "") or "").lower()
    phase_tokens = {p.lower() for p in phases if p}

    if has_softmax:
        if dag_phase == "prefill" or "prefill" in phase_tokens:
            return WorkloadKind.MHA_PREFILL
        if dag_phase == "decode" or "decode" in phase_tokens:
            return WorkloadKind.DECODE

        # 若没显式 phase，则尝试用名字或属性弱判断
        dag_name = str(getattr(dag, "name", "") or "").lower()
        if "prefill" in dag_name:
            return WorkloadKind.MHA_PREFILL
        if "decode" in dag_name:
            return WorkloadKind.DECODE

        # attention-like 但未显式写 phase，默认归 prefill
        return WorkloadKind.MHA_PREFILL

    # 无 softmax 时，若以 MatMul / elementwise 为主，则视作 GEMM-like
    if op_types:
        allowed = {"matmul", "identity", "biasadd", "activation", "add", "mul", "gelu", "relu", "norm"}
        lower_types = {t.lower() for t in op_types if t}
        if lower_types and lower_types.issubset(allowed):
            return WorkloadKind.GEMM_LIKE

    return WorkloadKind.UNKNOWN


# ============================================================
# fusion-space config
# ============================================================

@dataclass(frozen=True)
class FusionSpaceConfig:
    """
    定义 fusion 候选生成规则。
    它描述的是默认搜索边界，而不是唯一 fusion 策略。
    """
    max_group_size: int = 3
    allow_seq: bool = True
    allow_pipe: bool = True
    consecutive_only: bool = True
    allow_cross_phase: bool = False

    # 限制相邻算子类型是否允许进入同一 group
    allowed_adjacent_type_pairs: Optional[Set[Tuple[str, str]]] = None

    # 为防止第一版直接爆炸
    max_group_candidates: Optional[int] = 256
    max_gene_candidates: Optional[int] = 256


def default_fusion_config_for(dag: WorkloadDAGProtocol) -> FusionSpaceConfig:
    """
    根据 workload family 选择默认 fusion-space 规则。
    这是“默认搜索先验”，不是固定 fusion 策略。
    """
    kind = infer_workload_kind(dag)

    if kind == WorkloadKind.MHA_PREFILL:
        return FusionSpaceConfig(
            max_group_size=3,
            allow_seq=True,
            allow_pipe=True,
            consecutive_only=True,
            allow_cross_phase=False,
            allowed_adjacent_type_pairs={
                ("MatMul", "Softmax"),
                ("Softmax", "MatMul"),
                ("MatMul", "Identity"),
            },
            max_group_candidates=128,
            max_gene_candidates=64,
        )

    if kind == WorkloadKind.DECODE:
        return FusionSpaceConfig(
            max_group_size=2,
            allow_seq=True,
            allow_pipe=False,
            consecutive_only=True,
            allow_cross_phase=False,
            allowed_adjacent_type_pairs={
                ("MatMul", "Softmax"),
                ("Softmax", "MatMul"),
                ("MatMul", "Identity"),
            },
            max_group_candidates=96,
            max_gene_candidates=48,
        )

    if kind == WorkloadKind.GEMM_LIKE:
        return FusionSpaceConfig(
            max_group_size=3,
            allow_seq=True,
            allow_pipe=True,
            consecutive_only=True,
            allow_cross_phase=False,
            allowed_adjacent_type_pairs={
                ("MatMul", "BiasAdd"),
                ("BiasAdd", "Activation"),
                ("MatMul", "Activation"),
                ("MatMul", "Identity"),
                ("Identity", "Activation"),
            },
            max_group_candidates=128,
            max_gene_candidates=64,
        )

    # UNKNOWN：较保守
    return FusionSpaceConfig(
        max_group_size=2,
        allow_seq=True,
        allow_pipe=False,
        consecutive_only=True,
        allow_cross_phase=False,
        allowed_adjacent_type_pairs=None,
        max_group_candidates=64,
        max_gene_candidates=32,
    )


# ============================================================
# helpers
# ============================================================

def _get_all_ops(dag: WorkloadDAGProtocol) -> List[object]:
    """
    尽量兼容不同 workload 实现。
    优先读取 dag.ops / dag.op_map，否则尝试 dag.get_op(name)。
    """
    if hasattr(dag, "ops"):
        ops = getattr(dag, "ops")
        if isinstance(ops, dict):
            return list(ops.values())
        if isinstance(ops, list):
            return list(ops)

    if hasattr(dag, "op_map"):
        op_map = getattr(dag, "op_map")
        if isinstance(op_map, dict):
            return list(op_map.values())

    out = []
    if hasattr(dag, "op_names") and hasattr(dag, "get_op"):
        for name in dag.op_names():
            out.append(dag.get_op(name))
    return out


def _get_op(dag: WorkloadDAGProtocol, op_name: str):
    if hasattr(dag, "get_op"):
        return dag.get_op(op_name)
    if hasattr(dag, "ops"):
        ops = getattr(dag, "ops")
        if isinstance(ops, dict):
            return ops[op_name]
    if hasattr(dag, "op_map"):
        op_map = getattr(dag, "op_map")
        if isinstance(op_map, dict):
            return op_map[op_name]
    raise KeyError(f"Cannot find op '{op_name}' from workload DAG")


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _group_phase(dag: WorkloadDAGProtocol, ops: Sequence[str]) -> Optional[str]:
    phases = []
    for op in ops:
        op_obj = _get_op(dag, op)
        phase = getattr(op_obj, "phase", None)
        if phase is not None:
            phases.append(str(phase))
    phases = _unique_keep_order(phases)
    if not phases:
        return None
    if len(phases) == 1:
        return phases[0]
    return None


def _is_convex_subgraph(dag: WorkloadDAGProtocol, ops: Set[str]) -> bool:
    order = list(dag.op_names())
    graph: Dict[str, List[str]] = {op: [] for op in order}
    for u, v in dag.edges:
        graph[u].append(v)

    def reachable(src: str) -> Set[str]:
        vis = set()
        q = [src]
        while q:
            cur = q.pop(0)
            for nxt in graph.get(cur, []):
                if nxt not in vis:
                    vis.add(nxt)
                    q.append(nxt)
        vis.discard(src)
        return vis

    reach = {op: reachable(op) for op in order}
    for u in ops:
        in_group_desc = reach[u] & ops
        outside = reach[u] - ops
        for mid in outside:
            if reach[mid] & in_group_desc:
                return False
    return True


def _infer_group_boundary(
    dag: WorkloadDAGProtocol,
    ops: Sequence[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    返回:
        inputs, outputs, temps, shared
    """
    op_set = set(ops)
    cross_in: Set[str] = set()
    cross_out: Set[str] = set()
    internal_edge_tensors: Set[str] = set()

    for u, v in dag.edges:
        u_in = u in op_set
        v_in = v in op_set
        edge_tensors = list(dag.get_edge_tensors(u, v))

        if u_in and v_in:
            internal_edge_tensors.update(edge_tensors)
        elif (not u_in) and v_in:
            cross_in.update(edge_tensors)
        elif u_in and (not v_in):
            cross_out.update(edge_tensors)

    inputs = sorted(cross_in)
    outputs = sorted(cross_out)
    temps = sorted(internal_edge_tensors)
    shared = sorted(cross_in | cross_out)

    return inputs, outputs, temps, shared


def _infer_group_roles(
    dag: WorkloadDAGProtocol,
    inputs: Sequence[str],
    outputs: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    in_roles: Dict[str, str] = {}
    out_roles: Dict[str, str] = {}

    tensor_map = getattr(dag, "tensors", {})

    def role_of(tname: str) -> Optional[str]:
        t = tensor_map.get(tname)
        if t is None:
            return None
        special = getattr(t, "special_role", None)
        role = getattr(t, "role", None)
        if special:
            return str(special)
        if role:
            return str(role).upper()
        return tname

    for t in inputs:
        r = role_of(t)
        if r:
            in_roles[t] = r

    for t in outputs:
        r = role_of(t)
        if r:
            out_roles[t] = r

    return in_roles, out_roles


def _candidate_pipe_dims(dag: WorkloadDAGProtocol, ops: Sequence[str]) -> List[str]:
    op_set = set(ops)
    dims: Set[str] = set()
    reduce_dims: Set[str] = set()

    for u, v in dag.edges:
        if u in op_set and v in op_set:
            dims.update(str(x) for x in dag.get_edge_tensor_dims(u, v))
            red = dag.get_edge_reduction_dim(u, v)
            if red is not None:
                reduce_dims.add(str(red))

    return sorted(dims - reduce_dims)


# ============================================================
# FusionSpace
# ============================================================

@dataclass
class FusionSpace:
    dag: WorkloadDAGProtocol
    config: FusionSpaceConfig = field(default_factory=FusionSpaceConfig)

    @classmethod
    def from_workload(
        cls,
        dag: WorkloadDAGProtocol,
        config: Optional[FusionSpaceConfig] = None,
    ) -> "FusionSpace":
        """
        默认入口：
        - 若不给 config，则自动基于 workload family 选择默认规则
        - 若给 config，则保留消融实验/手工控制入口
        """
        if config is None:
            config = default_fusion_config_for(dag)
        return cls(dag=dag, config=config)

    def enumerate_groups(self) -> List[OpFusionGroup]:
        topo = list(self.dag.topo_order()) if hasattr(self.dag, "topo_order") else list(self.dag.op_names())
        out: List[OpFusionGroup] = []

        for size in range(1, self.config.max_group_size + 1):
            if self.config.consecutive_only:
                for i in range(0, len(topo) - size + 1):
                    ops = topo[i:i + size]
                    out.extend(self._build_groups_for_ops(ops))
            else:
                for ops in combinations(topo, size):
                    out.extend(self._build_groups_for_ops(list(ops)))

            if self.config.max_group_candidates is not None and len(out) >= self.config.max_group_candidates:
                return out[: self.config.max_group_candidates]

        return out

    def _build_groups_for_ops(self, ops: Sequence[str]) -> List[OpFusionGroup]:
        op_set = set(ops)

        if not _is_convex_subgraph(self.dag, op_set):
            return []

        phase = _group_phase(self.dag, ops)
        if (phase is None) and (not self.config.allow_cross_phase):
            phases = []
            for op in ops:
                op_obj = _get_op(self.dag, op)
                p = getattr(op_obj, "phase", None)
                if p is not None:
                    phases.append(str(p))
            phases = _unique_keep_order(phases)
            if len(phases) > 1:
                return []

        if not self._adjacent_type_rule_ok(ops):
            return []

        inputs, outputs, temps, shared = _infer_group_boundary(self.dag, ops)
        in_roles, out_roles = _infer_group_roles(self.dag, inputs, outputs)

        built: List[OpFusionGroup] = []
        gid_base = "__".join(ops)

        if self.config.allow_seq:
            try:
                built.append(
                    OpFusionGroup(
                        group_id=f"g_seq_{gid_base}",
                        ops=list(ops),
                        style=FusionStyle.SEQ,
                        phase=phase,
                        inputs=inputs,
                        outputs=outputs,
                        temps=temps,
                        in_roles=in_roles,
                        out_roles=out_roles,
                        shared=shared,
                    )
                )
            except Exception:
                pass

        if self.config.allow_pipe and len(ops) > 1:
            for pipe_dim in _candidate_pipe_dims(self.dag, ops):
                try:
                    built.append(
                        OpFusionGroup(
                            group_id=f"g_pipe_{gid_base}_{pipe_dim}",
                            ops=list(ops),
                            style=FusionStyle.PIPE,
                            phase=phase,
                            pipe_dim=pipe_dim,
                            inputs=inputs,
                            outputs=outputs,
                            temps=temps,
                            in_roles=in_roles,
                            out_roles=out_roles,
                            shared=shared,
                        )
                    )
                except Exception:
                    continue

        return built

    def _adjacent_type_rule_ok(self, ops: Sequence[str]) -> bool:
        rules = self.config.allowed_adjacent_type_pairs
        if not rules or len(ops) <= 1:
            return True

        for a, b in zip(ops[:-1], ops[1:]):
            op_a = _get_op(self.dag, a)
            op_b = _get_op(self.dag, b)
            type_a = getattr(op_a, "op_type", None)
            type_b = getattr(op_b, "op_type", None)
            if (type_a, type_b) not in rules:
                return False
        return True

    def enumerate_genes(self) -> List[FusionGene]:
        groups = self.enumerate_groups()
        dag_ops = set(self.dag.op_names())

        by_first_op: Dict[str, List[OpFusionGroup]] = {}
        for g in groups:
            first = g.ops[0]
            by_first_op.setdefault(first, []).append(g)

        topo = list(self.dag.topo_order()) if hasattr(self.dag, "topo_order") else sorted(dag_ops)
        genes: List[FusionGene] = []

        def backtrack(idx: int, chosen: List[OpFusionGroup], covered: Set[str]) -> None:
            if self.config.max_gene_candidates is not None and len(genes) >= self.config.max_gene_candidates:
                return

            if covered == dag_ops:
                gene = self._build_gene(chosen)
                if gene is not None:
                    genes.append(gene)
                return

            next_op = None
            while idx < len(topo):
                if topo[idx] not in covered:
                    next_op = topo[idx]
                    break
                idx += 1

            if next_op is None:
                return

            for g in by_first_op.get(next_op, []):
                g_ops = set(g.ops)
                if g_ops & covered:
                    continue
                chosen.append(g)
                backtrack(idx + 1, chosen, covered | g_ops)
                chosen.pop()

        backtrack(0, [], set())
        return genes

    def _build_gene(self, groups: Sequence[OpFusionGroup]) -> Optional[FusionGene]:
        op_to_gid: Dict[str, str] = {}
        for g in groups:
            for op in g.ops:
                if op in op_to_gid:
                    return None
                op_to_gid[op] = g.group_id

        coarse_edges: Set[Tuple[str, str]] = set()
        for u, v in self.dag.edges:
            gu = op_to_gid[u]
            gv = op_to_gid[v]
            if gu != gv:
                coarse_edges.add((gu, gv))

        ordered = sorted(groups, key=lambda x: x.group_id)
        gene_id = "gene__" + "__".join(g.group_id for g in ordered)

        try:
            gene = FusionGene(
                gene_id=gene_id,
                groups=list(ordered),
                edges=sorted(coarse_edges),
            )
            gene.validate(self.dag)
            return gene
        except Exception:
            return None


def enumerate_fusion_candidates(
    dag: WorkloadDAGProtocol,
    config: Optional[FusionSpaceConfig] = None,
) -> List[FusionGene]:
    """
    函数式快捷入口。
    """
    return FusionSpace.from_workload(dag, config=config).enumerate_genes()


__all__ = [
    "WorkloadKind",
    "infer_workload_kind",
    "FusionSpaceConfig",
    "default_fusion_config_for",
    "FusionSpace",
    "enumerate_fusion_candidates",
]
