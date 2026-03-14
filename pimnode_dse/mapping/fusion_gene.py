# fusion_gene.py

from __future__ import annotations
import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from pimnode_dse.workload import WorkloadDAG


# -------------------------
# 枚举定义
# -------------------------
class FusionStyle(Enum):
    SEQUENTIAL = auto()
    PIPELINE   = auto()


# -------------------------
# PipelineConstraint
# -------------------------
@dataclass
class PipelineConstraint:
    reduction_dims_in_pipeline: Set[str] = field(default_factory=set)

    @property
    def has_constraints(self) -> bool:
        return bool(self.reduction_dims_in_pipeline)

    def __repr__(self) -> str:
        if not self.has_constraints:
            return "PipelineConstraint(None)"
        return f"PipelineConstraint(reduction_dims={sorted(self.reduction_dims_in_pipeline)})"


# -------------------------
# OpFusionGroup
# -------------------------
@dataclass
class OpFusionGroup:
    op_names: List[str]
    fusion_style: FusionStyle
    pipeline_dim: Optional[str] = None

    pipeline_constraint: PipelineConstraint = field(init=False, default_factory=PipelineConstraint, compare=False)
    group_id: str = field(init=False, default="")

    # 新增
    phase: Optional[str] = None  # "prefill" / "decode"
    special_role: Optional[str] = None  # "KV_CACHE", "PARTIAL_O", "STATS", etc.

@dataclass
class OpFusionGroup:
    group_id: Optional[str] = None
    op_names: List[str] = field(default_factory=list)
    fusion_style: FusionStyle = FusionStyle.SEQUENTIAL
    pipeline_dim: Optional[str] = None

    pipeline_constraint: PipelineConstraint = field(
        default_factory=PipelineConstraint,
        compare=False,
    )

    phase: Optional[str] = None
    special_role: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.op_names:
            raise ValueError("OpFusionGroup 的 op_names 不能为空。")
        if len(self.op_names) != len(set(self.op_names)):
            duplicates = sorted({n for n in self.op_names if self.op_names.count(n) > 1})
            raise ValueError(f"OpFusionGroup 存在重复算子：{duplicates}")

        # 单算子组强制 SEQUENTIAL
        if len(self.op_names) == 1:
            self.fusion_style = FusionStyle.SEQUENTIAL
            self.pipeline_dim = None

        if self.fusion_style == FusionStyle.PIPELINE and self.pipeline_dim is None:
            raise ValueError(f"fusion_style 为 PIPELINE 时必须指定 pipeline_dim。算子: {self.op_names}")
        if self.fusion_style == FusionStyle.SEQUENTIAL and self.pipeline_dim is not None:
            raise ValueError(f"fusion_style 为 SEQUENTIAL 时 pipeline_dim 必须为 None。算子: {self.op_names}")

        # 如果未显式提供 group_id，则自动生成
        if not self.group_id:
            raw = "|".join(sorted(self.op_names))
            digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
            self.group_id = f"grp_{digest}"

    @property
    def op_name_set(self) -> FrozenSet[str]:
        return frozenset(self.op_names)

    def __repr__(self) -> str:
        dim_str = f", dim='{self.pipeline_dim}'" if self.pipeline_dim else ""
        const_str = f", constraint={self.pipeline_constraint}" if self.pipeline_constraint.has_constraints else ""
        phase_str = f", phase={self.phase}" if self.phase else ""
        role_str = f", role={self.special_role}" if self.special_role else ""
        return f"OpFusionGroup(id={self.group_id}, ops={self.op_names}, style={self.fusion_style.name}{dim_str}{const_str}{phase_str}{role_str})"


# -------------------------
# FusionGene
# -------------------------
@dataclass
class FusionGene:

    groups: List[OpFusionGroup]
    gene_id: Optional[str] = None
    group_edges: List[Tuple[str, str]] = field(init=False, default_factory=list)
    _op_to_group: Optional[Dict[str, str]] = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("FusionGene 的 groups 不能为空。")

        seen_gids = set()
        for g in self.groups:
            if g.group_id in seen_gids:
                raise ValueError(f"重复分组定义，group_id: {g.group_id}")
            seen_gids.add(g.group_id)

        if not self.gene_id:
            raw = "|".join(sorted(g.group_id for g in self.groups))
            self.gene_id = f"gene_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    def get_op_to_group_mapping(self) -> Dict[str, str]:
        if self._op_to_group is None:
            mapping: Dict[str, str] = {}
            for group in self.groups:
                for op in group.op_names:
                    if op in mapping:
                        raise ValueError(f"算子 '{op}' 同时存在于组 {mapping[op]} 和 {group.group_id}")
                    mapping[op] = group.group_id
            self._op_to_group = mapping
        return self._op_to_group

    def get_group_by_id(self, group_id: str) -> OpFusionGroup:
        for g in self.groups:
            if g.group_id == group_id:
                return g
        raise KeyError(f"未找到 group_id: {group_id}")

    # -------------------------
    # DAG 验证
    # -------------------------
    def validate_topology(self, dag: "WorkloadDAG") -> None:
        op_to_group = self.get_op_to_group_mapping()
        _check_coverage(dag.op_names(), op_to_group)
        reachability = _compute_reachability(dag.op_names(), dag.edges)
        for group in self.groups:
            _check_convex_subgraph(group, reachability)
        coarse_edges = _compute_coarse_edges(dag.edges, op_to_group)
        _check_coarse_dag_acyclic({g.group_id for g in self.groups}, coarse_edges)
        for group in self.groups:
            if group.fusion_style == FusionStyle.PIPELINE:
                _check_pipeline_dim(group, dag)

    def build_coarse_dag(self, dag: "WorkloadDAG") -> None:
        op_to_group = self.get_op_to_group_mapping()
        coarse_edges = _compute_coarse_edges(dag.edges, op_to_group)
        self.group_edges = sorted(coarse_edges)

    # -------------------------
    # 新增 group 边查询接口
    # -------------------------
    def get_group_successors(self, group_id: str) -> Set[str]:
        return {v for u, v in self.group_edges if u == group_id}

    def get_group_predecessors(self, group_id: str) -> Set[str]:
        return {u for u, v in self.group_edges if v == group_id}

    def __repr__(self) -> str:
        lines = [f"FusionGene(id={self.gene_id}, {len(self.groups)} groups):"]
        lines.extend(f"  {g}" for g in self.groups)
        edges_str = str(self.group_edges) if self.group_edges else "Not built"
        lines.append(f"  CoarseDAG edges: {edges_str}")
        return "\n".join(lines)


# ===========================
# 内部工具函数
# ===========================

def _compute_reachability(op_names: Set[str], edges: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    adjacency = {op: [] for op in op_names}
    for u, v in edges:
        adjacency[u].append(v)

    reachability = {}
    for start in op_names:
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nbr in adjacency[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        reachability[start] = visited - {start}
    return reachability


def _check_coverage(dag_op_names: Set[str], op_to_group: Dict[str, str]) -> None:
    mapped_ops = set(op_to_group.keys())
    missing = dag_op_names - mapped_ops
    extra = mapped_ops - dag_op_names
    if missing or extra:
        raise ValueError(f"覆盖性检查失败, 未分配: {sorted(missing)}, 外来: {sorted(extra)}")


def _check_convex_subgraph(group: OpFusionGroup, reachability: Dict[str, Set[str]]) -> None:
    op_set = group.op_name_set
    offending_edges = []
    for u in op_set:
        in_group_descendants = reachability[u] & op_set
        outside_reachable = reachability[u] - op_set
        for w in outside_reachable:
            bypass_targets = reachability[w] & in_group_descendants
            if bypass_targets:
                offending_edges.append((u, w, sorted(bypass_targets)))
    if offending_edges:
        raise ValueError(f"凸子集检查失败 group {group.group_id}, offending_edges={offending_edges}")


def _compute_coarse_edges(dag_edges: List[Tuple[str, str]], op_to_group: Dict[str, str]) -> Set[Tuple[str, str]]:
    coarse_edges = set()
    for u, v in dag_edges:
        u_grp, v_grp = op_to_group[u], op_to_group[v]
        if u_grp != v_grp:
            coarse_edges.add((u_grp, v_grp))
    return coarse_edges


def _check_coarse_dag_acyclic(group_ids: Set[str], coarse_edges: Set[Tuple[str, str]]) -> None:
    in_degree = {gid: 0 for gid in group_ids}
    adjacency = {gid: [] for gid in group_ids}
    for u, v in coarse_edges:
        adjacency[u].append(v)
        in_degree[v] += 1

    queue = deque(gid for gid, deg in in_degree.items() if deg == 0)
    visited_count = 0
    while queue:
        node = queue.popleft()
        visited_count += 1
        for nbr in adjacency[node]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    if visited_count != len(group_ids):
        cycles = sorted(gid for gid, deg in in_degree.items() if deg > 0)
        raise ValueError(f"Coarse DAG 存在环形依赖, 组: {cycles}")


def _check_pipeline_dim(group: OpFusionGroup, dag: "WorkloadDAG") -> None:
    assert group.pipeline_dim is not None
    pipeline_dim = group.pipeline_dim
    op_set = group.op_name_set

    intra_edges = [(u, v) for u, v in dag.edges if u in op_set and v in op_set]
    if not intra_edges:
        return

    all_dims = set()
    reduction_dims = set()
    for u, v in intra_edges:
        all_dims.update(dag.get_edge_tensor_dims(u, v))
        red_dim = dag.get_edge_reduction_dim(u, v)
        if red_dim is not None:
            reduction_dims.add(red_dim)

    if pipeline_dim not in all_dims:
        raise ValueError(
            f"OpFusionGroup {group.group_id} pipeline_dim '{pipeline_dim}' "
            f"不在组内数据流维度 {sorted(all_dims)}"
        )
    if pipeline_dim in reduction_dims:
        group.pipeline_constraint.reduction_dims_in_pipeline.add(pipeline_dim)
