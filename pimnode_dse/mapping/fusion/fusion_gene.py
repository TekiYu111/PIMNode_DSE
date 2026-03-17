from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Protocol, Set, Tuple


class FusionStyle(str, Enum):
    SEQ = "seq"
    PIPE = "pipe"


@dataclass
class OpFusionGroup:
    group_id: str
    ops: List[str]
    style: FusionStyle = FusionStyle.SEQ
    phase: Optional[str] = None
    role: Optional[str] = None
    pipe_dim: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    temps: List[str] = field(default_factory=list)
    in_roles: Dict[str, str] = field(default_factory=dict)
    out_roles: Dict[str, str] = field(default_factory=dict)
    shared: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._check_basic_fields()
        self._check_tensor_partition()
        self._check_roles()

    @property
    def op_set(self) -> Set[str]:
        return set(self.ops)

    @property
    def input_set(self) -> Set[str]:
        return set(self.inputs)

    @property
    def output_set(self) -> Set[str]:
        return set(self.outputs)

    @property
    def temp_set(self) -> Set[str]:
        return set(self.temps)

    @property
    def boundary_set(self) -> Set[str]:
        return self.input_set | self.output_set

    @property
    def shared_set(self) -> Set[str]:
        return set(self.shared)

    def _check_basic_fields(self) -> None:
        if not self.group_id:
            raise ValueError("group_id 不能为空")
        if not self.ops:
            raise ValueError(f"group '{self.group_id}' 的 ops 不能为空")
        if len(self.ops) != len(set(self.ops)):
            raise ValueError(f"group '{self.group_id}' 的 ops 存在重复项: {self.ops}")
        if self.style == FusionStyle.PIPE and not self.pipe_dim:
            raise ValueError(f"group '{self.group_id}' 为 PIPE 模式时，pipe_dim 不能为空")
        if self.style == FusionStyle.SEQ and self.pipe_dim is not None:
            raise ValueError(f"group '{self.group_id}' 为 SEQ 模式时，pipe_dim 必须为 None")

    def _check_tensor_partition(self) -> None:
        for name, values in {"inputs": self.inputs, "outputs": self.outputs, "temps": self.temps, "shared": self.shared}.items():
            if len(values) != len(set(values)):
                raise ValueError(f"group '{self.group_id}' 的 {name} 存在重复项: {values}")
        io_overlap = self.input_set & self.output_set
        if io_overlap:
            raise ValueError(f"group '{self.group_id}' 的张量不能同时属于 inputs 和 outputs: {sorted(io_overlap)}")
        boundary_temp_overlap = self.boundary_set & self.temp_set
        if boundary_temp_overlap:
            raise ValueError(f"group '{self.group_id}' 的张量不能同时属于 boundary 和 temps: {sorted(boundary_temp_overlap)}")
        illegal_shared = self.shared_set - self.boundary_set
        if illegal_shared:
            raise ValueError(f"group '{self.group_id}' 的 shared 必须属于 inputs 或 outputs: {sorted(illegal_shared)}")

    def _check_roles(self) -> None:
        illegal_in = set(self.in_roles) - self.input_set
        if illegal_in:
            raise ValueError(f"group '{self.group_id}' 的 in_roles 只能标注 inputs: {sorted(illegal_in)}")
        illegal_out = set(self.out_roles) - self.output_set
        if illegal_out:
            raise ValueError(f"group '{self.group_id}' 的 out_roles 只能标注 outputs: {sorted(illegal_out)}")
        empty_in = [k for k, v in self.in_roles.items() if not v]
        empty_out = [k for k, v in self.out_roles.items() if not v]
        if empty_in or empty_out:
            raise ValueError(f"group '{self.group_id}' 的 role 名不能为空: empty_in={empty_in}, empty_out={empty_out}")


@dataclass
class FusionGene:
    gene_id: str
    groups: List[OpFusionGroup]
    edges: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.gene_id:
            raise ValueError("gene_id 不能为空")
        if not self.groups:
            raise ValueError("groups 不能为空")
        group_ids = [g.group_id for g in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(f"FusionGene '{self.gene_id}' 的 group_id 存在重复")
        self._check_edge_endpoints()

    @property
    def group_map(self) -> Dict[str, OpFusionGroup]:
        return {g.group_id: g for g in self.groups}

    @property
    def op_to_group(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for g in self.groups:
            for op in g.ops:
                if op in mapping:
                    raise ValueError(f"算子 '{op}' 同时出现在 group '{mapping[op]}' 和 '{g.group_id}' 中")
                mapping[op] = g.group_id
        return mapping

    def topo_order(self) -> List[str]:
        graph = defaultdict(list)
        indeg = {g.group_id: 0 for g in self.groups}
        for u, v in self.edges:
            graph[u].append(v)
            indeg[v] += 1
        q = deque([gid for gid, deg in indeg.items() if deg == 0])
        order: List[str] = []
        while q:
            cur = q.popleft()
            order.append(cur)
            for nxt in graph[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        if len(order) != len(self.groups):
            raise ValueError(f"FusionGene '{self.gene_id}' 的 coarse DAG 存在环")
        return order

    def predecessors(self, group_id: str) -> Set[str]:
        return {u for u, v in self.edges if v == group_id}

    def successors(self, group_id: str) -> Set[str]:
        return {v for u, v in self.edges if u == group_id}

    def validate(self, dag: WorkloadDAGProtocol) -> None:
        self.check_structure(dag)
        self.check_boundary(dag)
        self.check_roles(dag)

    def check_structure(self, dag: WorkloadDAGProtocol) -> None:
        self._check_coverage(dag)
        self._check_convex(dag)
        self._check_edges_match_dag(dag)
        self.topo_order()
        self._check_pipe_dims(dag)

    def check_boundary(self, dag: WorkloadDAGProtocol) -> None:
        op_to_group = self.op_to_group
        cross_group_tensors: Dict[str, Set[str]] = defaultdict(set)
        for edge in dag.edge_list():
            src_gid = op_to_group[edge.src]
            dst_gid = op_to_group[edge.dst]
            if src_gid == dst_gid:
                continue
            cross_group_tensors[src_gid].add(edge.tensor)
            cross_group_tensors[dst_gid].add(edge.tensor)

        for g in self.groups:
            real_cross = cross_group_tensors.get(g.group_id, set())
            illegal_shared = g.shared_set - real_cross
            if illegal_shared:
                raise ValueError(f"group '{g.group_id}' 的 shared 不是实际跨 group 张量: {sorted(illegal_shared)}")
            bad_temps = g.temp_set & real_cross
            if bad_temps:
                raise ValueError(f"group '{g.group_id}' 的 temps 中包含真实跨 group 张量: {sorted(bad_temps)}")

    def check_roles(self, dag: WorkloadDAGProtocol) -> None:
        tensor_roles: Dict[str, str] = {}
        for g in self.groups:
            merged: Dict[str, str] = {}
            merged.update(g.in_roles)
            merged.update(g.out_roles)
            for t, r in merged.items():
                if t in tensor_roles and tensor_roles[t] != r:
                    raise ValueError(
                        f"共享张量 '{t}' 的 role 冲突: 已有 '{tensor_roles[t]}', 新增 '{r}', group='{g.group_id}'"
                    )
                tensor_roles[t] = r

    def _check_edge_endpoints(self) -> None:
        gids = {g.group_id for g in self.groups}
        for u, v in self.edges:
            if u not in gids or v not in gids:
                raise ValueError(f"FusionGene '{self.gene_id}' 的 edges 包含不存在的 group: {(u, v)}")
            if u == v:
                raise ValueError(f"FusionGene '{self.gene_id}' 的 edges 不允许自环: {(u, v)}")

    def _check_coverage(self, dag: WorkloadDAGProtocol) -> None:
        dag_ops = set(dag.op_names())
        mapped = set(self.op_to_group)
        missing = dag_ops - mapped
        extra = mapped - dag_ops
        if missing or extra:
            raise ValueError(f"FusionGene '{self.gene_id}' coverage 检查失败: missing={sorted(missing)}, extra={sorted(extra)}")

    def _check_convex(self, dag: WorkloadDAGProtocol) -> None:
        reach = _compute_reachability(dag.op_names(), [(e.src, e.dst) for e in dag.edge_list()])
        for g in self.groups:
            ops = g.op_set
            for u in ops:
                in_group_desc = reach[u] & ops
                outside = reach[u] - ops
                for mid in outside:
                    if reach[mid] & in_group_desc:
                        raise ValueError(f"group '{g.group_id}' 不是凸子图: 路径包含组外算子 '{mid}'")

    def _check_edges_match_dag(self, dag: WorkloadDAGProtocol) -> None:
        real_edges = self._build_coarse_edges_from_dag(dag)
        declared = set(self.edges)
        if real_edges != declared:
            raise ValueError(
                f"FusionGene '{self.gene_id}' 的 coarse edges 与 DAG 不一致: declared={sorted(declared)}, real={sorted(real_edges)}"
            )

    def _build_coarse_edges_from_dag(self, dag: WorkloadDAGProtocol) -> Set[Tuple[str, str]]:
        op_to_group = self.op_to_group
        coarse_edges: Set[Tuple[str, str]] = set()
        for edge in dag.edge_list():
            gu = op_to_group[edge.src]
            gv = op_to_group[edge.dst]
            if gu != gv:
                coarse_edges.add((gu, gv))
        return coarse_edges

    def _check_pipe_dims(self, dag: WorkloadDAGProtocol) -> None:
        for g in self.groups:
            if g.style != FusionStyle.PIPE:
                continue
            assert g.pipe_dim is not None
            dims: Set[str] = set()
            reduce_dims: Set[str] = set()
            for edge in dag.edge_list():
                if edge.src in g.op_set and edge.dst in g.op_set:
                    dims.update(edge.src_dims)
                    reduce_dims.update(edge.reduce_dims)
            if g.pipe_dim not in dims:
                raise ValueError(
                    f"group '{g.group_id}' 的 pipe_dim='{g.pipe_dim}' 不在组内数据流维度中: {sorted(dims)}"
                )
            if g.pipe_dim in reduce_dims:
                raise ValueError(
                    f"group '{g.group_id}' 的 pipe_dim='{g.pipe_dim}' 不能直接取 reduction dim: {sorted(reduce_dims)}"
                )


class EdgeProtocol(Protocol):
    src: str
    dst: str
    tensor: str
    src_dims: Tuple[str, ...]
    dst_dims: Tuple[str, ...]
    reduce_dims: Tuple[str, ...]


class WorkloadDAGProtocol(Protocol):
    def op_names(self) -> Set[str]: ...
    def edge_list(self) -> List[EdgeProtocol]: ...


def _compute_reachability(op_names: Iterable[str], edges: Iterable[Tuple[str, str]]) -> Dict[str, Set[str]]:
    op_set = set(op_names)
    graph: Dict[str, List[str]] = {op: [] for op in op_set}
    for u, v in edges:
        graph[u].append(v)
    reach: Dict[str, Set[str]] = {}
    for src in op_set:
        vis = set()
        q = deque([src])
        while q:
            cur = q.popleft()
            for nxt in graph[cur]:
                if nxt not in vis:
                    vis.add(nxt)
                    q.append(nxt)
        vis.discard(src)
        reach[src] = vis
    return reach
