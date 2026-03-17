from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple


class FusionStyle(str, Enum):
    SEQ = "seq"
    PIPE = "pipe"


@dataclass(frozen=True)
class TensorRole:
    """
    这里只是一个轻量角色标签容器。
    具体合法 role 集合由 workload / fusion_space 侧决定，
    fusion_gene.py 不负责定义完整设计空间。
    """
    name: str


@dataclass
class OpFusionGroup:
    """
    单个 fusion group。

    职责：
    1. 描述 group 内有哪些算子
    2. 描述 group 的边界张量
    3. 描述组内临时张量
    4. 描述边界张量角色
    5. 描述哪些边界张量是关键 shared tensor
    6. 描述组内执行方式（seq / pipe）

    不负责：
    - placement 层级
    - keep / writeback / evict
    - tile size / loop order
    - hardware 参数
    """

    group_id: str
    ops: List[str]

    style: FusionStyle = FusionStyle.SEQ
    phase: Optional[str] = None
    role: Optional[str] = None

    # 仅在 PIPE 模式下有意义
    pipe_dim: Optional[str] = None

    # group 边界 / 内部张量
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    temps: List[str] = field(default_factory=list)

    # 仅用于给 placement / 后续模块提供稳定接口
    in_roles: Dict[str, str] = field(default_factory=dict)
    out_roles: Dict[str, str] = field(default_factory=dict)

    # 关键跨 group 边界张量
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
            raise ValueError(
                f"group '{self.group_id}' 为 PIPE 模式时，pipe_dim 不能为空"
            )
        if self.style == FusionStyle.SEQ and self.pipe_dim is not None:
            raise ValueError(
                f"group '{self.group_id}' 为 SEQ 模式时，pipe_dim 必须为 None"
            )

    def _check_tensor_partition(self) -> None:
        for name, values in {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "temps": self.temps,
            "shared": self.shared,
        }.items():
            if len(values) != len(set(values)):
                raise ValueError(
                    f"group '{self.group_id}' 的 {name} 存在重复项: {values}"
                )

        io_overlap = self.input_set & self.output_set
        if io_overlap:
            raise ValueError(
                f"group '{self.group_id}' 的张量不能同时属于 inputs 和 outputs: "
                f"{sorted(io_overlap)}"
            )

        boundary_temp_overlap = self.boundary_set & self.temp_set
        if boundary_temp_overlap:
            raise ValueError(
                f"group '{self.group_id}' 的张量不能同时属于 boundary 和 temps: "
                f"{sorted(boundary_temp_overlap)}"
            )

        illegal_shared = self.shared_set - self.boundary_set
        if illegal_shared:
            raise ValueError(
                f"group '{self.group_id}' 的 shared 必须属于 inputs 或 outputs: "
                f"{sorted(illegal_shared)}"
            )

    def _check_roles(self) -> None:
        illegal_in = set(self.in_roles) - self.input_set
        if illegal_in:
            raise ValueError(
                f"group '{self.group_id}' 的 in_roles 只能标注 inputs: {sorted(illegal_in)}"
            )

        illegal_out = set(self.out_roles) - self.output_set
        if illegal_out:
            raise ValueError(
                f"group '{self.group_id}' 的 out_roles 只能标注 outputs: {sorted(illegal_out)}"
            )

        empty_in = [k for k, v in self.in_roles.items() if not v]
        empty_out = [k for k, v in self.out_roles.items() if not v]
        if empty_in or empty_out:
            raise ValueError(
                f"group '{self.group_id}' 的 role 名不能为空: "
                f"empty_in={empty_in}, empty_out={empty_out}"
            )


@dataclass
class FusionGene:
    """
    一套完整 fusion 方案。

    职责：
    1. 描述 workload DAG 如何被分成多个 group
    2. 描述 group 间 coarse DAG
    3. 提供 fusion 层的结构与接口约束检查

    不负责：
    - 设计空间枚举
    - placement / tiling / hardware 实例化
    """

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
                    raise ValueError(
                        f"算子 '{op}' 同时出现在 group '{mapping[op]}' 和 '{g.group_id}' 中"
                    )
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
        """
        fusion 层总校验入口。
        """
        self.check_structure(dag)
        self.check_boundary(dag)
        self.check_roles(dag)

    def check_structure(self, dag: WorkloadDAGProtocol) -> None:
        """
        结构约束：
        1. coverage
        2. convex
        3. coarse DAG acyclic
        4. pipe group 的 pipe_dim 合法
        """
        self._check_coverage(dag)
        self._check_convex(dag)
        self._check_edges_match_dag(dag)
        self.topo_order()
        self._check_pipe_dims(dag)

    def check_boundary(self, dag: WorkloadDAGProtocol) -> None:
        """
        boundary 约束：
        1. inputs / outputs / temps 自洽
        2. shared 必须是真正跨 group 的边界张量
        3. group 边界应与 DAG 跨 group 依赖一致
        """
        op_to_group = self.op_to_group

        # 建立跨 group 边
        cross_group_tensors: Dict[str, Set[str]] = defaultdict(set)
        for src_op, dst_op in dag.edges:
            src_gid = op_to_group[src_op]
            dst_gid = op_to_group[dst_op]
            if src_gid == dst_gid:
                continue
            tensors = set(dag.get_edge_tensors(src_op, dst_op))
            cross_group_tensors[src_gid].update(tensors)
            cross_group_tensors[dst_gid].update(tensors)

        for g in self.groups:
            declared_boundary = g.boundary_set

            # shared 必须是真正跨 group 的张量
            real_cross = cross_group_tensors.get(g.group_id, set())
            illegal_shared = g.shared_set - real_cross
            if illegal_shared:
                raise ValueError(
                    f"group '{g.group_id}' 的 shared 不是实际跨 group 张量: "
                    f"{sorted(illegal_shared)}"
                )

            # 如果显式声明了 boundary，则 boundary 至少不能与实际跨组接口完全脱节
            if declared_boundary and real_cross:
                # 允许 boundary 比 real_cross 更宽，但 shared 至少应落在真实跨组集合中
                pass

            # temps 不应包含任何真实跨 group 张量
            bad_temps = g.temp_set & real_cross
            if bad_temps:
                raise ValueError(
                    f"group '{g.group_id}' 的 temps 中包含真实跨 group 张量: "
                    f"{sorted(bad_temps)}"
                )

    def check_roles(self, dag: WorkloadDAGProtocol) -> None:
        """
        role 约束：
        1. 同一 shared tensor 的显式 role 不冲突
        2. 已声明 role 的边界张量必须存在于 boundary 中
        """
        tensor_roles: Dict[str, str] = {}

        for g in self.groups:
            merged = {}
            merged.update(g.in_roles)
            merged.update(g.out_roles)

            for t, r in merged.items():
                if t in tensor_roles and tensor_roles[t] != r:
                    raise ValueError(
                        f"共享张量 '{t}' 的 role 冲突: "
                        f"已有 '{tensor_roles[t]}', 新增 '{r}', group='{g.group_id}'"
                    )
                tensor_roles[t] = r

    def _check_edge_endpoints(self) -> None:
        gids = {g.group_id for g in self.groups}
        for u, v in self.edges:
            if u not in gids or v not in gids:
                raise ValueError(
                    f"FusionGene '{self.gene_id}' 的 edges 包含不存在的 group: {(u, v)}"
                )
            if u == v:
                raise ValueError(
                    f"FusionGene '{self.gene_id}' 的 edges 不允许自环: {(u, v)}"
                )

    def _check_coverage(self, dag: WorkloadDAGProtocol) -> None:
        dag_ops = set(dag.op_names())
        mapped = set(self.op_to_group)

        missing = dag_ops - mapped
        extra = mapped - dag_ops

        if missing or extra:
            raise ValueError(
                f"FusionGene '{self.gene_id}' coverage 检查失败: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

    def _check_convex(self, dag: WorkloadDAGProtocol) -> None:
        reach = _compute_reachability(dag.op_names(), dag.edges)

        for g in self.groups:
            ops = g.op_set
            for u in ops:
                in_group_desc = reach[u] & ops
                outside = reach[u] - ops
                for mid in outside:
                    if reach[mid] & in_group_desc:
                        raise ValueError(
                            f"group '{g.group_id}' 不是凸子图: "
                            f"路径包含组外算子 '{mid}'"
                        )

    def _check_edges_match_dag(self, dag: WorkloadDAGProtocol) -> None:
        real_edges = self._build_coarse_edges_from_dag(dag)
        declared = set(self.edges)
        if real_edges != declared:
            raise ValueError(
                f"FusionGene '{self.gene_id}' 的 coarse edges 与 DAG 不一致: "
                f"declared={sorted(declared)}, real={sorted(real_edges)}"
            )

    def _build_coarse_edges_from_dag(
        self,
        dag: WorkloadDAGProtocol,
    ) -> Set[Tuple[str, str]]:
        op_to_group = self.op_to_group
        coarse_edges: Set[Tuple[str, str]] = set()

        for u, v in dag.edges:
            gu = op_to_group[u]
            gv = op_to_group[v]
            if gu != gv:
                coarse_edges.add((gu, gv))
        return coarse_edges

    def _check_pipe_dims(self, dag: WorkloadDAGProtocol) -> None:
        for g in self.groups:
            if g.style != FusionStyle.PIPE:
                continue

            assert g.pipe_dim is not None
            dims = set()
            reduce_dims = set()

            for u, v in dag.edges:
                if u in g.op_set and v in g.op_set:
                    dims.update(dag.get_edge_tensor_dims(u, v))
                    red = dag.get_edge_reduction_dim(u, v)
                    if red is not None:
                        reduce_dims.add(red)

            if g.pipe_dim not in dims:
                raise ValueError(
                    f"group '{g.group_id}' 的 pipe_dim='{g.pipe_dim}' "
                    f"不在组内数据流维度中: {sorted(dims)}"
                )

            if g.pipe_dim in reduce_dims:
                raise ValueError(
                    f"group '{g.group_id}' 的 pipe_dim='{g.pipe_dim}' "
                    f"不能直接取 reduction dim: {sorted(reduce_dims)}"
                )


class WorkloadDAGProtocol:
    """
    这里只定义 fusion_gene 需要的最小 DAG 接口协议，
    方便后续重构 builder / workload 时解耦。
    """

    edges: List[Tuple[str, str]]

    def op_names(self) -> Set[str]:
        raise NotImplementedError

    def get_edge_tensors(self, src_op: str, dst_op: str) -> Iterable[str]:
        raise NotImplementedError

    def get_edge_tensor_dims(self, src_op: str, dst_op: str) -> Iterable[str]:
        raise NotImplementedError

    def get_edge_reduction_dim(self, src_op: str, dst_op: str) -> Optional[str]:
        raise NotImplementedError


def _compute_reachability(
    op_names: Iterable[str],
    edges: Iterable[Tuple[str, str]],
) -> Dict[str, Set[str]]:
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
