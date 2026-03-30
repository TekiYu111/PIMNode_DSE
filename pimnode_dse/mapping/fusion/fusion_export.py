from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from mapping_tree import Bind, GroupRel, GroupSpec, OpRel, OpSpec

from fusion_gene import FusionGene, FusionGroup
from graph_adapter import WorkloadFusionGraphAdapter

try:
    from sematic_adapter import WorkloadFusionSemanticAdapter
except Exception:  # pragma: no cover
    from semantic_adapter import WorkloadFusionSemanticAdapter  # type: ignore


EdgeKey = Tuple[str, str]
HoldKey = Tuple[str, str, str]


@dataclass(frozen=True)
class RelCfg:
    pipe: Tuple[EdgeKey, ...] = ()
    hold: Tuple[HoldKey, ...] = ()

    def __post_init__(self) -> None:
        pipe = tuple(sorted({_norm_edge(x) for x in self.pipe}))
        hold = tuple(sorted({_norm_hold(x) for x in self.hold}))
        object.__setattr__(self, "pipe", pipe)
        object.__setattr__(self, "hold", hold)

    def want_pipe(self, src: str, dst: str) -> bool:
        return (src, dst) in self.pipe

    def want_hold(self, src: str, dst: str, tens: str) -> bool:
        return (src, dst, tens) in self.hold


@dataclass(frozen=True)
class FusionOut:
    groups: Tuple[GroupSpec, ...]
    rels: Tuple[GroupRel, ...]

    def group_map(self) -> Dict[str, GroupSpec]:
        return {g.gid: g for g in self.groups}


def build_fusion_out(
    gene: FusionGene,
    dag: Any,
    cfg: RelCfg = RelCfg(),
) -> FusionOut:
    graph = WorkloadFusionGraphAdapter(dag)
    sem = WorkloadFusionSemanticAdapter(graph)
    groups = build_groups(gene=gene, dag=dag, graph=graph, sem=sem)
    rels = build_rels(gene=gene, graph=graph, sem=sem, cfg=cfg)
    validate_fusion_out(groups=groups, rels=rels)
    return FusionOut(groups=groups, rels=rels)


def build_groups(
    gene: FusionGene,
    dag: Any,
    graph: Optional[WorkloadFusionGraphAdapter] = None,
    sem: Optional[WorkloadFusionSemanticAdapter] = None,
) -> Tuple[GroupSpec, ...]:
    graph = graph or WorkloadFusionGraphAdapter(dag)
    sem = sem or WorkloadFusionSemanticAdapter(graph)
    out = []
    for group in gene.groups:
        out.append(_build_group(group=group, dag=dag, graph=graph, sem=sem))
    return tuple(out)


def build_rels(
    gene: FusionGene,
    graph: WorkloadFusionGraphAdapter,
    sem: Optional[WorkloadFusionSemanticAdapter] = None,
    cfg: RelCfg = RelCfg(),
) -> Tuple[GroupRel, ...]:
    sem = sem or WorkloadFusionSemanticAdapter(graph)
    group_ops = {group.group_id: tuple(group.ops) for group in gene.groups}
    edge_map = _edge_tens_map(gene)

    out = []
    for src, dst in sorted(gene.group_edges):
        tens = edge_map.get((src, dst), ())
        if not tens:
            continue
        src_ops = group_ops.get(src, ())
        dst_ops = group_ops.get(dst, ())
        bind = _pick_bind(
            src=src,
            dst=dst,
            tens=tens,
            src_ops=src_ops,
            dst_ops=dst_ops,
            graph=graph,
            sem=sem,
            cfg=cfg,
        )
        hold = _pick_hold(
            src=src,
            dst=dst,
            tens=tens,
            src_ops=src_ops,
            dst_ops=dst_ops,
            graph=graph,
            cfg=cfg,
        )
        out.append(GroupRel(src=src, dst=dst, bind=bind, tens=tens, hold=hold))
    return tuple(out)


def group_op_rels(
    group: FusionGroup,
    graph: WorkloadFusionGraphAdapter,
) -> Tuple[OpRel, ...]:
    op_set = set(group.ops)
    seen = set()
    out = []

    for src in sorted(group.ops):
        for dst in sorted(graph.succ(src)):
            if dst not in op_set:
                continue
            key = (src, dst)
            if key in seen:
                continue
            seen.add(key)
            tens = tuple(sorted(set(_edge_tensors(graph=graph, src=src, dst=dst))))
            out.append(OpRel(src=src, dst=dst, bind=Bind.SEQ, tens=tens))
    return tuple(out)


def validate_fusion_out(
    groups: Sequence[GroupSpec],
    rels: Sequence[GroupRel],
) -> None:
    groups = tuple(groups)
    rels = tuple(rels)

    gid_set = {g.gid for g in groups}
    if len(gid_set) != len(groups):
        raise ValueError("duplicate gid")

    seen_op = {}
    for group in groups:
        if not group.ops:
            raise ValueError("group ops must not be empty: %r" % group.gid)
        for op in group.ops:
            prev = seen_op.get(op.oid)
            if prev is not None:
                raise ValueError(
                    "op appears in multiple groups: %r in %r and %r"
                    % (op.oid, prev, group.gid)
                )
            seen_op[op.oid] = group.gid

    seen_rel = set()
    succ = {g.gid: [] for g in groups}
    indeg = {g.gid: 0 for g in groups}

    for rel in rels:
        if rel.src not in gid_set or rel.dst not in gid_set:
            raise ValueError(
                "rel references unknown group: %r -> %r" % (rel.src, rel.dst)
            )
        if rel.src == rel.dst:
            raise ValueError("self rel is not allowed: %r" % (rel.src,))
        key = (rel.src, rel.dst)
        if key in seen_rel:
            raise ValueError("duplicate rel: %r -> %r" % key)
        seen_rel.add(key)
        succ[rel.src].append(rel.dst)
        indeg[rel.dst] += 1

    ready = sorted([gid for gid, deg in indeg.items() if deg == 0])
    seen_cnt = 0
    while ready:
        gid = ready.pop(0)
        seen_cnt += 1
        for dst in sorted(succ[gid]):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
                ready.sort()

    if seen_cnt != len(groups):
        raise ValueError("group rels must form a DAG")


def _build_group(
    group: FusionGroup,
    dag: Any,
    graph: WorkloadFusionGraphAdapter,
    sem: WorkloadFusionSemanticAdapter,
) -> GroupSpec:
    ops = tuple(_build_op(op_id=op_id, dag=dag, sem=sem) for op_id in group.ops)
    rels = group_op_rels(group=group, graph=graph)
    return GroupSpec(
        gid=group.group_id,
        ops=ops,
        ins=tuple(group.inputs),
        outs=tuple(group.outputs),
        temps=tuple(group.temps),
        rels=rels,
    )


def _build_op(
    op_id: str,
    dag: Any,
    sem: WorkloadFusionSemanticAdapter,
) -> OpSpec:
    op_obj = _get_op(dag, op_id)
    ins = tuple(str(x) for x in getattr(op_obj, "inputs", ()))
    outs = tuple(str(x) for x in getattr(op_obj, "outputs", ()))
    try:
        op = sem.op_kind(op_id)
    except Exception:
        op = str(getattr(op_obj, "op_type", "unknown"))
    return OpSpec(oid=op_id, op=op, ins=ins, outs=outs)


def _pick_bind(
    src: str,
    dst: str,
    tens: Tuple[str, ...],
    src_ops: Tuple[str, ...],
    dst_ops: Tuple[str, ...],
    graph: WorkloadFusionGraphAdapter,
    sem: WorkloadFusionSemanticAdapter,
    cfg: RelCfg,
) -> Bind:
    if cfg.want_pipe(src, dst):
        if _can_pipe(
            tens=tens,
            src_ops=src_ops,
            dst_ops=dst_ops,
            graph=graph,
            sem=sem,
        ):
            return Bind.PIPE
    return Bind.SEQ


def _pick_hold(
    src: str,
    dst: str,
    tens: Tuple[str, ...],
    src_ops: Tuple[str, ...],
    dst_ops: Tuple[str, ...],
    graph: WorkloadFusionGraphAdapter,
    cfg: RelCfg,
) -> Tuple[str, ...]:
    src_shared = _shared_of_ops(ops=src_ops, graph=graph)
    dst_shared = _shared_of_ops(ops=dst_ops, graph=graph)

    out = []
    for name in tens:
        if name in src_shared or name in dst_shared or cfg.want_hold(src, dst, name):
            out.append(name)
    return tuple(sorted(set(out)))


def _can_pipe(
    tens: Tuple[str, ...],
    src_ops: Tuple[str, ...],
    dst_ops: Tuple[str, ...],
    graph: WorkloadFusionGraphAdapter,
    sem: WorkloadFusionSemanticAdapter,
) -> bool:
    if not tens:
        return False
    if not src_ops or not dst_ops:
        return False

    for name in tens:
        cat = sem.tensor_category(name)
        if cat in {"state", "output", "stat"}:
            return False

    hit = False
    for src_op in src_ops:
        for dst_op in dst_ops:
            for edge in graph.edges_between(src_op, dst_op):
                if edge.tensor not in tens:
                    continue
                hit = True
                if not getattr(edge, "src_dims", ()) or not getattr(edge, "dst_dims", ()):
                    return False
    return hit


def _edge_tens_map(gene: FusionGene) -> Dict[EdgeKey, Tuple[str, ...]]:
    out: Dict[EdgeKey, list] = {}
    for src, dst, tens in gene.group_edges_tensors:
        out.setdefault((src, dst), []).append(tens)
    return {key: tuple(sorted(set(val))) for key, val in out.items()}


def _edge_tensors(
    graph: WorkloadFusionGraphAdapter,
    src: str,
    dst: str,
) -> Tuple[str, ...]:
    out = []
    for edge in graph.edges_between(src, dst):
        out.append(str(edge.tensor))
    return tuple(out)


def _shared_of_ops(
    ops: Tuple[str, ...],
    graph: WorkloadFusionGraphAdapter,
) -> Tuple[str, ...]:
    if not ops:
        return ()
    bound = graph.boundary(set(ops))
    shared = getattr(bound, "shared", ())
    return tuple(sorted(str(x) for x in shared))


def _get_op(dag: Any, op_id: str) -> Any:
    if hasattr(dag, "op"):
        return dag.op(op_id)
    if hasattr(dag, "get_op"):
        return dag.get_op(op_id)
    raise AttributeError("dag must provide op() or get_op()")


def _norm_edge(val: EdgeKey) -> EdgeKey:
    if len(val) != 2:
        raise ValueError("pipe edge must be (src, dst)")
    src = str(val[0]).strip()
    dst = str(val[1]).strip()
    if not src or not dst or src == dst:
        raise ValueError("invalid pipe edge")
    return (src, dst)


def _norm_hold(val: HoldKey) -> HoldKey:
    if len(val) != 3:
        raise ValueError("hold item must be (src, dst, tens)")
    src = str(val[0]).strip()
    dst = str(val[1]).strip()
    tens = str(val[2]).strip()
    if not src or not dst or not tens or src == dst:
        raise ValueError("invalid hold item")
    return (src, dst, tens)


__all__ = [
    "RelCfg",
    "FusionOut",
    "build_fusion_out",
    "build_groups",
    "build_rels",
    "group_op_rels",
    "validate_fusion_out",
]
