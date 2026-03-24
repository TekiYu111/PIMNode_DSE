from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Set

from .fusion_gene import FusionGene, FusionGroup
from .graph_adapter import WorkloadFusionGraphAdapter

try:
    from .sematic_adapter import WorkloadFusionSemanticAdapter
except Exception:  # pragma: no cover
    from .semantic_adapter import WorkloadFusionSemanticAdapter  # type: ignore


class EdgeLike(Protocol):
    src: str
    dst: str


class WorkloadDAG(Protocol):
    def op_names(self) -> Iterable[str]:
        ...

    def edge_list(self) -> Iterable[EdgeLike]:
        ...

    def topo_order(self) -> Iterable[str]:
        ...

    def get_op(self, op_id: str) -> Any:
        ...

    def successors(self, op_id: str) -> Iterable[str]:
        ...

    def predecessors(self, op_id: str) -> Iterable[str]:
        ...

    def boundary_tensors(self, ops: Set[str]) -> Mapping[str, Set[str]]:
        ...


@dataclass(frozen=True)
class FusionSpaceConfig:
    max_depth: int = 6
    max_front: int = 32
    max_sig: int = 3
    max_out: int = 64
    max_group_ops: int = 4
    allow_cross_phase: bool = False
    allow_state_out_mix: bool = False
    keep_singleton: bool = True


@dataclass(frozen=True)
class StateProfile:
    cross_cnt: int
    state_cnt: int
    out_cnt: int
    bound_cnt: int
    max_ops: int
    group_cnt: int

    def key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.cross_cnt,
            self.state_cnt,
            self.out_cnt,
            self.bound_cnt,
            self.max_ops,
            self.group_cnt,
        )

    def dominates(self, other: "StateProfile") -> bool:
        a = self.key()
        b = other.key()
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


@dataclass(frozen=True)
class StateSig:
    main_shape: str
    max_ops: int
    state_bin: int
    out_bin: int
    bound_bin: int
    group_bin: int

    def key(self) -> tuple[str, int, int, int, int, int]:
        return (
            self.main_shape,
            self.max_ops,
            self.state_bin,
            self.out_bin,
            self.bound_bin,
            self.group_bin,
        )


@dataclass(frozen=True)
class MergeStep:
    src_gid: str
    dst_gid: str


@dataclass(frozen=True)
class FusionState:
    gene: FusionGene
    depth: int
    profile: StateProfile
    sig: StateSig

    def key(self) -> tuple[int, int, int, int, int, int]:
        return self.profile.key()


@dataclass
class FusionSpace:
    dag: WorkloadDAG
    config: FusionSpaceConfig = field(default_factory=FusionSpaceConfig)
    graph: Optional[WorkloadFusionGraphAdapter] = None
    semantic: Optional[WorkloadFusionSemanticAdapter] = None

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = WorkloadFusionGraphAdapter(self.dag)
        if self.semantic is None:
            self.semantic = WorkloadFusionSemanticAdapter(self.dag)
        self._topo_cache: Optional[tuple[str, ...]] = None
        self._topo_pos: dict[str, int] = {}
        self._succ_cache: Optional[dict[str, tuple[str, ...]]] = None

    def enumerate_genes(self) -> list[FusionGene]:
        seeds = [self.init_state()]
        done: list[FusionState] = []
        seen = {self._state_id(seeds[0].gene)}

        front = self._prune_states(seeds)
        for _ in range(self.config.max_depth):
            next_states: list[FusionState] = []
            grew = False
            for state in front:
                merges = self.list_merges(state)
                if not merges:
                    done.append(state)
                    continue
                for step in merges:
                    child = self.apply_merge(state, step)
                    if child is None:
                        continue
                    sid = self._state_id(child.gene)
                    if sid in seen:
                        continue
                    seen.add(sid)
                    next_states.append(child)
                    grew = True
            if not grew:
                done.extend(front)
                break
            front = self._prune_states(next_states)
            if not front:
                break
        else:
            done.extend(front)

        genes: list[FusionGene] = []
        out_seen: set[str] = set()
        init_gene = seeds[0].gene
        if self.config.keep_singleton:
            out_seen.add(init_gene.gene_id)
            genes.append(init_gene)

        for state in sorted(done, key=lambda item: item.key()):
            gene = state.gene
            if gene.gene_id in out_seen:
                continue
            out_seen.add(gene.gene_id)
            genes.append(gene)
            if len(genes) >= self.config.max_out:
                break
        return genes

    def enumerate_groups(self) -> list[FusionGroup]:
        seen: set[str] = set()
        out: list[FusionGroup] = []
        for gene in self.enumerate_genes():
            for group in gene.groups:
                if group.group_id in seen:
                    continue
                seen.add(group.group_id)
                out.append(group)
        return out

    def init_state(self) -> FusionState:
        parts = [[op] for op in self._topo()]
        gene = self._make_gene(parts, tag="init")
        profile = self.build_profile(gene)
        sig = self.build_sig(gene, profile)
        return FusionState(gene=gene, depth=0, profile=profile, sig=sig)

    def list_merges(self, state: FusionState) -> list[MergeStep]:
        groups = self._group_map(state.gene)
        out: list[MergeStep] = []
        for src_gid, dst_gid in self._gene_edges(state.gene):
            src = groups.get(src_gid)
            dst = groups.get(dst_gid)
            if src is None or dst is None:
                continue
            if not self._can_merge(src, dst):
                continue
            out.append(MergeStep(src_gid=src_gid, dst_gid=dst_gid))
        return out

    def apply_merge(self, state: FusionState, step: MergeStep) -> Optional[FusionState]:
        groups = self._group_map(state.gene)
        src = groups.get(step.src_gid)
        dst = groups.get(step.dst_gid)
        if src is None or dst is None:
            return None

        merged_ops = sorted(
            set(src.ops) | set(dst.ops),
            key=self._topo_index,
        )

        parts: list[list[str]] = []
        merged_added = False
        for group in state.gene.groups:
            if group.group_id in {step.src_gid, step.dst_gid}:
                if not merged_added:
                    parts.append(list(merged_ops))
                    merged_added = True
                continue
            parts.append(list(group.ops))
        if not merged_added:
            return None

        gene = self._make_gene(parts, tag=f"m{state.depth + 1}")
        profile = self.build_profile(gene)
        sig = self.build_sig(gene, profile)
        return FusionState(gene=gene, depth=state.depth + 1, profile=profile, sig=sig)

    def build_profile(self, gene: FusionGene) -> StateProfile:
        cross_cnt = 0
        state_cnt = 0
        out_cnt = 0
        for src_gid, dst_gid, tensor in gene.group_edges_tensors:
            if src_gid == dst_gid:
                continue
            cross_cnt += 1
            cat = self._tensor_category(tensor)
            if cat == "state":
                state_cnt += 1
            if cat == "output":
                out_cnt += 1

        bound_cnt = 0
        max_ops = 0
        for group in gene.groups:
            max_ops = max(max_ops, len(group.ops))
            bound_cnt += len(group.inputs) + len(group.outputs)

        return StateProfile(
            cross_cnt=cross_cnt,
            state_cnt=state_cnt,
            out_cnt=out_cnt,
            bound_cnt=bound_cnt,
            max_ops=max_ops,
            group_cnt=len(gene.groups),
        )

    def build_sig(self, gene: FusionGene, profile: StateProfile) -> StateSig:
        return StateSig(
            main_shape=self._main_shape(gene),
            max_ops=min(profile.max_ops, self.config.max_group_ops),
            state_bin=min(profile.state_cnt, 2),
            out_bin=min(profile.out_cnt, 2),
            bound_bin=min(profile.bound_cnt // 4, 3),
            group_bin=min(profile.group_cnt, 7),
        )

    def prune_states(self, states: Sequence[FusionState]) -> list[FusionState]:
        return self._prune_states(states)

    def _prune_states(self, states: Sequence[FusionState]) -> list[FusionState]:
        buckets: dict[tuple[str, int, int, int, int, int], list[FusionState]] = defaultdict(list)
        for state in states:
            buckets[state.sig.key()].append(state)

        kept: list[FusionState] = []
        for bucket in buckets.values():
            pareto = self._pareto_keep(bucket)
            pareto.sort(key=lambda item: item.key())
            kept.extend(pareto[: self.config.max_sig])

        kept.sort(key=lambda item: item.key())
        return kept[: self.config.max_front]

    def _pareto_keep(self, states: Sequence[FusionState]) -> list[FusionState]:
        out: list[FusionState] = []
        for cand in states:
            drop = False
            next_out: list[FusionState] = []
            for cur in out:
                if cur.profile.dominates(cand.profile):
                    drop = True
                    next_out.append(cur)
                    continue
                if cand.profile.dominates(cur.profile):
                    continue
                next_out.append(cur)
            if not drop:
                next_out.append(cand)
            out = next_out
        return out

    def _can_merge(self, src: FusionGroup, dst: FusionGroup) -> bool:
        merged = set(src.ops) | set(dst.ops)
        if len(merged) > self.config.max_group_ops:
            return False
        if not self._is_convex(merged):
            return False
        if not self._phase_ok(merged):
            return False
        if not self._boundary_ok(merged):
            return False
        return True

    def _phase_ok(self, ops: Set[str]) -> bool:
        if self.config.allow_cross_phase:
            return True
        vals = {self.semantic.op_phase(op) for op in ops if self.semantic.op_phase(op) not in {"", "unknown", None}}
        return len(vals) <= 1

    def _boundary_ok(self, ops: Set[str]) -> bool:
        if self.config.allow_state_out_mix:
            return True
        bound = self.graph.boundary(ops)
        seen_state = False
        seen_out = False
        for tensor in bound.outputs:
            cat = self._tensor_category(tensor)
            if cat == "state":
                seen_state = True
            if cat == "output":
                seen_out = True
        return not (seen_state and seen_out)

    def _is_convex(self, ops: Set[str]) -> bool:
        try:
            return bool(self.graph.is_convex(ops))
        except Exception:
            return self._is_convex_fallback(ops)

    def _is_convex_fallback(self, ops: Set[str]) -> bool:
        for src in ops:
            reach_in = self._reach(src, stop=None)
            inner = reach_in & ops
            outer = reach_in - ops
            for mid in outer:
                if self._reach(mid, stop=ops) & inner:
                    return False
        return True

    def _reach(self, src: str, stop: Optional[Set[str]]) -> set[str]:
        seen: set[str] = set()
        queue: deque[str] = deque([src])
        succ = self._succ_map()
        while queue:
            cur = queue.popleft()
            for nxt in succ.get(cur, ()): 
                if nxt in seen:
                    continue
                seen.add(nxt)
                if stop is not None and nxt in stop:
                    continue
                queue.append(nxt)
        seen.discard(src)
        return seen

    def _main_shape(self, gene: FusionGene) -> str:
        chain = self._main_chain()
        if not chain:
            return "none"

        op_to_gid = gene.op_to_group()
        groups = self._group_map(gene)
        seen: set[str] = set()
        parts: list[str] = []
        for op in chain:
            gid = op_to_gid.get(op)
            if gid is None or gid in seen:
                continue
            seen.add(gid)
            group = groups[gid]
            kinds = [self.semantic.op_kind(name) for name in group.ops if name in chain]
            if kinds:
                parts.append(f"[{'-'.join(kinds)}]")
        return "".join(parts) if parts else "none"

    def _main_chain(self) -> tuple[str, ...]:
        topo = self._topo()
        qk = self._find_first(topo, "qk")
        if qk is None:
            return tuple()
        softmax = self._find_succ(qk, "softmax")
        if softmax is None:
            return (qk,)
        av = self._find_succ(softmax, "av")
        if av is None:
            return (qk, softmax)
        return (qk, softmax, av)

    def _find_first(self, topo: Sequence[str], kind: str) -> Optional[str]:
        for op in topo:
            if self.semantic.op_kind(op) == kind:
                return op
        return None

    def _find_succ(self, op: str, kind: str) -> Optional[str]:
        hits = [name for name in self.graph.successors(op) if self.semantic.op_kind(name) == kind]
        if len(hits) == 1:
            return hits[0]
        return None

    def _make_gene(self, parts: Sequence[Sequence[str]], tag: str) -> FusionGene:
        groups = tuple(self._make_group(idx, ops) for idx, ops in enumerate(parts))
        gene_id = self._gene_id(groups, tag)
        return FusionGene.from_groups(gene_id=gene_id, groups=groups)

    def _make_group(self, idx: int, ops: Sequence[str]) -> FusionGroup:
        op_ids = tuple(sorted(set(ops), key=self._topo_index))
        bound = self.graph.boundary(set(op_ids))
        return FusionGroup(
            group_id=f"g{idx}",
            ops=op_ids,
            inputs=tuple(sorted(bound.inputs)),
            outputs=tuple(sorted(bound.outputs)),
            temps=tuple(sorted(bound.temps)),
        )

    def _gene_id(self, groups: Sequence[FusionGroup], tag: str) -> str:
        body = "__".join("-".join(group.ops) for group in groups)
        return f"gene__{tag}__{body}"

    def _state_id(self, gene: FusionGene) -> tuple[tuple[str, tuple[str, ...]], ...]:
        rows = [(group.group_id, tuple(group.ops)) for group in gene.groups]
        return tuple(sorted(rows))

    @staticmethod
    def _group_map(gene: FusionGene) -> dict[str, FusionGroup]:
        return {group.group_id: group for group in gene.groups}

    @staticmethod
    def _gene_edges(gene: FusionGene) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(set(gene.group_edges)))

    def _topo(self) -> tuple[str, ...]:
        if self._topo_cache is None:
            topo = tuple(self.graph.topo_order())
            if not topo:
                topo = tuple(sorted(self.graph.op_names()))
            self._topo_cache = topo
            self._topo_pos = {op: idx for idx, op in enumerate(topo)}
        return self._topo_cache

    def _topo_index(self, op: str) -> int:
        self._topo()
        return self._topo_pos.get(op, len(self._topo_pos))

    def _tensor_category(self, tensor: str) -> str:
        return self.semantic.tensor_category(tensor)

    def _succ_map(self) -> dict[str, tuple[str, ...]]:
        if self._succ_cache is None:
            out: dict[str, list[str]] = defaultdict(list)
            for edge in self.graph.edges():
                out[edge.src_op].append(edge.dst_op)
            self._succ_cache = {key: tuple(val) for key, val in out.items()}
        return self._succ_cache


__all__ = [
    "FusionSpaceConfig",
    "StateProfile",
    "StateSig",
    "MergeStep",
    "FusionState",
    "FusionSpace",
]
