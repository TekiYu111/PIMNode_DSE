from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pimnode_dse.hardware.arch_spec import HardwareSpec
from pimnode_dse.mapping.fusion.fusion_gene import FusionGene, FusionGroup
from pimnode_dse.mapping.placement.node import GroupDP, StoreNode
from pimnode_dse.mapping.tilling.tilling_gene import GroupTilingSpec
from pimnode_dse.mapping.workload.workload import WorkloadDAG


STORE_KIND = "store"
COMP_KIND = "comp"
PE_LEVEL = "pe"

LoopSet = Tuple[str, ...]
TensorSet = Tuple[str, ...]
SlotSet = Tuple["FlowSlot", ...]
BlkSet = Tuple["FlowBlk", ...]
NodeMap = Dict[str, Tuple[StoreNode, ...]]
CtxMap = Dict[str, "FlowCtx"]
DpMap = Dict[str, Tuple["FlowDP", ...]]
OutMap = Dict[str, Tuple["FlowBucket", ...]]
ContractMap = Dict[str, Tuple["FlowContract", ...]]
TensorLoopMap = Dict[str, LoopSet]


# --------------------------------------------------
# Core data classes
# --------------------------------------------------

@dataclass(frozen=True)
class FlowSlot:
    pos: int
    src: str
    dst: str

    def __post_init__(self) -> None:
        pos = int(self.pos)
        src = str(self.src).strip().lower()
        dst = str(self.dst).strip().lower()
        if pos < 0 or not src or not dst or src == dst:
            raise ValueError(f"invalid FlowSlot: pos={pos}, src={src}, dst={dst}")
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dst", dst)

    def kind(self) -> str:
        return COMP_KIND if self.dst == PE_LEVEL else STORE_KIND

    def level(self) -> str:
        return self.dst


@dataclass(frozen=True)
class FlowBlk:
    pos: int
    level: str
    loops: LoopSet
    tile_size: Dict[str, int] = field(default_factory=dict)
    temporal_loops: LoopSet = ()
    spatial_loops: LoopSet = ()
    repeat_hint: int = 1
    replication_hint: int = 1
    overlap_hint: bool = False

    def __post_init__(self) -> None:
        pos = int(self.pos)
        level = _norm_level(self.level)
        loops = _norm_loop_names(self.loops)
        tile_size = {str(k).strip().lower(): int(v) for k, v in dict(self.tile_size).items()}
        temporal_loops = _norm_loop_names(self.temporal_loops)
        spatial_loops = _norm_loop_names(self.spatial_loops)
        repeat_hint = int(self.repeat_hint)
        replication_hint = int(self.replication_hint)
        overlap_hint = bool(self.overlap_hint)

        if pos < 0:
            raise ValueError("FlowBlk.pos must be >= 0")
        if not level:
            raise ValueError("FlowBlk.level must not be empty")
        if not loops:
            raise ValueError("FlowBlk.loops must not be empty")
        if repeat_hint <= 0:
            raise ValueError("FlowBlk.repeat_hint must be > 0")
        if replication_hint <= 0:
            raise ValueError("FlowBlk.replication_hint must be > 0")

        for loop in loops:
            if loop not in tile_size:
                tile_size[loop] = 1

        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "loops", loops)
        object.__setattr__(self, "tile_size", tile_size)
        object.__setattr__(self, "temporal_loops", temporal_loops)
        object.__setattr__(self, "spatial_loops", spatial_loops)
        object.__setattr__(self, "repeat_hint", repeat_hint)
        object.__setattr__(self, "replication_hint", replication_hint)
        object.__setattr__(self, "overlap_hint", overlap_hint)

    def sig(self) -> Tuple:
        return (
            self.pos,
            self.level,
            self.loops,
            tuple(sorted(self.tile_size.items())),
            self.temporal_loops,
            self.spatial_loops,
            self.repeat_hint,
            self.replication_hint,
            self.overlap_hint,
        )


@dataclass(frozen=True)
class FlowDP:
    group: str
    place: GroupDP
    slots: SlotSet
    blks: BlkSet
    tiling_sig: Tuple = ()

    def __post_init__(self) -> None:
        group = str(self.group).strip()
        if not group:
            raise ValueError("FlowDP.group must not be empty")
        if not self.slots or not self.blks:
            raise ValueError("FlowDP slots/blks must not be empty")
        if len(self.slots) != len(self.blks):
            raise ValueError("FlowDP slots/blks size mismatch")

        slots = tuple(sorted(self.slots, key=lambda x: x.pos))
        blks = tuple(sorted(self.blks, key=lambda x: x.pos))
        want = tuple(range(len(slots)))

        if tuple(s.pos for s in slots) != want:
            raise ValueError("FlowDP slots pos must be compact")
        if tuple(b.pos for b in blks) != want:
            raise ValueError("FlowDP blks pos must be compact")

        for s, b in zip(slots, blks):
            if s.pos != b.pos or s.level() != b.level:
                raise ValueError("FlowDP slot/block mismatch")

        object.__setattr__(self, "group", group)
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "blks", blks)

    def eq_sig(self) -> Tuple:
        return (
            self.place.eq_sig(),
            tuple(b.sig() for b in self.blks),
            self.tiling_sig,
        )


@dataclass(frozen=True)
class FlowOut:
    group: str
    group_get: TensorSet = ()
    group_out: TensorSet = ()
    reuse_out: TensorSet = ()
    drop: TensorSet = ()

    def __post_init__(self) -> None:
        group = str(self.group).strip()
        group_get = _norm_tensors(self.group_get)
        group_out = _norm_tensors(self.group_out)
        reuse_out = _norm_tensors(self.reuse_out)
        drop = _norm_tensors(self.drop)

        if not group:
            raise ValueError("FlowOut.group must not be empty")

        out_set = set(group_out)
        reuse_set = set(reuse_out)
        drop_set = set(drop)

        if not reuse_set.issubset(out_set):
            raise ValueError("reuse_out must be subset of group_out")
        if not drop_set.issubset(out_set):
            raise ValueError("drop must be subset of group_out")
        if reuse_set & drop_set:
            raise ValueError("reuse_out/drop must be disjoint")
        if out_set != (reuse_set | drop_set):
            raise ValueError("group_out must equal reuse_out union drop")

        object.__setattr__(self, "group", group)
        object.__setattr__(self, "group_get", group_get)
        object.__setattr__(self, "group_out", group_out)
        object.__setattr__(self, "reuse_out", reuse_out)
        object.__setattr__(self, "drop", drop)

    def eq_sig(self) -> Tuple:
        return (
            self.group,
            tuple(sorted(self.group_get)),
            tuple(sorted(self.group_out)),
            tuple(sorted(self.reuse_out)),
            tuple(sorted(self.drop)),
        )


@dataclass(frozen=True)
class FlowBucket:
    out: FlowOut
    flows: Tuple[FlowDP, ...]

    def __post_init__(self) -> None:
        flows = tuple(self.flows)
        if not flows:
            raise ValueError("FlowBucket.flows must not be empty")
        if any(f.group != self.out.group for f in flows):
            raise ValueError("FlowBucket group mismatch")
        object.__setattr__(self, "flows", flows)

    def eq_sig(self) -> Tuple:
        return self.out.eq_sig()


@dataclass(frozen=True)
class FlowContract:
    group: str
    level_blocks: BlkSet
    group_get: TensorSet = ()
    group_out: TensorSet = ()
    reuse_out: TensorSet = ()
    drop: TensorSet = ()
    edge_tensors: TensorSet = ()
    hold_tensors: TensorSet = ()

    def __post_init__(self) -> None:
        group = str(self.group).strip()
        if not group:
            raise ValueError("FlowContract.group must not be empty")

        blocks = tuple(sorted(self.level_blocks, key=lambda x: x.pos))
        group_get = _norm_tensors(self.group_get)
        group_out = _norm_tensors(self.group_out)
        reuse_out = _norm_tensors(self.reuse_out)
        drop = _norm_tensors(self.drop)
        edge_tensors = _norm_tensors(self.edge_tensors)
        hold_tensors = _norm_tensors(self.hold_tensors)

        object.__setattr__(self, "group", group)
        object.__setattr__(self, "level_blocks", blocks)
        object.__setattr__(self, "group_get", group_get)
        object.__setattr__(self, "group_out", group_out)
        object.__setattr__(self, "reuse_out", reuse_out)
        object.__setattr__(self, "drop", drop)
        object.__setattr__(self, "edge_tensors", edge_tensors)
        object.__setattr__(self, "hold_tensors", hold_tensors)

    def canonical_state(self, place: GroupDP) -> Tuple:
        return (
            self.group,
            place.eq_sig(),
            tuple(
                (
                    blk.level,
                    blk.loops,
                    tuple(sorted(blk.tile_size.items())),
                    blk.temporal_loops,
                    blk.spatial_loops,
                    blk.repeat_hint,
                    blk.replication_hint,
                    blk.overlap_hint,
                )
                for blk in self.level_blocks
            ),
            self.group_get,
            self.group_out,
            self.reuse_out,
            self.drop,
            self.edge_tensors,
            self.hold_tensors,
        )

    def dominance_vector(self, workload: Optional[WorkloadDAG]) -> Tuple:
        dram_lb = _sum_tensor_bytes(self.group_get + self.group_out, workload)
        movement_lb = _sum_tensor_bytes(self.group_get + self.group_out + self.reuse_out, workload)
        repeat_total = 1
        overlap_penalty = 0
        replication_gain = 1

        for blk in self.level_blocks:
            repeat_total *= max(1, int(blk.repeat_hint))
            overlap_penalty += 0 if blk.overlap_hint else 1
            replication_gain = max(replication_gain, int(blk.replication_hint))

        return (
            int(dram_lb),
            int(movement_lb),
            int(repeat_total),
            int(overlap_penalty),
            -int(replication_gain),
        )


@dataclass(frozen=True)
class FlowCtx:
    group: str
    tens: TensorSet = ()
    loops: LoopSet = ()
    red: LoopSet = ()
    tmap: TensorLoopMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        group = str(self.group).strip()
        if not group:
            raise ValueError("FlowCtx.group empty")

        tens = _norm_tensors(self.tens)
        loops = _norm_loop_names(self.loops)
        red = tuple(x for x in _norm_loop_names(self.red) if x in set(loops))
        tmap = _norm_tmap(self.tmap)

        object.__setattr__(self, "group", group)
        object.__setattr__(self, "tens", tens)
        object.__setattr__(self, "loops", loops)
        object.__setattr__(self, "red", red)
        object.__setattr__(self, "tmap", tmap)

    def loops_of(self, tensor: str) -> LoopSet:
        name = str(tensor).strip()
        return self.tmap.get(name, self.loops)

    def red_set(self) -> Set[str]:
        return set(self.red)

    def keep_set(self) -> Set[str]:
        r = self.red_set()
        return {x for x in self.loops if x not in r}


# --------------------------------------------------
# Context builders
# --------------------------------------------------

def build_ctx(
    group: FusionGroup,
    workload: Optional[WorkloadDAG] = None,
    loops: Optional[Sequence[str]] = None,
) -> FlowCtx:
    tens = _norm_tensors((*group.inputs, *group.outputs, *group.temps))
    if loops is not None:
        return FlowCtx(group=group.group_id, tens=tens, loops=tuple(loops), red=(), tmap={})

    if workload is None:
        return FlowCtx(group=group.group_id, tens=tens, loops=(), red=(), tmap={})

    dims, red, tmap = _collect_group_dims(group, workload)
    return FlowCtx(group=group.group_id, tens=tens, loops=dims, red=red, tmap=tmap)


def build_ctxs(
    fusion: FusionGene,
    workload: Optional[WorkloadDAG] = None,
    loops: Optional[Mapping[str, Sequence[str]]] = None,
) -> CtxMap:
    out: CtxMap = {}
    loop_map = dict(loops or {})
    for g in fusion.groups:
        out[g.group_id] = build_ctx(g, workload=workload, loops=loop_map.get(g.group_id))
    return out


# --------------------------------------------------
# Deterministic flow derive
# --------------------------------------------------

def derive_flow_from_tiling(
    place: GroupDP,
    ctx: FlowCtx,
    tiling: GroupTilingSpec,
    hw: Optional[HardwareSpec] = None,
) -> Optional[FlowDP]:
    slots = build_slots(place, hw=hw)
    if not slots:
        return None

    blks: List[FlowBlk] = []
    prev: Optional[Set[str]] = None

    for slot in slots:
        lvl = slot.level()
        blk = _blk_from_tiling_level(tiling, lvl, ctx)
        if blk is None:
            return None

        cur = set(blk.loops)
        if prev is not None and not cur.issubset(prev):
            return None
        prev = cur

        if slot.kind() == STORE_KIND and _all_red(blk.loops, ctx.red_set()):
            return None
        if slot.kind() == COMP_KIND and not _need_compute_loop(blk.loops, ctx):
            return None

        blks.append(
            FlowBlk(
                pos=slot.pos,
                level=lvl,
                loops=blk.loops,
                tile_size=blk.tile_size,
                temporal_loops=blk.temporal_loops,
                spatial_loops=blk.spatial_loops,
                repeat_hint=blk.repeat_hint,
                replication_hint=blk.replication_hint,
                overlap_hint=blk.overlap_hint,
            )
        )

    flow = FlowDP(
        group=place.group,
        place=place,
        slots=slots,
        blks=tuple(blks),
        tiling_sig=_tiling_sig(tiling),
    )
    validate_flow_with_tiling(flow, ctx, tiling, hw=hw)
    return flow


def validate_flow_with_tiling(
    flow: FlowDP,
    ctx: FlowCtx,
    tiling: GroupTilingSpec,
    hw: Optional[HardwareSpec] = None,
) -> None:
    del hw

    split = set(str(x).strip().lower() for x in tiling.split_red if str(x).strip())
    if not split.issubset(ctx.red_set()):
        raise ValueError(f"split_red not subset of ctx.red: split={split}, red={ctx.red_set()}")

    if split:
        if tiling.acc_scope == "local":
            pe_blk = _last_comp_blk(flow)
            if not split.issubset(set(pe_blk.loops)):
                raise ValueError("acc_scope=local requires split_red covered by compute blk")
        elif tiling.acc_scope == "sram":
            sram_ok = False
            for s, b in zip(flow.slots, flow.blks):
                if s.level() == "sram" and (split & set(b.loops)):
                    sram_ok = True
                    break
            if not sram_ok:
                raise ValueError("acc_scope=sram requires split_red visible on sram blk")
        else:
            raise ValueError(f"unknown acc_scope: {tiling.acc_scope!r}")

    for s, b in zip(flow.slots, flow.blks):
        expect = _blk_from_tiling_level(tiling, s.level(), ctx)
        if expect is None:
            raise ValueError(f"missing tiling blk for level {s.level()}")
        if b.loops != expect.loops:
            raise ValueError(f"block loops mismatch at {s.level()}: got={b.loops}, expect={expect.loops}")
        if tuple(sorted(b.tile_size.items())) != tuple(sorted(expect.tile_size.items())):
            raise ValueError(f"block tile_size mismatch at {s.level()}")


def enum_dp(
    places: Mapping[str, Sequence[GroupDP]],
    tilings: Mapping[str, Sequence[GroupTilingSpec]],
    ctxs: Mapping[str, FlowCtx],
    hw: Optional[HardwareSpec] = None,
) -> DpMap:
    out: DpMap = {}
    for gid, group_places in places.items():
        ctx = ctxs.get(gid)
        ts = tuple(tilings.get(gid, ()))
        if ctx is None or not ts:
            out[str(gid)] = ()
            continue

        rows: List[FlowDP] = []
        seen: Set[Tuple] = set()
        for p in group_places:
            for t in ts:
                f = derive_flow_from_tiling(p, ctx, t, hw=hw)
                if f is None:
                    continue
                key = f.eq_sig()
                if key in seen:
                    continue
                seen.add(key)
                rows.append(f)
        out[str(gid)] = tuple(rows)
    return out


# --------------------------------------------------
# Flow outputs and contracts
# --------------------------------------------------

def build_flow_out(
    flow: FlowDP,
    ctx: FlowCtx,
    group: FusionGroup,
    succ_inputs: Optional[Sequence[str]] = None,
    hw: Optional[HardwareSpec] = None,
) -> FlowOut:
    group_get = calc_group_get(group, flow, ctx)
    group_out = calc_group_out(group, flow, ctx, hw=hw)
    reuse_out = calc_reuse_out(group_out, flow, ctx, succ_inputs or (), hw=hw)
    drop = calc_drop(group_out, reuse_out)
    return FlowOut(
        group=flow.group,
        group_get=group_get,
        group_out=group_out,
        reuse_out=reuse_out,
        drop=drop,
    )


def build_flow_contract(
    flow: FlowDP,
    ctx: FlowCtx,
    group: FusionGroup,
    succ_inputs: Optional[Sequence[str]] = None,
    hw: Optional[HardwareSpec] = None,
) -> FlowContract:
    out = build_flow_out(flow, ctx, group, succ_inputs=succ_inputs, hw=hw)
    edge = flow_edge_tens(flow, ctx, outs=group.outputs, hw=hw)
    hold = _hold_out_tens(edge, flow.blks, ctx)
    return FlowContract(
        group=flow.group,
        level_blocks=flow.blks,
        group_get=out.group_get,
        group_out=out.group_out,
        reuse_out=out.reuse_out,
        drop=out.drop,
        edge_tensors=edge,
        hold_tensors=hold,
    )


def enum_group_contracts(
    flows: Sequence[FlowDP],
    ctx: FlowCtx,
    group: FusionGroup,
    succ_inputs: Optional[Sequence[str]] = None,
    hw: Optional[HardwareSpec] = None,
    workload: Optional[WorkloadDAG] = None,
) -> Tuple[FlowContract, ...]:
    exact_seen: Set[Tuple] = set()
    buckets: Dict[Tuple, List[Tuple[FlowDP, FlowContract]]] = {}

    for flow in flows:
        contract = build_flow_contract(flow, ctx, group, succ_inputs=succ_inputs, hw=hw)
        exact_key = contract.canonical_state(flow.place)
        if exact_key in exact_seen:
            continue
        exact_seen.add(exact_key)

        bucket_key = (
            contract.group,
            flow.place.eq_sig(),
            contract.group_get,
            contract.group_out,
            contract.reuse_out,
            contract.drop,
            contract.edge_tensors,
            contract.hold_tensors,
        )
        buckets.setdefault(bucket_key, []).append((flow, contract))

    out: List[FlowContract] = []
    for key in sorted(buckets.keys(), key=str):
        pairs = buckets[key]
        keep = _non_dominated_contracts(pairs, workload=workload)
        out.extend(contract for _, contract in keep)
    return tuple(out)


def enum_contracts(
    dp_map: Mapping[str, Sequence[FlowDP]],
    fusion: FusionGene,
    ctxs: Mapping[str, FlowCtx],
    succ_inputs: Optional[Mapping[str, Sequence[str]]] = None,
    hw: Optional[HardwareSpec] = None,
    workload: Optional[WorkloadDAG] = None,
) -> ContractMap:
    gmap = {g.group_id: g for g in fusion.groups}
    smap = {str(k): tuple(v) for k, v in dict(succ_inputs or {}).items()}

    out: ContractMap = {}
    for gid, flows in dp_map.items():
        g = gmap.get(gid)
        c = ctxs.get(gid)
        if g is None or c is None:
            continue
        out[gid] = enum_group_contracts(
            flows=flows,
            ctx=c,
            group=g,
            succ_inputs=smap.get(gid, ()),
            hw=hw,
            workload=workload,
        )
    return out


# --------------------------------------------------
# Optional FlowOut grouping for inspection only
# --------------------------------------------------

def enum_group_out(
    flows: Sequence[FlowDP],
    ctx: FlowCtx,
    group: FusionGroup,
    succ_inputs: Optional[Sequence[str]] = None,
    hw: Optional[HardwareSpec] = None,
) -> Tuple[FlowBucket, ...]:
    by_sig: Dict[Tuple, List[Tuple[FlowDP, FlowOut]]] = {}
    for f in flows:
        o = build_flow_out(f, ctx, group, succ_inputs=succ_inputs, hw=hw)
        by_sig.setdefault(o.eq_sig(), []).append((f, o))

    return tuple(
        FlowBucket(out=pairs[0][1], flows=tuple(x[0] for x in pairs))
        for _, pairs in sorted(by_sig.items(), key=lambda x: str(x[0]))
    )


def enum_out(
    dp_map: Mapping[str, Sequence[FlowDP]],
    fusion: FusionGene,
    ctxs: Mapping[str, FlowCtx],
    succ_inputs: Optional[Mapping[str, Sequence[str]]] = None,
    hw: Optional[HardwareSpec] = None,
) -> OutMap:
    gmap = {g.group_id: g for g in fusion.groups}
    smap = {str(k): tuple(v) for k, v in dict(succ_inputs or {}).items()}

    out: OutMap = {}
    for gid, flows in dp_map.items():
        g = gmap.get(gid)
        c = ctxs.get(gid)
        if g is None or c is None:
            continue
        out[gid] = enum_group_out(
            flows=flows,
            ctx=c,
            group=g,
            succ_inputs=smap.get(gid, ()),
            hw=hw,
        )
    return out


# --------------------------------------------------
# Tensor I/O analysis
# --------------------------------------------------

def flow_read_tens(flow: FlowDP, ctx: FlowCtx, ins: Sequence[str] = ()) -> TensorSet:
    want = set(_norm_tensors(ins))
    node_map = _node_map(flow.place)

    first_store: Optional[Tuple[FlowSlot, FlowBlk]] = None
    for s, b in zip(flow.slots, flow.blks):
        if s.kind() == STORE_KIND:
            first_store = (s, b)
            break
    if first_store is None:
        return ()

    slot, blk = first_store
    live = set(blk.loops)

    out: List[str] = []
    seen: Set[str] = set()
    for ten in _slot_tens(slot, node_map):
        if want and ten not in want:
            continue
        dims = set(ctx.loops_of(ten))
        if dims and not (dims & live):
            continue
        if ten not in seen:
            seen.add(ten)
            out.append(ten)
    return _norm_tensors(out)


def flow_write_tens(
    flow: FlowDP,
    ctx: FlowCtx,
    outs: Sequence[str] = (),
    hw: Optional[HardwareSpec] = None,
) -> TensorSet:
    last_mem = _last_store_mem(flow, hw=hw)
    if not last_mem:
        return ()

    node_map = _node_map(flow.place)
    want = set(_norm_tensors(outs))
    comp_blk = _last_comp_blk(flow)
    live = set(comp_blk.loops)

    out: List[str] = []
    seen: Set[str] = set()
    for node in node_map.get(last_mem, ()):
        for ten in node.tens:
            if want and ten not in want:
                continue
            dims = set(ctx.loops_of(ten))
            if dims and not (dims & live):
                continue
            if ten not in seen:
                seen.add(ten)
                out.append(ten)
    return _norm_tensors(out)


def flow_edge_tens(
    flow: FlowDP,
    ctx: FlowCtx,
    outs: Sequence[str] = (),
    hw: Optional[HardwareSpec] = None,
) -> TensorSet:
    edge = flow_write_tens(flow=flow, ctx=ctx, outs=outs, hw=hw)
    hold = set(_hold_out_tens(edge, flow.blks, ctx))
    return _norm_tensors(x for x in edge if x in hold)


def calc_group_get(group: FusionGroup, flow: FlowDP, ctx: FlowCtx) -> TensorSet:
    return flow_read_tens(flow, ctx, ins=group.inputs)


def calc_group_out(
    group: FusionGroup,
    flow: FlowDP,
    ctx: FlowCtx,
    hw: Optional[HardwareSpec] = None,
) -> TensorSet:
    return flow_write_tens(flow, ctx, outs=group.outputs, hw=hw)


def calc_reuse_out(
    group_out: Sequence[str],
    flow: FlowDP,
    ctx: FlowCtx,
    succ_inputs: Sequence[str],
    hw: Optional[HardwareSpec] = None,
) -> TensorSet:
    outs = _norm_tensors(group_out)
    want = set(_norm_tensors(succ_inputs))
    edge = set(flow_edge_tens(flow, ctx, outs=outs, hw=hw))
    return _norm_tensors(x for x in outs if x in want and x in edge)


def calc_drop(group_out: Sequence[str], reuse_out: Sequence[str]) -> TensorSet:
    reuse = set(_norm_tensors(reuse_out))
    return _norm_tensors(x for x in _norm_tensors(group_out) if x not in reuse)


# --------------------------------------------------
# Canonical + dominance pruning
# --------------------------------------------------

def _non_dominated_contracts(
    rows: Sequence[Tuple[FlowDP, FlowContract]],
    *,
    workload: Optional[WorkloadDAG],
) -> List[Tuple[FlowDP, FlowContract]]:
    kept: List[Tuple[FlowDP, FlowContract]] = []
    for cand in rows:
        _, cand_contract = cand
        cand_vec = cand_contract.dominance_vector(workload)

        drop = False
        next_kept: List[Tuple[FlowDP, FlowContract]] = []

        for cur in kept:
            _, cur_contract = cur
            cur_vec = cur_contract.dominance_vector(workload)

            if _dominates(cur_vec, cand_vec):
                drop = True
                next_kept.append(cur)
                continue

            if _dominates(cand_vec, cur_vec):
                continue

            next_kept.append(cur)

        if not drop:
            next_kept.append(cand)
        kept = next_kept

    return kept


def _dominates(a: Tuple, b: Tuple) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def build_slots(place: GroupDP, hw: Optional[HardwareSpec] = None) -> SlotSet:
    base = _store_slots(place)
    comp = _compute_level(hw)
    if not comp:
        return base
    last_store = _last_store_level(base, place)
    if not last_store:
        return base
    if base and base[-1].dst == comp:
        return base
    return tuple((*base, FlowSlot(pos=len(base), src=last_store, dst=comp)))


def _blk_from_tiling_level(
    tiling: GroupTilingSpec,
    level: str,
    ctx: FlowCtx,
) -> Optional[FlowBlk]:
    lv = _norm_level(level)
    try:
        tier = tiling.get(lv)
    except KeyError:
        return None

    order = tuple(x for x in _norm_loop_names(tier.loop_order) if x in set(ctx.loops))
    if not order:
        return None

    tile_size = {str(k).strip().lower(): int(v) for k, v in getattr(tier, "tile_size", {}).items()}
    temporal_loops, spatial_loops = _split_loops_for_level(lv, order)
    repeat_hint = _repeat_hint(temporal_loops, tile_size)
    replication_hint = _replication_hint(spatial_loops, tile_size)
    overlap_hint = bool(getattr(tier, "rw_overlap", False))

    return FlowBlk(
        pos=0,
        level=lv,
        loops=order,
        tile_size=tile_size,
        temporal_loops=temporal_loops,
        spatial_loops=spatial_loops,
        repeat_hint=repeat_hint,
        replication_hint=replication_hint,
        overlap_hint=overlap_hint,
    )


def _collect_group_dims(
    group: FusionGroup,
    workload: WorkloadDAG,
) -> Tuple[LoopSet, LoopSet, TensorLoopMap]:
    dims: List[str] = []
    red: List[str] = []
    tmap: TensorLoopMap = {}
    seen_dim: Set[str] = set()
    seen_red: Set[str] = set()

    for op_id in group.ops:
        op = workload.op(op_id)

        for d in getattr(op, "iter_dims", ()):
            n = str(d).strip().lower()
            if n and n not in seen_dim:
                seen_dim.add(n)
                dims.append(n)

        for d in getattr(op, "reduce_dims", ()):
            n = str(d).strip().lower()
            if n and n not in seen_red:
                seen_red.add(n)
                red.append(n)

        for ten, tdims in getattr(op, "tensor_dims", {}).items():
            ten_name = str(ten).strip()
            vals = _norm_loop_names(tdims)
            prev = tmap.get(ten_name)
            if prev is None:
                tmap[ten_name] = vals
            else:
                keep = tuple(x for x in prev if x in set(vals))
                tmap[ten_name] = keep or prev

    red_filtered = tuple(x for x in dims if x in set(red))
    return tuple(dims), red_filtered, tmap


def _node_map(place: GroupDP) -> NodeMap:
    out: Dict[str, List[StoreNode]] = {}
    for n in place.nodes:
        out.setdefault(_norm_level(n.level), []).append(n)
    return {k: tuple(v) for k, v in out.items()}


def _slot_tens(slot: FlowSlot, node_map: NodeMap) -> Tuple[str, ...]:
    out: List[str] = []
    seen: Set[str] = set()
    for lv in (slot.src, slot.dst):
        for node in node_map.get(lv, ()):
            for ten in node.tens:
                if ten not in seen:
                    seen.add(ten)
                    out.append(ten)
    return tuple(out)


def _store_slots(place: GroupDP) -> SlotSet:
    levels = _level_chain(place.nodes)
    if len(levels) < 2:
        return ()
    out: List[FlowSlot] = []
    for i, (src, dst) in enumerate(zip(levels, levels[1:])):
        if src != dst:
            out.append(FlowSlot(pos=i, src=src, dst=dst))
    return tuple(out)


def _level_chain(nodes: Sequence[StoreNode]) -> Tuple[str, ...]:
    out: List[str] = []
    for n in sorted(nodes, key=lambda x: x.pos):
        lv = _norm_level(n.level)
        if not out or out[-1] != lv:
            out.append(lv)
    return tuple(out)


def _last_store_level(slots: SlotSet, place: GroupDP) -> str:
    if slots:
        return slots[-1].dst
    if not place.nodes:
        return ""
    return _norm_level(sorted(place.nodes, key=lambda x: x.pos)[-1].level)


def _last_store_mem(flow: FlowDP, hw: Optional[HardwareSpec] = None) -> str:
    vals: List[str] = []
    if hw is not None:
        raw = hw.storage_levels
        vals = [_norm_level(x) for x in raw if str(x).strip()]
    if vals:
        have = {_norm_level(n.level) for n in flow.place.nodes}
        vals = [x for x in vals if x in have]
    if vals:
        return vals[-1]

    nodes = tuple(sorted(flow.place.nodes, key=lambda x: x.pos))
    if not nodes:
        return ""
    chain = _level_chain(nodes)
    return chain[-1] if chain else ""


def _compute_level(hw: Optional[HardwareSpec] = None) -> str:
    if hw is None:
        return PE_LEVEL
    raw = hw.compute_levels
    vals = _norm_loop_names(raw)
    return vals[0] if vals else PE_LEVEL


def _last_comp_blk(flow: FlowDP) -> FlowBlk:
    for s, b in reversed(tuple(zip(flow.slots, flow.blks))):
        if s.kind() == COMP_KIND:
            return b
    return flow.blks[-1]


def _hold_out_tens(outs: Sequence[str], blks: BlkSet, ctx: FlowCtx) -> TensorSet:
    want = _norm_tensors(outs)
    keep = ctx.keep_set()
    out: List[str] = []
    for ten in want:
        dims = set(ctx.loops_of(ten))
        if not dims:
            continue
        if any((set(b.loops) & dims & keep) for b in blks):
            out.append(ten)
    return _norm_tensors(out)


def _need_compute_loop(loops: LoopSet, ctx: FlowCtx) -> bool:
    if not loops:
        return False
    if not ctx.red:
        return True
    r = ctx.red_set()
    return any(x not in r for x in loops)


def _all_red(loops: LoopSet, red: Set[str]) -> bool:
    return bool(loops) and all(x in red for x in loops)


def _tiling_sig(tiling: GroupTilingSpec) -> Tuple:
    rows = []
    for lv, spec in sorted(tiling.tier_tiles.items()):
        rows.append(
            (
                _norm_level(lv),
                tuple(_norm_loop_names(spec.loop_order)),
                tuple(sorted((k, int(v)) for k, v in spec.tile_size.items())),
                getattr(spec, "buf_mode", "single"),
                bool(getattr(spec, "rw_overlap", False)),
            )
        )
    return (
        str(tiling.group_id).strip(),
        tuple(_norm_loop_names(getattr(tiling, "split_red", ()))),
        str(getattr(tiling, "acc_scope", "sram")).strip().lower(),
        tuple(rows),
    )


def _split_loops_for_level(level: str, loops: LoopSet) -> Tuple[LoopSet, LoopSet]:
    if str(level).strip().lower() == "pe":
        return (), loops
    return loops, ()


def _repeat_hint(temporal_loops: Sequence[str], tile_size: Mapping[str, int]) -> int:
    # dataflow层只保存局部repeat hint；严格全局repeat由estimate结合full extent再解释
    repeat = 1
    for loop in temporal_loops:
        repeat *= max(1, int(tile_size.get(loop, 1)))
    return max(1, repeat)


def _replication_hint(spatial_loops: Sequence[str], tile_size: Mapping[str, int]) -> int:
    repl = 1
    for loop in spatial_loops:
        repl *= max(1, int(tile_size.get(loop, 1)))
    return max(1, repl)


def _sum_tensor_bytes(names: Sequence[str], workload: Optional[WorkloadDAG]) -> int:
    if workload is None:
        return len(tuple(names))
    total = 0
    for name in names:
        try:
            total += int(workload.tensor(name).size_bytes())
        except Exception:
            total += 1
    return total


def _norm_level(x: object) -> str:
    s = str(x).strip().lower()
    return s[:-5] if s.endswith("_tier") else s


def _norm_loop_names(items: Iterable[object]) -> LoopSet:
    out: List[str] = []
    seen: Set[str] = set()
    for it in items:
        n = str(it).strip().lower()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def _norm_tensors(items: Iterable[object]) -> TensorSet:
    out: List[str] = []
    seen: Set[str] = set()
    for it in items:
        n = str(it).strip()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def _norm_tmap(raw: Mapping[str, Iterable[object]]) -> TensorLoopMap:
    out: TensorLoopMap = {}
    for k, v in raw.items():
        name = str(k).strip()
        if name:
            out[name] = _norm_loop_names(v)
    return out


def explain(flow: FlowDP) -> Dict[str, object]:
    return {
        "group": flow.group,
        "slots": [{"pos": s.pos, "src": s.src, "dst": s.dst, "kind": s.kind()} for s in flow.slots],
        "blks": [
            {
                "pos": b.pos,
                "level": b.level,
                "loops": list(b.loops),
                "tile_size": dict(b.tile_size),
                "temporal_loops": list(b.temporal_loops),
                "spatial_loops": list(b.spatial_loops),
                "repeat_hint": b.repeat_hint,
                "replication_hint": b.replication_hint,
                "overlap_hint": b.overlap_hint,
            }
            for b in flow.blks
        ],
    }


def explain_out(out: FlowOut) -> Dict[str, object]:
    return {
        "group": out.group,
        "group_get": list(out.group_get),
        "group_out": list(out.group_out),
        "reuse_out": list(out.reuse_out),
        "drop": list(out.drop),
    }


def explain_contract(contract: FlowContract, place: Optional[GroupDP] = None) -> Dict[str, object]:
    return {
        "group": contract.group,
        "canonical_state": None if place is None else contract.canonical_state(place),
        "group_get": list(contract.group_get),
        "group_out": list(contract.group_out),
        "reuse_out": list(contract.reuse_out),
        "drop": list(contract.drop),
        "edge_tensors": list(contract.edge_tensors),
        "hold_tensors": list(contract.hold_tensors),
        "level_blocks": [
            {
                "level": blk.level,
                "loops": list(blk.loops),
                "tile_size": dict(blk.tile_size),
                "temporal_loops": list(blk.temporal_loops),
                "spatial_loops": list(blk.spatial_loops),
                "repeat_hint": blk.repeat_hint,
                "replication_hint": blk.replication_hint,
                "overlap_hint": blk.overlap_hint,
            }
            for blk in contract.level_blocks
        ],
    }


def explain_bucket(row: FlowBucket) -> Dict[str, object]:
    return {
        "out": explain_out(row.out),
        "flow_count": len(row.flows),
        "flows": [explain(f) for f in row.flows],
    }


__all__ = [
    "FlowSlot",
    "FlowBlk",
    "FlowDP",
    "FlowOut",
    "FlowBucket",
    "FlowContract",
    "FlowCtx",
    "build_ctx",
    "build_ctxs",
    "derive_flow_from_tiling",
    "validate_flow_with_tiling",
    "enum_dp",
    "build_flow_out",
    "build_flow_contract",
    "enum_group_contracts",
    "enum_contracts",
    "enum_group_out",
    "enum_out",
    "calc_group_get",
    "calc_group_out",
    "calc_reuse_out",
    "calc_drop",
    "explain",
    "explain_out",
    "explain_contract",
    "explain_bucket",
]
