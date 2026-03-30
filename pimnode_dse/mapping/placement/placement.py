from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pimnode_dse.hardware.arch_spec import HardwareSpec
from pimnode_dse.mapping.fusion.fusion_gene import FusionGene, FusionGroup
from pimnode_dse.mapping.placement.node import (
    VALID_RESIDENCE,
    GroupDP,
    PlacementPlan,
    ResidenceMode,
    StoreNode,
)

if TYPE_CHECKING:
    from pimnode_dse.mapping.workload.workload import WorkloadDAG


PLAN_ID = "placement"
NODE_ID = "n"
NODE_ID_MAX = 64
STORE_LEVELS = ("dram", "sram")
PE_LEVEL = "pe"

STATE_TAGS = ("cache", "state", "ctx")

Cover = Tuple[Tuple[str, ...], ...]
NodeSet = Tuple[StoreNode, ...]
DpMap = Dict[str, Tuple[GroupDP, ...]]
AllowMap = Dict[str, Tuple[str, ...]]
ResidenceHint = Dict[str, ResidenceMode]

# Key used for local node-set de-duplication inside _enum_level_nodes.
# Must include residence so buffering/policy variants are not collapsed.
NodeKey = Tuple[Tuple[str, Tuple[str, ...], ResidenceMode], ...]

# Key used for position-ordered de-duplication inside enum_group_dp.
PosKey = Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[str, ...], ResidenceMode], ...]]

# Role signatures used for placement signature.
RoleSig = Tuple[Tuple[str, str, ResidenceMode], ...]
PlaceSig = Tuple[Tuple[str, ...], RoleSig, RoleSig, RoleSig]


@dataclass(frozen=True)
class GroupCtx:
    group: str
    tens: Tuple[str, ...]
    levels: Tuple[str, ...]
    ins: Tuple[str, ...] = ()
    outs: Tuple[str, ...] = ()
    temps: Tuple[str, ...] = ()
    allow: AllowMap = field(default_factory=dict)
    # Per-tensor residence hints: callers supply explicit policies;
    # defaults are inferred from tensor role when absent.
    residence_hint: ResidenceHint = field(default_factory=dict)

    def __post_init__(self) -> None:
        group = str(self.group).strip()
        if not group:
            raise ValueError("group must not be empty")

        tens = _uniq(self.tens)
        if not tens:
            raise ValueError("tens must not be empty")

        levels = _norm_store_levels(self.levels)
        ins = _subset_keep(self.ins, tens)
        outs = _subset_keep(self.outs, tens)
        temps = _subset_keep(self.temps, tens)
        allow = _norm_allow(self.allow, levels)
        residence_hint = _norm_residence_hint(self.residence_hint, tens)

        object.__setattr__(self, "group", group)
        object.__setattr__(self, "tens", tens)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "ins", ins)
        object.__setattr__(self, "outs", outs)
        object.__setattr__(self, "temps", temps)
        object.__setattr__(self, "allow", allow)
        object.__setattr__(self, "residence_hint", residence_hint)

    def levels_of(self, ten: str) -> Tuple[str, ...]:
        name = str(ten).strip()
        return self.allow.get(name, self.levels)

    def residence_of(self, ten: str) -> ResidenceMode:
        """Return the residence policy for a tensor.

        Priority: explicit hint > role-based default.
        Role defaults:
          output  → evict   (written-back immediately, not retained)
          state   → pinned  (permanently resident across iterations)
          input   → single  (fetched once per tile, standard)
          temp    → single
        """
        name = str(ten).strip()
        if name in self.residence_hint:
            return self.residence_hint[name]
        if name in set(self.outs):
            return "evict"
        if _is_state_name(name):
            return "pinned"
        return "single"


def build_ctx(
    group: FusionGroup,
    workload: Optional["WorkloadDAG"] = None,
    hw: Optional[HardwareSpec] = None,
    levels: Optional[Sequence[str]] = None,
    allow: Optional[AllowMap] = None,
    residence_hint: Optional[ResidenceHint] = None,
) -> GroupCtx:
    tens = _group_tens(group, workload)
    vals = _pick_store_levels(hw, levels)
    return GroupCtx(
        group=group.group_id,
        tens=tens,
        levels=vals,
        ins=_uniq(group.inputs),
        outs=_uniq(group.outputs),
        temps=_uniq(group.temps),
        allow=allow or {},
        residence_hint=residence_hint or {},
    )


def build_ctxs(
    workload: Optional["WorkloadDAG"],
    fusion: FusionGene,
    hw: Optional[HardwareSpec] = None,
    levels: Optional[Sequence[str]] = None,
    allow: Optional[Dict[str, AllowMap]] = None,
    residence_hint: Optional[Dict[str, ResidenceHint]] = None,
) -> Dict[str, GroupCtx]:
    out: Dict[str, GroupCtx] = {}
    allow = allow or {}
    residence_hint = residence_hint or {}
    for group in fusion.groups:
        out[group.group_id] = build_ctx(
            group=group,
            workload=workload,
            hw=hw,
            levels=levels,
            allow=allow.get(group.group_id),
            residence_hint=residence_hint.get(group.group_id),
        )
    return out


def enum_group_dp(ctx: GroupCtx) -> Tuple[GroupDP, ...]:
    by_sig: Dict[PlaceSig, GroupDP] = {}
    seen_pos: Set[PosKey] = set()

    for cover in _enum_cover(ctx):
        for nodes in _enum_level_nodes(ctx, cover):
            for placed in _enum_pos(ctx, nodes):
                if not _legal_dp(ctx, placed):
                    continue

                pos_key = _pos_key(placed, ctx.levels)
                if pos_key in seen_pos:
                    continue
                seen_pos.add(pos_key)

                dp = GroupDP(group=ctx.group, nodes=placed)
                sig = _place_sig(ctx, dp)

                prev = by_sig.get(sig)
                if prev is None:
                    by_sig[sig] = dp
                    continue

                by_sig[sig] = _pick_place_rep(prev, dp)

    return tuple(by_sig[sig] for sig in sorted(by_sig.keys()))


def enum_dp(
    workload: Optional["WorkloadDAG"],
    fusion: FusionGene,
    hw: Optional[HardwareSpec] = None,
    levels: Optional[Sequence[str]] = None,
    allow: Optional[Dict[str, AllowMap]] = None,
    residence_hint: Optional[Dict[str, ResidenceHint]] = None,
) -> DpMap:
    out: DpMap = {}
    ctxs = build_ctxs(
        workload=workload,
        fusion=fusion,
        hw=hw,
        levels=levels,
        allow=allow,
        residence_hint=residence_hint,
    )
    for group_id, ctx in ctxs.items():
        out[group_id] = enum_group_dp(ctx)
    return out


def build_plan(selected: Sequence[GroupDP], plan_id: str = PLAN_ID) -> PlacementPlan:
    return PlacementPlan(id=plan_id, groups=tuple(selected))


def explain(dp: GroupDP) -> Dict[str, object]:
    return {
        "group": dp.group,
        "nodes": [
            {
                "id": node.id,
                "level": node.level,
                "pos": node.pos,
                "residence": getattr(node, "residence", "single"),
                "tens": list(node.tens),
            }
            for node in dp.nodes
        ],
    }


def _place_sig(ctx: GroupCtx, dp: GroupDP) -> PlaceSig:
    home = {ten: dp.node_of(ten) for ten in ctx.tens}

    chain = tuple(
        level
        for level in ctx.levels
        if any(str(home[ten].level).strip().lower() == level for ten in ctx.tens)
    )

    def role_sig(names: Sequence[str]) -> RoleSig:
        return tuple(
            sorted(
                (ten, str(home[ten].level).strip().lower(), home[ten].residence)
                for ten in names
            )
        )

    return (
        chain,
        role_sig(ctx.ins),
        role_sig(ctx.outs),
        role_sig(ctx.temps),
    )


def _place_rank(dp: GroupDP) -> tuple:
    nodes = tuple(sorted(dp.nodes, key=lambda item: item.pos))
    return (
        len(nodes),
        tuple(
            (
                str(node.level).strip().lower(),
                int(node.pos),
                str(getattr(node, "residence", "single")).strip().lower(),
                tuple(sorted(str(ten).strip() for ten in node.tens)),
            )
            for node in nodes
        ),
    )


def _pick_place_rep(old: GroupDP, new: GroupDP) -> GroupDP:
    return new if _place_rank(new) < _place_rank(old) else old


def _group_tens(group: FusionGroup, workload: Optional["WorkloadDAG"]) -> Tuple[str, ...]:
    vals = _uniq((*group.inputs, *group.outputs, *group.temps))
    if vals:
        return vals
    if workload is None:
        raise ValueError(f"group {group.group_id!r} has no tensor boundary and workload is missing")
    raw = workload.subgraph_tensors(set(group.ops))
    return _uniq((*raw.get("inputs", ()), *raw.get("outputs", ()), *raw.get("temps", ())))


def _pick_store_levels(
    hw: Optional[HardwareSpec],
    levels: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    if levels:
        return _norm_store_levels(levels)
    if hw is not None:
        raw = getattr(hw, "levels", ())
        if callable(raw):
            raw = raw()
        return _norm_store_levels(raw)
    return STORE_LEVELS


def _enum_cover(ctx: GroupCtx) -> Tuple[Cover, ...]:
    out: List[Cover] = []

    whole = _pack_cover(ctx.tens)
    if whole:
        out.append(whole)

    split = _split_cover(ctx.tens)
    if split:
        out.append(split)

    role = _role_cover(ctx)
    if role:
        out.append(role)

    io = _io_cover(ctx)
    if io:
        out.append(io)

    state = _state_cover(ctx)
    if state:
        out.append(state)

    mc = _multicopy_cover(ctx)
    if mc:
        out.append(mc)

    return _dedup_cover(out)


def _role_cover(ctx: GroupCtx) -> Cover:
    return _pack_cover(ctx.ins, ctx.outs, ctx.temps)


def _io_cover(ctx: GroupCtx) -> Cover:
    left = _uniq((*ctx.ins, *ctx.temps))
    return _pack_cover(left, ctx.outs)


def _state_cover(ctx: GroupCtx) -> Cover:
    state = tuple(name for name in ctx.tens if _is_state_name(name))
    other = tuple(name for name in ctx.tens if name not in set(state))
    return _pack_cover(state, other)


def _multicopy_cover(ctx: GroupCtx) -> Cover:
    """Generate a double-buffer cover for eligible input tensors.

    IMPORTANT: in design A, double-buffer is a *buffering mode* for a tensor at
    its home level (typically SRAM). It does NOT introduce additional logical
    replicas, so tensors must still appear exactly once across the cover.

    We generate an alternative cover that splits each eligible input tensor into
    its own singleton bucket, so it can be independently assigned to SRAM and
    tagged as residence="double".
    """
    if "sram" not in ctx.levels:
        return ()

    eligible = tuple(t for t in ctx.ins if ctx.residence_hint.get(t, "") == "double")
    if not eligible:
        return ()

    parts: list[tuple[str, ...]] = [(t,) for t in eligible]
    other = tuple(t for t in ctx.tens if t not in set(eligible))
    if other:
        parts.append(other)

    return _canon_cover(parts)


def _pack_cover(*parts: Sequence[str]) -> Cover:
    clean = [tuple(_uniq(part)) for part in parts if part]
    if not clean:
        return ()
    return _canon_cover(clean)


def _split_cover(items: Sequence[str]) -> Cover:
    vals = _uniq(items)
    if len(vals) <= 1:
        return ()
    return tuple((name,) for name in vals)


def _dedup_cover(rows: Sequence[Cover]) -> Tuple[Cover, ...]:
    out: List[Cover] = []
    seen: Set[Cover] = set()
    for row in rows:
        if not row:
            continue
        if row in seen:
            continue
        flat = tuple(name for part in row for name in part)
        if len(flat) != len(set(flat)):
            continue
        seen.add(row)
        out.append(row)
    return tuple(out)


def _is_state_name(name: str) -> bool:
    text = str(name).strip().lower()
    return any(tag in text for tag in STATE_TAGS)


def _norm_residence_hint(raw: ResidenceHint, tens: Sequence[str]) -> ResidenceHint:
    keep = set(tens)
    out: ResidenceHint = {}
    for ten, mode in dict(raw or {}).items():
        name = str(ten).strip()
        if not name or name not in keep:
            continue
        val = str(mode).strip().lower()
        if val not in VALID_RESIDENCE:
            continue
        out[name] = val
    return out


def _node_residence(ctx: GroupCtx, tens: Sequence[str], level: str) -> ResidenceMode:
    """Decide residence for a node given the bucket and chosen level."""
    lv = str(level).strip().lower()

    # If the bucket is exactly one input tensor and it is hinted as double,
    # treat it as a double-buffer resident in SRAM.
    if lv == "sram" and len(tens) == 1:
        ten = str(tens[0]).strip()
        if ten and ctx.residence_hint.get(ten) == "double":
            return "double"

    # Otherwise fall back to role-based default; if bucket contains mixed roles,
    # pick the 'strongest' (most constraining) mode.
    strength = {"single": 0, "evict": 1, "pinned": 2, "double": 3}
    mode = "single"
    for ten in tens:
        m = ctx.residence_of(ten)
        if strength.get(m, 0) > strength[mode]:
            mode = m
    return mode


def _enum_level_nodes(ctx: GroupCtx, cover: Cover) -> Tuple[NodeSet, ...]:
    if not cover:
        return ()

    modes: List[Tuple[str, ...]] = []
    for tens in cover:
        vals = _shared_levels(ctx, tens)
        if not vals:
            return ()
        modes.append(vals)

    out: List[NodeSet] = []
    seen: Set[NodeKey] = set()

    for combo in product(*modes):
        nodes = tuple(
            StoreNode(
                id=_node_id(tens),
                level=level,
                pos=0,
                tens=tens,
                residence=_node_residence(ctx, tens, level),
            )
            for tens, level in zip(cover, combo)
        )
        key = _node_key(nodes)
        if key in seen:
            continue
        seen.add(key)
        out.append(nodes)

    return tuple(out)


def _enum_pos(ctx: GroupCtx, nodes: Sequence[StoreNode]) -> Tuple[NodeSet, ...]:
    if not nodes:
        return ()

    placed = tuple(
        StoreNode(
            id=node.id,
            level=node.level,
            pos=pos,
            tens=node.tens,
            residence=node.residence,
        )
        for pos, node in enumerate(sorted(nodes, key=_pos_sort_key(ctx)))
    )
    return (placed,)


def _pos_sort_key(ctx: GroupCtx):
    def key(node: StoreNode) -> Tuple[int, Tuple[str, ...], str]:
        return (_level_pos(ctx.levels, node.level), node.tens, node.id)
    return key


def _legal_dp(ctx: GroupCtx, nodes: Sequence[StoreNode]) -> bool:
    if not nodes:
        return False

    ids: Set[str] = set()
    pos: Set[int] = set()
    hit: Dict[str, int] = {}

    for node in nodes:
        if node.id in ids:
            return False
        ids.add(node.id)

        if node.pos in pos:
            return False
        pos.add(node.pos)

        if node.level not in ctx.levels:
            return False

        for ten in node.tens:
            if node.level not in ctx.levels_of(ten):
                return False
            hit[ten] = hit.get(ten, 0) + 1

        # Double-buffer only makes sense in SRAM.
        if node.residence == "double" and node.level != "sram":
            return False

    if pos != set(range(len(nodes))):
        return False

    # Design A: no logical replicas at placement level.
    for ten in ctx.tens:
        if hit.get(ten, 0) != 1:
            return False

    return True


def _shared_levels(ctx: GroupCtx, tens: Sequence[str]) -> Tuple[str, ...]:
    keep: Optional[Tuple[str, ...]] = None
    for ten in tens:
        vals = ctx.levels_of(ten)
        keep = vals if keep is None else tuple(level for level in keep if level in vals)
    return keep or ()


def _canon_cover(parts: Sequence[Sequence[str]]) -> Cover:
    clean = [tuple(sorted(_uniq(part))) for part in parts if part]
    clean.sort(key=lambda tens: (len(tens), tens))
    return tuple(clean)


def _node_id(tens: Sequence[str]) -> str:
    parts = [name.lower().replace(":", "_").replace("-", "_") for name in tens]
    parts = [part for part in parts if part]
    if not parts:
        return NODE_ID
    text = "_".join(parts)
    return text if len(text) <= NODE_ID_MAX else text[:NODE_ID_MAX]


def _node_key(nodes: Sequence[StoreNode]) -> NodeKey:
    rows = [(str(node.level).strip().lower(), tuple(node.tens), node.residence) for node in nodes]
    rows.sort(key=lambda item: (_level_rank(item[0]), item[1], item[2]))
    return tuple(rows)


def _pos_key(nodes: Sequence[StoreNode], levels: Sequence[str]) -> PosKey:
    chain = _slot_chain(nodes, levels)
    rows = tuple((str(node.level).strip().lower(), tuple(node.tens), node.residence) for node in nodes)
    return (chain, rows)


def _slot_chain(nodes: Sequence[StoreNode], levels: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for node in sorted(nodes, key=lambda item: item.pos):
        level = str(node.level).strip().lower()
        if not out or out[-1] != level:
            out.append(level)
    if out:
        return tuple(out)
    vals = tuple(str(name).strip().lower() for name in levels)
    return vals[:0]


def _level_rank(level: str) -> Tuple[int, str]:
    name = str(level).strip().lower()
    return (0 if name in STORE_LEVELS else 1, name)


def _level_pos(levels: Sequence[str], level: str) -> int:
    want = str(level).strip().lower()
    vals = tuple(str(name).strip().lower() for name in levels)
    try:
        return vals.index(want)
    except ValueError:
        return len(vals)


def _uniq(items: Iterable[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in items:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _subset_keep(items: Iterable[str], full: Sequence[str]) -> Tuple[str, ...]:
    keep = set(full)
    out: List[str] = []
    seen: Set[str] = set()
    for raw in items:
        name = str(raw).strip()
        if not name or name not in keep or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _norm_store_levels(levels: Iterable[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in levels:
        name = str(raw).strip().lower()
        if not name or name == PE_LEVEL or name in seen:
            continue
        seen.add(name)
        out.append(name)
    if not out:
        raise ValueError("store levels must not be empty")
    return tuple(out)


def _norm_allow(allow: AllowMap, levels: Sequence[str]) -> AllowMap:
    vals = tuple(levels)
    keep = set(vals)
    out: AllowMap = {}
    for ten, raw in allow.items():
        name = str(ten).strip()
        if not name:
            continue
        part = tuple(level for level in _norm_store_levels(raw) if level in keep)
        out[name] = part or vals
    return out
