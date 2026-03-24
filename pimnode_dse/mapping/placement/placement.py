from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Sequence

from pimnode_dse.mapping.fusion.fusion_gene import FusionGene, FusionGroup
from pimnode_dse.mapping.placement.node import GroupDP, PlacementPlan, StoreNode


@dataclass(frozen=True)
class GroupCtx:
    """Small placement-side group view.

    Placement should only consume this view. Workload and fusion specifics are
    adapted at the file boundary.
    """

    id: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    temps: tuple[str, ...] = ()
    edge_in: tuple[str, ...] = ()
    edge_out: tuple[str, ...] = ()
    ops: tuple[str, ...] = ()
    phase: str | None = None
    has_state: bool = False
    has_update: bool = False
    roles: dict[str, str] = field(default_factory=dict)
    dims: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reuse: tuple[str, ...] = ()

    def tensors(self) -> tuple[str, ...]:
        items: list[str] = []
        seen: set[str] = set()
        for bucket in (self.inputs, self.outputs, self.temps, self.edge_in, self.edge_out):
            for tensor in bucket:
                if tensor not in seen:
                    seen.add(tensor)
                    items.append(tensor)
        return tuple(items)

    def is_cross(self, tensor: str) -> bool:
        return tensor in self.edge_in or tensor in self.edge_out

    def role_of(self, tensor: str) -> str:
        return self.roles.get(tensor, "UNK")

    def dims_of(self, tensor: str) -> tuple[str, ...]:
        return self.dims.get(tensor, ())


# -----------------------------
# public API
# -----------------------------

def build_ctx(group: FusionGroup, workload: Any | None = None, gene: FusionGene | None = None) -> GroupCtx:
    edge_in, edge_out = _find_edges(group.group_id, gene)
    roles = _build_roles(group, workload)
    dims = _build_dims(group, workload)
    phase = _read_phase(group, workload)
    reuse = _guess_reuse(group, workload)
    has_state = any(_is_state_tensor(name) for name in (*edge_in, *edge_out, *group.inputs, *group.outputs))
    has_update = any(_is_update_tensor(name) for name in (*group.outputs, *edge_out))
    return GroupCtx(
        id=group.group_id,
        inputs=tuple(group.inputs),
        outputs=tuple(group.outputs),
        temps=tuple(group.temps),
        edge_in=edge_in,
        edge_out=edge_out,
        ops=tuple(group.ops),
        phase=phase,
        has_state=has_state,
        has_update=has_update,
        roles=roles,
        dims=dims,
        reuse=reuse,
    )


def enum_group_dp(ctx: GroupCtx, top: int = 8) -> tuple[GroupDP, ...]:
    """Enumerate placement candidates from structure, not templates.

    Search dimensions are intentionally small:
    - KV split or merge
    - mid tensor keep or pass
    - state update isolated or folded
    - same-level slot order normalization
    """

    parts = _enum_part(ctx)
    out: list[GroupDP] = []
    seen: set[tuple[tuple[str, str, int, tuple[str, ...]], ...]] = set()

    for part in parts:
        nodes = _make_nodes(ctx, part)
        if not nodes:
            continue
        nodes = _enum_slot(nodes)
        dp = GroupDP(group=ctx.id, nodes=nodes)
        if not _legal_dp(dp, ctx):
            continue
        key = _dp_key(dp)
        if key in seen:
            continue
        seen.add(key)
        out.append(dp)

    out.sort(key=lambda dp: (_score_dp(dp, ctx), len(dp.nodes), dp.group))
    return tuple(out[: max(1, top)])


def build_plan(
    groups: Sequence[FusionGroup],
    workload: Any | None = None,
    gene: FusionGene | None = None,
    plan_id: str = "placement",
    top: int = 1,
) -> PlacementPlan:
    out: list[GroupDP] = []
    for group in groups:
        ctx = build_ctx(group, workload=workload, gene=gene)
        cands = enum_group_dp(ctx, top=max(1, top))
        if not cands:
            continue
        out.append(cands[0])
    return PlacementPlan(id=plan_id, groups=tuple(out))


def lower_plan(plan: PlacementPlan, tree: Any) -> Any:
    """Attach placement metadata to an existing tree.

    Tree mutation stays outside placement. This function only exposes placement
    results for later builder work.
    """

    setattr(tree, "placement_plan", plan)
    return tree


def explain(dp: GroupDP) -> dict[str, list[dict[str, object]]]:
    return {
        dp.group: [
            {
                "id": node.id,
                "level": node.level,
                "kind": node.kind,
                "slot": node.slot,
                "axis": list(node.axis),
                "tensors": list(node.tensors),
            }
            for node in dp.nodes
        ]
    }


# -----------------------------
# structure enum
# -----------------------------

def _enum_part(ctx: GroupCtx) -> tuple[tuple[dict[str, object], ...], ...]:
    inputs = tuple(sorted(ctx.inputs))
    outputs = tuple(sorted(ctx.outputs))
    mids = tuple(sorted(ctx.temps))
    state_in = tuple(sorted(t for t in ctx.tensors() if _node_kind(ctx, t) == "state" and t not in ctx.outputs))
    state_out = tuple(sorted(t for t in outputs if _node_kind(ctx, t) == "state"))
    acc = tuple(sorted(t for t in outputs if _node_kind(ctx, t) == "acc"))
    main_out = tuple(sorted(t for t in outputs if t not in state_out and t not in acc))

    hold_inputs = tuple(t for t in inputs if t not in state_in)
    kv = tuple(t for t in hold_inputs if ctx.role_of(t) in {"K", "V"})
    rest_in = tuple(t for t in hold_inputs if t not in kv)

    kv_modes: list[tuple[tuple[str, ...], ...]] = [()]
    if kv:
        kv_modes = [(kv,), *((t,) for t in kv)] if len(kv) <= 2 else [(kv,)]

    mid_modes: list[tuple[tuple[str, ...], ...]] = [()]
    if mids:
        mid_modes = [(), (mids,)]
        if len(mids) > 1:
            mid_modes.append(tuple((t,) for t in mids))

    state_out_modes: list[tuple[tuple[str, ...], ...]] = [()]
    if state_out:
        state_out_modes = [(state_out,)]
        if len(state_out) > 1:
            state_out_modes.append(tuple((t,) for t in state_out))

    out: list[tuple[dict[str, object], ...]] = []
    for kv_part, mid_part, state_part in product(kv_modes, mid_modes, state_out_modes):
        spec: list[dict[str, object]] = []

        for group in kv_part:
            spec.append({"kind": "hold", "level": "sram", "tensors": tuple(group)})
        if rest_in:
            spec.append({"kind": "hold", "level": "sram", "tensors": rest_in})
        if state_in:
            spec.append({"kind": "state", "level": "sram", "tensors": state_in})
        for group in mid_part:
            spec.append({"kind": "flow", "level": "sram", "tensors": tuple(group)})
        if acc:
            spec.append({"kind": "acc", "level": "pe", "tensors": acc})
        if main_out:
            spec.append({"kind": "hold", "level": "sram", "tensors": main_out})
        for group in state_part:
            spec.append({"kind": "state", "level": "sram", "tensors": tuple(group)})

        if spec:
            out.append(tuple(spec))

    if not out:
        return ((),)
    return tuple(dict.fromkeys(out))


def _make_nodes(ctx: GroupCtx, part: Sequence[dict[str, object]]) -> tuple[StoreNode, ...]:
    out: list[StoreNode] = []
    for idx, item in enumerate(part):
        tensors = tuple(item["tensors"])
        if not tensors:
            continue
        kind = str(item["kind"])
        level = str(item["level"])
        axis = _pick_axis(ctx, tensors, kind)
        out.append(
            StoreNode(
                id=f"{ctx.id}:{level}:{kind}:{idx}",
                group=ctx.id,
                level=level,
                tensors=tensors,
                slot=idx,
                kind=kind,
                axis=axis,
            )
        )
    return tuple(out)


def _enum_slot(nodes: Sequence[StoreNode]) -> tuple[StoreNode, ...]:
    by_level: dict[str, list[StoreNode]] = {}
    for node in nodes:
        by_level.setdefault(node.level, []).append(node)

    out: list[StoreNode] = []
    for level in sorted(by_level):
        ordered = sorted(by_level[level], key=_slot_key)
        for slot, node in enumerate(ordered):
            out.append(
                StoreNode(
                    id=node.id,
                    group=node.group,
                    level=node.level,
                    tensors=node.tensors,
                    slot=slot,
                    kind=node.kind,
                    axis=node.axis,
                )
            )
    return tuple(sorted(out, key=lambda n: (n.level, n.slot, n.kind, n.id)))


def _slot_key(node: StoreNode) -> tuple[int, int, str]:
    kind_rank = {"state": 0, "hold": 1, "flow": 2, "acc": 3}
    cross_rank = 0 if any(_is_state_tensor(t) for t in node.tensors) else 1
    return (kind_rank.get(node.kind, 9), cross_rank, node.id)


# -----------------------------
# legality and score
# -----------------------------

def _legal_dp(dp: GroupDP, ctx: GroupCtx) -> bool:
    seen: dict[str, int] = {}
    acc_seen = False
    state_seen = False

    for node in dp.nodes:
        if node.kind == "acc":
            if node.level != "pe":
                return False
            acc_seen = True
        if node.kind == "state":
            state_seen = True
        for tensor in node.tensors:
            seen[tensor] = seen.get(tensor, 0) + 1

    if ctx.outputs:
        out_hit = set()
        for node in dp.nodes:
            for tensor in node.tensors:
                if tensor in ctx.outputs:
                    out_hit.add(tensor)
        if set(ctx.outputs) - out_hit:
            return False

    if ctx.has_state and not state_seen:
        return False
    if any(_node_kind(ctx, t) == "acc" for t in ctx.outputs) and not acc_seen:
        return False

    for tensor in ctx.inputs:
        if seen.get(tensor, 0) > 1 and _node_kind(ctx, tensor) != "state":
            return False
    for tensor in ctx.temps:
        if seen.get(tensor, 0) > 1:
            return False

    return True


def _score_dp(dp: GroupDP, ctx: GroupCtx) -> tuple[int, int, int, int]:
    state_cnt = sum(1 for node in dp.nodes if node.kind == "state")
    flow_cnt = sum(1 for node in dp.nodes if node.kind == "flow")
    hold_cnt = sum(1 for node in dp.nodes if node.kind == "hold")
    cross_front = 0
    for node in dp.nodes:
        if node.level == "sram" and node.slot == 0 and any(ctx.is_cross(t) for t in node.tensors):
            cross_front = 1
            break
    return (-cross_front, state_cnt, flow_cnt, hold_cnt)


def _dp_key(dp: GroupDP) -> tuple[tuple[str, str, int, tuple[str, ...]], ...]:
    return tuple((node.level, node.kind, node.slot, node.tensors) for node in dp.nodes)


# -----------------------------
# internal helpers
# -----------------------------

def _find_edges(group_id: str, gene: FusionGene | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if gene is None:
        return (), ()
    edge_in: list[str] = []
    edge_out: list[str] = []
    for src, dst, tensor in gene.group_edges_tensors:
        if dst == group_id and tensor not in edge_in:
            edge_in.append(tensor)
        if src == group_id and tensor not in edge_out:
            edge_out.append(tensor)
    return tuple(edge_in), tuple(edge_out)


def _build_roles(group: FusionGroup, workload: Any | None) -> dict[str, str]:
    roles: dict[str, str] = {}
    if workload is not None and hasattr(workload, "get_tensor"):
        for tensor in (*group.inputs, *group.outputs, *group.temps):
            try:
                spec = workload.get_tensor(tensor)
                role = getattr(spec, "role", None)
                if role is not None:
                    roles[tensor] = str(role).split(".")[-1]
            except Exception:
                pass
    for tensor in (*group.inputs, *group.outputs, *group.temps):
        roles.setdefault(tensor, _guess_role(tensor))
    return roles


def _build_dims(group: FusionGroup, workload: Any | None) -> dict[str, tuple[str, ...]]:
    dims: dict[str, tuple[str, ...]] = {}
    if workload is not None and hasattr(workload, "get_op"):
        for op_id in group.ops:
            try:
                op = workload.get_op(op_id)
            except Exception:
                continue
            iter_dims = tuple(getattr(op, "iter_dims", ()) or ())
            tensor_dims = dict(getattr(op, "tensor_dims", {}) or {})
            for tensor, tensor_axes in tensor_dims.items():
                if tensor in (*group.inputs, *group.outputs, *group.temps) and tensor_axes:
                    dims[tensor] = tuple(tensor_axes)
                elif tensor in (*group.inputs, *group.outputs, *group.temps) and iter_dims:
                    dims.setdefault(tensor, iter_dims)
    return dims


def _read_phase(group: FusionGroup, workload: Any | None) -> str | None:
    phase = getattr(group, "phase", None)
    if phase:
        return str(phase)
    spec = getattr(workload, "spec", None)
    if spec is not None and hasattr(spec, "mode"):
        return str(spec.mode)
    return None


def _guess_reuse(group: FusionGroup, workload: Any | None) -> tuple[str, ...]:
    axes: list[str] = []
    if workload is not None and hasattr(workload, "get_op"):
        for op_id in group.ops:
            try:
                op = workload.get_op(op_id)
            except Exception:
                continue
            for axis in getattr(op, "iter_dims", ()) or ():
                if axis not in axes:
                    axes.append(axis)
            for axis in getattr(op, "reduce_dims", ()) or ():
                if axis not in axes:
                    axes.append(axis)
    if not axes:
        return ("q", "k", "dh")
    return tuple(axes)


def _node_kind(ctx: GroupCtx, tensor: str) -> str:
    role = ctx.role_of(tensor)
    if role in {"STATE", "CACHE"} or _is_state_tensor(tensor):
        return "state"
    if role in {"SCORES", "PROBS", "STATS"}:
        return "flow"
    if role in {"O", "ACC"} and tensor in ctx.outputs:
        return "acc"
    return "hold"


def _pick_axis(ctx: GroupCtx, tensors: Sequence[str], kind: str) -> tuple[str, ...]:
    axes: list[str] = []
    for tensor in tensors:
        for axis in ctx.dims_of(tensor):
            if axis in ctx.reuse and axis not in axes:
                axes.append(axis)
    if axes:
        return tuple(axes[:2])
    if kind == "flow" and len(ctx.reuse) >= 2:
        return tuple(ctx.reuse[:2])
    if ctx.reuse:
        return (ctx.reuse[0],)
    return ()


def _guess_role(tensor: str) -> str:
    name = tensor.upper()
    if "CACHE" in name or "STATE" in name:
        return "STATE"
    if "SCORE" in name:
        return "SCORES"
    if "PROB" in name:
        return "PROBS"
    if "STAT" in name:
        return "STATS"
    if "PARTIAL" in name:
        return "ACC"
    if name.startswith("Q") or "_Q" in name:
        return "Q"
    if name.startswith("K") or "_K" in name:
        return "K"
    if name.startswith("V") or "_V" in name:
        return "V"
    if name.startswith("O") or "_O" in name:
        return "O"
    return "UNK"


def _is_state_tensor(name: str) -> bool:
    upper = name.upper()
    return "CACHE" in upper or "STATE" in upper


def _is_update_tensor(name: str) -> bool:
    upper = name.upper()
    return "APPEND" in upper or "CACHE_OUT" in upper or "STATE_OUT" in upper
