from __future__ import annotations

from dataclasses import dataclass, field

from pimnode_dse.mapping.placement.node import GroupDP, StoreNode
from pimnode_dse.mapping.placement.placement import GroupCtx


LEVELS = ("dram", "sram", "pe")
KINDS = ("hold", "state", "acc", "flow")


@dataclass(frozen=True)
class RuleErr:
    kind: str
    msg: str
    node: str | None = None


@dataclass(frozen=True)
class RuleOut:
    ok: bool
    errs: tuple[RuleErr, ...] = field(default_factory=tuple)

    def text(self) -> tuple[str, ...]:
        out: list[str] = []
        for err in self.errs:
            if err.node:
                out.append(f"[{err.kind}] {err.node}: {err.msg}")
            else:
                out.append(f"[{err.kind}] {err.msg}")
        return tuple(out)


# -----------------------------
# public
# -----------------------------

def node_key(node: StoreNode) -> tuple[object, ...]:
    return (node.group, node.level, node.slot, node.kind, node.axis, node.tensors)



def dp_key(dp: GroupDP) -> tuple[tuple[object, ...], ...]:
    return tuple(node_key(node) for node in dp.nodes)



def check_node(node: StoreNode, ctx: GroupCtx) -> tuple[RuleErr, ...]:
    errs: list[RuleErr] = []
    group_tensors = set(ctx.tensors())
    reuse = set(ctx.reuse)

    if node.group != ctx.id:
        errs.append(RuleErr("group", f"node group {node.group!r} != ctx {ctx.id!r}", node.id))

    if node.level not in LEVELS:
        errs.append(RuleErr("level", f"bad level {node.level!r}", node.id))

    if node.kind not in KINDS:
        errs.append(RuleErr("kind", f"bad kind {node.kind!r}", node.id))

    if not node.tensors:
        errs.append(RuleErr("tensor", "node must cover at least one tensor", node.id))

    for tensor in node.tensors:
        if tensor not in group_tensors:
            errs.append(RuleErr("tensor", f"tensor {tensor!r} not in group", node.id))
        if tensor in ctx.temps and node.level == "dram":
            errs.append(RuleErr("level", f"temp tensor {tensor!r} cannot live at dram", node.id))

    bad_axis = [axis for axis in node.axis if axis not in reuse]
    if bad_axis:
        errs.append(RuleErr("axis", f"axis {tuple(bad_axis)!r} not in reuse set {tuple(ctx.reuse)!r}", node.id))

    if node.kind == "acc":
        if node.level != "pe":
            errs.append(RuleErr("acc", "acc node must be placed at pe", node.id))
        if not any(tensor in ctx.outputs for tensor in node.tensors):
            errs.append(RuleErr("acc", "acc node must cover at least one output tensor", node.id))

    if node.kind == "state":
        if not ctx.has_state:
            errs.append(RuleErr("state", "state node in non-state group", node.id))
        if not any(_is_state_like(ctx, tensor) for tensor in node.tensors):
            errs.append(RuleErr("state", "state node must cover state-like tensor", node.id))

    if node.kind == "flow":
        if node.level == "dram":
            errs.append(RuleErr("flow", "flow node cannot be placed at dram", node.id))
        if not any(tensor in ctx.temps for tensor in node.tensors):
            errs.append(RuleErr("flow", "flow node must cover at least one temp tensor", node.id))

    return tuple(errs)



def check_group(dp: GroupDP, ctx: GroupCtx) -> tuple[RuleErr, ...]:
    errs: list[RuleErr] = []
    group_tensors = set(ctx.tensors())
    out_tensors = set(ctx.outputs)
    state_tensors = {tensor for tensor in group_tensors if _is_state_like(ctx, tensor)}

    if dp.group != ctx.id:
        errs.append(RuleErr("group", f"dp group {dp.group!r} != ctx {ctx.id!r}"))

    if not dp.nodes:
        errs.append(RuleErr("group", "group dp must contain at least one node"))
        return tuple(errs)

    seen_id: set[str] = set()
    slot_used: set[tuple[str, int]] = set()
    cover: dict[str, list[StoreNode]] = {}

    for node in dp.nodes:
        if node.id in seen_id:
            errs.append(RuleErr("node", f"duplicate node id {node.id!r}", node.id))
        seen_id.add(node.id)

        slot_key = (node.level, node.slot)
        if slot_key in slot_used:
            errs.append(RuleErr("slot", f"duplicate slot {slot_key!r}", node.id))
        slot_used.add(slot_key)

        for tensor in node.tensors:
            cover.setdefault(tensor, []).append(node)

    stray = sorted(set(cover) - group_tensors)
    if stray:
        errs.append(RuleErr("tensor", f"dp covers tensors outside group: {stray!r}"))

    for tensor in group_tensors:
        nodes = cover.get(tensor, [])
        if not nodes:
            continue
        if len(nodes) == 1:
            continue

        kinds = {node.kind for node in nodes}
        levels = {node.level for node in nodes}

        if len(levels) == 1:
            errs.append(
                RuleErr(
                    "cover",
                    f"tensor {tensor!r} has multiple nodes at the same level: {[node.id for node in nodes]!r}",
                )
            )
            continue

        if tensor in ctx.temps:
            errs.append(
                RuleErr(
                    "cover",
                    f"temp tensor {tensor!r} must not be kept by multiple levels: {[node.id for node in nodes]!r}",
                )
            )
            continue

        if kinds == {"state", "hold"} and _is_state_like(ctx, tensor):
            errs.append(
                RuleErr(
                    "cover",
                    f"state-like tensor {tensor!r} must use a single kind, got {sorted(kinds)!r}",
                )
            )
            continue

        if len(kinds) > 1:
            errs.append(
                RuleErr(
                    "cover",
                    f"tensor {tensor!r} has mixed node kinds {sorted(kinds)!r}",
                )
            )

    level_slots: dict[str, list[int]] = {}
    for node in dp.nodes:
        level_slots.setdefault(node.level, []).append(node.slot)
    for level, slots in level_slots.items():
        have = sorted(slots)
        want = list(range(len(have)))
        if have != want:
            errs.append(RuleErr("slot", f"level {level!r} slots must be compact {want!r}, got {have!r}"))

    miss_out = sorted(tensor for tensor in out_tensors if tensor not in cover)
    if miss_out:
        errs.append(RuleErr("out", f"output tensors not covered: {miss_out!r}"))

    if ctx.has_state:
        miss_state = sorted(tensor for tensor in state_tensors if tensor not in cover)
        if miss_state:
            errs.append(RuleErr("state", f"state-like tensors not covered: {miss_state!r}"))
        if not any(node.kind == "state" for node in dp.nodes):
            errs.append(RuleErr("state", "state group must keep at least one state node"))

    if any(tensor in out_tensors for tensor in ctx.tensors()):
        acc_need = any(_needs_acc(ctx, tensor) for tensor in out_tensors)
        if acc_need and not any(node.kind == "acc" for node in dp.nodes):
            errs.append(RuleErr("acc", "group with acc-like outputs must keep at least one acc node"))

    return tuple(errs)



def legal_dp(dp: GroupDP, ctx: GroupCtx) -> bool:
    return not explain_dp(dp, ctx).errs



def explain_dp(dp: GroupDP, ctx: GroupCtx) -> RuleOut:
    errs: list[RuleErr] = []
    for node in dp.nodes:
        errs.extend(check_node(node, ctx))
    errs.extend(check_group(dp, ctx))
    return RuleOut(ok=not errs, errs=tuple(errs))


# -----------------------------
# helpers
# -----------------------------

def _is_state_like(ctx: GroupCtx, tensor: str) -> bool:
    role = ctx.role_of(tensor)
    if role in {"STATE", "CACHE"}:
        return True
    upper = tensor.upper()
    return "CACHE" in upper or "STATE" in upper or tensor in ctx.edge_in or tensor in ctx.edge_out



def _needs_acc(ctx: GroupCtx, tensor: str) -> bool:
    role = ctx.role_of(tensor)
    if role in {"ACC", "O"}:
        return True
    upper = tensor.upper()
    return "PARTIAL" in upper or upper.startswith("O")
