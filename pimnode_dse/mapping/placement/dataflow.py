from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pimnode_dse.mapping.placement.node import GroupDP, StoreNode


@dataclass(frozen=True)
class FlowStep:
    """One loop block in executable order."""

    loops: tuple[str, ...]
    action: str

    def __post_init__(self) -> None:
        clean = tuple(dict.fromkeys(self.loops))
        object.__setattr__(self, "loops", clean)
        if not self.action:
            raise ValueError("FlowStep.action must not be empty")


@dataclass(frozen=True)
class FlowDP:
    """Loop blocks for one fused group."""

    group: str
    steps: tuple[FlowStep, ...]

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("FlowDP.group must not be empty")
        if not self.steps:
            raise ValueError("FlowDP.steps must not be empty")
        clean = tuple(step for step in self.steps if step.loops)
        if not clean:
            raise ValueError("FlowDP.steps must contain at least one non-empty step")
        object.__setattr__(self, "steps", clean)

    def render(self, indent: str = "    ") -> str:
        lines: list[str] = []
        pad = ""
        for step in self.steps:
            loops = ", ".join(step.loops)
            lines.append(f"{pad}for {loops}:")
            lines.append(f"{pad}{indent}{step.action}")
            pad += indent
        return "\n".join(lines)


# -----------------------------
# public API
# -----------------------------


def enum_flow_dp(dp: GroupDP, ctx: Any, arch: Any) -> tuple[FlowDP, ...]:
    """Build canonical loop blocks for one placement result.

    Input
    - dp: fixed placement
    - ctx: group-side loop facts
    - arch: hardware levels

    Output
    - tuple[FlowDP, ...]

    Current policy returns one canonical result. It removes redundant order by
    sorting loops with the same structural effect under the same step.
    """

    levels = _levels(dp, arch)
    loops = _loops(ctx)
    if len(levels) < 2 or not loops:
        return ()

    slots = levels[1:]
    by_step: dict[str, list[str]] = {level: [] for level in slots}

    for loop in loops:
        dst = _loop_dst(loop, dp, ctx, levels)
        if dst not in by_step:
            dst = slots[-1]
        by_step[dst].append(loop)

    steps: list[FlowStep] = []
    for dst in slots:
        group = by_step[dst]
        if not group:
            continue
        order = _canon_loops(group, dst, dp, ctx)
        steps.append(FlowStep(loops=order, action=_action(dst, dp)))

    if not steps:
        return ()
    return (FlowDP(group=dp.group, steps=tuple(steps)),)


# -----------------------------
# internal helpers
# -----------------------------


def _levels(dp: GroupDP, arch: Any) -> tuple[str, ...]:
    raw = None
    if hasattr(arch, "levels"):
        raw = arch.levels() if callable(arch.levels) else arch.levels
    elif hasattr(arch, "mem_levels"):
        raw = arch.mem_levels() if callable(arch.mem_levels) else arch.mem_levels
    if raw:
        levels = tuple(str(x) for x in raw)
    else:
        levels = _default_level_order(dp)
    if len(levels) < 2:
        raise ValueError("dataflow needs at least two hardware levels")
    return levels


def _default_level_order(dp: GroupDP) -> tuple[str, ...]:
    rank = {
        "dram": 0,
        "hbm": 0,
        "global": 1,
        "sram": 2,
        "local": 3,
        "rf": 4,
        "reg": 4,
        "pe": 5,
    }
    levels = tuple(dict.fromkeys(node.level for node in dp.nodes))
    return tuple(sorted(levels, key=lambda x: (rank.get(x, 99), x)))


def _loops(ctx: Any) -> tuple[str, ...]:
    if hasattr(ctx, "loops") and callable(ctx.loops):
        return tuple(dict.fromkeys(str(x) for x in ctx.loops()))
    if hasattr(ctx, "iter_loops"):
        raw = ctx.iter_loops() if callable(ctx.iter_loops) else ctx.iter_loops
        return tuple(dict.fromkeys(str(x) for x in raw))
    if hasattr(ctx, "reuse"):
        raw = ctx.reuse() if callable(ctx.reuse) else ctx.reuse
        return tuple(dict.fromkeys(str(x) for x in raw))
    raise ValueError("ctx must provide loops() or reuse")


def _reduce_loops(ctx: Any) -> tuple[str, ...]:
    if hasattr(ctx, "reduce_loops") and callable(ctx.reduce_loops):
        return tuple(dict.fromkeys(str(x) for x in ctx.reduce_loops()))
    if hasattr(ctx, "reduce"):
        raw = ctx.reduce() if callable(ctx.reduce) else ctx.reduce
        return tuple(dict.fromkeys(str(x) for x in raw))
    return ()


def _tensor_loops(ctx: Any, tensor: str) -> tuple[str, ...]:
    if hasattr(ctx, "tensor_loops") and callable(ctx.tensor_loops):
        return tuple(dict.fromkeys(str(x) for x in ctx.tensor_loops(tensor)))
    if hasattr(ctx, "dims"):
        dims = ctx.dims(tensor) if callable(ctx.dims) else getattr(ctx, "dims", {})
        if isinstance(dims, dict):
            return tuple(dict.fromkeys(str(x) for x in dims.get(tensor, ())))
    return ()


def _loop_dst(loop: str, dp: GroupDP, ctx: Any, levels: Sequence[str]) -> str:
    level_idx = {level: idx for idx, level in enumerate(levels)}
    touched = _loop_tensors(loop, dp, ctx)
    if not touched:
        return levels[-1]

    max_idx = 1
    for node in dp.nodes:
        if any(t in touched for t in node.tensors):
            idx = level_idx.get(node.level)
            if idx is None:
                continue
            if idx > max_idx:
                max_idx = idx
    return levels[max_idx]


def _loop_tensors(loop: str, dp: GroupDP, ctx: Any) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for tensor in dp.tensors():
        if loop in _tensor_loops(ctx, tensor) and tensor not in seen:
            seen.add(tensor)
            out.append(tensor)
    return tuple(out)


def _canon_loops(loops: Iterable[str], dst: str, dp: GroupDP, ctx: Any) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(loops), key=lambda x: _loop_key(x, dst, dp, ctx)))


def _loop_key(loop: str, dst: str, dp: GroupDP, ctx: Any) -> tuple[object, ...]:
    reduce_set = set(_reduce_loops(ctx))
    touched = _loop_tensors(loop, dp, ctx)
    levels = tuple(sorted(dict.fromkeys(node.level for node in dp.nodes if any(t in touched for t in node.tensors))))
    kinds = tuple(sorted(dict.fromkeys(node.kind for node in dp.nodes if any(t in touched for t in node.tensors))))
    return (
        dst,
        1 if loop in reduce_set else 0,
        levels,
        kinds,
        touched,
        loop,
    )


def _action(dst: str, dp: GroupDP) -> str:
    if dst == "pe" or any(node.level == dst and node.kind == "acc" for node in dp.nodes):
        return "run pe / acc nodes"
    return f"load {dst} nodes"


def explain(flow: FlowDP) -> list[dict[str, object]]:
    return [{"loops": list(step.loops), "action": step.action} for step in flow.steps]


def render(flow: FlowDP, indent: str = "    ") -> str:
    return flow.render(indent=indent)
