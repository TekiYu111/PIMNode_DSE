from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple


Level = str
Kind = str
Axis = Tuple[str, ...]


@dataclass(frozen=True)
class StoreNode:
    """Placement-side storage node inside one fused group.

    Fields stay intentionally small:
    - level: target memory level
    - tensors: tensor subset covered by this node
    - slot: local order among nodes at the same level
    - kind: hold / flow / state / acc
    - axis: main reuse axis set
    """

    id: str
    group: str
    level: Level
    tensors: tuple[str, ...]
    slot: int
    kind: Kind
    axis: Axis = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("StoreNode.id must not be empty")
        if not self.group:
            raise ValueError("StoreNode.group must not be empty")
        if not self.level:
            raise ValueError("StoreNode.level must not be empty")
        if self.slot < 0:
            raise ValueError("StoreNode.slot must be >= 0")
        if self.kind not in {"hold", "flow", "state", "acc"}:
            raise ValueError(f"Unsupported StoreNode.kind: {self.kind!r}")
        uniq = tuple(sorted(dict.fromkeys(self.tensors)))
        if not uniq:
            raise ValueError("StoreNode.tensors must not be empty")
        object.__setattr__(self, "tensors", uniq)
        object.__setattr__(self, "axis", tuple(dict.fromkeys(self.axis)))

    def covers(self, tensor: str) -> bool:
        return tensor in self.tensors


@dataclass(frozen=True)
class GroupDP:
    """Placement result for one fused group."""

    group: str
    nodes: tuple[StoreNode, ...]

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("GroupDP.group must not be empty")
        if not self.nodes:
            raise ValueError("GroupDP.nodes must not be empty")
        seen: set[str] = set()
        clean: list[StoreNode] = []
        for node in self.nodes:
            if node.group != self.group:
                raise ValueError(
                    f"StoreNode {node.id!r} belongs to group {node.group!r}, expected {self.group!r}"
                )
            if node.id in seen:
                raise ValueError(f"Duplicate StoreNode id in group {self.group!r}: {node.id!r}")
            seen.add(node.id)
            clean.append(node)
        clean.sort(key=lambda n: (n.level, n.slot, n.kind, n.id))
        object.__setattr__(self, "nodes", tuple(clean))

    def levels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(node.level for node in self.nodes))

    def tensors(self) -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for node in self.nodes:
            for tensor in node.tensors:
                if tensor not in seen:
                    seen.add(tensor)
                    out.append(tensor)
        return tuple(out)

    def nodes_at(self, level: str) -> tuple[StoreNode, ...]:
        return tuple(node for node in self.nodes if node.level == level)


@dataclass(frozen=True)
class PlacementPlan:
    """Placement result across all fused groups."""

    id: str
    groups: tuple[GroupDP, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("PlacementPlan.id must not be empty")
        seen: set[str] = set()
        ordered: list[GroupDP] = []
        for group in self.groups:
            if group.group in seen:
                raise ValueError(f"Duplicate GroupDP for group {group.group!r}")
            seen.add(group.group)
            ordered.append(group)
        ordered.sort(key=lambda g: g.group)
        object.__setattr__(self, "groups", tuple(ordered))

    def group_map(self) -> dict[str, GroupDP]:
        return {group.group: group for group in self.groups}

    def get(self, group_id: str) -> GroupDP:
        for group in self.groups:
            if group.group == group_id:
                return group
        raise KeyError(group_id)


def new_group_dp(group: str, nodes: Iterable[StoreNode]) -> GroupDP:
    return GroupDP(group=group, nodes=tuple(nodes))


def new_plan(plan_id: str, groups: Iterable[GroupDP]) -> PlacementPlan:
    return PlacementPlan(id=plan_id, groups=tuple(groups))
