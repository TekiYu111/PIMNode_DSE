from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple


Level = str
TensorSet = Tuple[str, ...]

# Residence mode for a StoreNode:
#   single  – one copy, default behaviour (current semantics)
#   double  – double-buffer: two physical slots alternate; SRAM cost ×2,
#             but compute and data-fetch can overlap
#   pinned  – permanently resident in this level; no fetch/evict cost across
#             iterations (state tensors that never leave SRAM)
#   evict   – written back immediately after last use; occupies SRAM only
#             during the computation that produced it (output tensors)
ResidenceMode = str
VALID_RESIDENCE: Tuple[ResidenceMode, ...] = ("single", "double", "pinned", "evict")


@dataclass(frozen=True)
class StoreNode:
    """Stable storage node inside one fused group.

    This type only describes placement structure:
    - id: short stable name
    - level: normalized hardware level
    - pos: order on the storage chain
    - tens: covered tensors
    - residence: data-residency policy for the node
    """

    id: str
    level: Level
    pos: int
    tens: TensorSet
    residence: ResidenceMode = "single"

    def __post_init__(self) -> None:
        node_id = str(self.id).strip()
        level = str(self.level).strip().lower()

        if not node_id:
            raise ValueError("StoreNode.id must not be empty")
        if not level:
            raise ValueError("StoreNode.level must not be empty")
        if self.pos < 0:
            raise ValueError("StoreNode.pos must be >= 0")

        residence = str(self.residence).strip().lower()
        if residence not in VALID_RESIDENCE:
            raise ValueError(
                f"StoreNode.residence must be one of {VALID_RESIDENCE}, got {residence!r}"
            )

        clean: list[str] = []
        seen: set[str] = set()
        for tensor in self.tens:
            name = str(tensor).strip()
            if not name:
                raise ValueError("StoreNode.tens must not contain empty tensor")
            if name not in seen:
                seen.add(name)
                clean.append(name)

        if not clean:
            raise ValueError("StoreNode.tens must not be empty")

        object.__setattr__(self, "id", node_id)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "residence", residence)
        object.__setattr__(self, "tens", tuple(sorted(clean)))

    def has(self, tensor: str) -> bool:
        return tensor in self.tens

    def overlap(self, other: "StoreNode") -> tuple[str, ...]:
        other_set = set(other.tens)
        return tuple(t for t in self.tens if t in other_set)

    def covers(self, tensors: Iterable[str]) -> bool:
        own = set(self.tens)
        return all(str(tensor).strip() in own for tensor in tensors)

    def sig(self) -> tuple[str, str, int, TensorSet, ResidenceMode]:
        return (self.id, self.level, self.pos, self.tens, self.residence)

    def eq_sig(self) -> tuple[str, int, TensorSet, ResidenceMode]:
        return (self.level, self.pos, self.tens, self.residence)


@dataclass(frozen=True)
class GroupDP:
    """Placement result for one fused group."""

    group: str
    nodes: tuple[StoreNode, ...]

    def __post_init__(self) -> None:
        group = str(self.group).strip()
        if not group:
            raise ValueError("GroupDP.group must not be empty")
        if not self.nodes:
            raise ValueError("GroupDP.nodes must not be empty")

        clean = tuple(sorted(self.nodes, key=lambda node: node.pos))

        ids = [node.id for node in clean]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate StoreNode id in group {group!r}")

        pos = [node.pos for node in clean]
        if len(pos) != len(set(pos)):
            raise ValueError(f"Duplicate StoreNode pos in group {group!r}")

        want = list(range(len(clean)))
        if pos != want:
            raise ValueError(
                f"StoreNode.pos must be continuous in group {group!r}: got {pos}, want {want}"
            )

        object.__setattr__(self, "group", group)
        object.__setattr__(self, "nodes", clean)

    def levels(self) -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for node in self.nodes:
            if node.level not in seen:
                seen.add(node.level)
                out.append(node.level)
        return tuple(out)

    def tens(self) -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for node in self.nodes:
            for tensor in node.tens:
                if tensor not in seen:
                    seen.add(tensor)
                    out.append(tensor)
        return tuple(out)

    def nodes_at(self, level: str) -> tuple[StoreNode, ...]:
        want = str(level).strip().lower()
        return tuple(node for node in self.nodes if node.level == want)

    def node_of(self, tensor: str) -> StoreNode:
        name = str(tensor).strip()
        for node in self.nodes:
            if node.has(name):
                return node
        raise KeyError(name)

    def sig(self) -> tuple[tuple[str, str, int, TensorSet, ResidenceMode], ...]:
        return tuple(node.sig() for node in self.nodes)

    def eq_sig(self) -> tuple[tuple[str, int, TensorSet, ResidenceMode], ...]:
        return tuple(node.eq_sig() for node in self.nodes)


@dataclass(frozen=True)
class PlacementPlan:
    """Placement result across all fused groups."""

    id: str
    groups: tuple[GroupDP, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        plan_id = str(self.id).strip()
        if not plan_id:
            raise ValueError("PlacementPlan.id must not be empty")

        seen: set[str] = set()
        ordered: list[GroupDP] = []
        for group in self.groups:
            if group.group in seen:
                raise ValueError(f"Duplicate GroupDP for group {group.group!r}")
            seen.add(group.group)
            ordered.append(group)

        ordered.sort(key=lambda group: group.group)
        object.__setattr__(self, "id", plan_id)
        object.__setattr__(self, "groups", tuple(ordered))

    def group_map(self) -> dict[str, GroupDP]:
        return {group.group: group for group in self.groups}

    def get(self, group_id: str) -> GroupDP:
        key = str(group_id).strip()
        for group in self.groups:
            if group.group == key:
                return group
        raise KeyError(key)


def new_group_dp(group: str, nodes: Iterable[StoreNode]) -> GroupDP:
    return GroupDP(group=group, nodes=tuple(nodes))


def new_plan(plan_id: str, groups: Iterable[GroupDP]) -> PlacementPlan:
    return PlacementPlan(id=plan_id, groups=tuple(groups))
