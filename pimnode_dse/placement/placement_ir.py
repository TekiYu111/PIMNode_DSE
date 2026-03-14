from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

# -----------------------------------------------------------------------------
# Core aliases / defaults
# -----------------------------------------------------------------------------
MemoryLevel = str
TensorName = str
TensorRole = str

DEFAULT_MEMORY_LEVELS: Sequence[MemoryLevel] = ("DRAM", "SRAM", "PE")


# -----------------------------------------------------------------------------
# Instantiated placement objects (consume by builder / attach to TileNode)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ResidentSet:
    """Actual tensors resident in a concrete memory level for one scope."""

    mem: MemoryLevel
    tensors: Set[TensorName] = field(default_factory=set)


@dataclass(frozen=True)
class BoundaryAction:
    """Concrete boundary actions for a memory level at scope entry / exit."""

    mem: MemoryLevel
    prefetch: Set[TensorName] = field(default_factory=set)
    writeback: Set[TensorName] = field(default_factory=set)
    evict: Set[TensorName] = field(default_factory=set)


@dataclass
class PlacementScope:
    """Concrete, instantiated placement for a scope / fusion group.

    This is the object the MappingBuilder should consume. It uses *real tensor
    names* after template instantiation.
    """

    scope_name: str
    resident_sets: List[ResidentSet] = field(default_factory=list)
    boundary_actions: List[BoundaryAction] = field(default_factory=list)
    phase: Optional[str] = None
    special_role: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def get_resident_tensors(self, mem: MemoryLevel) -> Set[TensorName]:
        for rs in self.resident_sets:
            if rs.mem == mem:
                return set(rs.tensors)
        return set()

    def get_boundary_action(self, mem: MemoryLevel) -> BoundaryAction:
        for ba in self.boundary_actions:
            if ba.mem == mem:
                return BoundaryAction(
                    mem=ba.mem,
                    prefetch=set(ba.prefetch),
                    writeback=set(ba.writeback),
                    evict=set(ba.evict),
                )
        return empty_boundary_action(mem)

    def memories(self) -> List[MemoryLevel]:
        mems = {rs.mem for rs in self.resident_sets}
        mems.update(ba.mem for ba in self.boundary_actions)
        return sorted(mems)


@dataclass
class PlacementPlan:
    """Concrete placement plan used by builder / tree materialization."""

    scopes: Dict[str, PlacementScope] = field(default_factory=dict)

    def get_scope(self, scope_name: str) -> PlacementScope:
        return self.scopes[scope_name]

    def get_resident_tensors(self, scope_name: str, mem: MemoryLevel) -> Set[TensorName]:
        return self.get_scope(scope_name).get_resident_tensors(mem)

    def get_boundary_action(self, scope_name: str, mem: MemoryLevel) -> BoundaryAction:
        return self.get_scope(scope_name).get_boundary_action(mem)

    def add_scope(self, scope: PlacementScope) -> None:
        self.scopes[scope.scope_name] = scope


# -----------------------------------------------------------------------------
# Template-level placement objects (role-based; instantiate before builder)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RoleResidentSet:
    """Template resident set keyed by logical tensor roles, not real names."""

    mem: MemoryLevel
    tensor_roles: Set[TensorRole] = field(default_factory=set)


@dataclass(frozen=True)
class RoleBoundaryAction:
    """Template boundary action keyed by logical tensor roles."""

    mem: MemoryLevel
    prefetch_roles: Set[TensorRole] = field(default_factory=set)
    writeback_roles: Set[TensorRole] = field(default_factory=set)
    evict_roles: Set[TensorRole] = field(default_factory=set)


@dataclass
class PlacementTemplateScope:
    """Role-based template before binding to a specific fusion group."""

    scope_name: str
    resident_sets: List[RoleResidentSet] = field(default_factory=list)
    boundary_actions: List[RoleBoundaryAction] = field(default_factory=list)
    supported_phases: Set[str] = field(default_factory=set)
    supported_roles: Set[str] = field(default_factory=set)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PlacementTemplatePlan:
    """Template library keyed by template/scope name."""

    scopes: Dict[str, PlacementTemplateScope] = field(default_factory=dict)

    def get_scope(self, scope_name: str) -> PlacementTemplateScope:
        return self.scopes[scope_name]

    def add_scope(self, scope: PlacementTemplateScope) -> None:
        self.scopes[scope.scope_name] = scope


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def empty_boundary_action(mem: MemoryLevel) -> BoundaryAction:
    return BoundaryAction(mem=mem, prefetch=set(), writeback=set(), evict=set())


def normalize_scope(
    scope: PlacementScope,
    memory_levels: Sequence[MemoryLevel] = DEFAULT_MEMORY_LEVELS,
    tensor_universe: Optional[Set[TensorName]] = None,
    *,
    memories: Optional[Sequence[MemoryLevel]] = None,
) -> PlacementScope:
    """Return a normalized concrete scope.

    Guarantees:
    - every requested memory level has one ResidentSet and one BoundaryAction
    - duplicate memory entries are merged
    - optional tensor_universe filtering keeps only tensors visible in this group
    """
    if memories is not None:
        memory_levels = memories

    resident_by_mem: Dict[MemoryLevel, Set[TensorName]] = {mem: set() for mem in memory_levels}
    action_by_mem: Dict[MemoryLevel, BoundaryAction] = {
        mem: empty_boundary_action(mem) for mem in memory_levels
    }

    for rs in scope.resident_sets:
        resident_by_mem.setdefault(rs.mem, set()).update(rs.tensors)

    for ba in scope.boundary_actions:
        existing = action_by_mem.setdefault(ba.mem, empty_boundary_action(ba.mem))
        action_by_mem[ba.mem] = BoundaryAction(
            mem=ba.mem,
            prefetch=set(existing.prefetch) | set(ba.prefetch),
            writeback=set(existing.writeback) | set(ba.writeback),
            evict=set(existing.evict) | set(ba.evict),
        )

    if tensor_universe is not None:
        for mem in list(resident_by_mem.keys()):
            resident_by_mem[mem] &= tensor_universe
        for mem, ba in list(action_by_mem.items()):
            action_by_mem[mem] = BoundaryAction(
                mem=mem,
                prefetch=set(ba.prefetch) & tensor_universe,
                writeback=set(ba.writeback) & tensor_universe,
                evict=set(ba.evict) & tensor_universe,
            )

    return PlacementScope(
        scope_name=scope.scope_name,
        resident_sets=[ResidentSet(mem=mem, tensors=tensors) for mem, tensors in resident_by_mem.items()],
        boundary_actions=[action_by_mem[mem] for mem in resident_by_mem.keys()],
        phase=scope.phase,
        special_role=scope.special_role,
        metadata=dict(scope.metadata),
    )


def instantiate_template_scope(
    template_scope: PlacementTemplateScope,
    role_bindings: Mapping[TensorRole, Iterable[TensorName]],
    *,
    scope_name: Optional[str] = None,
    phase: Optional[str] = None,
    special_role: Optional[str] = None,
    memory_levels: Sequence[MemoryLevel] = DEFAULT_MEMORY_LEVELS,
    tensor_universe: Optional[Set[TensorName]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> PlacementScope:
    """Instantiate a role-based placement template into a concrete scope.

    Parameters
    ----------
    template_scope:
        Role-based template.
    role_bindings:
        Mapping from logical role (e.g. "Q", "K", "STATS") to actual tensor
        names visible in the current fusion group.
    scope_name:
        Concrete scope name. Defaults to template_scope.scope_name.
    """

    def resolve_roles(role_set: Iterable[TensorRole]) -> Set[TensorName]:
        out: Set[TensorName] = set()
        for role in role_set:
            out.update(role_bindings.get(role, set()))
        return out

    instantiated = PlacementScope(
        scope_name=scope_name or template_scope.scope_name,
        resident_sets=[
            ResidentSet(mem=rs.mem, tensors=resolve_roles(rs.tensor_roles))
            for rs in template_scope.resident_sets
        ],
        boundary_actions=[
            BoundaryAction(
                mem=ba.mem,
                prefetch=resolve_roles(ba.prefetch_roles),
                writeback=resolve_roles(ba.writeback_roles),
                evict=resolve_roles(ba.evict_roles),
            )
            for ba in template_scope.boundary_actions
        ],
        phase=phase,
        special_role=special_role,
        metadata={**template_scope.metadata, **dict(metadata or {})},
    )
    return normalize_scope(
        instantiated,
        memory_levels=memory_levels,
        tensor_universe=tensor_universe,
    )


def role_bindings_from_mapping(mapping: Mapping[str, Iterable[str]]) -> Dict[str, Set[str]]:
    """Small utility to canonicalize role bindings into Set[str] values."""
    return {key: set(values) for key, values in mapping.items()}
