# placement/rules.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from pimnode_dse.placement.placement_ir import (
    DEFAULT_MEMORY_LEVELS,
    BoundaryAction,
    PlacementScope,
    ResidentSet,
    empty_boundary_action,
    normalize_scope,
)

# -----------------------------------------------------------------------------
# Canonical logical roles
# -----------------------------------------------------------------------------
ROLE_Q = "Q"
ROLE_K = "K"
ROLE_V = "V"
ROLE_SCORES = "SCORES"
ROLE_STATS = "STATS"
ROLE_PROBS = "PROBS"
ROLE_PARTIAL_O = "PARTIAL_O"
ROLE_O = "O"
ROLE_KV_CACHE = "KV_CACHE"

READ_ONLY_ROLES: Set[str] = {
    ROLE_Q,
    ROLE_K,
    ROLE_V,
    ROLE_KV_CACHE,
    ROLE_SCORES,
    ROLE_PROBS,
}

WRITABLE_ROLES: Set[str] = {
    ROLE_STATS,
    ROLE_PARTIAL_O,
    ROLE_O,
}

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"


# -----------------------------------------------------------------------------
# Action node (lightweight, builder-consumable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionNode:
    tensor: str
    mem: str                     # 'PE', 'SRAM', 'DRAM'
    action: str                  # 'LOAD', 'PREFETCH', 'WRITEBACK', 'EVICT'
    scope: str
    phase: str = ""
    role: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Metadata / role helpers
# -----------------------------------------------------------------------------
def canonicalize_phase(phase: Optional[str]) -> str:
    if not phase:
        return ""
    p = phase.strip().lower()
    if p == "prefill":
        return PHASE_PREFILL
    if p == "decode":
        return PHASE_DECODE
    return phase


def _safe_get_tensor_meta(
    tensor: str,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Mapping[str, object]:
    if not tensor_meta:
        return {}
    return tensor_meta.get(tensor, {})


def get_tensor_role(
    tensor: str,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Optional[str]:
    """
    Resolve the logical role of a concrete tensor name.

    Priority:
      1) explicit tensor_to_role mapping
      2) tensor_meta[tensor]['role']
      3) tensor_meta[tensor]['special_role']
      4) lightweight name fallback
    """
    if tensor_to_role and tensor in tensor_to_role:
        return tensor_to_role[tensor]

    meta = _safe_get_tensor_meta(tensor, tensor_meta)
    if "role" in meta and meta["role"]:
        return str(meta["role"])
    if "special_role" in meta and meta["special_role"]:
        return str(meta["special_role"])

    # Very light fallback only for debugging / partial migration.
    upper_name = tensor.upper()
    if upper_name in {
        ROLE_Q,
        ROLE_K,
        ROLE_V,
        ROLE_SCORES,
        ROLE_STATS,
        ROLE_PROBS,
        ROLE_PARTIAL_O,
        ROLE_O,
        ROLE_KV_CACHE,
    }:
        return upper_name

    # Common substring fallback during transition.
    if "KV_CACHE" in upper_name:
        return ROLE_KV_CACHE
    if "PARTIAL" in upper_name and "O" in upper_name:
        return ROLE_PARTIAL_O
    if "STAT" in upper_name:
        return ROLE_STATS
    if upper_name.startswith("Q"):
        return ROLE_Q
    if upper_name.startswith("K"):
        return ROLE_K
    if upper_name.startswith("V"):
        return ROLE_V
    if "SCORE" in upper_name:
        return ROLE_SCORES
    if "PROB" in upper_name:
        return ROLE_PROBS
    if upper_name == "O" or upper_name.startswith("O_") or upper_name.endswith("_O"):
        return ROLE_O

    return None


def is_read_only(
    tensor: str,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> bool:
    role = get_tensor_role(tensor, tensor_to_role=tensor_to_role, tensor_meta=tensor_meta)
    return role in READ_ONLY_ROLES


def is_writable(
    tensor: str,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> bool:
    role = get_tensor_role(tensor, tensor_to_role=tensor_to_role, tensor_meta=tensor_meta)
    return role in WRITABLE_ROLES


def is_kv_cache(
    tensor: str,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> bool:
    role = get_tensor_role(tensor, tensor_to_role=tensor_to_role, tensor_meta=tensor_meta)
    return role == ROLE_KV_CACHE


def should_load_resident(
    tensor: str,
    mem: str,
    phase: Optional[str],
    *,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> bool:
    """
    Decide whether a resident tensor should materialize as an explicit LOAD action.

    Current minimal policy:
      - SRAM / PE resident tensors get LOAD actions
      - DRAM residents do not
      - final O resident in DRAM is backing-store only, not a LOAD
    """
    if mem not in {"SRAM", "PE"}:
        return False

    role = get_tensor_role(tensor, tensor_to_role=tensor_to_role, tensor_meta=tensor_meta)
    if role == ROLE_O and mem == "PE":
        return True
    return True


# -----------------------------------------------------------------------------
# Hard rules
# -----------------------------------------------------------------------------
def _sanitize_boundary_action(
    ba: BoundaryAction,
    phase: str,
    *,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> BoundaryAction:
    """
    Apply local hard rules on one boundary action and return a sanitized copy.
    """
    phase = canonicalize_phase(phase)

    prefetch = set(ba.prefetch)
    writeback = set(ba.writeback)
    evict = set(ba.evict)

    # Rule 1: RO tensors are never writeback targets.
    writeback = {
        t for t in writeback
        if not is_read_only(t, tensor_to_role=tensor_to_role, tensor_meta=tensor_meta)
    }

    # Rule 2: phase-aware writeback / retention policy.
    if phase == PHASE_DECODE:
        # decode tends to preserve KV cache in-place; do not write back KV cache.
        writeback = {
            t for t in writeback
            if not is_kv_cache(t, tensor_to_role=tensor_to_role, tensor_meta=tensor_meta)
        }
        # Also avoid evicting KV cache from fast memory in decode scopes.
        evict = {
            t for t in evict
            if not is_kv_cache(t, tensor_to_role=tensor_to_role, tensor_meta=tensor_meta)
        }

    elif phase == PHASE_PREFILL:
        # prefill allows rolling outputs / partial results to be written back.
        # No extra action needed beyond removing RO tensors.
        pass

    # Rule 3: do not both prefetch and evict the same tensor at the same boundary.
    evict -= prefetch

    # Rule 4: avoid writeback + evict duplication for writable state.
    # If a tensor is written back at this boundary, eviction is redundant in this IR.
    evict -= writeback

    return BoundaryAction(
        mem=ba.mem,
        prefetch=prefetch,
        writeback=writeback,
        evict=evict,
    )


def apply_hard_rules(
    scope: PlacementScope,
    phase: Optional[str],
    *,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
    memories: Sequence[str] = DEFAULT_MEMORY_LEVELS,
) -> PlacementScope:
    """
    Return a new PlacementScope with hard placement constraints applied.

    This function is intentionally pure: it does not mutate the input scope.
    """
    phase = canonicalize_phase(phase or scope.phase)

    normalized = normalize_scope(scope, memory_levels=memories)

    new_boundary_actions: List[BoundaryAction] = []
    for mem in memories:
        ba = normalized.get_boundary_action(mem)
        new_boundary_actions.append(
            _sanitize_boundary_action(
                ba,
                phase=phase,
                tensor_to_role=tensor_to_role,
                tensor_meta=tensor_meta,
            )
        )

    out = PlacementScope(
        scope_name=normalized.scope_name,
        resident_sets=[
            ResidentSet(mem=rs.mem, tensors=set(rs.tensors))
            for rs in normalized.resident_sets
        ],
        boundary_actions=new_boundary_actions,
        phase=phase or normalized.phase,
        special_role=normalized.special_role,
        metadata=dict(normalized.metadata),
    )
    return normalize_scope(out, memory_levels=memories)



# -----------------------------------------------------------------------------
# Derivation rules -> explicit actions
# -----------------------------------------------------------------------------
def _append_actions(
    actions: List[ActionNode],
    tensors: Iterable[str],
    *,
    mem: str,
    action: str,
    scope: PlacementScope,
    phase: str,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> None:
    for tensor in sorted(set(tensors)):
        role = get_tensor_role(tensor, tensor_to_role=tensor_to_role, tensor_meta=tensor_meta)
        metadata = dict(extra_metadata or {})
        if role is not None:
            metadata["role"] = role
        actions.append(
            ActionNode(
                tensor=tensor,
                mem=mem,
                action=action,
                scope=scope.scope_name,
                phase=phase,
                role=role,
                metadata=metadata,
            )
        )


def derive_actions(
    scope: PlacementScope,
    phase: Optional[str],
    *,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
    memories: Sequence[str] = DEFAULT_MEMORY_LEVELS,
    include_loads: bool = True,
) -> List[ActionNode]:
    """
    Convert a concrete PlacementScope into explicit ActionNode objects.

    Recommended flow:
        scope2 = apply_hard_rules(scope, phase, tensor_to_role=..., tensor_meta=...)
        actions = derive_actions(scope2, phase, tensor_to_role=..., tensor_meta=...)

    This function is phase-aware and role-aware, but it consumes only concrete
    tensor names, keeping builder integration simple.
    """
    phase = canonicalize_phase(phase or scope.phase)
    normalized = normalize_scope(scope, memories=memories)

    actions: List[ActionNode] = []

    # Boundary actions first.
    for mem in memories:
        ba = normalized.get_boundary_action(mem)

        _append_actions(
            actions,
            ba.prefetch,
            mem=mem,
            action="PREFETCH",
            scope=normalized,
            phase=phase,
            tensor_to_role=tensor_to_role,
            tensor_meta=tensor_meta,
            extra_metadata={"boundary": "entry"},
        )
        _append_actions(
            actions,
            ba.writeback,
            mem=mem,
            action="WRITEBACK",
            scope=normalized,
            phase=phase,
            tensor_to_role=tensor_to_role,
            tensor_meta=tensor_meta,
            extra_metadata={"boundary": "exit"},
        )
        _append_actions(
            actions,
            ba.evict,
            mem=mem,
            action="EVICT",
            scope=normalized,
            phase=phase,
            tensor_to_role=tensor_to_role,
            tensor_meta=tensor_meta,
            extra_metadata={"boundary": "exit"},
        )

    # Resident loads after boundary actions.
    if include_loads:
        for rs in normalized.resident_sets:
            for tensor in sorted(rs.tensors):
                if should_load_resident(
                    tensor,
                    rs.mem,
                    phase,
                    tensor_to_role=tensor_to_role,
                    tensor_meta=tensor_meta,
                ):
                    role = get_tensor_role(
                        tensor,
                        tensor_to_role=tensor_to_role,
                        tensor_meta=tensor_meta,
                    )
                    actions.append(
                        ActionNode(
                            tensor=tensor,
                            mem=rs.mem,
                            action="LOAD",
                            scope=normalized.scope_name,
                            phase=phase,
                            role=role,
                            metadata={"resident": True},
                        )
                    )

    return dedupe_actions(actions)


# -----------------------------------------------------------------------------
# Convenience pipeline
# -----------------------------------------------------------------------------
def apply_rules_and_derive_actions(
    scope: PlacementScope,
    phase: Optional[str],
    *,
    tensor_to_role: Optional[Mapping[str, str]] = None,
    tensor_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
    memories: Sequence[str] = DEFAULT_MEMORY_LEVELS,
    include_loads: bool = True,
) -> Tuple[PlacementScope, List[ActionNode]]:
    """
    Convenience helper for builder / tests.
    """
    scoped = apply_hard_rules(
        scope,
        phase,
        tensor_to_role=tensor_to_role,
        tensor_meta=tensor_meta,
        memories=memories,
    )
    actions = derive_actions(
        scoped,
        phase,
        tensor_to_role=tensor_to_role,
        tensor_meta=tensor_meta,
        memories=memories,
        include_loads=include_loads,
    )
    return scoped, actions


# -----------------------------------------------------------------------------
# Dedupe / formatting helpers
# -----------------------------------------------------------------------------
def dedupe_actions(actions: Sequence[ActionNode]) -> List[ActionNode]:
    seen: Set[Tuple[str, str, str, str, str]] = set()
    out: List[ActionNode] = []
    for a in actions:
        key = (a.tensor, a.mem, a.action, a.scope, a.phase)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def group_actions_by_mem(actions: Sequence[ActionNode]) -> Dict[str, List[ActionNode]]:
    grouped: Dict[str, List[ActionNode]] = {}
    for action in actions:
        grouped.setdefault(action.mem, []).append(action)
    return grouped


__all__ = [
    "ROLE_Q",
    "ROLE_K",
    "ROLE_V",
    "ROLE_SCORES",
    "ROLE_STATS",
    "ROLE_PROBS",
    "ROLE_PARTIAL_O",
    "ROLE_O",
    "ROLE_KV_CACHE",
    "PHASE_PREFILL",
    "PHASE_DECODE",
    "ActionNode",
    "canonicalize_phase",
    "get_tensor_role",
    "is_read_only",
    "is_writable",
    "is_kv_cache",
    "apply_hard_rules",
    "derive_actions",
    "apply_rules_and_derive_actions",
    "dedupe_actions",
    "group_actions_by_mem",
]
