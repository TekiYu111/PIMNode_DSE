# placement/templates.py

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Set

from pimnode_dse.placement.placement_ir import (
    DEFAULT_MEMORY_LEVELS,
    PlacementTemplatePlan,
    PlacementTemplateScope,
    RoleBoundaryAction,
    RoleResidentSet,
)

# -----------------------------------------------------------------------------
# Canonical logical tensor roles used by placement templates
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

ALL_KNOWN_ROLES: Set[str] = {
    ROLE_Q,
    ROLE_K,
    ROLE_V,
    ROLE_SCORES,
    ROLE_STATS,
    ROLE_PROBS,
    ROLE_PARTIAL_O,
    ROLE_O,
    ROLE_KV_CACHE,
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _rr(mem: str, roles: Iterable[str]) -> RoleResidentSet:
    return RoleResidentSet(mem=mem, tensor_roles=set(roles))


def _ba(
    mem: str,
    *,
    prefetch: Optional[Iterable[str]] = None,
    writeback: Optional[Iterable[str]] = None,
    evict: Optional[Iterable[str]] = None,
) -> RoleBoundaryAction:
    return RoleBoundaryAction(
        mem=mem,
        prefetch_roles=set(prefetch or ()),
        writeback_roles=set(writeback or ()),
        evict_roles=set(evict or ()),
    )

def _template(
    scope_name: str,
    resident_sets: Sequence[RoleResidentSet],
    boundary_actions: Sequence[RoleBoundaryAction],
    *,
    description: Optional[str] = None,
    supported_phases: Optional[Iterable[str]] = None,
    supported_special_roles: Optional[Iterable[str]] = None,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> PlacementTemplateScope:
    metadata: Dict[str, object] = {
        "template_name": scope_name,
        "description": description or "",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return PlacementTemplateScope(
        scope_name=scope_name,
        resident_sets=list(resident_sets),
        boundary_actions=list(boundary_actions),
        supported_phases=set(supported_phases or ()),
        supported_roles=set(supported_special_roles or ()),
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# MaxReuse
# -----------------------------------------------------------------------------
def max_reuse_template(scope_name: str = "MaxReuse") -> PlacementTemplateScope:
    """
    Keep as many hot tensors resident as possible.

    Intended use:
      - smaller tiles
      - decode-like settings
      - high reuse scenarios where SRAM can hold most working-set tensors

    Interpretation:
      - SRAM tries to hold Q/K/V plus lightweight running state (stats / partial-O)
      - PE keeps only the current micro-tile operands / accumulators
      - DRAM mainly acts as backing store with no extra boundary churn by default
    """
    resident_sets = [
        _rr("SRAM", {ROLE_Q, ROLE_K, ROLE_V, ROLE_STATS, ROLE_PARTIAL_O}),
        _rr("PE", {ROLE_PARTIAL_O}),
        _rr("DRAM", {ROLE_O, ROLE_KV_CACHE}),
    ]
    boundary_actions = [
        _ba("SRAM"),
        _ba("PE"),
        _ba("DRAM"),
    ]
    return _template(
        scope_name,
        resident_sets,
        boundary_actions,
        description="Keep most hot tensors in SRAM; minimal boundary traffic.",
        supported_phases={"decode", "prefill"},
        supported_special_roles={ROLE_KV_CACHE, ROLE_PARTIAL_O, ROLE_STATS},
        extra_metadata={
            "family": "max_reuse",
            "design_intent": "all-stream / high local reuse",
        },
    )


# -----------------------------------------------------------------------------
# Balanced
# -----------------------------------------------------------------------------
def balanced_template(scope_name: str = "Balanced") -> PlacementTemplateScope:
    """
    Keep producer-side / frequently reused tensors resident, stream the rest.

    Intended use:
      - general-purpose prefill
      - SRAM-constrained cases
      - keep-Q/K/stats style mappings

    Interpretation:
      - SRAM keeps Q/K and rolling state (stats / partial-O)
      - V / probs / final output are more likely to cross boundaries
      - boundary actions are deliberately mild and can be refined later per phase
    """
    resident_sets = [
        _rr("SRAM", {ROLE_Q, ROLE_K, ROLE_STATS, ROLE_PARTIAL_O}),
        _rr("PE", {ROLE_PARTIAL_O}),
        _rr("DRAM", {ROLE_V, ROLE_PROBS, ROLE_O, ROLE_KV_CACHE}),
    ]
    boundary_actions = [
        _ba(
            "SRAM",
            prefetch={ROLE_V, ROLE_PROBS},
            writeback={ROLE_O},
            evict={ROLE_SCORES},
        ),
        _ba("PE"),
        _ba("DRAM"),
    ]
    return _template(
        scope_name,
        resident_sets,
        boundary_actions,
        description="Keep producer-side tensors resident; stream larger consumers.",
        supported_phases={"prefill", "decode"},
        supported_special_roles={ROLE_STATS, ROLE_PARTIAL_O},
        extra_metadata={
            "family": "balanced",
            "design_intent": "keep-Q/K/stats, stream V/output as needed",
        },
    )


# -----------------------------------------------------------------------------
# MinReuse
# -----------------------------------------------------------------------------
def min_reuse_template(scope_name: str = "MinReuse") -> PlacementTemplateScope:
    """
    Stream most tensors through SRAM, keep only the absolute minimum resident.

    Intended use:
      - large tiles
      - tight SRAM budget
      - bandwidth-heavy but capacity-safe fallback mappings

    Interpretation:
      - SRAM only keeps Q and tiny rolling state if needed
      - K/V/probs/output mostly move across boundaries
      - this template is intentionally aggressive on boundary actions
    """
    resident_sets = [
        _rr("SRAM", {ROLE_Q, ROLE_STATS}),
        _rr("PE", {ROLE_PARTIAL_O}),
        _rr("DRAM", {ROLE_K, ROLE_V, ROLE_PROBS, ROLE_O, ROLE_KV_CACHE}),
    ]
    boundary_actions = [
        _ba(
            "SRAM",
            prefetch={ROLE_K, ROLE_V, ROLE_PROBS},
            writeback={ROLE_O, ROLE_PARTIAL_O},
            evict={ROLE_K, ROLE_V, ROLE_SCORES, ROLE_PROBS},
        ),
        _ba("PE"),
        _ba("DRAM"),
    ]
    return _template(
        scope_name,
        resident_sets,
        boundary_actions,
        description="Stream-heavy fallback template for tight SRAM budgets.",
        supported_phases={"prefill"},
        supported_special_roles={ROLE_PARTIAL_O, ROLE_STATS},
        extra_metadata={
            "family": "min_reuse",
            "design_intent": "capacity-first / stream-most-tensors",
        },
    )


# -----------------------------------------------------------------------------
# K-D Fusion
# -----------------------------------------------------------------------------
def kd_fusion_template(scope_name: str = "K-D Fusion") -> PlacementTemplateScope:
    """
    Favor K/V or KV-cache reuse across fused decode-like or partial-O-heavy scopes.

    Intended use:
      - decode
      - KV-cache aware mappings
      - keep-partial-O / keep-KV style fused scopes

    Interpretation:
      - SRAM prioritizes K/V and running accumulators / stats
      - Q is allowed to stream in
      - boundary actions are tuned toward KV-aware fused dataflow
    """
    resident_sets = [
        _rr("SRAM", {ROLE_K, ROLE_V, ROLE_KV_CACHE, ROLE_PARTIAL_O, ROLE_STATS}),
        _rr("PE", {ROLE_PARTIAL_O}),
        _rr("DRAM", {ROLE_Q, ROLE_O, ROLE_PROBS}),
    ]
    boundary_actions = [
        _ba(
            "SRAM",
            prefetch={ROLE_Q},
            writeback={ROLE_O, ROLE_PARTIAL_O},
            evict={ROLE_SCORES, ROLE_PROBS},
        ),
        _ba("PE"),
        _ba("DRAM"),
    ]
    return _template(
        scope_name,
        resident_sets,
        boundary_actions,
        description="KV-aware fused template; ideal for decode and partial-O-heavy flows.",
        supported_phases={"decode", "prefill"},
        supported_special_roles={ROLE_KV_CACHE, ROLE_PARTIAL_O, ROLE_STATS},
        extra_metadata={
            "family": "kd_fusion",
            "design_intent": "keep-KV / keep-partial-O across fused scopes",
        },
    )


# -----------------------------------------------------------------------------
# Factory / registry
# -----------------------------------------------------------------------------
def build_core_templates() -> PlacementTemplatePlan:
    """
    Build the canonical four core placement templates.

    Returns:
        PlacementTemplatePlan containing:
          - MaxReuse
          - Balanced
          - MinReuse
          - K-D Fusion
    """
    plan = PlacementTemplatePlan(scopes={})
    for scope in (
        max_reuse_template("MaxReuse"),
        balanced_template("Balanced"),
        min_reuse_template("MinReuse"),
        kd_fusion_template("K-D Fusion"),
    ):
        plan.add_scope(scope)
    return plan


def build_templates_by_phase(phase: str) -> PlacementTemplatePlan:
    """
    Convenience filter for phase-aware template enumeration.

    This does not instantiate templates. It only filters template-level scopes.
    """
    base = build_core_templates()
    if not phase:
        return base

    filtered = PlacementTemplatePlan(scopes={})
    for scope_name, scope in base.scopes.items():
        supported = set(scope.supported_phases)
        if not supported or phase in supported:
            filtered.add_scope(scope)
    return filtered


def get_template_names() -> Sequence[str]:
    return ("MaxReuse", "Balanced", "MinReuse", "K-D Fusion")


def get_known_roles() -> Set[str]:
    return set(ALL_KNOWN_ROLES)


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
    "ALL_KNOWN_ROLES",
    "max_reuse_template",
    "balanced_template",
    "min_reuse_template",
    "kd_fusion_template",
    "build_core_templates",
    "build_templates_by_phase",
    "get_template_names",
    "get_known_roles",
]
