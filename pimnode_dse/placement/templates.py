from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Set

from pimnode_dse.placement.placement_ir import (
    DEFAULT_SCOPE,
    LifetimeScope,
    PlacementBoundary,
    RolePlacementTemplate,
    RolePlacementTemplateLibrary,
    RoleResidencySpec,
    RoleTransferSpec,
    Site,
    TransferAction,
    default_site_map,
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

def _rs(site: Site, roles: Iterable[str]) -> RoleResidencySpec:
    return RoleResidencySpec(tensor_roles=set(roles), site=site)


def _ts(
    site: Site,
    boundary: PlacementBoundary,
    action: TransferAction,
    roles: Iterable[str],
) -> RoleTransferSpec:
    return RoleTransferSpec(
        tensor_roles=set(roles),
        site=site,
        boundary=boundary,
        action=action,
    )


def _template(
    template_name: str,
    residency: Sequence[RoleResidencySpec],
    transfers: Sequence[RoleTransferSpec],
    *,
    description: Optional[str] = None,
    supported_phases: Optional[Iterable[str]] = None,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> RolePlacementTemplate:
    metadata: Dict[str, object] = {
        "template_name": template_name,
        "description": description or "",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return RolePlacementTemplate(
        template_name=template_name,
        residency=list(residency),
        transfers=list(transfers),
        supported_phases=set(supported_phases or ()),
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Canonical attention sites
# -----------------------------------------------------------------------------

def canonical_attention_sites(scope: LifetimeScope = DEFAULT_SCOPE) -> Dict[str, Site]:
    """Return the minimal canonical site set used by attention templates.

    This is a curated subset for coarse-grained attention placement templates,
    not an exhaustive enumeration of all representable sites in the IR.
    """

    return default_site_map(scope)


# -----------------------------------------------------------------------------
# Template families
# -----------------------------------------------------------------------------

def operand_resident_template(
    template_name: str = "OperandResident",
    *,
    scope: LifetimeScope = DEFAULT_SCOPE,
) -> RolePlacementTemplate:
    """Keep Q/K/V close to compute while accumulating outputs locally.

    Intended use:
    - prefill or decode when operand reuse is strong
    - smaller SRAM-feasible tiles
    - mappings that want operand-downstream reuse without full KV-state residency
    """

    s = canonical_attention_sites(scope)
    residency = [
        _rs(s["DRAM_OPERAND"], {ROLE_Q, ROLE_K, ROLE_V}),
        _rs(s["SRAM_OPERAND"], {ROLE_Q, ROLE_K, ROLE_V}),
        _rs(s["PE_OPERAND"], {ROLE_Q, ROLE_K, ROLE_V}),
        _rs(s["PE_ACCUM"], {ROLE_PARTIAL_O}),
        _rs(s["SRAM_ACCUM"], {ROLE_STATS, ROLE_PARTIAL_O}),
        _rs(s["DRAM_FORWARD"], {ROLE_O}),
        _rs(s["DRAM_STATE"], {ROLE_KV_CACHE}),
    ]
    transfers = [
        _ts(s["SRAM_OPERAND"], PlacementBoundary.ENTER, TransferAction.LOAD, {ROLE_Q, ROLE_K, ROLE_V}),
        _ts(s["PE_OPERAND"], PlacementBoundary.ENTER, TransferAction.LOAD, {ROLE_Q, ROLE_K, ROLE_V}),
        _ts(s["SRAM_FORWARD"], PlacementBoundary.EXIT, TransferAction.EVICT, {ROLE_SCORES, ROLE_PROBS}),
        _ts(s["SRAM_ACCUM"], PlacementBoundary.EXIT, TransferAction.WRITEBACK, {ROLE_PARTIAL_O, ROLE_O}),
    ]
    return _template(
        template_name,
        residency,
        transfers,
        description="Keep operand tiles near compute; accumulate outputs locally.",
        supported_phases={"prefill", "decode"},
        extra_metadata={
            "family": "operand_resident",
            "design_intent": "downstream operand reuse with local accumulation",
        },
    )



def producer_consumer_keep_template(
    template_name: str = "ProducerConsumerKeep",
    *,
    scope: LifetimeScope = DEFAULT_SCOPE,
) -> RolePlacementTemplate:
    """Keep forward intermediates in SRAM across producer-consumer boundaries."""

    s = canonical_attention_sites(scope)
    residency = [
        _rs(s["DRAM_OPERAND"], {ROLE_Q, ROLE_K, ROLE_V}),
        _rs(s["SRAM_OPERAND"], {ROLE_Q, ROLE_K, ROLE_V}),
        _rs(s["PE_OPERAND"], {ROLE_Q, ROLE_K, ROLE_V}),
        _rs(s["SRAM_FORWARD"], {ROLE_SCORES, ROLE_PROBS}),
        _rs(s["SRAM_ACCUM"], {ROLE_STATS, ROLE_PARTIAL_O}),
        _rs(s["PE_ACCUM"], {ROLE_PARTIAL_O}),
        _rs(s["DRAM_FORWARD"], {ROLE_O}),
        _rs(s["DRAM_STATE"], {ROLE_KV_CACHE}),
    ]
    transfers = [
        _ts(s["SRAM_OPERAND"], PlacementBoundary.ENTER, TransferAction.LOAD, {ROLE_Q, ROLE_K, ROLE_V}),
        _ts(s["PE_OPERAND"], PlacementBoundary.ENTER, TransferAction.LOAD, {ROLE_Q, ROLE_K, ROLE_V}),
        _ts(s["SRAM_FORWARD"], PlacementBoundary.ENTER, TransferAction.PREFETCH, {ROLE_SCORES, ROLE_PROBS}),
        _ts(s["SRAM_ACCUM"], PlacementBoundary.EXIT, TransferAction.WRITEBACK, {ROLE_PARTIAL_O, ROLE_O}),
    ]
    return _template(
        template_name,
        residency,
        transfers,
        description="Keep producer-consumer intermediates in SRAM across local boundaries.",
        supported_phases={"prefill", "decode"},
        extra_metadata={
            "family": "producer_consumer_keep",
            "design_intent": "retain forward intermediates to reduce churn",
        },
    )



def local_accumulate_template(
    template_name: str = "LocalAccumulate",
    *,
    scope: LifetimeScope = DEFAULT_SCOPE,
) -> RolePlacementTemplate:
    """Prioritize local accumulation; stream larger operands and forwards more aggressively."""

    s = canonical_attention_sites(scope)
    residency = [
        _rs(s["DRAM_OPERAND"], {ROLE_Q, ROLE_K, ROLE_V}),
        _rs(s["SRAM_OPERAND"], {ROLE_Q}),
        _rs(s["PE_OPERAND"], {ROLE_Q}),
        _rs(s["SRAM_ACCUM"], {ROLE_STATS, ROLE_PARTIAL_O, ROLE_O}),
        _rs(s["PE_ACCUM"], {ROLE_PARTIAL_O}),
        _rs(s["DRAM_FORWARD"], {ROLE_SCORES, ROLE_PROBS}),
        _rs(s["DRAM_STATE"], {ROLE_KV_CACHE}),
    ]
    transfers = [
        _ts(s["SRAM_OPERAND"], PlacementBoundary.ENTER, TransferAction.PREFETCH, {ROLE_K, ROLE_V}),
        _ts(s["PE_OPERAND"], PlacementBoundary.ENTER, TransferAction.LOAD, {ROLE_Q}),
        _ts(s["SRAM_FORWARD"], PlacementBoundary.EXIT, TransferAction.EVICT, {ROLE_SCORES, ROLE_PROBS}),
        _ts(s["SRAM_ACCUM"], PlacementBoundary.EXIT, TransferAction.WRITEBACK, {ROLE_PARTIAL_O, ROLE_O}),
    ]
    return _template(
        template_name,
        residency,
        transfers,
        description="Favor local accumulation; stream bulky operands and forwards as needed.",
        supported_phases={"prefill", "decode"},
        extra_metadata={
            "family": "local_accumulate",
            "design_intent": "capacity-conscious accumulation-centric placement",
        },
    )



def state_persistent_template(
    template_name: str = "StatePersistent",
    *,
    scope: LifetimeScope = DEFAULT_SCOPE,
) -> RolePlacementTemplate:
    """Keep KV-related state resident across decode-like scopes."""

    s = canonical_attention_sites(scope)
    residency = [
        _rs(s["DRAM_OPERAND"], {ROLE_Q, ROLE_K, ROLE_V}),
        _rs(s["SRAM_OPERAND"], {ROLE_Q}),
        _rs(s["PE_OPERAND"], {ROLE_Q}),
        _rs(s["SRAM_STATE"], {ROLE_KV_CACHE}),
        _rs(s["DRAM_STATE"], {ROLE_KV_CACHE}),
        _rs(s["SRAM_ACCUM"], {ROLE_STATS, ROLE_PARTIAL_O}),
        _rs(s["PE_ACCUM"], {ROLE_PARTIAL_O}),
        _rs(s["DRAM_FORWARD"], {ROLE_O, ROLE_PROBS}),
    ]
    transfers = [
        _ts(s["SRAM_STATE"], PlacementBoundary.ENTER, TransferAction.PREFETCH, {ROLE_KV_CACHE}),
        _ts(s["SRAM_OPERAND"], PlacementBoundary.ENTER, TransferAction.LOAD, {ROLE_Q}),
        _ts(s["SRAM_FORWARD"], PlacementBoundary.EXIT, TransferAction.EVICT, {ROLE_SCORES, ROLE_PROBS}),
        _ts(s["SRAM_ACCUM"], PlacementBoundary.EXIT, TransferAction.WRITEBACK, {ROLE_PARTIAL_O, ROLE_O}),
    ]
    return _template(
        template_name,
        residency,
        transfers,
        description="Persist KV-like state in SRAM while streaming query-side work.",
        supported_phases={"decode"},
        extra_metadata={
            "family": "state_persistent",
            "design_intent": "decode-oriented persistent state residency",
        },
    )


# -----------------------------------------------------------------------------
# Factory / registry
# -----------------------------------------------------------------------------

def build_core_templates(
    *,
    scope: LifetimeScope = DEFAULT_SCOPE,
) -> RolePlacementTemplateLibrary:
    """Build the canonical coarse-grained attention placement templates."""

    library = RolePlacementTemplateLibrary()
    for template in (
        operand_resident_template("OperandResident", scope=scope),
        producer_consumer_keep_template("ProducerConsumerKeep", scope=scope),
        local_accumulate_template("LocalAccumulate", scope=scope),
        state_persistent_template("StatePersistent", scope=scope),
    ):
        library.add(template)
    return library



def build_templates_by_phase(
    phase: str,
    *,
    scope: LifetimeScope = DEFAULT_SCOPE,
) -> RolePlacementTemplateLibrary:
    """Filter canonical templates by supported phase without instantiating them."""

    base = build_core_templates(scope=scope)
    if not phase:
        return base

    phase_key = phase.strip().lower()
    filtered = RolePlacementTemplateLibrary()
    for template in base.templates.values():
        if template.supports_phase(phase_key):
            filtered.add(template)
    return filtered



def get_template_names() -> Sequence[str]:
    return (
        "OperandResident",
        "ProducerConsumerKeep",
        "LocalAccumulate",
        "StatePersistent",
    )



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
    "canonical_attention_sites",
    "operand_resident_template",
    "producer_consumer_keep_template",
    "local_accumulate_template",
    "state_persistent_template",
    "build_core_templates",
    "build_templates_by_phase",
    "get_template_names",
    "get_known_roles",
]
