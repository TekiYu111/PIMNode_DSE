from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

from pimnode_dse.placement.placement_ir import (
    PlacementBoundary,
    PlacementDomain,
    ResidencySpec,
    Site,
    StorageTier,
    TensorPlacementSpec,
    TransferAction,
    TransferSpec,
)

# -----------------------------------------------------------------------------
# Canonical logical tensor roles used by placement templates / instantiated specs
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

ALL_KNOWN_ROLES = {
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

ROLE_TO_DOMAIN_SPACE: Dict[str, set[PlacementDomain]] = {
    ROLE_Q: {PlacementDomain.OPERAND},
    ROLE_K: {PlacementDomain.OPERAND},
    ROLE_V: {PlacementDomain.OPERAND},
    ROLE_SCORES: {PlacementDomain.FORWARD},
    ROLE_STATS: {PlacementDomain.ACCUM},
    ROLE_PROBS: {PlacementDomain.FORWARD},
    ROLE_PARTIAL_O: {PlacementDomain.ACCUM},
    ROLE_O: {PlacementDomain.ACCUM, PlacementDomain.FORWARD},
    ROLE_KV_CACHE: {PlacementDomain.STATE},
}


# -----------------------------------------------------------------------------
# Public context / diagnostics / result objects
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionNode:
    tensor: str
    site: Site
    boundary: PlacementBoundary
    action: TransferAction
    phase: str = ""
    role: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleViolation:
    family: str  # semantic / placement / flow / hardware
    tensor: str
    message: str
    severity: str = "error"  # error / warning
    site: Site | None = None
    boundary: PlacementBoundary | None = None
    action: TransferAction | None = None


@dataclass
class RuleResult:
    normalized_spec: TensorPlacementSpec
    violations: list[RuleViolation]
    actions: list[ActionNode]

    @property
    def is_valid(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)


@dataclass
class RuleContext:
    phase: str = ""
    tensor_to_role: Dict[str, str] = field(default_factory=dict)
    tensor_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hardware_limits: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def canonicalize_phase(phase: str | None) -> str:
    return (phase or "").strip().lower()


def get_tensor_role(tensor: str, tensor_to_role: Mapping[str, str] | None = None) -> str:
    if tensor_to_role and tensor in tensor_to_role:
        return tensor_to_role[tensor]

    upper = tensor.upper()
    if "KV" in upper and "CACHE" in upper:
        return ROLE_KV_CACHE
    if "PARTIAL" in upper and "O" in upper:
        return ROLE_PARTIAL_O
    if upper.endswith("SCORES") or "SCORE" in upper:
        return ROLE_SCORES
    if upper.endswith("PROBS") or "PROB" in upper:
        return ROLE_PROBS
    if upper.endswith("STATS") or "STAT" in upper:
        return ROLE_STATS
    if upper == "Q" or upper.startswith("Q_") or "QUERY" in upper:
        return ROLE_Q
    if upper == "K" or upper.startswith("K_") or "KEY" in upper:
        return ROLE_K
    if upper == "V" or upper.startswith("V_") or "VALUE" in upper:
        return ROLE_V
    if upper == "O" or upper.endswith("_O") or "OUTPUT" in upper:
        return ROLE_O
    return tensor


def get_allowed_domains(role: str) -> set[PlacementDomain]:
    return set(ROLE_TO_DOMAIN_SPACE.get(role, set()))


def _unique_sorted_residency(items: Iterable[ResidencySpec]) -> list[ResidencySpec]:
    uniq: Dict[tuple[str, Site], ResidencySpec] = {}
    for item in items:
        uniq[(item.tensor, item.site)] = item
    return sorted(
        uniq.values(),
        key=lambda x: (
            x.tensor,
            str(x.site.storage_tier),
            str(x.site.lifetime_scope),
            str(x.site.domain),
        ),
    )


def _unique_sorted_transfers(items: Iterable[TransferSpec]) -> list[TransferSpec]:
    uniq: Dict[tuple[str, Site, PlacementBoundary, TransferAction], TransferSpec] = {}
    for item in items:
        uniq[(item.tensor, item.site, item.boundary, item.action)] = item
    return sorted(
        uniq.values(),
        key=lambda x: (
            x.tensor,
            str(x.site.storage_tier),
            str(x.site.lifetime_scope),
            str(x.site.domain),
            str(x.boundary),
            str(x.action),
        ),
    )


def normalize_tensor_placement_spec(spec: TensorPlacementSpec) -> TensorPlacementSpec:
    normalized = TensorPlacementSpec(tensor=spec.tensor)
    normalized.residency = _unique_sorted_residency(spec.residency)
    normalized.transfers = _unique_sorted_transfers(spec.transfers)
    return normalized


def _filtered_spec_by_domain(spec: TensorPlacementSpec, domain: PlacementDomain) -> TensorPlacementSpec:
    filtered = TensorPlacementSpec(tensor=spec.tensor)
    filtered.residency = [r for r in spec.residency if r.site.domain == domain]
    filtered.transfers = [t for t in spec.transfers if t.site.domain == domain]
    return filtered


# -----------------------------------------------------------------------------
# Domain policies (default allow-list style) + flow constraints
# -----------------------------------------------------------------------------
class DomainPolicy(Protocol):
    domain: PlacementDomain

    def is_valid_site(self, site: Site, ctx: RuleContext, tensor: str) -> bool: ...

    def allowed_actions(
        self,
        site: Site,
        boundary: PlacementBoundary,
        ctx: RuleContext,
        tensor: str,
    ) -> set[TransferAction]: ...

    def check_flow(
        self,
        tensor: str,
        spec: TensorPlacementSpec,
        ctx: RuleContext,
    ) -> list[RuleViolation]: ...


class BaseDomainPolicy:
    domain: PlacementDomain
    allowed_storage_tiers: set[StorageTier] = set()
    allowed_domains: set[PlacementDomain] = set()
    enter_actions: set[TransferAction] = set()
    exit_actions: set[TransferAction] = set()

    def is_valid_site(self, site: Site, ctx: RuleContext, tensor: str) -> bool:
        return site.domain in self.allowed_domains and site.storage_tier in self.allowed_storage_tiers

    def allowed_actions(
        self,
        site: Site,
        boundary: PlacementBoundary,
        ctx: RuleContext,
        tensor: str,
    ) -> set[TransferAction]:
        if boundary == PlacementBoundary.ENTER:
            return set(self.enter_actions)
        if boundary == PlacementBoundary.EXIT:
            return set(self.exit_actions)
        return set()

    def check_flow(
        self,
        tensor: str,
        spec: TensorPlacementSpec,
        ctx: RuleContext,
    ) -> list[RuleViolation]:
        return []


class OperandPolicy(BaseDomainPolicy):
    domain = PlacementDomain.OPERAND
    allowed_storage_tiers = {StorageTier.DRAM, StorageTier.SRAM, StorageTier.PE}
    allowed_domains = {PlacementDomain.OPERAND}
    enter_actions = {TransferAction.LOAD, TransferAction.PREFETCH}
    exit_actions = {TransferAction.EVICT}


class ForwardPolicy(BaseDomainPolicy):
    domain = PlacementDomain.FORWARD
    allowed_storage_tiers = {StorageTier.DRAM, StorageTier.SRAM}
    allowed_domains = {PlacementDomain.FORWARD}
    enter_actions = {TransferAction.LOAD, TransferAction.PREFETCH}
    exit_actions = {TransferAction.EVICT}


class AccumPolicy(BaseDomainPolicy):
    domain = PlacementDomain.ACCUM
    allowed_storage_tiers = {StorageTier.SRAM, StorageTier.PE}
    allowed_domains = {PlacementDomain.ACCUM}
    enter_actions = {TransferAction.LOAD}
    exit_actions = {TransferAction.WRITEBACK, TransferAction.EVICT}

    def check_flow(
        self,
        tensor: str,
        spec: TensorPlacementSpec,
        ctx: RuleContext,
    ) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        has_accum_residency = any(r.site.domain == PlacementDomain.ACCUM for r in spec.residency)
        has_writeback = any(
            t.boundary == PlacementBoundary.EXIT and t.action == TransferAction.WRITEBACK
            for t in spec.transfers
        )
        if has_accum_residency and not has_writeback:
            violations.append(
                RuleViolation(
                    family="flow",
                    tensor=tensor,
                    message="accum-domain tensor has no EXIT->WRITEBACK path",
                    severity="warning",
                )
            )
        return violations


class StatePolicy(BaseDomainPolicy):
    domain = PlacementDomain.STATE
    allowed_storage_tiers = {StorageTier.DRAM, StorageTier.SRAM}
    allowed_domains = {PlacementDomain.STATE}
    enter_actions = {TransferAction.LOAD, TransferAction.PREFETCH}
    exit_actions = {TransferAction.WRITEBACK, TransferAction.EVICT}

    def allowed_actions(
        self,
        site: Site,
        boundary: PlacementBoundary,
        ctx: RuleContext,
        tensor: str,
    ) -> set[TransferAction]:
        allowed = super().allowed_actions(site, boundary, ctx, tensor)
        meta = ctx.tensor_meta.get(tensor, {})
        has_backing = meta.get("has_backing", site.storage_tier == StorageTier.DRAM)
        if boundary == PlacementBoundary.EXIT and not has_backing:
            allowed.discard(TransferAction.EVICT)
        return allowed

    def check_flow(
        self,
        tensor: str,
        spec: TensorPlacementSpec,
        ctx: RuleContext,
    ) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        meta = ctx.tensor_meta.get(tensor, {})
        has_backing = meta.get(
            "has_backing",
            any(r.site.storage_tier == StorageTier.DRAM for r in spec.residency),
        )
        for transfer in spec.transfers:
            if transfer.boundary == PlacementBoundary.EXIT and transfer.action == TransferAction.EVICT and not has_backing:
                violations.append(
                    RuleViolation(
                        family="flow",
                        tensor=tensor,
                        message="state-domain tensor cannot be evicted without a backing copy",
                        site=transfer.site,
                        boundary=transfer.boundary,
                        action=transfer.action,
                    )
                )
        return violations


DOMAIN_POLICY_REGISTRY: Dict[PlacementDomain, DomainPolicy] = {
    PlacementDomain.OPERAND: OperandPolicy(),
    PlacementDomain.FORWARD: ForwardPolicy(),
    PlacementDomain.ACCUM: AccumPolicy(),
    PlacementDomain.STATE: StatePolicy(),
}


# -----------------------------------------------------------------------------
# Constraint objects
# -----------------------------------------------------------------------------
class Constraint(Protocol):
    def check(self, spec: TensorPlacementSpec, ctx: RuleContext) -> list[RuleViolation]: ...


class SemanticConstraint:
    def check(self, spec: TensorPlacementSpec, ctx: RuleContext) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        role = get_tensor_role(spec.tensor, ctx.tensor_to_role)
        allowed_domains = get_allowed_domains(role)
        if not allowed_domains:
            violations.append(
                RuleViolation(
                    family="semantic",
                    tensor=spec.tensor,
                    message=f"unknown tensor role '{role}'",
                    severity="warning",
                )
            )
            return violations

        allowed_domain_names = ", ".join(sorted(d.name for d in allowed_domains))
        for residency in spec.residency:
            if residency.site.domain not in allowed_domains:
                violations.append(
                    RuleViolation(
                        family="semantic",
                        tensor=spec.tensor,
                        message=(
                            f"role {role} does not allow residency domain {residency.site.domain.name}; "
                            f"allowed domains: {allowed_domain_names}"
                        ),
                        site=residency.site,
                    )
                )
        for transfer in spec.transfers:
            if transfer.site.domain not in allowed_domains:
                violations.append(
                    RuleViolation(
                        family="semantic",
                        tensor=spec.tensor,
                        message=(
                            f"role {role} does not allow transfer domain {transfer.site.domain.name}; "
                            f"allowed domains: {allowed_domain_names}"
                        ),
                        site=transfer.site,
                        boundary=transfer.boundary,
                        action=transfer.action,
                    )
                )
                continue

            policy = DOMAIN_POLICY_REGISTRY[transfer.site.domain]
            allowed = policy.allowed_actions(transfer.site, transfer.boundary, ctx, spec.tensor)
            if transfer.action not in allowed:
                violations.append(
                    RuleViolation(
                        family="semantic",
                        tensor=spec.tensor,
                        message=(
                            f"action {transfer.action.name} is not allowed for domain {transfer.site.domain.name} "
                            f"at {transfer.boundary.name}"
                        ),
                        site=transfer.site,
                        boundary=transfer.boundary,
                        action=transfer.action,
                    )
                )
        return violations


class PlacementConstraint:
    def check(self, spec: TensorPlacementSpec, ctx: RuleContext) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        role = get_tensor_role(spec.tensor, ctx.tensor_to_role)
        allowed_domains = get_allowed_domains(role)
        if not allowed_domains:
            return violations

        residency_sites = {r.site for r in spec.residency}
        for residency in spec.residency:
            if residency.site.domain not in allowed_domains:
                continue
            policy = DOMAIN_POLICY_REGISTRY[residency.site.domain]
            if not policy.is_valid_site(residency.site, ctx, spec.tensor):
                violations.append(
                    RuleViolation(
                        family="placement",
                        tensor=spec.tensor,
                        message=f"site is not a valid placement location for domain {residency.site.domain.name}",
                        site=residency.site,
                    )
                )
        for transfer in spec.transfers:
            if transfer.site not in residency_sites:
                violations.append(
                    RuleViolation(
                        family="placement",
                        tensor=spec.tensor,
                        message="transfer refers to a site where the tensor is not resident",
                        site=transfer.site,
                        boundary=transfer.boundary,
                        action=transfer.action,
                    )
                )
                continue
            if transfer.site.domain not in allowed_domains:
                continue
            policy = DOMAIN_POLICY_REGISTRY[transfer.site.domain]
            if not policy.is_valid_site(transfer.site, ctx, spec.tensor):
                violations.append(
                    RuleViolation(
                        family="placement",
                        tensor=spec.tensor,
                        message=f"transfer site is not a valid placement location for domain {transfer.site.domain.name}",
                        site=transfer.site,
                        boundary=transfer.boundary,
                        action=transfer.action,
                    )
                )
        return violations


class FlowConstraint:
    def check(self, spec: TensorPlacementSpec, ctx: RuleContext) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        role = get_tensor_role(spec.tensor, ctx.tensor_to_role)
        allowed_domains = get_allowed_domains(role)
        if not allowed_domains:
            return violations

        domains_present = {
            domain
            for domain in allowed_domains
            if any(r.site.domain == domain for r in spec.residency) or any(t.site.domain == domain for t in spec.transfers)
        }

        if spec.residency and not any(t.boundary == PlacementBoundary.ENTER for t in spec.transfers):
            violations.append(
                RuleViolation(
                    family="flow",
                    tensor=spec.tensor,
                    message="resident tensor has no ENTER action",
                    severity="warning",
                )
            )

        for domain in domains_present:
            domain_spec = _filtered_spec_by_domain(spec, domain)
            policy = DOMAIN_POLICY_REGISTRY[domain]
            violations.extend(policy.check_flow(spec.tensor, domain_spec, ctx))

            exit_actions = [t for t in domain_spec.transfers if t.boundary == PlacementBoundary.EXIT]
            if domain in {PlacementDomain.ACCUM, PlacementDomain.STATE} and domain_spec.residency and not exit_actions:
                violations.append(
                    RuleViolation(
                        family="flow",
                        tensor=spec.tensor,
                        message=f"{domain.name}-domain tensor has no EXIT action",
                        severity="warning",
                    )
                )
        return violations


class HardwareFeasibilityConstraint:
    """Lightweight feasibility checks using optional hardware_limits.

    Expected optional keys in ctx.hardware_limits:
      - dram_bytes, sram_bytes, pe_bytes
      - tensor_sizes: {tensor_name: bytes}
    """

    _TIER_KEY = {
        StorageTier.DRAM: "dram_bytes",
        StorageTier.SRAM: "sram_bytes",
        StorageTier.PE: "pe_bytes",
    }

    def check(self, spec: TensorPlacementSpec, ctx: RuleContext) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        limits = ctx.hardware_limits or {}
        tensor_sizes = limits.get("tensor_sizes", {})
        size = tensor_sizes.get(spec.tensor) or ctx.tensor_meta.get(spec.tensor, {}).get("size_bytes")
        if size is None:
            return violations

        for residency in spec.residency:
            limit_key = self._TIER_KEY.get(residency.site.storage_tier)
            if not limit_key:
                continue
            cap = limits.get(limit_key)
            if cap is not None and size > cap:
                violations.append(
                    RuleViolation(
                        family="hardware",
                        tensor=spec.tensor,
                        message=(
                            f"tensor replica size {size} exceeds {residency.site.storage_tier.name} "
                            f"limit {cap}"
                        ),
                        site=residency.site,
                    )
                )
        return violations


DEFAULT_CONSTRAINTS: tuple[Constraint, ...] = (
    SemanticConstraint(),
    PlacementConstraint(),
    FlowConstraint(),
    HardwareFeasibilityConstraint(),
)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def derive_actions(spec: TensorPlacementSpec, ctx: RuleContext) -> list[ActionNode]:
    role = get_tensor_role(spec.tensor, ctx.tensor_to_role)
    phase = canonicalize_phase(ctx.phase)
    actions: list[ActionNode] = []
    for transfer in spec.transfers:
        actions.append(
            ActionNode(
                tensor=spec.tensor,
                site=transfer.site,
                boundary=transfer.boundary,
                action=transfer.action,
                phase=phase,
                role=role,
                metadata=dict(ctx.tensor_meta.get(spec.tensor, {})),
            )
        )
    return actions


def apply_rules(
    spec: TensorPlacementSpec,
    *,
    phase: str = "",
    tensor_to_role: Mapping[str, str] | None = None,
    tensor_meta: Mapping[str, Mapping[str, Any]] | None = None,
    hardware_limits: Mapping[str, Any] | None = None,
    constraints: Sequence[Constraint] | None = None,
) -> RuleResult:
    ctx = RuleContext(
        phase=canonicalize_phase(phase),
        tensor_to_role=dict(tensor_to_role or {}),
        tensor_meta={k: dict(v) for k, v in (tensor_meta or {}).items()},
        hardware_limits=dict(hardware_limits or {}),
    )
    normalized = normalize_tensor_placement_spec(spec)
    violations: list[RuleViolation] = []
    for constraint in constraints or DEFAULT_CONSTRAINTS:
        violations.extend(constraint.check(normalized, ctx))
    actions = derive_actions(normalized, ctx)
    return RuleResult(normalized_spec=normalized, violations=violations, actions=actions)


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
    "ROLE_TO_DOMAIN_SPACE",
    "ActionNode",
    "RuleViolation",
    "RuleResult",
    "RuleContext",
    "DomainPolicy",
    "DOMAIN_POLICY_REGISTRY",
    "Constraint",
    "SemanticConstraint",
    "PlacementConstraint",
    "FlowConstraint",
    "HardwareFeasibilityConstraint",
    "normalize_tensor_placement_spec",
    "canonicalize_phase",
    "get_tensor_role",
    "get_allowed_domains",
    "derive_actions",
    "apply_rules",
]
