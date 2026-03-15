from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


class StorageTier(Enum):
    """Physical storage tiers in the PIM hierarchy."""

    DRAM = "DRAM"
    SRAM = "SRAM"
    PE = "PE"


class LifetimeScope(Enum):
    """How long a resident replica is intended to survive."""

    GROUP = "group"
    TILE = "tile"
    INNER_TILE = "inner_tile"
    OP = "op"


class PlacementDomain(Enum):
    """Minimal behavioral classes consumed by downstream analysis."""

    OPERAND = "operand"
    ACCUM = "accum"
    FORWARD = "forward"
    STATE = "state"



class PlacementBoundary(Enum):
    ENTER = "enter"
    EXIT = "exit"


class TransferAction(Enum):
    LOAD = "LOAD"
    PREFETCH = "PREFETCH"
    WRITEBACK = "WRITEBACK"
    EVICT = "EVICT"


@dataclass(frozen=True)
class Site:
    """A placement site factorized into storage tier, lifetime scope and domain."""

    storage_tier: StorageTier
    lifetime_scope: LifetimeScope
    domain: PlacementDomain


    def short_name(self) -> str:
        return f"{self.storage_tier.value}/{self.lifetime_scope.value}/{self.domain.value}"


@dataclass(frozen=True)
class ResidencySpec:
    """A concrete tensor has a resident replica at a given site."""

    tensor: str
    site: Site


@dataclass(frozen=True)
class TransferSpec:
    """A concrete tensor transfer action is allowed/required at a site boundary.

    Notes
    -----
    The movement direction is intentionally not encoded here. It is derived later
    from the execution tree / scope nesting (TileFlow-style lowering).
    """

    tensor: str
    site: Site
    boundary: PlacementBoundary
    action: TransferAction


@dataclass
class TensorPlacementSpec:
    """Tensor-level placement skeleton for one concrete scope/group."""

    scope_name: str
    phase: Optional[str] = None
    residency: List[ResidencySpec] = field(default_factory=list)
    transfers: List[TransferSpec] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def all_tensors(self) -> Set[str]:
        tensors = {spec.tensor for spec in self.residency}
        tensors.update(spec.tensor for spec in self.transfers)
        return tensors

    def sites_for_tensor(self, tensor: str) -> Set[Site]:
        sites = {spec.site for spec in self.residency if spec.tensor == tensor}
        sites.update(spec.site for spec in self.transfers if spec.tensor == tensor)
        return sites

    def residency_for_tensor(self, tensor: str) -> List[ResidencySpec]:
        return [spec for spec in self.residency if spec.tensor == tensor]

    def transfers_for_tensor(self, tensor: str) -> List[TransferSpec]:
        return [spec for spec in self.transfers if spec.tensor == tensor]

    def validate_unique_specs(self) -> None:
        resid_keys = {(spec.tensor, spec.site) for spec in self.residency}
        if len(resid_keys) != len(self.residency):
            raise ValueError(f"Duplicate residency specs found in scope '{self.scope_name}'")

        transfer_keys = {
            (spec.tensor, spec.site, spec.boundary, spec.action)
            for spec in self.transfers
        }
        if len(transfer_keys) != len(self.transfers):
            raise ValueError(f"Duplicate transfer specs found in scope '{self.scope_name}'")


# -----------------------------
# Role-level template objects
# -----------------------------


@dataclass(frozen=True)
class RoleResidencySpec:
    tensor_roles: Set[str]
    site: Site


@dataclass(frozen=True)
class RoleTransferSpec:
    tensor_roles: Set[str]
    site: Site
    boundary: PlacementBoundary
    action: TransferAction


@dataclass
class RolePlacementTemplate:
    """Role-based template to be instantiated per concrete scope/group."""

    template_name: str
    residency: List[RoleResidencySpec] = field(default_factory=list)
    transfers: List[RoleTransferSpec] = field(default_factory=list)
    supported_phases: Set[str] = field(default_factory=set)
    metadata: Dict[str, object] = field(default_factory=dict)

    def supports_phase(self, phase: Optional[str]) -> bool:
        if not self.supported_phases or phase is None:
            return True
        return phase in self.supported_phases


@dataclass
class RolePlacementTemplateLibrary:
    templates: Dict[str, RolePlacementTemplate] = field(default_factory=dict)

    def add(self, template: RolePlacementTemplate) -> None:
        if template.template_name in self.templates:
            raise ValueError(f"Duplicate template name: {template.template_name}")
        self.templates[template.template_name] = template

    def names(self) -> List[str]:
        return list(self.templates.keys())

    def get(self, name: str) -> RolePlacementTemplate:
        return self.templates[name]


# -----------------------------
# Canonical site helpers
# -----------------------------


def make_site(
    storage_tier: StorageTier,
    lifetime_scope: LifetimeScope,
    domain: PlacementDomain,
) -> Site:
    return Site(
        storage_tier=storage_tier,
        lifetime_scope=lifetime_scope,
        domain=domain,
    )


DEFAULT_SCOPE = LifetimeScope.TILE
DEFAULT_STORAGE_TIERS: Sequence[StorageTier] = (
    StorageTier.DRAM,
    StorageTier.SRAM,
    StorageTier.PE,
)


def default_site_map(
    scope: LifetimeScope = DEFAULT_SCOPE,
) -> Dict[str, Site]:
    return {
        "DRAM_OPERAND": make_site(StorageTier.DRAM, scope, PlacementDomain.OPERAND),
        "DRAM_FORWARD": make_site(StorageTier.DRAM, scope, PlacementDomain.FORWARD),
        "DRAM_STATE": make_site(StorageTier.DRAM, scope, PlacementDomain.STATE),
        "SRAM_OPERAND": make_site(StorageTier.SRAM, scope, PlacementDomain.OPERAND),
        "SRAM_FORWARD": make_site(StorageTier.SRAM, scope, PlacementDomain.FORWARD),
        "SRAM_ACCUM": make_site(StorageTier.SRAM, scope, PlacementDomain.ACCUM),
        "SRAM_STATE": make_site(StorageTier.SRAM, scope, PlacementDomain.STATE),
        "PE_OPERAND": make_site(StorageTier.PE, scope, PlacementDomain.OPERAND),
        "PE_ACCUM": make_site(StorageTier.PE, scope, PlacementDomain.ACCUM),
    }


# -----------------------------
# Helper utilities
# -----------------------------


def canonicalize_phase(phase: Optional[str]) -> Optional[str]:
    if phase is None:
        return None
    p = phase.strip().lower()
    if p in {"prefill", "decode"}:
        return p
    raise ValueError(f"Unsupported phase: {phase}")


def _resolve_roles(
    role_bindings: Mapping[str, Sequence[str]],
    roles: Iterable[str],
) -> Set[str]:
    tensors: Set[str] = set()
    for role in roles:
        tensors.update(role_bindings.get(role, ()))
    return tensors


def instantiate_tensor_placement(
    template: RolePlacementTemplate,
    scope_name: str,
    role_bindings: Mapping[str, Sequence[str]],
    *,
    phase: Optional[str] = None,
    tensor_universe: Optional[Iterable[str]] = None,
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> TensorPlacementSpec:
    """Instantiate a role-based template into concrete tensor placement."""

    phase_canonical = canonicalize_phase(phase)
    if not template.supports_phase(phase_canonical):
        raise ValueError(
            f"Template '{template.template_name}' does not support phase '{phase_canonical}'"
        )

    allowed = set(tensor_universe) if tensor_universe is not None else None

    residency: List[ResidencySpec] = []
    for spec in template.residency:
        tensors = _resolve_roles(role_bindings, spec.tensor_roles)
        if allowed is not None:
            tensors &= allowed
        residency.extend(ResidencySpec(tensor=t, site=spec.site) for t in sorted(tensors))

    transfers: List[TransferSpec] = []
    for spec in template.transfers:
        tensors = _resolve_roles(role_bindings, spec.tensor_roles)
        if allowed is not None:
            tensors &= allowed
        transfers.extend(
            TransferSpec(
                tensor=t,
                site=spec.site,
                boundary=spec.boundary,
                action=spec.action,
            )
            for t in sorted(tensors)
        )

    metadata: Dict[str, object] = {
        "template_name": template.template_name,
        **template.metadata,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    placement = TensorPlacementSpec(
        scope_name=scope_name,
        phase=phase_canonical,
        residency=residency,
        transfers=transfers,
        metadata=metadata,
    )
    placement.validate_unique_specs()
    return placement


__all__ = [
    "StorageTier",
    "LifetimeScope",
    "PlacementDomain",
    "PlacementBoundary",
    "TransferAction",
    "Site",
    "ResidencySpec",
    "TransferSpec",
    "TensorPlacementSpec",
    "RoleResidencySpec",
    "RoleTransferSpec",
    "RolePlacementTemplate",
    "RolePlacementTemplateLibrary",
    "make_site",
    "DEFAULT_SCOPE",
    "DEFAULT_STORAGE_TIERS",
    "default_site_map",
    "canonicalize_phase",
    "instantiate_tensor_placement",
]
