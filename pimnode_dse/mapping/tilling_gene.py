from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pimnode_dse.placement.placement_ir import StorageTier


OpScheduleStyle = Literal["sequential", "micro_pipeline"]


@dataclass
class TierTileSpec:
    """
    Tier-level static tiling result.

    This object only describes how a single storage tier is tiled.
    It does NOT encode:
      - placement semantics
      - residency / writeback / boundary actions
      - hardware budgets
      - runtime tile generation logic

    Fields:
      storage_tier:
          Which storage tier this tiling spec belongs to.

      tile_size:
          Per-dimension tile extent at this tier.

      loop_order:
          Traversal order of dimensions at this tier.

      problem_size:
          The local problem extent visible at this tier.
          This is useful for debugging and later analysis.
    """

    storage_tier: StorageTier
    tile_size: dict[str, int]
    loop_order: list[str]
    problem_size: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Defensive copy to avoid accidental aliasing.
        self.tile_size = dict(self.tile_size)
        self.loop_order = list(self.loop_order)
        self.problem_size = dict(self.problem_size)
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.storage_tier, StorageTier):
            raise TypeError("storage_tier must be a StorageTier")

        if not self.tile_size:
            raise ValueError(f"{self.storage_tier.name}: tile_size must not be empty")

        for dim, size in self.tile_size.items():
            if not isinstance(dim, str) or not dim:
                raise ValueError(f"{self.storage_tier.name}: invalid tile dimension name {dim!r}")
            if not isinstance(size, int) or size <= 0:
                raise ValueError(
                    f"{self.storage_tier.name}: tile_size[{dim!r}] must be a positive int, got {size!r}"
                )

        if not self.loop_order:
            raise ValueError(f"{self.storage_tier.name}: loop_order must not be empty")

        seen: set[str] = set()
        valid_dims = set(self.tile_size.keys()) | set(self.problem_size.keys())

        for dim in self.loop_order:
            if not isinstance(dim, str) or not dim:
                raise ValueError(f"{self.storage_tier.name}: invalid loop dimension name {dim!r}")
            if dim in seen:
                raise ValueError(f"{self.storage_tier.name}: duplicate dimension {dim!r} in loop_order")
            seen.add(dim)
            if dim not in valid_dims:
                raise ValueError(
                    f"{self.storage_tier.name}: loop_order dimension {dim!r} "
                    "does not appear in tile_size or problem_size"
                )

        for dim, size in self.problem_size.items():
            if not isinstance(dim, str) or not dim:
                raise ValueError(f"{self.storage_tier.name}: invalid problem dimension name {dim!r}")
            if not isinstance(size, int) or size <= 0:
                raise ValueError(
                    f"{self.storage_tier.name}: problem_size[{dim!r}] must be a positive int, got {size!r}"
                )

        # When both are present, problem_size should not be smaller than tile_size.
        for dim, tile_extent in self.tile_size.items():
            if dim in self.problem_size and self.problem_size[dim] < tile_extent:
                raise ValueError(
                    f"{self.storage_tier.name}: problem_size[{dim!r}]={self.problem_size[dim]} "
                    f"is smaller than tile_size[{dim!r}]={tile_extent}"
                )

    def clone(self) -> "TierTileSpec":
        return TierTileSpec(
            storage_tier=self.storage_tier,
            tile_size=dict(self.tile_size),
            loop_order=list(self.loop_order),
            problem_size=dict(self.problem_size),
        )


@dataclass
class GroupTilingSpec:
    """
    Static tiling result for one fusion/op group.

    Fields:
      group_id:
          Identifier of the fusion group.

      tier_tiles:
          Static tiling specs for tiers used by this group.

      op_schedule_style:
          Scheduling relation among the internal stages / ops of this group.
          - "sequential": stages execute strictly in sequence
          - "micro_pipeline": adjacent stages may form tile-granular pipeline
    """

    group_id: str
    tier_tiles: dict[StorageTier, TierTileSpec]
    op_schedule_style: OpScheduleStyle = "sequential"

    def __post_init__(self) -> None:
        self.tier_tiles = dict(self.tier_tiles)
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")

        if self.op_schedule_style not in ("sequential", "micro_pipeline"):
            raise ValueError(
                "op_schedule_style must be either 'sequential' or 'micro_pipeline'"
            )

        if not self.tier_tiles:
            raise ValueError(f"{self.group_id}: tier_tiles must not be empty")

        for tier, spec in self.tier_tiles.items():
            if not isinstance(tier, StorageTier):
                raise TypeError(f"{self.group_id}: tier key must be a StorageTier, got {tier!r}")
            if not isinstance(spec, TierTileSpec):
                raise TypeError(
                    f"{self.group_id}: tier_tiles[{tier!r}] must be a TierTileSpec, got {type(spec)!r}"
                )
            if spec.storage_tier != tier:
                raise ValueError(
                    f"{self.group_id}: tier key {tier.name} does not match "
                    f"spec.storage_tier {spec.storage_tier.name}"
                )
            spec.validate()

    def get_tier_tile_spec(self, storage_tier: StorageTier) -> TierTileSpec:
        try:
            return self.tier_tiles[storage_tier]
        except KeyError as exc:
            raise KeyError(
                f"{self.group_id}: no tiling spec for tier {storage_tier.name}"
            ) from exc

    def clone(self) -> "GroupTilingSpec":
        return GroupTilingSpec(
            group_id=self.group_id,
            tier_tiles={tier: spec.clone() for tier, spec in self.tier_tiles.items()},
            op_schedule_style=self.op_schedule_style,
        )


@dataclass
class TilingGene:
    """
    A complete static tiling design point across all groups.

    Fields:
      group_tiles:
          Mapping from group_id to its static tier-level tiling result.

      gene_id:
          Optional identifier for external search / bookkeeping.
          This module does not generate or interpret it.
    """

    group_tiles: dict[str, GroupTilingSpec]
    gene_id: str = ""

    def __post_init__(self) -> None:
        self.group_tiles = dict(self.group_tiles)
        self.validate()

    def validate(self) -> None:
        if not self.group_tiles:
            raise ValueError("group_tiles must not be empty")

        for group_id, spec in self.group_tiles.items():
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"invalid group_id key: {group_id!r}")
            if not isinstance(spec, GroupTilingSpec):
                raise TypeError(
                    f"group_tiles[{group_id!r}] must be a GroupTilingSpec, got {type(spec)!r}"
                )
            if spec.group_id != group_id:
                raise ValueError(
                    f"group_tiles key {group_id!r} does not match spec.group_id {spec.group_id!r}"
                )
            spec.validate()

    def get_group_spec(self, group_id: str) -> GroupTilingSpec:
        try:
            return self.group_tiles[group_id]
        except KeyError as exc:
            raise KeyError(f"no tiling spec found for group {group_id!r}") from exc

    def add_group_spec(self, group_spec: GroupTilingSpec) -> None:
        if not isinstance(group_spec, GroupTilingSpec):
            raise TypeError(f"group_spec must be a GroupTilingSpec, got {type(group_spec)!r}")
        group_spec.validate()
        self.group_tiles[group_spec.group_id] = group_spec

    def clone(self) -> "TilingGene":
        return TilingGene(
            group_tiles={gid: spec.clone() for gid, spec in self.group_tiles.items()},
            gene_id=self.gene_id,
        )


__all__ = [
    "OpScheduleStyle",
    "TierTileSpec",
    "GroupTilingSpec",
    "TilingGene",
]
