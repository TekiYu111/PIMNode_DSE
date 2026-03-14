from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pimnode_dse.mapping.fusion_gene import FusionGene, OpFusionGroup
from pimnode_dse.mapping.tilling_gene import GroupTilingSpec, MemTileSpec, TilingGene
from pimnode_dse.placement.placement_ir import PlacementTemplateScope


# =============================================================================
# Helpers
# =============================================================================


def _deep_copy_mapping(obj: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if obj is None:
        return {}
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, Mapping):
            out[str(k)] = _deep_copy_mapping(v)
        elif isinstance(v, list):
            out[str(k)] = list(v)
        else:
            out[str(k)] = v
    return out

def _expand_dotted_mapping(raw: Mapping[str, Any]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or "." not in key:
            nested[str(key)] = value
            continue
        cur = nested
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = value
    return nested



def _to_runtime_dim(dim: str) -> str:
    d = str(dim).lower()
    if d == "q":
        return "m"
    if d == "k":
        return "n"
    if d == "dh":
        return "k"
    return str(dim)



def _normalize_loop_order(loop_order: Iterable[str]) -> List[str]:
    return [_to_runtime_dim(dim) for dim in loop_order]



def _normalize_tile_dims(tile_obj: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for raw_dim, value in tile_obj.items():
        if raw_dim == "description":
            continue
        if value is None:
            continue
        out[_to_runtime_dim(str(raw_dim))] = int(value)
    return out



def _extract_level_tile(raw_tiling: Mapping[str, Any], level: str) -> Dict[str, int]:
    """
    Accept both nested shape:
        raw_tiling["dram_tile"] = {"q": 64, "k": 64, "dh": 32}
    and already-flattened/dotted shape if any legacy caller still passes it.
    """
    level_key = f"{str(level).lower()}_tile"

    nested = raw_tiling.get(level_key)
    if isinstance(nested, Mapping):
        return _normalize_tile_dims(nested)

    legacy: Dict[str, Any] = {}
    prefix = level_key + "."
    for k, v in raw_tiling.items():
        if isinstance(k, str) and k.startswith(prefix):
            legacy[k[len(prefix):]] = v
    if legacy:
        return _normalize_tile_dims(legacy)

    return {}



def _extract_spatial_flag(raw_tiling: Mapping[str, Any], level: str) -> bool:
    spatial_mapping = raw_tiling.get("spatial_mapping")
    if isinstance(spatial_mapping, Mapping):
        if level == "PE":
            return bool(spatial_mapping.get("pe", False))
        if level == "SRAM":
            return bool(spatial_mapping.get("sram", False))
        return False

    if level == "PE":
        return bool(raw_tiling.get("spatial_mapping.pe", False))
    if level == "SRAM":
        return bool(raw_tiling.get("spatial_mapping.sram", False))
    return False



def _copy_template_scope(scope: PlacementTemplateScope) -> PlacementTemplateScope:
    return PlacementTemplateScope(
        scope_name=scope.scope_name,
        resident_sets=list(scope.resident_sets),
        boundary_actions=list(scope.boundary_actions),
        supported_phases=set(scope.supported_phases),
        supported_roles=set(scope.supported_roles),
        metadata=dict(scope.metadata),
    )



# =============================================================================
# Main candidate
# =============================================================================


@dataclass
class DSECandidate:
    """
    Unified runtime DSE candidate.

    This is the only materialized candidate object that downstream code should
    consume. It intentionally owns:
      - raw search-point payloads
      - normalized runtime objects (tiling_spec / tiling_gene / template_scope)
      - candidate identity / indices for ranking, logging, debugging
    """

    hardware_index: int = -1
    fusion_index: int = -1
    placement_index: int = -1
    tiling_index: int = -1

    hardware_config: Dict[str, Any] = field(default_factory=dict)

    fusion_gene: Optional[FusionGene] = None
    fusion_group: Optional[OpFusionGroup] = None

    template_name: str = ""
    template_scope: Optional[PlacementTemplateScope] = None

    tiling_spec: Optional[GroupTilingSpec] = None
    tiling_gene: Optional[TilingGene] = None

    raw_hardware: Dict[str, Any] = field(default_factory=dict)
    raw_fusion: Dict[str, Any] = field(default_factory=dict)
    raw_placement: Dict[str, Any] = field(default_factory=dict)
    raw_tiling: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def group_id(self) -> str:
        if self.fusion_group is None:
            return ""
        return self.fusion_group.group_id

    @property
    def phase(self) -> Optional[str]:
        if self.fusion_group is not None and self.fusion_group.phase is not None:
            return self.fusion_group.phase
        if self.tiling_spec is not None and self.tiling_spec.phase is not None:
            return self.tiling_spec.phase
        return None

    @property
    def special_role(self) -> Optional[str]:
        if self.fusion_group is not None and self.fusion_group.special_role is not None:
            return self.fusion_group.special_role
        if self.tiling_spec is not None and self.tiling_spec.special_role is not None:
            return self.tiling_spec.special_role
        return None

    def clone(self) -> "DSECandidate":
        return DSECandidate(
            hardware_index=self.hardware_index,
            fusion_index=self.fusion_index,
            placement_index=self.placement_index,
            tiling_index=self.tiling_index,
            hardware_config=_deep_copy_mapping(self.hardware_config),
            fusion_gene=self.fusion_gene,
            fusion_group=self.fusion_group,
            template_name=self.template_name,
            template_scope=None if self.template_scope is None else _copy_template_scope(self.template_scope),
            tiling_spec=None if self.tiling_spec is None else self.tiling_spec.clone(),
            tiling_gene=None if self.tiling_gene is None else self.tiling_gene.clone(),
            raw_hardware=_deep_copy_mapping(self.raw_hardware),
            raw_fusion=_deep_copy_mapping(self.raw_fusion),
            raw_placement=_deep_copy_mapping(self.raw_placement),
            raw_tiling=_deep_copy_mapping(self.raw_tiling),
            metadata=_deep_copy_mapping(self.metadata),
        )

    def to_tiling_gene(self, gene_id: Optional[str] = None) -> TilingGene:
        if self.tiling_gene is not None:
            cloned = self.tiling_gene.clone()
            if gene_id:
                cloned.gene_id = gene_id
            return cloned

        if self.tiling_spec is None:
            raise ValueError("DSECandidate has no tiling_spec / tiling_gene.")

        return TilingGene(
            gene_id=gene_id or self._default_tiling_gene_id(),
            group_tiles={self.tiling_spec.group_id: self.tiling_spec.clone()},
        )

    def to_debug_dict(self) -> Dict[str, Any]:
        tile_summary: Dict[str, Any] = {}
        if self.tiling_spec is not None:
            for lvl, spec in self.tiling_spec.tiles.items():
                tile_summary[lvl] = {
                    "tile_size": dict(spec.tile_size or {}),
                    "loop_order": list(spec.loop_order or []),
                    "is_spatial": bool(spec.is_spatial),
                }

        return {
            "hardware_index": self.hardware_index,
            "fusion_index": self.fusion_index,
            "placement_index": self.placement_index,
            "tiling_index": self.tiling_index,
            "group_id": self.group_id,
            "template_name": self.template_name,
            "phase": self.phase,
            "special_role": self.special_role,
            "hardware_config": _deep_copy_mapping(self.hardware_config),
            "tiles": tile_summary,
            "metadata": _deep_copy_mapping(self.metadata),
        }

    def _default_tiling_gene_id(self) -> str:
        gid = self.group_id or "unknown_group"
        return (
            f"tiling_hw{self.hardware_index}_f{self.fusion_index}"
            f"_p{self.placement_index}_t{self.tiling_index}_{gid}"
        )


# =============================================================================
# Materialization entrypoints
# =============================================================================


def build_hardware_config(raw_hw: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Candidate-side normalization for hardware payload.

    Runtime DRAM truth must come from an existing dram.cfg file referenced by
    ``dram_cfg_path``. candidate.py no longer materializes dram.cfg from search
    fields and no longer accepts dram_search as a fallback dependency.
    """
    hw = _expand_dotted_mapping(_deep_copy_mapping(raw_hw))

    dram_cfg_path = hw.get("dram_cfg_path")
    if not dram_cfg_path:
        raise ValueError(
            "hardware raw point must provide dram_cfg_path; "
            "candidate.py no longer derives it from dram_search or other DRAM fields."
        )
    if not Path(str(dram_cfg_path)).is_file():
        raise ValueError(f"dram_cfg_path does not point to an existing cfg file: {dram_cfg_path}")

    global_cfg = hw.pop("global", None)
    if isinstance(global_cfg, Mapping) and "bytes_per_element_default" in global_cfg and "bytes_per_element_default" not in hw:
        hw["bytes_per_element_default"] = int(global_cfg["bytes_per_element_default"])

    hw.pop("dram_search", None)
    return hw



def build_tiling_spec(
    raw_tiling: Mapping[str, Any],
    *,
    group_id: str,
    phase: Optional[str] = None,
    special_role: Optional[str] = None,
) -> GroupTilingSpec:
    hierarchy = list(raw_tiling.get("hierarchy") or ["DRAM", "SRAM", "PE"])
    raw_loop_order = raw_tiling.get("loop_order") or []
    loop_order = _normalize_loop_order(raw_loop_order)

    searched_dims = list(raw_tiling.get("searched_dims") or [])
    fixed_strategy = raw_tiling.get("fixed_strategy")
    if not isinstance(fixed_strategy, Mapping):
        fixed_strategy = {}

    tiles: Dict[str, MemTileSpec] = {}
    for level in hierarchy:
        upper = str(level).upper()
        tile_size = _extract_level_tile(raw_tiling, upper)
        is_spatial = _extract_spatial_flag(raw_tiling, upper)

        if not tile_size and upper == "DRAM":
            continue

        if loop_order and tile_size:
            local_loop_order = [dim for dim in loop_order if dim in tile_size]
            if not local_loop_order:
                local_loop_order = list(tile_size.keys())
        else:
            local_loop_order = list(tile_size.keys())

        tiles[upper] = MemTileSpec(
            mem_level=upper,
            tile_size=tile_size or None,
            loop_order=local_loop_order or None,
            is_spatial=is_spatial,
            active_dims=list(tile_size.keys()) if tile_size else None,
        )

    return GroupTilingSpec(
        group_id=group_id,
        tiles=tiles,
        phase=phase,
        special_role=special_role,
        metadata={
            "source": "design_space",
            "searched_dims": list(searched_dims),
            "fixed_strategy": dict(fixed_strategy),
            "raw_candidate": _deep_copy_mapping(raw_tiling),
        },
    )



def build_tiling_gene(
    raw_tiling: Mapping[str, Any],
    *,
    group_id: str,
    phase: Optional[str] = None,
    special_role: Optional[str] = None,
    gene_id: Optional[str] = None,
) -> TilingGene:
    spec = build_tiling_spec(
        raw_tiling,
        group_id=group_id,
        phase=phase,
        special_role=special_role,
    )
    return TilingGene(
        gene_id=gene_id or f"tiling_{group_id}",
        group_tiles={group_id: spec},
    )



def build_candidate(
    *,
    hardware_raw: Mapping[str, Any],
    fusion_gene: FusionGene,
    fusion_group: OpFusionGroup,
    template_name: str,
    template_scope: PlacementTemplateScope,
    tiling_raw: Mapping[str, Any],
    hardware_index: int = -1,
    fusion_index: int = -1,
    placement_index: int = -1,
    tiling_index: int = -1,
    fusion_raw: Optional[Mapping[str, Any]] = None,
    placement_raw: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> DSECandidate:
    hw_cfg = build_hardware_config(hardware_raw)

    tiling_spec = build_tiling_spec(
        tiling_raw,
        group_id=fusion_group.group_id,
        phase=fusion_group.phase,
        special_role=fusion_group.special_role,
    )

    tiling_gene = TilingGene(
        gene_id=(
            f"tiling_hw{hardware_index}_f{fusion_index}"
            f"_p{placement_index}_t{tiling_index}_{fusion_group.group_id}"
        ),
        group_tiles={fusion_group.group_id: tiling_spec.clone()},
    )

    return DSECandidate(
        hardware_index=hardware_index,
        fusion_index=fusion_index,
        placement_index=placement_index,
        tiling_index=tiling_index,
        hardware_config=hw_cfg,
        fusion_gene=fusion_gene,
        fusion_group=fusion_group,
        template_name=template_name,
        template_scope=_copy_template_scope(template_scope),
        tiling_spec=tiling_spec,
        tiling_gene=tiling_gene,
        raw_hardware=_deep_copy_mapping(hardware_raw),
        raw_fusion=_deep_copy_mapping(fusion_raw),
        raw_placement=_deep_copy_mapping(placement_raw),
        raw_tiling=_deep_copy_mapping(tiling_raw),
        metadata=_deep_copy_mapping(metadata),
    )


# =============================================================================
# Batch builders
# =============================================================================


def build_candidates_from_stage_a_subspace(
    *,
    hardware_raw: Mapping[str, Any],
    hardware_index: int,
    fusion_gene: FusionGene,
    fusion_index: int,
    placement_index_base: int,
    tiling_index_base: int,
    subspace: Any,
    design_space: Any,
) -> List[DSECandidate]:
    """
    Materialize final DSECandidate objects from the feasible Stage-A subspace.

    Contract:
      - Stage A is responsible for pruning and for deciding which tiling specs are
        feasible under each (group, template) key.
      - candidate.py is the only place that materializes final DSECandidate
        objects, but it must not silently regenerate tilings behind Stage A.

    Therefore, if a kept (group, template) key has no feasible tiling specs,
    this function raises instead of falling back to design_space expansion.
    """
    del design_space

    out: List[DSECandidate] = []
    group_index: Dict[str, OpFusionGroup] = {g.group_id: g for g in fusion_gene.groups}
    next_placement_index = placement_index_base
    next_tiling_index = tiling_index_base
    hw_cfg = build_hardware_config(hardware_raw)

    for group_id, template_names in subspace.placement_options_by_group.items():
        group = group_index.get(group_id)
        if group is None:
            continue

        for template_name in template_names:
            template_scope = subspace.placement_templates_by_name[template_name]
            feasible_specs = subspace.tiling_subspace_by_key.get((group_id, template_name), [])
            if not feasible_specs:
                raise ValueError(
                    "Stage A kept placement key without any feasible tiling candidates: "
                    f"group_id={group_id}, template_name={template_name}. "
                    "This violates the pipeline contract that groups/templates without tiling candidates "
                    "must be pruned before candidate materialization."
                )

            for local_idx, spec in enumerate(feasible_specs):
                cand = DSECandidate(
                    hardware_index=hardware_index,
                    fusion_index=fusion_index,
                    placement_index=next_placement_index,
                    tiling_index=next_tiling_index,
                    hardware_config=_deep_copy_mapping(hw_cfg),
                    fusion_gene=fusion_gene,
                    fusion_group=group,
                    template_name=template_name,
                    template_scope=_copy_template_scope(template_scope),
                    tiling_spec=spec.clone(),
                    tiling_gene=TilingGene(
                        gene_id=(
                            f"tiling_hw{hardware_index}_f{fusion_index}"
                            f"_p{next_placement_index}_t{next_tiling_index}_{group_id}"
                        ),
                        group_tiles={group_id: spec.clone()},
                    ),
                    raw_hardware=_deep_copy_mapping(hardware_raw),
                    raw_placement={"templates": template_name},
                    metadata={
                        "source": "stage_a",
                        "stage_a_local_tiling_index": local_idx,
                    },
                )
                out.append(cand)
                next_tiling_index += 1

            next_placement_index += 1

    return out
