from __future__ import annotations

"""
Stage A: hardware-aware feasible-subspace restriction for PIM-node DSE.

Responsibilities:
- accept a runtime hardware configuration or an already-built PIMNodeArchSpec
- prune infeasible fusion / placement / tiling choices
- output a feasible partial subspace for downstream enumeration

Non-responsibilities:
- raw search-point expansion
- candidate materialization
- DRAM cfg generation
- legacy HardwareGene compatibility / schema patching
"""

import math
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "hardware"))

from arch_spec import PIMNodeArchSpec, arch_from_mapping

from pimnode_dse.mapping.fusion_gene import FusionGene, OpFusionGroup
from pimnode_dse.mapping.tilling_gene import TilingGene, GroupTilingSpec, MemTileSpec
from pimnode_dse.placement.placement_ir import PlacementScope, PlacementTemplatePlan, PlacementTemplateScope


@dataclass(frozen=True)
class ConstraintResult:
    ok: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RejectedItem:
    kind: str  # "fusion" | "placement" | "tiling"
    key: Tuple[str, ...]
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeasibleSubspace:
    arch: PIMNodeArchSpec
    fusion_options: List[OpFusionGroup] = field(default_factory=list)
    placement_options_by_group: Dict[str, List[str]] = field(default_factory=dict)
    placement_templates_by_name: Dict[str, PlacementTemplateScope] = field(default_factory=dict)
    tiling_subspace_by_key: Dict[Tuple[str, str], List[GroupTilingSpec]] = field(default_factory=dict)
    rejected: List[RejectedItem] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageAConfig:
    bytes_per_element_default: Optional[int] = None
    allow_empty_tiling_for_group: bool = True
    min_required_tiling_candidates_per_key: int = 0
    conservative_full_tensor_if_unknown: bool = True
    prefer_phase_role_overlay: bool = True
    max_tiling_candidates_per_key: Optional[int] = None


class StageASpaceRestriction:
    def __init__(self, arch: Any, config: Optional[StageAConfig] = None):
        self.arch = self._normalize_arch(arch)
        self.cfg = config or StageAConfig()

    def run(
        self,
        workload: Any,
        fusion_options: FusionGene | Sequence[OpFusionGroup],
        placement_templates: PlacementTemplatePlan,
        tiling_gene: Optional[TilingGene] = None,
    ) -> FeasibleSubspace:
        groups = self._normalize_fusion_options(fusion_options)
        out = FeasibleSubspace(arch=self.arch)

        template_scopes = dict(getattr(placement_templates, "scopes", {}) or {})
        out.diagnostics["total_groups_in"] = len(groups)
        out.diagnostics["total_templates_in"] = len(template_scopes)

        accepted_groups = 0
        kept_placements = 0
        kept_tilings = 0

        for template_name, template_scope in template_scopes.items():
            out.placement_templates_by_name[template_name] = self._copy_template_scope(template_scope)

        for group in groups:
            gshape = self.check_basic_shape_feasibility(workload, group)
            if not gshape.ok:
                out.rejected.append(
                    RejectedItem(
                        kind="fusion",
                        key=(group.group_id,),
                        reason=gshape.reason,
                        details=gshape.details,
                    )
                )
                continue

            out.fusion_options.append(group)
            accepted_groups += 1
            feasible_templates: List[str] = []

            for template_name, template_scope in template_scopes.items():
                pres = self.check_basic_shape_feasibility(
                    workload=workload,
                    group=group,
                    placement_scope=template_scope,
                    template_name=template_name,
                )
                if not pres.ok:
                    out.rejected.append(
                        RejectedItem(
                            kind="placement",
                            key=(group.group_id, template_name),
                            reason=pres.reason,
                            details=pres.details,
                        )
                    )
                    continue

                cres = self.check_sram_capacity(
                    workload=workload,
                    group=group,
                    placement_scope=template_scope,
                    template_name=template_name,
                )
                if not cres.ok:
                    out.rejected.append(
                        RejectedItem(
                            kind="placement",
                            key=(group.group_id, template_name),
                            reason=cres.reason,
                            details=cres.details,
                        )
                    )
                    continue

                feasible_templates.append(template_name)
                kept_placements += 1

                tiling_candidates = self._get_tiling_candidates_for_group(tiling_gene, group)
                if not tiling_candidates:
                    if self.cfg.allow_empty_tiling_for_group:
                        out.tiling_subspace_by_key[(group.group_id, template_name)] = []
                    elif self.cfg.min_required_tiling_candidates_per_key > 0:
                        out.rejected.append(
                            RejectedItem(
                                kind="tiling",
                                key=(group.group_id, template_name),
                                reason="no_tiling_candidates_for_group",
                                details={"group_id": group.group_id, "template_name": template_name},
                            )
                        )
                    continue

                accepted_tilings: List[GroupTilingSpec] = []
                for spec in tiling_candidates:
                    tres0 = self.check_basic_shape_feasibility(
                        workload=workload,
                        group=group,
                        placement_scope=template_scope,
                        tiling_spec=spec,
                        template_name=template_name,
                    )
                    if not tres0.ok:
                        out.rejected.append(
                            RejectedItem(
                                kind="tiling",
                                key=(group.group_id, template_name, self._tiling_key(spec)),
                                reason=tres0.reason,
                                details=tres0.details,
                            )
                        )
                        continue

                    tres1 = self.check_sram_capacity(
                        workload=workload,
                        group=group,
                        placement_scope=template_scope,
                        tiling_spec=spec,
                        template_name=template_name,
                    )
                    if not tres1.ok:
                        out.rejected.append(
                            RejectedItem(
                                kind="tiling",
                                key=(group.group_id, template_name, self._tiling_key(spec)),
                                reason=tres1.reason,
                                details=tres1.details,
                            )
                        )
                        continue

                    tres2 = self.check_bandwidth_upper_bound(
                        workload=workload,
                        group=group,
                        placement_scope=template_scope,
                        tiling_spec=spec,
                        template_name=template_name,
                    )
                    if not tres2.ok:
                        out.rejected.append(
                            RejectedItem(
                                kind="tiling",
                                key=(group.group_id, template_name, self._tiling_key(spec)),
                                reason=tres2.reason,
                                details=tres2.details,
                            )
                        )
                        continue

                    accepted_tilings.append(spec)
                    kept_tilings += 1
                    if (
                        self.cfg.max_tiling_candidates_per_key is not None
                        and len(accepted_tilings) >= self.cfg.max_tiling_candidates_per_key
                    ):
                        break

                if accepted_tilings or self.cfg.allow_empty_tiling_for_group:
                    out.tiling_subspace_by_key[(group.group_id, template_name)] = accepted_tilings
                elif len(accepted_tilings) < self.cfg.min_required_tiling_candidates_per_key:
                    out.rejected.append(
                        RejectedItem(
                            kind="tiling",
                            key=(group.group_id, template_name),
                            reason="insufficient_tiling_candidates_after_filtering",
                            details={
                                "group_id": group.group_id,
                                "template_name": template_name,
                                "accepted_count": len(accepted_tilings),
                                "required_min": self.cfg.min_required_tiling_candidates_per_key,
                            },
                        )
                    )

            out.placement_options_by_group[group.group_id] = feasible_templates

        out.diagnostics.update(
            {
                "groups_kept": accepted_groups,
                "placements_kept": kept_placements,
                "tilings_kept": kept_tilings,
                "groups_rejected": len([r for r in out.rejected if r.kind == "fusion"]),
                "placements_rejected": len([r for r in out.rejected if r.kind == "placement"]),
                "tilings_rejected": len([r for r in out.rejected if r.kind == "tiling"]),
            }
        )
        return out

    def check_basic_shape_feasibility(
        self,
        workload: Any,
        group: OpFusionGroup,
        placement_scope: Optional[Any] = None,
        tiling_spec: Optional[GroupTilingSpec] = None,
        template_name: Optional[str] = None,
    ) -> ConstraintResult:
        if not group.op_names:
            return ConstraintResult(False, "empty_group", {"group_id": group.group_id})

        workload_op_names = self._workload_op_names(workload)
        missing_ops = [op for op in group.op_names if op not in workload_op_names]
        if missing_ops:
            return ConstraintResult(False, "group_contains_unknown_ops", {"group_id": group.group_id, "missing_ops": missing_ops})

        if placement_scope is not None:
            unknown_tensors: List[str] = []
            for t in self._collect_scope_tensors(placement_scope):
                if self._is_known_template_placeholder(t):
                    continue
                if not self._tensor_exists(workload, t):
                    unknown_tensors.append(t)
            if unknown_tensors:
                return ConstraintResult(
                    False,
                    "placement_references_unknown_tensors",
                    {
                        "group_id": group.group_id,
                        "template_name": template_name or placement_scope.scope_name,
                        "scope_name": placement_scope.scope_name,
                        "unknown_tensors": sorted(set(unknown_tensors)),
                    },
                )

        if tiling_spec is not None:
            problem_dims = self._problem_dims(workload)
            invalid_levels: Dict[str, Any] = {}
            for level, mspec in (tiling_spec.tiles or {}).items():
                tile_size = dict(getattr(mspec, "tile_size", {}) or {})
                loop_order = list(getattr(mspec, "loop_order", []) or [])
                if any(v <= 0 for v in tile_size.values()):
                    invalid_levels[level] = {"reason": "non_positive_tile_size", "tile_size": tile_size}
                    continue
                if loop_order:
                    missing_in_loop = sorted(set(tile_size.keys()) - set(loop_order))
                    if missing_in_loop:
                        invalid_levels[level] = {
                            "reason": "tile_dims_missing_from_loop_order",
                            "tile_size": tile_size,
                            "loop_order": loop_order,
                            "missing_dims": missing_in_loop,
                        }
                        continue
                too_large: Dict[str, Tuple[int, int]] = {}
                for dim, size in tile_size.items():
                    if dim in problem_dims and size > problem_dims[dim]:
                        too_large[dim] = (size, problem_dims[dim])
                if too_large:
                    invalid_levels[level] = {
                        "reason": "tile_size_exceeds_problem_extent",
                        "offending_dims": too_large,
                    }
            if invalid_levels:
                return ConstraintResult(
                    False,
                    "invalid_tiling_shape",
                    {"group_id": group.group_id, "template_name": template_name, "invalid_levels": invalid_levels},
                )

        return ConstraintResult(True)

    def check_sram_capacity(
        self,
        workload: Any,
        group: OpFusionGroup,
        placement_scope: Any,
        tiling_spec: Optional[GroupTilingSpec] = None,
        template_name: Optional[str] = None,
    ) -> ConstraintResult:
        required_bytes, breakdown = self._estimate_required_sram_bytes(
            workload=workload,
            group=group,
            placement_scope=placement_scope,
            tiling_spec=tiling_spec,
        )
        capacity = self.arch.sram.total_capacity_bytes
        ok = required_bytes <= capacity
        return ConstraintResult(
            ok=ok,
            reason="" if ok else "sram_capacity_exceeded",
            details={
                "group_id": group.group_id,
                "template_name": template_name or placement_scope.scope_name,
                "scope_name": placement_scope.scope_name,
                "required_bytes": required_bytes,
                "capacity_bytes": capacity,
                "breakdown": breakdown,
            },
        )

    def check_bandwidth_upper_bound(
        self,
        workload: Any,
        group: OpFusionGroup,
        placement_scope: Any,
        tiling_spec: GroupTilingSpec,
        template_name: Optional[str] = None,
    ) -> ConstraintResult:
        req = self._estimate_bandwidth_requirements(workload, group, placement_scope, tiling_spec)

        reasons: List[str] = []
        avail = {
            "dram_total_bandwidth_bpc": self.arch.dram.total_bandwidth_bytes_per_cycle,
            "sram_total_bandwidth_bpc": self.arch.sram.total_bandwidth_bytes_per_cycle,
            "de_bus_bandwidth_bpc": self.arch.de.bus_bandwidth_bytes_per_cycle,
            "pe_input_bandwidth_bpc": self.arch.pe.input_bandwidth_bytes_per_cycle,
            "pe_output_bandwidth_bpc": self.arch.pe.output_bandwidth_bytes_per_cycle,
            "dram_bank_count": self.arch.dram.bank_count,
        }

        if req["required_dram_bpc"] > avail["dram_total_bandwidth_bpc"]:
            reasons.append("dram_bandwidth_exceeded")
        if req["required_dram_parallel_streams"] > avail["dram_bank_count"]:
            reasons.append("dram_parallelism_exceeded")
        if req["required_sram_bpc"] > avail["sram_total_bandwidth_bpc"]:
            reasons.append("sram_bandwidth_exceeded")
        if req["required_de_bpc"] > avail["de_bus_bandwidth_bpc"]:
            reasons.append("de_bus_bandwidth_exceeded")

        pe_in_avail = avail["pe_input_bandwidth_bpc"]
        if pe_in_avail is not None and req["required_pe_input_bpc"] > pe_in_avail:
            reasons.append("pe_input_bandwidth_exceeded")

        pe_out_avail = avail["pe_output_bandwidth_bpc"]
        if pe_out_avail is not None and req["required_pe_output_bpc"] > pe_out_avail:
            reasons.append("pe_output_bandwidth_exceeded")

        ok = not reasons
        return ConstraintResult(
            ok=ok,
            reason="" if ok else "+".join(reasons),
            details={
                "group_id": group.group_id,
                "template_name": template_name or placement_scope.scope_name,
                "scope_name": placement_scope.scope_name,
                **req,
                **avail,
            },
        )

    def _normalize_arch(self, arch: Any) -> PIMNodeArchSpec:
        if isinstance(arch, PIMNodeArchSpec):
            return arch
        if isinstance(arch, Mapping):
            return arch_from_mapping(dict(arch))
        hardware_config = getattr(arch, "hardware_config", None)
        if isinstance(hardware_config, Mapping):
            return arch_from_mapping(dict(hardware_config))
        raise TypeError("StageA requires a PIMNodeArchSpec, a hardware_config mapping, or an object exposing .hardware_config")

    def _normalize_fusion_options(self, fusion_options: FusionGene | Sequence[OpFusionGroup]) -> List[OpFusionGroup]:
        if isinstance(fusion_options, FusionGene):
            return list(fusion_options.groups)
        return list(fusion_options)

    def _copy_template_scope(self, scope: PlacementTemplateScope) -> PlacementTemplateScope:
        return copy.deepcopy(scope)

    def _get_tiling_candidates_for_group(
        self,
        tiling_gene: Optional[TilingGene],
        group: OpFusionGroup,
    ) -> List[GroupTilingSpec]:
        if tiling_gene is None:
            return []
        specs: List[GroupTilingSpec] = []
        if self.cfg.prefer_phase_role_overlay:
            spec = tiling_gene.get_group_spec(group.group_id, phase=group.phase, role=group.special_role)
            if spec is not None:
                specs.append(spec)
        fallback = tiling_gene.get_group_spec(group.group_id)
        if fallback is not None and all(self._tiling_key(s) != self._tiling_key(fallback) for s in specs):
            specs.append(fallback)
        return specs

    def _tiling_key(self, spec: GroupTilingSpec) -> str:
        parts = [spec.group_id, spec.phase or "", spec.special_role or ""]
        for lvl in sorted(spec.tiles.keys()):
            ms = spec.tiles[lvl]
            parts.append(
                f"{lvl}:{sorted((getattr(ms, 'tile_size', {}) or {}).items())}:"
                f"{getattr(ms, 'loop_order', None)}:{getattr(ms, 'is_spatial', None)}"
            )
        return "|".join(parts)

    def _collect_scope_tensors(self, scope: Any) -> List[str]:
        out: List[str] = []
        for rs in getattr(scope, "resident_sets", []) or []:
            out.extend(sorted(self._resident_items(rs)))
        for ba in getattr(scope, "boundary_actions", []) or []:
            out.extend(sorted(self._boundary_prefetch_items(ba)))
            out.extend(sorted(self._boundary_writeback_items(ba)))
            out.extend(sorted(self._boundary_evict_items(ba)))
        return out

    def _is_known_template_placeholder(self, tensor_name: str) -> bool:
        lname = tensor_name.lower()
        return lname in {
            "q_tile", "k_tile", "v_tile", "h_tile", "o_tile", "output_tile",
            "current_tile", "other_tensor", "q", "k", "v", "h", "o",
            "scores", "stats", "probs", "partial_o", "kv_cache",
        }

    def _mem_name(self, obj: Any) -> str:
        return str(getattr(obj, "mem", "")).upper()

    def _resident_items(self, rs: Any) -> set[str]:
        vals = getattr(rs, "tensors", None)
        if vals is None:
            vals = getattr(rs, "tensor_roles", None)
        return set(vals or ())

    def _boundary_prefetch_items(self, ba: Any) -> set[str]:
        vals = getattr(ba, "prefetch", None)
        if vals is None:
            vals = getattr(ba, "prefetch_roles", None)
        return set(vals or ())

    def _boundary_writeback_items(self, ba: Any) -> set[str]:
        vals = getattr(ba, "writeback", None)
        if vals is None:
            vals = getattr(ba, "writeback_roles", None)
        return set(vals or ())

    def _boundary_evict_items(self, ba: Any) -> set[str]:
        vals = getattr(ba, "evict", None)
        if vals is None:
            vals = getattr(ba, "evict_roles", None)
        return set(vals or ())

    def _estimate_required_sram_bytes(
        self,
        workload: Any,
        group: OpFusionGroup,
        placement_scope: PlacementScope,
        tiling_spec: Optional[GroupTilingSpec],
    ) -> Tuple[int, Dict[str, int]]:
        resident_sram = set()
        for rs in getattr(placement_scope, "resident_sets", []) or []:
            if self._mem_name(rs) == "SRAM":
                resident_sram |= self._resident_items(rs)

        transient_sram = set()
        for ba in getattr(placement_scope, "boundary_actions", []) or []:
            if self._mem_name(ba) == "SRAM":
                transient_sram |= self._boundary_prefetch_items(ba) | self._boundary_writeback_items(ba)

        breakdown: Dict[str, int] = {}
        total = 0
        for t in sorted(resident_sram):
            nbytes = self._estimate_tensor_or_tile_bytes(workload, t, tiling_spec)
            breakdown[f"resident:{t}"] = nbytes
            total += nbytes
        for t in sorted(transient_sram - resident_sram):
            nbytes = self._estimate_tensor_or_tile_bytes(workload, t, tiling_spec)
            half = max(1, nbytes // 2)
            breakdown[f"transient:{t}"] = half
            total += half
        return total, breakdown

    def _estimate_bandwidth_requirements(
        self,
        workload: Any,
        group: OpFusionGroup,
        placement_scope: Any,
        tiling_spec: GroupTilingSpec,
    ) -> Dict[str, Any]:
        bytes_per_elem = self._bytes_per_element(workload)
        inner = tiling_spec.tiles.get("PE") or tiling_spec.tiles.get("SRAM")
        tile_work = self._estimate_tile_work(workload, group, inner)
        peak_compute = max(1, self.arch.pe.total_mac_per_cycle)
        estimated_tile_cycles = max(1, math.ceil(tile_work / peak_compute))

        required_pe_input_bpc = 2.0 * bytes_per_elem * peak_compute
        required_pe_output_bpc = 1.0 * bytes_per_elem * peak_compute
        required_de_bpc = max(required_pe_input_bpc, required_pe_output_bpc)
        required_sram_bpc = max(required_pe_input_bpc, required_pe_output_bpc)

        dram_source_tensors = set()
        sram_resident = set()
        for rs in getattr(placement_scope, "resident_sets", []) or []:
            if self._mem_name(rs) == "SRAM":
                sram_resident |= self._resident_items(rs)
            if self._mem_name(rs) == "DRAM":
                dram_source_tensors |= self._resident_items(rs)
        for ba in getattr(placement_scope, "boundary_actions", []) or []:
            if self._mem_name(ba) == "SRAM":
                dram_source_tensors |= self._boundary_prefetch_items(ba)

        dram_bytes_per_tile = 0
        for t in dram_source_tensors - sram_resident:
            dram_bytes_per_tile += self._estimate_tensor_or_tile_bytes(workload, t, tiling_spec)
        required_dram_bpc = dram_bytes_per_tile / float(estimated_tile_cycles)
        required_dram_parallel_streams = max(1, min(len(dram_source_tensors - sram_resident), 8))

        return {
            "estimated_tile_work": tile_work,
            "estimated_tile_cycles": estimated_tile_cycles,
            "required_pe_input_bpc": required_pe_input_bpc,
            "required_pe_output_bpc": required_pe_output_bpc,
            "required_de_bpc": required_de_bpc,
            "required_sram_bpc": required_sram_bpc,
            "required_dram_bpc": required_dram_bpc,
            "required_dram_parallel_streams": required_dram_parallel_streams,
            "bytes_per_element": bytes_per_elem,
        }

    def _estimate_tile_work(self, workload: Any, group: OpFusionGroup, inner: Optional[MemTileSpec]) -> int:
        spec = getattr(workload, "spec", None)
        if spec is not None and hasattr(spec, "Dh"):
            tdims = dict((getattr(inner, "tile_size", {}) if inner is not None else {}) or {})
            m = tdims.get("m", tdims.get("M", getattr(spec, "Q_len", 1)))
            n = tdims.get("n", tdims.get("N", getattr(spec, "KV_total", getattr(spec, "KV_len", 1))))
            k = tdims.get("k", tdims.get("K", getattr(spec, "Dh", 1)))
            op_factor = max(1, len(group.op_names))
            return int(max(1, m) * max(1, n) * max(1, k) * op_factor)
        tdims = dict((getattr(inner, "tile_size", {}) if inner is not None else {}) or {})
        volume = 1
        for v in tdims.values():
            volume *= max(1, int(v))
        return max(1, volume * max(1, len(group.op_names)))

    def _estimate_tensor_or_tile_bytes(self, workload: Any, tensor_name: str, tiling_spec: Optional[GroupTilingSpec]) -> int:
        if self._tensor_exists(workload, tensor_name):
            return self._tensor_nbytes(workload, tensor_name)

        problem = self._problem_dims(workload)
        bytes_per_elem = self._bytes_per_element(workload)
        tile_fraction = self._tile_fraction(workload, tiling_spec)
        lname = tensor_name.lower()

        if lname in {"q_tile", "q"}:
            full = self._attention_tensor_bytes(problem, bytes_per_elem, kind="Q")
            return max(1, math.ceil(full * tile_fraction.get("Q", 1.0)))
        if lname in {"k_tile", "k"}:
            full = self._attention_tensor_bytes(problem, bytes_per_elem, kind="K")
            return max(1, math.ceil(full * tile_fraction.get("K", 1.0)))
        if lname in {"v_tile", "v"}:
            full = self._attention_tensor_bytes(problem, bytes_per_elem, kind="V")
            return max(1, math.ceil(full * tile_fraction.get("V", 1.0)))
        if lname in {"h_tile", "o_tile", "output_tile", "h", "o", "partial_o"}:
            full = self._attention_tensor_bytes(problem, bytes_per_elem, kind="O")
            return max(1, math.ceil(full * tile_fraction.get("O", 1.0)))
        if lname in {"stats", "scores", "probs", "kv_cache"}:
            return max(1, bytes_per_elem)
        if lname == "current_tile":
            pe_spec = None if tiling_spec is None else (tiling_spec.tiles.get("PE") or tiling_spec.tiles.get("SRAM"))
            pe_volume = 1
            if pe_spec is not None:
                for v in (getattr(pe_spec, "tile_size", {}) or {}).values():
                    pe_volume *= max(1, int(v))
                return max(1, pe_volume * bytes_per_elem)
            return max(1, problem.get("Dh", 1) * bytes_per_elem)
        if lname == "other_tensor":
            return max(1, bytes_per_elem)

        alias = self._resolve_tensor_alias(workload, tensor_name)
        if alias is not None:
            base = self._tensor_nbytes(workload, alias)
            return max(1, math.ceil(base * tile_fraction.get("GENERIC", 1.0)))

        if self.cfg.conservative_full_tensor_if_unknown:
            full_sum = sum(self._tensor_nbytes(workload, name) for name in self._workload_tensor_names(workload))
            return max(bytes_per_elem, full_sum)
        return max(1, bytes_per_elem)

    def _tile_fraction(self, workload: Any, tiling_spec: Optional[GroupTilingSpec]) -> Dict[str, float]:
        problem = self._problem_dims(workload)
        out = {"Q": 1.0, "K": 1.0, "V": 1.0, "O": 1.0, "GENERIC": 1.0}
        if tiling_spec is None:
            return out
        ms = tiling_spec.tiles.get("SRAM") or tiling_spec.tiles.get("PE")
        if ms is None:
            return out

        tile = dict(getattr(ms, "tile_size", {}) or {})
        q_frac = 1.0
        k_frac = 1.0
        for dim_key, global_key in (("b", "B"), ("m", "Q_len"), ("n", "KV_total"), ("k", "Dh"), ("h", "Hq"), ("g", "Hkv")):
            if dim_key in tile and global_key in problem and problem[global_key] > 0:
                frac = min(1.0, float(tile[dim_key]) / float(problem[global_key]))
                if dim_key in {"b", "m", "k", "h"}:
                    q_frac *= frac
                if dim_key in {"b", "n", "k", "g"}:
                    k_frac *= frac
        out["Q"] = max(1e-9, q_frac)
        out["K"] = max(1e-9, k_frac)
        out["V"] = max(1e-9, k_frac)
        out["O"] = max(1e-9, q_frac)
        out["GENERIC"] = max(1e-9, min(q_frac, k_frac))
        return out

    def _attention_tensor_bytes(self, problem: Mapping[str, int], bytes_per_elem: int, kind: str) -> int:
        B = int(problem.get("B", 1))
        Q_len = int(problem.get("Q_len", 1))
        KV_total = int(problem.get("KV_total", problem.get("KV_len", 1)))
        Dh = int(problem.get("Dh", 1))
        Hq = int(problem.get("Hq", 1))
        Hkv = int(problem.get("Hkv", Hq))
        if kind == "Q":
            elems = B * Q_len * Hq * Dh
        elif kind == "K":
            elems = B * KV_total * Hkv * Dh
        elif kind == "V":
            elems = B * KV_total * Hkv * Dh
        else:
            elems = B * Q_len * Hq * Dh
        return max(1, elems * bytes_per_elem)

    def _problem_dims(self, workload: Any) -> Dict[str, int]:
        spec = getattr(workload, "spec", None)
        if spec is None:
            return {}
        out: Dict[str, int] = {}
        for key in ("B", "Q_len", "KV_len", "Dh", "Hq", "Hkv"):
            if hasattr(spec, key):
                out[key] = int(getattr(spec, key))
        if hasattr(spec, "KV_total"):
            try:
                out["KV_total"] = int(spec.KV_total)
            except Exception:
                pass
        if "KV_total" not in out and "KV_len" in out:
            out["KV_total"] = out["KV_len"]
        return out

    def _bytes_per_element(self, workload: Any) -> int:
        spec = getattr(workload, "spec", None)
        if spec is not None and hasattr(spec, "dtype_bytes"):
            try:
                val = int(getattr(spec, "dtype_bytes"))
                if val > 0:
                    return val
            except Exception:
                pass
        if self.cfg.bytes_per_element_default is not None:
            return self.cfg.bytes_per_element_default
        return int(self.arch.bytes_per_element_default)

    def _workload_op_names(self, workload: Any) -> set[str]:
        names = getattr(workload, "op_names", None)
        if callable(names):
            return set(names())
        ops = getattr(workload, "ops", [])
        out = set()
        for op in ops:
            op_id = getattr(op, "op_id", None)
            if op_id:
                out.add(op_id)
        return out

    def _workload_tensor_names(self, workload: Any) -> List[str]:
        tensors = getattr(workload, "tensors", {})
        if isinstance(tensors, Mapping):
            return list(tensors.keys())
        return []

    def _tensor_exists(self, workload: Any, tensor_name: str) -> bool:
        tensors = getattr(workload, "tensors", {})
        return isinstance(tensors, Mapping) and tensor_name in tensors

    def _tensor_nbytes(self, workload: Any, tensor_name: str) -> int:
        tensors = getattr(workload, "tensors", {})
        if not isinstance(tensors, Mapping) or tensor_name not in tensors:
            raise KeyError(f"Unknown tensor: {tensor_name}")
        ts = tensors[tensor_name]
        shape = tuple(getattr(ts, "shape", ()) or ())
        if not shape:
            return self._bytes_per_element(workload)
        elems = 1
        for d in shape:
            elems *= max(1, int(d))
        dtype = getattr(ts, "dtype", None)
        dtype_bytes = self._dtype_to_bytes(dtype) or self._bytes_per_element(workload)
        return elems * dtype_bytes

    def _dtype_to_bytes(self, dtype: Optional[str]) -> Optional[int]:
        if not dtype:
            return None
        d = dtype.lower()
        if d in {"float16", "fp16", "half", "bfloat16", "bf16", "int16", "uint16"}:
            return 2
        if d in {"float32", "fp32", "int32", "uint32"}:
            return 4
        if d in {"float64", "fp64", "int64", "uint64"}:
            return 8
        if d in {"int8", "uint8", "byte"}:
            return 1
        return None

    def _resolve_tensor_alias(self, workload: Any, tensor_name: str) -> Optional[str]:
        lname = tensor_name.lower()
        tensors = self._workload_tensor_names(workload)
        for t in tensors:
            if t.lower() == lname:
                return t
        prefix_map = {
            "q_tile": ["q"],
            "k_tile": ["k"],
            "v_tile": ["v"],
            "h_tile": ["h", "o", "out"],
        }
        for key, prefixes in prefix_map.items():
            if lname == key:
                matches = [t for t in tensors if any(t.lower().startswith(p) for p in prefixes)]
                if len(matches) == 1:
                    return matches[0]
        return None


__all__ = [
    "ConstraintResult",
    "RejectedItem",
    "FeasibleSubspace",
    "StageAConfig",
    "StageASpaceRestriction",
]
