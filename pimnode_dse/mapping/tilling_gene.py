from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple
import copy
import hashlib
import json
import math


MemLevel = Literal["DRAM", "SRAM", "RF", "PE"]
OpScheduleStyle = Literal["sequential", "micro_pipeline"]


@dataclass
class GlobalTilingPolicy:
    """Global knobs for tiling search / interpretation.

    This object intentionally stays lightweight. Placement-specific semantics
    such as residency, writeback and lifetime must not be stored here.
    """

    prefer_spatial_levels: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemTileSpec:
    """Tiling description for a single memory level.

    Upgraded semantics:
    - tile_size:    local tile extent processed per iteration
    - loop_order:   loop nesting / traversal order for this level
    - problem_size: full visible problem size at this level
    - loop_count:   actual loop count for this level

    Backward compatibility:
    - old code may still construct with:
        fixed_tile_size=...
        fixed_loop_order=...
    - existing builder code can still call get_tile_spec() and receive
      (tile_size, loop_order)
    - runtime_tile_fn may return:
        1) (tile_size, loop_order)
        2) (tile_size, loop_order, problem_size, loop_count)
        3) {
             "tile_size": ...,
             "loop_order": ...,
             "problem_size": ...,
             "loop_count": ...
           }
    """

    mem_level: str

    # New canonical fields
    tile_size: Optional[Dict[str, int]] = None
    loop_order: Optional[List[str]] = None
    problem_size: Optional[Dict[str, int]] = None
    loop_count: Optional[Dict[str, int]] = None

    is_spatial: bool = False
    active_dims: Optional[List[str]] = None
    runtime_tile_fn: Optional[Callable[[Dict[str, Any]], Any]] = None

    # Legacy compatibility fields
    fixed_tile_size: Optional[Dict[str, int]] = None
    fixed_loop_order: Optional[List[str]] = None

    def __post_init__(self) -> None:
        # Normalize legacy -> canonical
        if self.tile_size is None and self.fixed_tile_size is not None:
            self.tile_size = dict(self.fixed_tile_size)
        if self.loop_order is None and self.fixed_loop_order is not None:
            self.loop_order = list(self.fixed_loop_order)

        # Keep legacy fields mirrored for older callers that may inspect attrs
        if self.fixed_tile_size is None and self.tile_size is not None:
            self.fixed_tile_size = dict(self.tile_size)
        if self.fixed_loop_order is None and self.loop_order is not None:
            self.fixed_loop_order = list(self.loop_order)

    def get_tile_desc(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = context or {}

        if self.runtime_tile_fn is not None:
            runtime_out = self.runtime_tile_fn(ctx)
            return self._normalize_runtime_output(runtime_out)

        tile_size = dict(self.tile_size or {})
        loop_order = list(self.loop_order or tile_size.keys())
        problem_size = dict(self.problem_size or {})
        loop_count = dict(self.loop_count or {})

        if not loop_count and problem_size and tile_size:
            loop_count = self._derive_loop_count(problem_size, tile_size)

        return {
            "tile_size": tile_size,
            "loop_order": loop_order,
            "problem_size": problem_size,
            "loop_count": loop_count,
        }

    def get_tile_spec(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, int], List[str]]:
        """Backward-compatible API used by current builder code."""
        desc = self.get_tile_desc(context)
        return dict(desc.get("tile_size", {})), list(desc.get("loop_order", []))

    def get_problem_size(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        return dict(self.get_tile_desc(context).get("problem_size", {}))

    def get_loop_count(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        return dict(self.get_tile_desc(context).get("loop_count", {}))

    def clone(self) -> "MemTileSpec":
        return MemTileSpec(
            mem_level=self.mem_level,
            tile_size=dict(self.tile_size or {}) or None,
            loop_order=list(self.loop_order or []) or None,
            problem_size=dict(self.problem_size or {}) or None,
            loop_count=dict(self.loop_count or {}) or None,
            is_spatial=bool(self.is_spatial),
            active_dims=list(self.active_dims) if self.active_dims else None,
            runtime_tile_fn=self.runtime_tile_fn,
            fixed_tile_size=dict(self.tile_size or {}) or None,
            fixed_loop_order=list(self.loop_order or []) or None,
        )

    @staticmethod
    def _derive_loop_count(
        problem_size: Mapping[str, int],
        tile_size: Mapping[str, int],
    ) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for dim, total in problem_size.items():
            t = int(tile_size.get(dim, total))
            total_i = int(total)
            if t <= 0:
                raise ValueError(f"tile_size[{dim!r}] must be > 0, got {t}")
            out[dim] = int(math.ceil(total_i / t))
        return out

    def _normalize_runtime_output(self, runtime_out: Any) -> Dict[str, Any]:
        if isinstance(runtime_out, dict):
            tile_size = dict(runtime_out.get("tile_size", {}) or {})
            loop_order = list(runtime_out.get("loop_order", []) or tile_size.keys())
            problem_size = dict(runtime_out.get("problem_size", {}) or {})
            loop_count = dict(runtime_out.get("loop_count", {}) or {})
            if not loop_count and problem_size and tile_size:
                loop_count = self._derive_loop_count(problem_size, tile_size)
            return {
                "tile_size": tile_size,
                "loop_order": loop_order,
                "problem_size": problem_size,
                "loop_count": loop_count,
            }

        if isinstance(runtime_out, tuple) and len(runtime_out) == 2:
            tile_size, loop_order = runtime_out
            return {
                "tile_size": dict(tile_size or {}),
                "loop_order": list(loop_order or []),
                "problem_size": {},
                "loop_count": {},
            }

        if isinstance(runtime_out, tuple) and len(runtime_out) == 4:
            tile_size, loop_order, problem_size, loop_count = runtime_out
            tile_size = dict(tile_size or {})
            loop_order = list(loop_order or tile_size.keys())
            problem_size = dict(problem_size or {})
            loop_count = dict(loop_count or {})
            if not loop_count and problem_size and tile_size:
                loop_count = self._derive_loop_count(problem_size, tile_size)
            return {
                "tile_size": tile_size,
                "loop_order": loop_order,
                "problem_size": problem_size,
                "loop_count": loop_count,
            }

        raise TypeError(
            "runtime_tile_fn must return either "
            "(tile_size, loop_order), "
            "(tile_size, loop_order, problem_size, loop_count), "
            "or a dict with keys tile_size/loop_order/problem_size/loop_count"
        )


@dataclass
class GroupTilingSpec:
    """Pure tiling spec for one fusion group.

    Aligned with placement-first architecture:
    - keeps only tiling/scheduling information
    - does NOT encode residency/writeback/lifetime/storage policy
    - optional phase/special_role are descriptive metadata or lookup keys

    Suggested metadata usage for current short-term plan:
    metadata = {
        "searched_dims": ["Sq", "Skv", "Dh"],
        "fixed_dim_policy": {
            "B": "no_tile",
            "Hq": "full",
            "Hkv": "full",
        }
    }
    """

    group_id: str
    tiles: Dict[str, MemTileSpec] = field(default_factory=dict)
    op_schedule_style: OpScheduleStyle = "sequential"
    sram_budget_bytes: Optional[int] = None
    phase: Optional[str] = None
    special_role: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_mem_tile_spec(self, mem_level: str) -> MemTileSpec:
        if mem_level not in self.tiles:
            raise KeyError(f"MemTileSpec not found for level {mem_level}")
        return self.tiles[mem_level]

    def set_mem_tile_spec(self, mem_tile_spec: MemTileSpec) -> None:
        self.tiles[mem_tile_spec.mem_level] = mem_tile_spec

    def clone(self) -> "GroupTilingSpec":
        return GroupTilingSpec(
            group_id=self.group_id,
            tiles={lvl: spec.clone() for lvl, spec in self.tiles.items()},
            op_schedule_style=self.op_schedule_style,
            sram_budget_bytes=self.sram_budget_bytes,
            phase=self.phase,
            special_role=self.special_role,
            metadata=copy.deepcopy(self.metadata),
        )


@dataclass
class TilingGene:
    gene_id: str = ""
    group_tiles: Dict[str, GroupTilingSpec] = field(default_factory=dict)
    global_policy: GlobalTilingPolicy = field(default_factory=GlobalTilingPolicy)
    phase_role_mapping: Dict[str, Dict[str, Dict[str, Dict[str, MemTileSpec]]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gene_id:
            self.gene_id = f"tiling_{self._hash_short(self.to_canonical_dict_without_gene_id())}"
        for gid in self.group_tiles.keys():
            if not isinstance(gid, str) or not gid:
                raise ValueError(f"TilingGene.group_tiles contains invalid group_id: {gid!r}")

    @property
    def group_specs(self) -> Dict[str, GroupTilingSpec]:
        return self.group_tiles

    def get_group_spec(
        self,
        group_id: str,
        phase: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[GroupTilingSpec]:
        base = self.group_tiles.get(group_id)
        if base is None:
            return None

        spec = base.clone()
        if phase or role:
            mem_map = self.phase_role_mapping.get(group_id, {}).get(phase or "", {}).get(role or "", {})
            if mem_map:
                for mem_level, mem_tile_spec in mem_map.items():
                    spec.tiles[mem_level] = mem_tile_spec.clone()
                if phase is not None:
                    spec.phase = phase
                if role is not None:
                    spec.special_role = role
        return spec

    def add_group_spec(self, group_spec: GroupTilingSpec) -> None:
        self.group_tiles[group_spec.group_id] = group_spec

    def add_phase_role_mapping(
        self,
        group_id: str,
        phase: str,
        role: str,
        mem_tile_specs: Mapping[str, MemTileSpec],
    ) -> None:
        cloned = {mem: spec.clone() for mem, spec in mem_tile_specs.items()}
        self.phase_role_mapping.setdefault(group_id, {}).setdefault(phase, {})[role] = cloned

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "gene_id": self.gene_id,
            "group_tiles": self._canonical_group_tiles(),
            "global_policy": {
                "prefer_spatial_levels": list(self.global_policy.prefer_spatial_levels),
                "notes": copy.deepcopy(self.global_policy.notes),
            },
            "phase_role_mapping": self._canonical_phase_role_mapping(),
        }

    def to_canonical_dict_without_gene_id(self) -> Dict[str, Any]:
        return {
            "group_tiles": self._canonical_group_tiles(),
            "global_policy": {
                "prefer_spatial_levels": list(self.global_policy.prefer_spatial_levels),
                "notes": copy.deepcopy(self.global_policy.notes),
            },
            "phase_role_mapping": self._canonical_phase_role_mapping(),
        }

    def _canonical_group_tiles(self) -> Dict[str, Any]:
        canon_groups: Dict[str, Any] = {}
        for gid, gspec in sorted(self.group_tiles.items()):
            tiles_out: Dict[str, Any] = {}
            for lvl, spec in sorted(gspec.tiles.items()):
                tiles_out[lvl] = {
                    "tile_size": dict(spec.tile_size or {}),
                    "loop_order": list(spec.loop_order or []),
                    "problem_size": dict(spec.problem_size or {}),
                    "loop_count": dict(spec.loop_count or {}),
                    "is_spatial": bool(spec.is_spatial),
                    "active_dims": list(spec.active_dims) if spec.active_dims else None,
                }
            canon_groups[gid] = {
                "tiles": tiles_out,
                "op_schedule_style": gspec.op_schedule_style,
                "sram_budget_bytes": gspec.sram_budget_bytes,
                "phase": gspec.phase,
                "special_role": gspec.special_role,
                "metadata": copy.deepcopy(gspec.metadata),
            }
        return canon_groups

    def _canonical_phase_role_mapping(self) -> Dict[str, Any]:
        canon_mapping: Dict[str, Any] = {}
        for gid, by_phase in sorted(self.phase_role_mapping.items()):
            phase_out: Dict[str, Any] = {}
            for phase, by_role in sorted(by_phase.items()):
                role_out: Dict[str, Any] = {}
                for role, mem_map in sorted(by_role.items()):
                    role_out[role] = {
                        mem: {
                            "tile_size": dict(spec.tile_size or {}),
                            "loop_order": list(spec.loop_order or []),
                            "problem_size": dict(spec.problem_size or {}),
                            "loop_count": dict(spec.loop_count or {}),
                            "is_spatial": bool(spec.is_spatial),
                            "active_dims": list(spec.active_dims) if spec.active_dims else None,
                        }
                        for mem, spec in sorted(mem_map.items())
                    }
                phase_out[phase] = role_out
            canon_mapping[gid] = phase_out
        return canon_mapping

    @staticmethod
    def _hash_short(obj: Any) -> str:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]
