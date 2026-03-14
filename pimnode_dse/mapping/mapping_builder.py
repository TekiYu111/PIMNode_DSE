from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pimnode_dse.workload import WorkloadDAG
from pimnode_dse.mapping.fusion_gene import FusionGene, FusionStyle, OpFusionGroup, PipelineConstraint
from pimnode_dse.mapping.mapping_tree import (
    MappingTree,
    OpNode,
    PipelineSpec,
    ScopeNode,
    ScopeType,
    TileNode,
    LoopNode,
)
from pimnode_dse.mapping.tilling_gene import TilingGene

try:
    from pimnode_dse.placement.templates import build_core_templates
except Exception:  # pragma: no cover
    build_core_templates = None

from pimnode_dse.placement.placement_ir import (
    DEFAULT_MEMORY_LEVELS,
    BoundaryAction,
    PlacementScope,
    PlacementTemplatePlan,
    PlacementTemplateScope,
    ResidentSet,
    instantiate_template_scope,
    normalize_scope,
)
from pimnode_dse.placement.rules import apply_hard_rules


@dataclass
class BuildConfig:
    program_scope_type: ScopeType = ScopeType.Sequential
    outer_mem_level: str = "DRAM"
    inner_mem_level: str = "SRAM"
    default_template_name: Optional[str] = None
    allow_template_fallback: bool = False
    include_storage_nodes: bool = True
    include_load_for_resident: bool = False
    apply_hard_rules_before_attach: bool = True
    normalize_memories: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_MEMORY_LEVELS))


@dataclass
class BuildReport:
    gene_id: str
    group_topo: List[str] = field(default_factory=list)
    op_to_group: Dict[str, str] = field(default_factory=dict)
    pipeline_constraints: Dict[str, PipelineConstraint] = field(default_factory=dict)
    placement_selection: Dict[str, str] = field(default_factory=dict)
    role_bindings: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)


@dataclass
class BuildResult:
    tree: MappingTree
    group_scopes: Dict[str, ScopeNode]
    report: BuildReport


class MappingBuilder:
    """
    Build MappingTree from WorkloadDAG + FusionGene + TilingGene + placement templates/scopes.

    Updated loop semantics:
      - tile_size   : local tile extent processed per iteration
      - problem_size: full visible problem size at this level
      - loop_count  : actual loop count for this level

    Compatibility:
      - if tiling spec only provides tile_size / loop_order, builder still works
      - if problem_size / loop_count exist, builder now lowers them into TileNode / LoopNode
    """

    def __init__(
        self,
        dag: WorkloadDAG,
        fusion: FusionGene,
        tiling: TilingGene,
        placement_plan: Optional[PlacementTemplatePlan] = None,
        *,
        selected_placements: Optional[Mapping[str, PlacementScope]] = None,
        selected_templates: Optional[Mapping[str, str]] = None,
        config: Optional[BuildConfig] = None,
    ) -> None:
        self.dag = dag
        self.fusion = fusion
        self.tiling = tiling
        self.cfg = config or BuildConfig()

        if self.tiling is None:
            raise ValueError("TilingGene 是必选输入")

        if placement_plan is None:
            if self.cfg.allow_template_fallback and build_core_templates is not None:
                placement_plan = build_core_templates()
            else:
                raise ValueError(
                    "placement_plan 是必选输入。请在 placement 阶段先完成模板生成/筛选，"
                    "再把 PlacementTemplatePlan 传给 MappingBuilder。"
                )

        self.placement_plan = placement_plan
        self.selected_placements: Dict[str, PlacementScope] = dict(selected_placements or {})
        self.selected_templates: Dict[str, str] = dict(selected_templates or {})
        self.tree_root: Optional[ScopeNode] = None

        self._validate_placement_inputs()

    def build(self, phase: str = "prefill") -> BuildResult:
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"Unsupported phase: {phase!r}")

        self.dag.validate()
        self.fusion.validate_topology(self.dag)
        self.fusion.build_coarse_dag(self.dag)

        op_to_group = self.fusion.get_op_to_group_mapping()
        group_topo = self._topo_groups()
        placement_selection, role_binding_report = self._resolve_all_group_placements(phase)

        report = BuildReport(
            gene_id=self.fusion.gene_id,
            group_topo=group_topo,
            op_to_group=dict(op_to_group),
            pipeline_constraints={
                g.group_id: g.pipeline_constraint
                for g in self.fusion.groups
                if getattr(g.pipeline_constraint, "has_constraints", False)
            },
            placement_selection={gid: scope.scope_name for gid, scope in placement_selection.items()},
            role_bindings=role_binding_report,
        )

        self.tree_root = ScopeNode(
            scope_type=self.cfg.program_scope_type,
            name=f"Program({self.dag.name})",
            phase=phase,
        )

        group_scopes: Dict[str, ScopeNode] = {}
        for group in self.fusion.groups:
            group_scopes[group.group_id] = self._make_group_scope(group, placement_selection[group.group_id])

        for gid in group_topo:
            self.tree_root.add_child(group_scopes[gid])

        op_index: Dict[str, OpNode] = {}
        topo_ops = self.dag.topo_order()

        for group in self.fusion.groups:
            group_scope = group_scopes[group.group_id]
            placement_scope = placement_selection[group.group_id]
            _, inner_tile, op_attach_point = self._attach_tiles_from_tiling(
                group=group,
                group_scope=group_scope,
                group_id=group.group_id,
                placement_scope=placement_scope,
            )
            self._attach_ops_in_group(
                op_parent=op_attach_point,
                group=group,
                topo_ops=topo_ops,
                op_to_group=op_to_group,
                op_index_out=op_index,
            )

        tree = MappingTree(root=self.tree_root, op_index=op_index)
        return BuildResult(tree=tree, group_scopes=group_scopes, report=report)

    def _validate_placement_inputs(self) -> None:
        if self.placement_plan is None:
            raise ValueError("placement_plan 不能为空")

        missing_from_plan = [
            name for name in self.selected_templates.values()
            if name not in self.placement_plan.scopes
        ]
        if missing_from_plan:
            raise KeyError(f"selected_templates 引用了未定义模板: {sorted(set(missing_from_plan))}")

        if self.cfg.default_template_name is not None:
            if self.cfg.default_template_name not in self.placement_plan.scopes:
                raise KeyError(
                    f"default_template_name={self.cfg.default_template_name!r} 不在 placement_plan 中"
                )

    def _resolve_all_group_placements(
        self,
        phase: str,
    ) -> Tuple[Dict[str, PlacementScope], Dict[str, Dict[str, List[str]]]]:
        placements: Dict[str, PlacementScope] = {}
        role_report: Dict[str, Dict[str, List[str]]] = {}
        for group in self.fusion.groups:
            scope, bindings = self._resolve_group_placement(group, phase)
            placements[group.group_id] = scope
            role_report[group.group_id] = {
                role: sorted(tensors) for role, tensors in bindings.items() if tensors
            }
        return placements, role_report

    def _validate_template_compatibility(
        self,
        group: OpFusionGroup,
        template_scope: PlacementTemplateScope,
        *,
        resolved_phase: Optional[str],
        resolved_special_role: Optional[str],
    ) -> None:
        if template_scope.supported_phases:
            if resolved_phase and resolved_phase not in template_scope.supported_phases:
                raise ValueError(
                    f"group {group.group_id} phase={resolved_phase!r} "
                    f"is not supported by template {template_scope.scope_name!r}. "
                    f"supported_phases={sorted(template_scope.supported_phases)}"
                )

        if template_scope.supported_roles:
            if resolved_special_role and resolved_special_role not in template_scope.supported_roles:
                raise ValueError(
                    f"group {group.group_id} special_role={resolved_special_role!r} "
                    f"is not supported by template {template_scope.scope_name!r}. "
                    f"supported_roles={sorted(template_scope.supported_roles)}"
                )

    def _resolve_group_placement(
        self,
        group: OpFusionGroup,
        phase: str,
    ) -> Tuple[PlacementScope, Dict[str, Set[str]]]:
        resolved_phase = getattr(group, "phase", None) or phase
        resolved_special_role = getattr(group, "special_role", None)

        role_to_tensors = self._build_role_to_tensors(group)
        tensor_to_role = self._invert_role_bindings(role_to_tensors)
        tensor_meta = self._build_tensor_meta_for_group(group)

        if group.group_id in self.selected_placements:
            concrete = self._clone_scope(
                self.selected_placements[group.group_id],
                phase=resolved_phase,
            )
            concrete = normalize_scope(
                concrete,
                memory_levels=self.cfg.normalize_memories,
            )
            if self.cfg.apply_hard_rules_before_attach:
                concrete = apply_hard_rules(
                    concrete,
                    phase=resolved_phase,
                    tensor_to_role=tensor_to_role,
                    tensor_meta=tensor_meta,
                    memories=self.cfg.normalize_memories,
                )
            return concrete, role_to_tensors

        scope_name = self._select_template_name_for_group(group)
        template_scope = self.placement_plan.scopes[scope_name]

        self._validate_template_compatibility(
            group,
            template_scope,
            resolved_phase=resolved_phase,
            resolved_special_role=resolved_special_role,
        )

        concrete = instantiate_template_scope(
            template_scope,
            role_bindings=role_to_tensors,
            phase=resolved_phase,
            special_role=resolved_special_role,
        )
        concrete = normalize_scope(
            concrete,
            memory_levels=self.cfg.normalize_memories,
        )

        if self.cfg.apply_hard_rules_before_attach:
            concrete = apply_hard_rules(
                concrete,
                phase=resolved_phase,
                tensor_to_role=tensor_to_role,
                tensor_meta=tensor_meta,
                memories=self.cfg.normalize_memories,
            )

        return concrete, role_to_tensors

    def _select_template_name_for_group(self, group: OpFusionGroup) -> str:
        if group.group_id in self.selected_templates:
            return self.selected_templates[group.group_id]

        if self.cfg.default_template_name is not None:
            return self.cfg.default_template_name

        if len(self.placement_plan.scopes) == 1:
            return next(iter(self.placement_plan.scopes.keys()))

        available = sorted(self.placement_plan.scopes.keys())
        raise ValueError(
            f"group {group.group_id} 缺少显式 placement 选择。可用模板: {available}. "
            "请通过 selected_templates / selected_placements 传入每个 group 的模板，"
            "或在 BuildConfig.default_template_name 中指定默认模板。"
        )

    def _clone_scope(self, scope: PlacementScope, phase: Optional[str] = None) -> PlacementScope:
        resident_sets = [ResidentSet(mem=rs.mem, tensors=set(rs.tensors)) for rs in scope.resident_sets]
        boundary_actions = [
            BoundaryAction(
                mem=ba.mem,
                prefetch=set(ba.prefetch),
                writeback=set(ba.writeback),
                evict=set(ba.evict),
            )
            for ba in scope.boundary_actions
        ]
        cloned = PlacementScope(
            scope_name=scope.scope_name,
            resident_sets=resident_sets,
            boundary_actions=boundary_actions,
            phase=phase if phase is not None else getattr(scope, "phase", None),
            special_role=getattr(scope, "special_role", None),
            metadata=dict(getattr(scope, "metadata", {}) or {}),
        )
        return cloned

    def _build_role_to_tensors(self, group: OpFusionGroup) -> Dict[str, Set[str]]:
        group_tensor_names = self._collect_group_tensor_names(group)

        role_to_tensors: Dict[str, Set[str]] = {
            "Q": set(),
            "K": set(),
            "V": set(),
            "SCORES": set(),
            "STATS": set(),
            "PROBS": set(),
            "PARTIAL_O": set(),
            "O": set(),
            "KV_CACHE": set(),
        }

        for tensor_name in group_tensor_names:
            role = self._infer_tensor_role(tensor_name)
            if role is None:
                continue
            role_to_tensors.setdefault(role, set()).add(tensor_name)

            meta = self._lookup_tensor_spec(tensor_name)
            special_role = self._get_attr(meta, "special_role")
            if special_role == "KV_CACHE":
                role_to_tensors.setdefault("KV_CACHE", set()).add(tensor_name)

        return role_to_tensors

    def _invert_role_bindings(self, role_to_tensors: Mapping[str, Iterable[str]]) -> Dict[str, str]:
        tensor_to_role: Dict[str, str] = {}
        for role, tensors in role_to_tensors.items():
            for tensor in tensors:
                if tensor in tensor_to_role and tensor_to_role[tensor] != "KV_CACHE":
                    continue
                tensor_to_role[tensor] = role
        return tensor_to_role

    def _build_tensor_meta_for_group(self, group: OpFusionGroup) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for tensor_name in self._collect_group_tensor_names(group):
            spec = self._lookup_tensor_spec(tensor_name)
            if spec is None:
                out[tensor_name] = {}
                continue
            meta = {
                "role": self._get_attr(spec, "role"),
                "special_role": self._get_attr(spec, "special_role"),
                "name": self._get_attr(spec, "name") or tensor_name,
            }
            out[tensor_name] = {k: v for k, v in meta.items() if v is not None}
        return out

    def _collect_group_tensor_names(self, group: OpFusionGroup) -> Set[str]:
        tensors: Set[str] = set()
        for op_id in group.op_names:
            op = self.dag.get_op(op_id)
            for name in self._iter_op_tensors(op, "inputs"):
                tensors.add(name)
            for name in self._iter_op_tensors(op, "outputs"):
                tensors.add(name)
        return tensors

    def _iter_op_tensors(self, op: Any, field_name: str) -> Iterable[str]:
        vals = getattr(op, field_name, None)
        if vals is None and isinstance(op, Mapping):
            vals = op.get(field_name)

        if vals is None:
            return []

        if isinstance(vals, Mapping):
            out: List[str] = []
            for v in vals.values():
                if isinstance(v, (list, tuple, set)):
                    out.extend(str(x) for x in v)
                else:
                    out.append(str(v))
            return out

        if isinstance(vals, (list, tuple, set)):
            return [str(x) for x in vals]

        return [str(vals)]

    def _lookup_tensor_spec(self, tensor_name: str) -> Any:
        tensor_table = getattr(self.dag, "tensors", None)
        if isinstance(tensor_table, Mapping):
            return tensor_table.get(tensor_name)

        if hasattr(self.dag, "get_tensor"):
            try:
                return self.dag.get_tensor(tensor_name)
            except Exception:
                return None
        return None

    def _infer_tensor_role(self, tensor_name: str) -> Optional[str]:
        spec = self._lookup_tensor_spec(tensor_name)
        if spec is not None:
            special_role = self._get_attr(spec, "special_role")
            role = self._get_attr(spec, "role")
            name = str(self._get_attr(spec, "name") or tensor_name)

            if special_role in {
                "KV_CACHE",
                "PARTIAL_O",
                "STATS",
                "SCORES",
                "PROBS",
            }:
                return str(special_role)
            if role in {
                "Q",
                "K",
                "V",
                "O",
                "SCORES",
                "STATS",
                "PROBS",
                "PARTIAL_O",
                "KV_CACHE",
            }:
                return str(role)

            guessed = self._infer_role_from_name(name)
            if guessed is not None:
                return guessed

        return self._infer_role_from_name(tensor_name)

    def _infer_role_from_name(self, name: str) -> Optional[str]:
        u = str(name).upper()
        if "KV_CACHE" in u:
            return "KV_CACHE"
        if "PARTIAL" in u and "O" in u:
            return "PARTIAL_O"
        if "STAT" in u:
            return "STATS"
        if "SCORE" in u:
            return "SCORES"
        if "PROB" in u:
            return "PROBS"
        if u == "Q" or u.startswith("Q_"):
            return "Q"
        if u == "K" or u.startswith("K_"):
            return "K"
        if u == "V" or u.startswith("V_"):
            return "V"
        if u == "O" or u.startswith("O_") or u.endswith("_O"):
            return "O"
        return None

    def _get_attr(self, obj: Any, attr: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, Mapping):
            return obj.get(attr)
        return getattr(obj, attr, None)

    def _topo_groups(self) -> List[str]:
        if hasattr(self.fusion, "coarse_topo_order"):
            try:
                order = list(self.fusion.coarse_topo_order())
                if order:
                    return order
            except Exception:
                pass
        return [g.group_id for g in self.fusion.groups]

    def _make_group_scope(self, group: OpFusionGroup, placement_scope: PlacementScope) -> ScopeNode:
        scope_type = ScopeType.Group
        pipeline_spec: Optional[PipelineSpec] = None
        if group.fusion_style == FusionStyle.PIPELINE:
            scope_type = ScopeType.Pipeline
            pc = group.pipeline_constraint
            pipeline_spec = PipelineSpec(
                pipeline_dim=getattr(group, "pipeline_dim", None),
                num_stages=max(2, int(getattr(pc, "num_stages", 2))),
                buffering=getattr(pc, "buffering", "pingpong"),
                allow_interleave=bool(getattr(pc, "allow_interleave", True)),
                constraint=pc,
            )

        return ScopeNode(
            scope_type=scope_type,
            name=f"Group({group.group_id})",
            pipeline_spec=pipeline_spec,
            phase=group.phase,
            special_role=group.special_role,
            placement_state=placement_scope,
        )

    def _get_group_tiling_spec(self, group: OpFusionGroup) -> Any:
        group_id = group.group_id
        spec = self.tiling.get_group_spec(group_id, phase=group.phase, role=group.special_role)
        if spec is not None:
            return spec

        for attr in ("group_tiles", "group_specs"):
            table = getattr(self.tiling, attr, None)
            if isinstance(table, Mapping) and group_id in table:
                return table[group_id]

        raise KeyError(f"TilingGene 中找不到 group {group_id!r} 的 tiling spec")

    def _lookup_tile_spec(self, gspec: Any, mem_level: str) -> Any:
        tiles = getattr(gspec, "tiles", None)
        if tiles is None and isinstance(gspec, Mapping):
            tiles = gspec.get("tiles")
        if tiles is None:
            return None
        if isinstance(tiles, Mapping):
            return tiles.get(mem_level)
        try:
            return getattr(tiles, mem_level)
        except Exception:
            return None

    def _extract_tile_attrs(
        self,
        tile_spec: Any,
    ) -> Tuple[
        Optional[Dict[str, int]],
        Optional[List[str]],
        Optional[Dict[str, int]],
        Optional[Dict[str, int]],
        bool,
    ]:
        if tile_spec is None:
            return None, None, None, None, False

        if hasattr(tile_spec, "get_tile_desc"):
            desc = tile_spec.get_tile_desc()
            tile_size = desc.get("tile_size")
            loop_order = desc.get("loop_order")
            problem_size = desc.get("problem_size")
            loop_count = desc.get("loop_count")
            is_spatial = bool(getattr(tile_spec, "is_spatial", False))
            return tile_size, loop_order, problem_size, loop_count, is_spatial

        if isinstance(tile_spec, Mapping):
            tile_size = tile_spec.get("tile_size")
            loop_order = tile_spec.get("loop_order")
            problem_size = tile_spec.get("problem_size")
            loop_count = tile_spec.get("loop_count")
            is_spatial = bool(tile_spec.get("is_spatial", False))
            return tile_size, loop_order, problem_size, loop_count, is_spatial

        tile_size = getattr(tile_spec, "tile_size", None)
        loop_order = getattr(tile_spec, "loop_order", None)
        problem_size = getattr(tile_spec, "problem_size", None)
        loop_count = getattr(tile_spec, "loop_count", None)
        is_spatial = bool(getattr(tile_spec, "is_spatial", False))
        return tile_size, loop_order, problem_size, loop_count, is_spatial

    def _attach_tiles_from_tiling(
        self,
        group: OpFusionGroup,
        group_scope: ScopeNode,
        group_id: str,
        placement_scope: PlacementScope,
    ) -> Tuple[TileNode, TileNode, Any]:
        gspec = self._get_group_tiling_spec(group)
        phase = getattr(gspec, "phase", None) or group.phase
        special_role = getattr(gspec, "special_role", None) or group.special_role

        role_to_tensors = self._build_role_to_tensors(group)
        tensor_to_role = self._invert_role_bindings(role_to_tensors)
        tensor_meta = self._build_tensor_meta_for_group(group)

        outer_tile_spec = self._lookup_tile_spec(gspec, self.cfg.outer_mem_level)
        (
            outer_tile_size,
            outer_loop_order,
            outer_problem_size,
            outer_loop_count,
            outer_is_spatial,
        ) = self._extract_tile_attrs(outer_tile_spec)
        outer = TileNode(
            mem_level=self.cfg.outer_mem_level,
            tile_size=outer_tile_size,
            loop_order=outer_loop_order,
            problem_size=outer_problem_size,
            loop_count=outer_loop_count,
            is_spatial=outer_is_spatial,
            name=f"{group_id}:{self.cfg.outer_mem_level}_TILE",
            placement_state=placement_scope,
            phase=phase,
            special_role=special_role,
        )
        group_scope.add_child(outer)

        inner_tile_spec = self._lookup_tile_spec(gspec, self.cfg.inner_mem_level)
        (
            inner_tile_size,
            inner_loop_order,
            inner_problem_size,
            inner_loop_count,
            inner_is_spatial,
        ) = self._extract_tile_attrs(inner_tile_spec)
        inner = TileNode(
            mem_level=self.cfg.inner_mem_level,
            tile_size=inner_tile_size,
            loop_order=inner_loop_order,
            problem_size=inner_problem_size,
            loop_count=inner_loop_count,
            is_spatial=inner_is_spatial,
            name=f"{group_id}:{self.cfg.inner_mem_level}_TILE",
            placement_state=placement_scope,
            phase=phase,
            special_role=special_role,
        )
        outer.add_child(inner)

        outer.materialize_placement(
            default_src=self._default_src_for(self.cfg.outer_mem_level),
            default_dst=self._default_dst_for(self.cfg.outer_mem_level),
            include_storage=self.cfg.include_storage_nodes,
            include_load_for_resident=self.cfg.include_load_for_resident,
            tensor_to_role=tensor_to_role,
            tensor_meta=tensor_meta,
            use_rules=True,
        )
        inner.materialize_placement(
            default_src=self._default_src_for(self.cfg.inner_mem_level),
            default_dst=self._default_dst_for(self.cfg.inner_mem_level),
            include_storage=self.cfg.include_storage_nodes,
            include_load_for_resident=self.cfg.include_load_for_resident,
            tensor_to_role=tensor_to_role,
            tensor_meta=tensor_meta,
            use_rules=True,
        )

        op_attach_point = self._ensure_loopnest(inner)
        return outer, inner, op_attach_point

    def _ensure_loopnest(self, tile: TileNode) -> Any:
        for child in tile.children:
            if isinstance(child, LoopNode):
                cur = child
                while True:
                    nxt = next((c for c in getattr(cur, "children", []) if isinstance(c, LoopNode)), None)
                    if nxt is None:
                        break
                    cur = nxt
                return cur

        parent: Any = tile
        last: Any = tile
        for dim in tile.loop_order or []:
            tile_extent = tile.tile_size.get(dim) if tile.tile_size else None
            problem_size = tile.problem_size.get(dim) if tile.problem_size else None
            loop_count = tile.loop_count.get(dim) if tile.loop_count else None

            if loop_count is None and problem_size is not None and tile_extent not in (None, 0):
                try:
                    loop_count = -(-int(problem_size) // int(tile_extent))
                except Exception:
                    loop_count = None

            kind = "spatial" if tile.is_spatial else "temporal"

            loop = LoopNode(
                iter_dim=dim,
                tile_extent=tile_extent,
                problem_size=problem_size,
                loop_count=loop_count,
                domain=None,
                kind=kind,
                name=f"{tile.name}:loop_{dim}",
            )
            parent.add_child(loop)
            parent = loop
            last = loop
        return last

    def _level_order(self) -> List[str]:
        return ["DRAM", "SRAM", "RF", "PE"]

    def _default_src_for(self, mem_level: str) -> Optional[str]:
        order = self._level_order()
        if mem_level not in order:
            return None
        idx = order.index(mem_level)
        return order[idx - 1] if idx > 0 else None

    def _default_dst_for(self, mem_level: str) -> Optional[str]:
        order = self._level_order()
        if mem_level not in order:
            return None
        idx = order.index(mem_level)
        return order[idx - 1] if idx > 0 else None

    def _attach_ops_in_group(
        self,
        op_parent: Any,
        group: OpFusionGroup,
        topo_ops: List[str],
        op_to_group: Dict[str, str],
        op_index_out: Dict[str, OpNode],
    ) -> None:
        for op_id in topo_ops:
            if op_to_group.get(op_id) != group.group_id:
                continue
            op = self.dag.get_op(op_id)
            node = OpNode(
                op_id=op_id,
                op_type=op.op_type,
                name=f"Op({op_id})",
                phase=group.phase,
                special_role=group.special_role,
            )
            op_parent.add_child(node)
            op_index_out[op_id] = node
