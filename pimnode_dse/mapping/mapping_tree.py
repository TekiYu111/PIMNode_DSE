from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pimnode_dse.placement.placement_ir import BoundaryAction, PlacementPlan, PlacementScope, ResidentSet
from pimnode_dse.placement.rules import derive_actions as derive_rule_actions


# ============================================================
# Enums
# ============================================================


class ScopeType(Enum):
    Sequential = "Sequential"
    Parallel = "Parallel"
    Pipeline = "Pipeline"
    Pipelined = "Pipeline"  # alias
    Phase = "Phase"
    Group = "Group"
    Program = "Program"


class NodeType(Enum):
    Scope = "Scope"
    Tile = "Tile"
    Loop = "Loop"
    Storage = "Storage"
    Action = "Action"
    Op = "Op"


class ActionType(Enum):
    LOAD = "LOAD"
    STORE = "STORE"
    PREFETCH = "PREFETCH"
    WRITEBACK = "WRITEBACK"
    WRITEBACK_DRAM = "WRITEBACK_DRAM"
    WRITEBACK_SRAM = "WRITEBACK_SRAM"
    EVICT = "EVICT"
    BARRIER = "BARRIER"
    COMPUTE = "COMPUTE"


DataPlacement = PlacementPlan


# ============================================================
# PipelineSpec
# ============================================================


@dataclass
class PipelineSpec:
    pipeline_dim: Optional[str] = None
    num_stages: int = 2
    buffering: str = "pingpong"
    allow_interleave: bool = True
    constraint: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.num_stages < 2:
            raise ValueError("num_stages must be >= 2")
        if self.buffering not in ("single", "pingpong", "multi"):
            raise ValueError("buffering must be one of: single, pingpong, multi")


# ============================================================
# Stats containers
# ============================================================


@dataclass
class OpStats:
    compute_cycles: int = 0
    mac_count: int = 0
    energy: Optional[float] = None


@dataclass
class TileStats:
    active_tensors: List[str] = field(default_factory=list)
    working_set_bytes: Dict[str, int] = field(default_factory=dict)
    access_count: Dict[str, int] = field(default_factory=dict)
    traffic_bytes: Dict[str, int] = field(default_factory=dict)
    energy_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScopeStats:
    total_cycles: int = 0
    overlap_ratio: Optional[float] = None
    bytes_moved: Dict[str, int] = field(default_factory=dict)
    energy_breakdown: Dict[str, float] = field(default_factory=dict)


# ============================================================
# Visitor
# ============================================================


class MappingTreeVisitor:
    def visit(self, node: "Node") -> None:
        node.accept(self)

    def visit_scope(self, node: "ScopeNode") -> None:
        for child in node.children:
            self.visit(child)

    def visit_tile(self, node: "TileNode") -> None:
        for child in node.children:
            self.visit(child)

    def visit_loop(self, node: "LoopNode") -> None:
        for child in node.children:
            self.visit(child)

    def visit_storage(self, node: "StorageNode") -> None:
        for child in node.children:
            self.visit(child)

    def visit_action(self, node: "ActionNode") -> None:
        for child in node.children:
            self.visit(child)

    def visit_op(self, node: "OpNode") -> None:
        return None


# ============================================================
# Base node
# ============================================================


class Node:
    def __init__(self, node_type: NodeType, name: str = "") -> None:
        self.node_type: NodeType = node_type
        self.name: str = name
        self.children: List["Node"] = []
        self.parent: Optional["Node"] = None
        self.attrs: Dict[str, Any] = {}

    def add_child(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return self

    def extend_children(self, children: Iterable["Node"]) -> "Node":
        for child in children:
            self.add_child(child)
        return self

    def remove_child(self, child: "Node") -> None:
        self.children.remove(child)
        child.parent = None

    def accept(self, visitor: MappingTreeVisitor) -> None:
        raise NotImplementedError

    def display(self, indent: int = 0) -> None:
        raise NotImplementedError

    def walk(self) -> Iterable["Node"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def ancestors(self) -> Iterable["Node"]:
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def find_children(self, node_cls: type) -> List["Node"]:
        return [child for child in self.children if isinstance(child, node_cls)]

    def root(self) -> "Node":
        node: Node = self
        while node.parent is not None:
            node = node.parent
        return node

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ============================================================
# Scope node
# ============================================================


class ScopeNode(Node):
    def __init__(
        self,
        scope_type: ScopeType,
        name: str = "",
        pipeline_spec: Optional[PipelineSpec] = None,
        phase: Optional[str] = None,
        special_role: Optional[str] = None,
        placement_state: Optional[PlacementScope] = None,
    ) -> None:
        super().__init__(NodeType.Scope, name or scope_type.value)
        self.scope_type: ScopeType = scope_type
        self.pipeline_spec: Optional[PipelineSpec] = None
        self.phase: Optional[str] = phase
        self.special_role: Optional[str] = special_role
        self.placement_state: Optional[PlacementScope] = placement_state
        self.stats: Optional[ScopeStats] = None

        if scope_type in (ScopeType.Pipeline, ScopeType.Pipelined):
            self.pipeline_spec = pipeline_spec if pipeline_spec is not None else PipelineSpec()
        elif pipeline_spec is not None:
            raise ValueError("pipeline_spec is only valid for Pipeline scope")

    def accept(self, visitor: MappingTreeVisitor) -> None:
        visitor.visit_scope(self)

    def display(self, indent: int = 0) -> None:
        prefix = "  " * indent
        spec_str = ""
        if self.scope_type in (ScopeType.Pipeline, ScopeType.Pipelined) and self.pipeline_spec:
            ps = self.pipeline_spec
            spec_str = (
                f"  [dim={ps.pipeline_dim} stages={ps.num_stages} "
                f"buf={ps.buffering} interleave={ps.allow_interleave}]"
            )
        tags: List[str] = []
        if self.phase is not None:
            tags.append(f"phase={self.phase}")
        if self.special_role is not None:
            tags.append(f"role={self.special_role}")
        if self.placement_state is not None:
            tags.append(f"placement_scope={self.placement_state.scope_name}")
        tag_str = f"  [{' | '.join(tags)}]" if tags else ""
        stats_str = ""
        if self.stats is not None:
            stats_str = f"  => cycles={self.stats.total_cycles} bytes={self.stats.bytes_moved}"
            if self.stats.overlap_ratio is not None:
                stats_str += f" overlap={self.stats.overlap_ratio:.2f}"
            if self.stats.energy_breakdown:
                stats_str += f" energy={self.stats.energy_breakdown}"
        print(f"{prefix}Scope[{self.scope_type.value}] '{self.name}'{tag_str}{spec_str}{stats_str}")
        for child in self.children:
            child.display(indent + 1)


# ============================================================
# Tile node
# ============================================================


class TileNode(Node):
    def __init__(
        self,
        mem_level: str,
        tile_size: Optional[Dict[str, int]] = None,
        loop_order: Optional[List[str]] = None,
        problem_size: Optional[Dict[str, int]] = None,
        loop_count: Optional[Dict[str, int]] = None,
        is_spatial: bool = False,
        name: str = "",
        placement_state: Optional[PlacementScope] = None,
        phase: Optional[str] = None,
        special_role: Optional[str] = None,
        binding: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(NodeType.Tile, name or mem_level)
        self.mem_level: str = mem_level
        self.tile_size: Dict[str, int] = dict(tile_size or {})
        self.loop_order: List[str] = list(loop_order or [])
        self.problem_size: Dict[str, int] = dict(problem_size or {})
        self.loop_count: Dict[str, int] = dict(loop_count or {})
        self.is_spatial: bool = is_spatial
        self.phase: Optional[str] = phase
        self.special_role: Optional[str] = special_role
        self.binding: Dict[str, Any] = dict(binding or {})
        self.stats: Optional[TileStats] = None
        self.placement_state: Optional[PlacementScope] = placement_state

    @property
    def placement_level(self) -> Optional[Dict[str, Tuple[str, ...]]]:
        if self.placement_state is None:
            return None
        resident = self.resident_tensors()
        boundary = self.boundary_action()
        return {
            "resident": resident,
            "prefetch": tuple(sorted(boundary.prefetch)) if boundary else (),
            "writeback": tuple(sorted(boundary.writeback)) if boundary else (),
            "evict": tuple(sorted(boundary.evict)) if boundary else (),
        }

    def set_placement_scope(self, placement_scope: PlacementScope) -> None:
        self.placement_state = placement_scope

    def resident_set(self) -> Optional[ResidentSet]:
        if self.placement_state is None:
            return None
        return next((rs for rs in self.placement_state.resident_sets if rs.mem == self.mem_level), None)

    def boundary_action(self) -> Optional[BoundaryAction]:
        if self.placement_state is None:
            return None
        return next((ba for ba in self.placement_state.boundary_actions if ba.mem == self.mem_level), None)

    def resident_tensors(self) -> Tuple[str, ...]:
        rs = self.resident_set()
        return tuple(sorted(rs.tensors)) if rs is not None else ()

    def prefetch_tensors(self) -> Tuple[str, ...]:
        ba = self.boundary_action()
        return tuple(sorted(ba.prefetch)) if ba is not None else ()

    def writeback_tensors(self) -> Tuple[str, ...]:
        ba = self.boundary_action()
        return tuple(sorted(ba.writeback)) if ba is not None else ()

    def evict_tensors(self) -> Tuple[str, ...]:
        ba = self.boundary_action()
        return tuple(sorted(ba.evict)) if ba is not None else ()

    def attach_storage(self, storage: "StorageNode") -> "TileNode":
        self.add_child(storage)
        return self

    def attach_action(self, action: "ActionNode") -> "TileNode":
        self.add_child(action)
        return self

    def attach_loop_chain(
        self,
        loop_dims: Sequence[str],
        tile_extents: Optional[Dict[str, int]] = None,
        problem_sizes: Optional[Dict[str, int]] = None,
        loop_counts: Optional[Dict[str, int]] = None,
    ) -> Optional["LoopNode"]:
        tile_extents = dict(tile_extents or {})
        problem_sizes = dict(problem_sizes or {})
        loop_counts = dict(loop_counts or {})
        head: Optional[LoopNode] = None
        tail: Optional[LoopNode] = None
        for dim in loop_dims:
            node = LoopNode(
                iter_dim=dim,
                tile_extent=tile_extents.get(dim),
                problem_size=problem_sizes.get(dim),
                loop_count=loop_counts.get(dim),
            )
            if head is None:
                head = node
                self.add_child(node)
            else:
                tail.add_child(node)
            tail = node
        return head

    def derive_storage_nodes(self, *, anchor: str = "scope") -> List["StorageNode"]:
        return [
            StorageNode(
                mem_level=self.mem_level,
                tensor=tensor,
                keep=True,
                anchor=anchor,
                resident=True,
                bypass=False,
            )
            for tensor in self.resident_tensors()
        ]

    def derive_actions_legacy(
        self,
        default_src: Optional[str] = None,
        default_dst: Optional[str] = None,
        *,
        include_load_for_resident: bool = False,
    ) -> List["ActionNode"]:
        if self.placement_state is None:
            return []

        derived: List[ActionNode] = []

        if include_load_for_resident:
            for tensor in self.resident_tensors():
                derived.append(
                    ActionNode.load(
                        tensors=[tensor],
                        src_level=default_src,
                        dst_level=self.mem_level,
                        anchor="tile_entry",
                    )
                )

        for tensor in self.prefetch_tensors():
            derived.append(
                ActionNode.prefetch(
                    tensors=[tensor],
                    src_level=default_src,
                    dst_level=self.mem_level,
                    anchor="tile_entry",
                )
            )

        for tensor in self.writeback_tensors():
            derived.append(
                ActionNode.writeback(
                    tensors=[tensor],
                    src_level=self.mem_level,
                    dst_level=default_dst,
                    anchor="tile_exit",
                )
            )

        for tensor in self.evict_tensors():
            derived.append(
                ActionNode.evict(
                    tensors=[tensor],
                    src_level=self.mem_level,
                    dst_level=default_dst,
                    anchor="tile_exit",
                )
            )

        return derived

    def derive_actions_via_rules(
        self,
        *,
        tensor_to_role: Optional[Dict[str, str]] = None,
        tensor_meta: Optional[Dict[str, Dict[str, object]]] = None,
        include_load_for_resident: bool = False,
        default_src: Optional[str] = None,
        default_dst: Optional[str] = None,
    ) -> List["ActionNode"]:
        if self.placement_state is None:
            return []

        rule_actions = derive_rule_actions(
            self.placement_state,
            phase=self.phase,
            tensor_to_role=tensor_to_role,
            tensor_meta=tensor_meta,
            include_loads=include_load_for_resident,
        )

        out: List["ActionNode"] = []
        for ra in rule_actions:
            if getattr(ra, "mem", None) != self.mem_level:
                continue

            action_type = str(ra.action).upper()
            src_level: Optional[str] = None
            dst_level: Optional[str] = None

            if action_type in {"LOAD", "PREFETCH"}:
                src_level = default_src
                dst_level = self.mem_level
            elif action_type in {"WRITEBACK", "STORE", "EVICT"}:
                src_level = self.mem_level
                dst_level = default_dst
            elif action_type == "BARRIER":
                src_level = None
                dst_level = None

            attrs = dict(getattr(ra, "metadata", {}) or {})
            if getattr(ra, "role", None) is not None:
                attrs.setdefault("role", ra.role)
            if getattr(ra, "scope", None) is not None:
                attrs.setdefault("scope", ra.scope)
            if getattr(ra, "phase", None) is not None:
                attrs.setdefault("phase", ra.phase)

            anchor = "tile_entry" if action_type in {"LOAD", "PREFETCH"} else "tile_exit"

            if action_type == "LOAD":
                out.append(
                    ActionNode.load(
                        tensors=[ra.tensor],
                        src_level=src_level,
                        dst_level=dst_level,
                        anchor=anchor,
                        attrs=attrs,
                    )
                )
            elif action_type == "PREFETCH":
                out.append(
                    ActionNode.prefetch(
                        tensors=[ra.tensor],
                        src_level=src_level,
                        dst_level=dst_level,
                        anchor=anchor,
                        attrs=attrs,
                    )
                )
            elif action_type in {"WRITEBACK", "STORE"}:
                out.append(
                    ActionNode.writeback(
                        tensors=[ra.tensor],
                        src_level=src_level,
                        dst_level=dst_level,
                        anchor=anchor,
                        attrs=attrs,
                    )
                )
            elif action_type == "EVICT":
                out.append(
                    ActionNode.evict(
                        tensors=[ra.tensor],
                        src_level=src_level,
                        dst_level=dst_level,
                        anchor=anchor,
                        attrs=attrs,
                    )
                )
            elif action_type == "BARRIER":
                out.append(
                    ActionNode.barrier(
                        tensors=[ra.tensor] if getattr(ra, "tensor", None) else (),
                        anchor="scope_boundary",
                        attrs=attrs,
                    )
                )

        return out

    def derive_actions(
        self,
        default_src: Optional[str] = None,
        default_dst: Optional[str] = None,
        *,
        include_load_for_resident: bool = False,
        tensor_to_role: Optional[Dict[str, str]] = None,
        tensor_meta: Optional[Dict[str, Dict[str, object]]] = None,
        use_rules: bool = True,
    ) -> List["ActionNode"]:
        if use_rules:
            return self.derive_actions_via_rules(
                tensor_to_role=tensor_to_role,
                tensor_meta=tensor_meta,
                include_load_for_resident=include_load_for_resident,
                default_src=default_src,
                default_dst=default_dst,
            )
        return self.derive_actions_legacy(
            default_src=default_src,
            default_dst=default_dst,
            include_load_for_resident=include_load_for_resident,
        )

    def materialize_placement(
        self,
        default_src: Optional[str] = None,
        default_dst: Optional[str] = None,
        *,
        include_storage: bool = True,
        include_load_for_resident: bool = False,
        tensor_to_role: Optional[Dict[str, str]] = None,
        tensor_meta: Optional[Dict[str, Dict[str, object]]] = None,
        use_rules: bool = True,
    ) -> "TileNode":
        if include_storage:
            self.extend_children(self.derive_storage_nodes(anchor="scope"))
        self.extend_children(
            self.derive_actions(
                default_src=default_src,
                default_dst=default_dst,
                include_load_for_resident=include_load_for_resident,
                tensor_to_role=tensor_to_role,
                tensor_meta=tensor_meta,
                use_rules=use_rules,
            )
        )
        return self

    def accept(self, visitor: MappingTreeVisitor) -> None:
        visitor.visit_tile(self)

    def display(self, indent: int = 0) -> None:
        prefix = "  " * indent
        kind = "Spatial" if self.is_spatial else "Temporal"
        tile_str = ", ".join(f"{k}={v}" for k, v in self.tile_size.items()) or "-"
        order_str = "->".join(self.loop_order) if self.loop_order else "?"
        tags = []
        if self.problem_size:
            tags.append(f"problem={self.problem_size}")
        if self.loop_count:
            tags.append(f"loops={self.loop_count}")
        if self.phase is not None:
            tags.append(f"phase={self.phase}")
        if self.special_role is not None:
            tags.append(f"role={self.special_role}")
        if self.binding:
            tags.append(f"binding={self.binding}")
        if self.placement_state is not None:
            resident = list(self.resident_tensors())
            prefetch = list(self.prefetch_tensors())
            writeback = list(self.writeback_tensors())
            evict = list(self.evict_tensors())
            if resident:
                tags.append(f"resident={resident}")
            if prefetch:
                tags.append(f"prefetch={prefetch}")
            if writeback:
                tags.append(f"writeback={writeback}")
            if evict:
                tags.append(f"evict={evict}")
            if not (resident or prefetch or writeback or evict):
                tags.append(f"placement_scope={self.placement_state.scope_name}")
        stats_str = ""
        if self.stats is not None:
            stats_str = f"  => ws={self.stats.working_set_bytes} acc={self.stats.access_count}"
            if self.stats.traffic_bytes:
                stats_str += f" traffic={self.stats.traffic_bytes}"
            if self.stats.energy_breakdown:
                stats_str += f" energy={self.stats.energy_breakdown}"
        tag_str = f"  [{' | '.join(tags)}]" if tags else ""
        print(
            f"{prefix}Tile[{kind}@{self.mem_level}] '{self.name}'  tile=({tile_str})  "
            f"order={order_str}{tag_str}{stats_str}"
        )
        for child in self.children:
            child.display(indent + 1)


TileScopeNode = TileNode


# ============================================================
# Loop node
# ============================================================


class LoopNode(Node):
    def __init__(
        self,
        iter_dim: str,
        tile_extent: Optional[int] = None,
        problem_size: Optional[int] = None,
        loop_count: Optional[int] = None,
        domain: Optional[Dict[str, Any]] = None,
        kind: str = "temporal",
        name: str = "",
        extent: Optional[int] = None,  # backward-compatible alias
    ) -> None:
        super().__init__(NodeType.Loop, name or f"loop:{iter_dim}")
        self.iter_dim: str = iter_dim

        # backward compatibility:
        # if old caller still passes extent, treat it as tile_extent
        if tile_extent is None and extent is not None:
            tile_extent = extent

        self.tile_extent: Optional[int] = tile_extent
        self.problem_size: Optional[int] = problem_size
        self.loop_count: Optional[int] = loop_count
        self.domain: Optional[Dict[str, Any]] = dict(domain) if domain is not None else None
        self.kind: str = kind

    @property
    def extent(self) -> Optional[int]:
        # compatibility alias; old code may still read .extent
        return self.tile_extent

    def accept(self, visitor: MappingTreeVisitor) -> None:
        visitor.visit_loop(self)

    def display(self, indent: int = 0) -> None:
        prefix = "  " * indent
        tile_str = "?" if self.tile_extent is None else str(self.tile_extent)
        prob_str = "?" if self.problem_size is None else str(self.problem_size)
        loop_str = "?" if self.loop_count is None else str(self.loop_count)
        dom_str = f" domain={self.domain.get('kind', '?')}" if self.domain else ""
        print(
            f"{prefix}Loop[{self.kind}] {self.iter_dim}  "
            f"tile={tile_str} problem={prob_str} loops={loop_str}{dom_str}"
        )
        for child in self.children:
            child.display(indent + 1)


# ============================================================
# Storage node
# ============================================================


class StorageNode(Node):
    def __init__(
        self,
        mem_level: str,
        tensor: str,
        keep: bool,
        writeback: str = "end_of_group",
        lifetime: str = "within_group",
        anchor: str = "scope_end",
        policy: Optional[Dict[str, Any]] = None,
        resident: Optional[bool] = None,
        bypass: Optional[bool] = None,
        name: str = "",
    ) -> None:
        super().__init__(NodeType.Storage, name or f"Storage@{mem_level}:{tensor}")
        self.mem_level: str = mem_level
        self.tensor: str = tensor
        self.keep: bool = keep
        self.resident: bool = keep if resident is None else resident
        self.bypass: bool = (not self.resident) if bypass is None else bypass
        self.writeback: str = writeback
        self.lifetime: str = lifetime
        self.anchor: str = anchor
        self.policy: Dict[str, Any] = dict(policy or {})

    def accept(self, visitor: MappingTreeVisitor) -> None:
        visitor.visit_storage(self)

    def display(self, indent: int = 0) -> None:
        prefix = "  " * indent
        mode = "KEEP" if self.keep else "BYPASS"
        extra = []
        if self.resident:
            extra.append("resident=True")
        if self.bypass:
            extra.append("bypass=True")
        if self.policy:
            extra.append(f"policy={self.policy}")
        extra_str = f"  [{' | '.join(extra)}]" if extra else ""
        print(
            f"{prefix}Storage[{mode}@{self.mem_level}] {self.tensor}  "
            f"writeback={self.writeback} lifetime={self.lifetime} anchor={self.anchor}{extra_str}"
        )


# ============================================================
# Action node
# ============================================================


class ActionNode(Node):
    def __init__(
        self,
        action_type: str | ActionType,
        tensors: Sequence[str] = (),
        src_level: Optional[str] = None,
        dst_level: Optional[str] = None,
        anchor: str = "tile_entry",
        attrs: Optional[Dict[str, Any]] = None,
        name: str = "",
    ) -> None:
        normalized = _normalize_action_type(action_type)
        super().__init__(NodeType.Action, name or f"Action:{normalized.value}")
        self.action_type: str = normalized.value
        self.src_level: Optional[str] = src_level
        self.dst_level: Optional[str] = dst_level
        self.tensors: List[str] = list(_stable_unique(tensors))
        self.anchor: str = anchor
        self.attrs: Dict[str, Any] = dict(attrs or {})

    @property
    def kind(self) -> ActionType:
        return ActionType(self.action_type)

    @classmethod
    def load(
        cls,
        tensors: Sequence[str],
        src_level: Optional[str] = None,
        dst_level: Optional[str] = None,
        anchor: str = "tile_entry",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "ActionNode":
        return cls(ActionType.LOAD, tensors, src_level, dst_level, anchor, attrs)

    @classmethod
    def prefetch(
        cls,
        tensors: Sequence[str],
        src_level: Optional[str] = None,
        dst_level: Optional[str] = None,
        anchor: str = "tile_entry",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "ActionNode":
        return cls(ActionType.PREFETCH, tensors, src_level, dst_level, anchor, attrs)

    @classmethod
    def store(
        cls,
        tensors: Sequence[str],
        src_level: Optional[str] = None,
        dst_level: Optional[str] = None,
        anchor: str = "tile_exit",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "ActionNode":
        return cls(ActionType.STORE, tensors, src_level, dst_level, anchor, attrs)

    @classmethod
    def writeback(
        cls,
        tensors: Sequence[str],
        src_level: Optional[str] = None,
        dst_level: Optional[str] = None,
        anchor: str = "tile_exit",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "ActionNode":
        if dst_level is not None:
            upper = dst_level.upper()
            if upper == "DRAM":
                return cls(ActionType.WRITEBACK_DRAM, tensors, src_level, dst_level, anchor, attrs)
            if upper == "SRAM":
                return cls(ActionType.WRITEBACK_SRAM, tensors, src_level, dst_level, anchor, attrs)
        return cls(ActionType.WRITEBACK, tensors, src_level, dst_level, anchor, attrs)

    @classmethod
    def evict(
        cls,
        tensors: Sequence[str],
        src_level: Optional[str] = None,
        dst_level: Optional[str] = None,
        anchor: str = "tile_exit",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "ActionNode":
        return cls(ActionType.EVICT, tensors, src_level, dst_level, anchor, attrs)

    @classmethod
    def barrier(
        cls,
        tensors: Sequence[str] = (),
        anchor: str = "scope_boundary",
        attrs: Optional[Dict[str, Any]] = None,
        name: str = "",
    ) -> "ActionNode":
        return cls(ActionType.BARRIER, tensors, None, None, anchor, attrs, name)

    def accept(self, visitor: MappingTreeVisitor) -> None:
        visitor.visit_action(self)

    def display(self, indent: int = 0) -> None:
        prefix = "  " * indent
        path = f" {self.src_level}->{self.dst_level}" if self.src_level or self.dst_level else ""
        attrs_str = f" attrs={self.attrs}" if self.attrs else ""
        print(f"{prefix}Action[{self.action_type}]{path} tensors={self.tensors} anchor={self.anchor}{attrs_str}")
        for child in self.children:
            child.display(indent + 1)


# ============================================================
# Op node
# ============================================================


class OpNode(Node):
    def __init__(
        self,
        op_id: str,
        op_type: Optional[str] = None,
        parallel_dims: Optional[List[str]] = None,
        name: str = "",
        phase: Optional[str] = None,
        special_role: Optional[str] = None,
    ) -> None:
        super().__init__(NodeType.Op, name or op_id)
        self.op_id: str = op_id
        self.op_type: Optional[str] = op_type
        self.parallel_dims: List[str] = list(parallel_dims or [])
        self.phase: Optional[str] = phase
        self.special_role: Optional[str] = special_role
        self.stats: Optional[OpStats] = None

    def accept(self, visitor: MappingTreeVisitor) -> None:
        visitor.visit_op(self)

    def display(self, indent: int = 0) -> None:
        prefix = "  " * indent
        prim_str = f" type={self.op_type}" if self.op_type else ""
        pdim_str = f" parallel={self.parallel_dims}" if self.parallel_dims else ""
        phase_str = f" phase={self.phase}" if self.phase else ""
        role_str = f" role={self.special_role}" if self.special_role else ""
        stats_str = ""
        if self.stats:
            stats_str = f"  => cycles={self.stats.compute_cycles} macs={self.stats.mac_count}"
            if self.stats.energy is not None:
                stats_str += f" energy={self.stats.energy}"
        print(f"{prefix}Op '{self.op_id}'{prim_str}{pdim_str}{phase_str}{role_str}{stats_str}")


# ============================================================
# MappingTree container
# ============================================================


@dataclass
class MappingTree:
    root: Node
    op_index: Dict[str, OpNode] = field(default_factory=dict)

    def display(self) -> None:
        print("=== MappingTree ===")
        self.root.display(indent=0)
        print("===================")

    def accept(self, visitor: MappingTreeVisitor) -> None:
        visitor.visit(self.root)

    def walk(self) -> Iterable[Node]:
        yield from self.root.walk()

    def collect_tiles(self, mem_level: Optional[str] = None) -> List[TileNode]:
        tiles = [node for node in self.walk() if isinstance(node, TileNode)]
        if mem_level is None:
            return tiles
        return [tile for tile in tiles if tile.mem_level == mem_level]

    def collect_actions(self, action_type: Optional[str | ActionType] = None) -> List[ActionNode]:
        actions = [node for node in self.walk() if isinstance(node, ActionNode)]
        if action_type is None:
            return actions
        normalized = _normalize_action_type(action_type).value
        return [action for action in actions if action.action_type == normalized]

    def collect_scopes(self, scope_type: Optional[ScopeType] = None) -> List[ScopeNode]:
        scopes = [node for node in self.walk() if isinstance(node, ScopeNode)]
        if scope_type is None:
            return scopes
        return [scope for scope in scopes if scope.scope_type == scope_type]

    def rebuild_op_index(self) -> Dict[str, OpNode]:
        self.op_index = {
            node.op_id: node
            for node in self.walk()
            if isinstance(node, OpNode)
        }
        return self.op_index

    def find_op(self, op_id: str) -> Optional[OpNode]:
        if not self.op_index:
            self.rebuild_op_index()
        return self.op_index.get(op_id)


# ============================================================
# Helpers
# ============================================================


def _stable_unique(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _normalize_action_type(action_type: str | ActionType) -> ActionType:
    if isinstance(action_type, ActionType):
        return action_type
    try:
        return ActionType(action_type)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ActionType)
        raise ValueError(f"unsupported action_type: {action_type!r}; valid types: {valid}") from exc
