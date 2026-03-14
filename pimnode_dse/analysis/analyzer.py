from __future__ import annotations

"""TileFlow-style analyzer with lightweight pipeline-aware timing.

This module is intentionally *not* a cycle-accurate simulator. It follows a
TileFlow-like philosophy:
- recursively / structurally estimate data movement volume
- estimate compute work from tiled loop domains
- derive resource-pressure metrics from memory / compute capacities
- optionally apply a lightweight pipeline-aware timing correction on SRAM-PE
  execution scopes

What it is good for:
- DSE ranking and pruning
- top-K selection before detailed DRAM simulation
- latency / energy trend analysis
- comparing sequential vs pipeline-friendly candidates

What it does not model precisely:
- SRAM bank / port conflicts
- fine-grained event scheduling
- cycle-accurate bubbles between SRAM/DE/PE stages
- exact DRAM timing (delegate to Ramulator)
"""

from dataclasses import dataclass, field
from math import ceil, prod
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hardware.arch_spec import PIMNodeArchSpec
from pimnode_dse.mapping.mapping_tree import (
    ActionNode,
    ActionType,
    LoopNode,
    MappingTree,
    Node,
    OpNode,
    PipelineSpec,
    ScopeNode,
    ScopeStats,
    ScopeType,
    TileNode,
    TileStats,
    OpStats,
)
from pimnode_dse.workload.workload import OpSpec, TensorSpec, WorkloadDAG


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------


@dataclass
class AnalyzerConfig:
    bytes_per_element_default: int = 2
    enable_pipeline_correction: bool = True
    enable_energy_estimation: bool = True
    optimistic_pipeline_overlap: bool = True
    overlap_load_compute: bool = True
    overlap_compute_store: bool = True
    count_prefetch_as_load: bool = True
    assume_identity_compute_is_free: bool = False
    # Lightweight energy coefficients for ranking only.
    energy_per_byte_by_level: Dict[str, float] = field(
        default_factory=lambda: {"DRAM": 20.0, "SRAM": 2.0, "PE": 0.5}
    )
    energy_per_mac: float = 1.0
    # If True, total cycles use max(onchip, dram); else add them conservatively.
    overlap_onchip_with_dram: bool = True


@dataclass
class LevelTraffic:
    level_name: str
    read_bytes: int = 0
    write_bytes: int = 0
    action_bytes: Dict[str, int] = field(default_factory=dict)
    peak_live_bytes: int = 0
    bandwidth_demand_bytes_per_cycle: float = 0.0

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes


@dataclass
class ComputeSummary:
    op_count: int = 0
    mac_count: int = 0
    compute_cycles: float = 0.0
    pe_utilization: float = 0.0
    op_cycles: Dict[str, float] = field(default_factory=dict)


@dataclass
class FeasibilityReport:
    feasible: bool = True
    reasons: List[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.feasible = False
        self.reasons.append(reason)


@dataclass
class PipelineTimingSummary:
    pipeline_enabled: bool = False
    scope_name: str = ""
    tile_count: int = 1
    load_cycles_per_tile: float = 0.0
    compute_cycles_per_tile: float = 0.0
    store_cycles_per_tile: float = 0.0
    sequential_cycles: float = 0.0
    startup_cycles: float = 0.0
    steady_state_cycles: float = 0.0
    drain_cycles: float = 0.0
    overlapped_cycles: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class AnalysisSummary:
    candidate_id: str
    feasibility: FeasibilityReport
    per_level: Dict[str, LevelTraffic] = field(default_factory=dict)
    compute: ComputeSummary = field(default_factory=ComputeSummary)
    pipeline: Dict[str, PipelineTimingSummary] = field(default_factory=dict)

    dram_bytes: int = 0
    onchip_bytes: int = 0
    peak_sram_bytes: int = 0

    dram_cycles_est: float = 0.0
    onchip_cycles_est: float = 0.0
    total_cycles_est: float = 0.0
    energy_est: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analyzer implementation
# ---------------------------------------------------------------------------


class Analyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.config = config or AnalyzerConfig()

    # ----------------------------- public API -----------------------------
    def analyze(
        self,
        candidate_id: str,
        tree: MappingTree,
        hardware_spec: PIMNodeArchSpec,
        workload: Optional[WorkloadDAG] = None,
    ) -> AnalysisSummary:
        feasibility = self.check_feasibility(tree, hardware_spec, workload=workload)
        per_level = self.estimate_level_traffic(tree, hardware_spec, workload=workload)
        compute = self.estimate_compute(tree, hardware_spec, workload=workload)
        pipeline = self.estimate_pipeline_timing(tree, hardware_spec, workload=workload)

        dram_bytes = per_level.get("DRAM", LevelTraffic("DRAM")).total_bytes
        onchip_bytes = sum(v.total_bytes for k, v in per_level.items() if k.upper() != "DRAM")
        peak_sram_bytes = per_level.get("SRAM", LevelTraffic("SRAM")).peak_live_bytes
        dram_cycles_est = self.estimate_dram_cycles(dram_bytes, hardware_spec)
        onchip_cycles_est = self.compose_onchip_cycles(compute, pipeline)
        total_cycles_est = self.compose_total_cycles(onchip_cycles_est, dram_cycles_est)
        energy_est = self.estimate_energy(per_level, compute)

        summary = AnalysisSummary(
            candidate_id=candidate_id,
            feasibility=feasibility,
            per_level=per_level,
            compute=compute,
            pipeline=pipeline,
            dram_bytes=dram_bytes,
            onchip_bytes=onchip_bytes,
            peak_sram_bytes=peak_sram_bytes,
            dram_cycles_est=dram_cycles_est,
            onchip_cycles_est=onchip_cycles_est,
            total_cycles_est=total_cycles_est,
            energy_est=energy_est,
            metadata={
                "model": "tileflow_like_with_pipeline_correction",
                "limitations": [
                    "No SRAM bank/port conflict simulation",
                    "No event-level stage scheduling",
                    "DRAM timing is bandwidth-only coarse estimate",
                    "Pipeline overlap inferred from tree structure and PipelineSpec",
                ],
            },
        )

        self._annotate_tree(tree, summary)
        return summary

    def check_feasibility(
        self,
        tree: MappingTree,
        hardware_spec: PIMNodeArchSpec,
        workload: Optional[WorkloadDAG] = None,
    ) -> FeasibilityReport:
        report = FeasibilityReport(feasible=True)
        sram_cap = hardware_spec.sram.total_capacity_bytes

        for tile in tree.collect_tiles("SRAM"):
            tile_bytes = self._estimate_tile_live_bytes(tile, workload, prefer_resident=True)
            if tile_bytes > sram_cap:
                report.fail(
                    f"SRAM tile '{tile.name}' requires {tile_bytes} B > capacity {sram_cap} B"
                )

        # Transfer-path sanity on explicit actions.
        for action in tree.collect_actions():
            if action.src_level and action.dst_level:
                if not hardware_spec.validate_transfer_path(action.src_level, action.dst_level):
                    report.fail(
                        f"Illegal transfer path for action {action.action_type}: "
                        f"{action.src_level}->{action.dst_level}"
                    )

        # Pipeline sanity.
        for scope in tree.collect_scopes(ScopeType.Pipeline):
            if scope.pipeline_spec and not hardware_spec.de.supports_pipeline:
                report.fail(f"Pipeline scope '{scope.name}' present but hardware DE does not support pipeline")

        return report

    def estimate_level_traffic(
        self,
        tree: MappingTree,
        hardware_spec: PIMNodeArchSpec,
        workload: Optional[WorkloadDAG] = None,
    ) -> Dict[str, LevelTraffic]:
        levels = {lvl: LevelTraffic(level_name=lvl) for lvl in hardware_spec.memory_levels}
        # Track tile live bytes as a TileFlow-like peak occupancy proxy.
        for tile in tree.collect_tiles():
            lvl = tile.mem_level.upper()
            if lvl not in levels:
                levels[lvl] = LevelTraffic(level_name=lvl)
            levels[lvl].peak_live_bytes = max(
                levels[lvl].peak_live_bytes,
                self._estimate_tile_live_bytes(tile, workload, prefer_resident=False),
            )

        for action in tree.collect_actions():
            action_kind = action.kind
            bytes_moved = self._estimate_action_bytes(action, workload)

            # Route bytes to source / destination memory levels.
            if action_kind in {ActionType.LOAD, ActionType.PREFETCH}:
                src = (action.src_level or "DRAM").upper()
                dst = (action.dst_level or "SRAM").upper()
                levels.setdefault(src, LevelTraffic(src)).read_bytes += bytes_moved
                levels.setdefault(dst, LevelTraffic(dst)).write_bytes += bytes_moved
                levels[src].action_bytes[action_kind.value] = levels[src].action_bytes.get(action_kind.value, 0) + bytes_moved
                levels[dst].action_bytes[action_kind.value] = levels[dst].action_bytes.get(action_kind.value, 0) + bytes_moved
            elif action_kind in {
                ActionType.STORE,
                ActionType.WRITEBACK,
                ActionType.WRITEBACK_DRAM,
                ActionType.WRITEBACK_SRAM,
                ActionType.EVICT,
            }:
                src = (action.src_level or "SRAM").upper()
                dst = (action.dst_level or "DRAM").upper()
                levels.setdefault(src, LevelTraffic(src)).read_bytes += bytes_moved
                levels.setdefault(dst, LevelTraffic(dst)).write_bytes += bytes_moved
                levels[src].action_bytes[action_kind.value] = levels[src].action_bytes.get(action_kind.value, 0) + bytes_moved
                levels[dst].action_bytes[action_kind.value] = levels[dst].action_bytes.get(action_kind.value, 0) + bytes_moved
            elif action_kind == ActionType.COMPUTE:
                levels.setdefault("PE", LevelTraffic("PE")).action_bytes[action_kind.value] = (
                    levels.setdefault("PE", LevelTraffic("PE")).action_bytes.get(action_kind.value, 0) + bytes_moved
                )
            else:
                # Barrier carries no bytes.
                continue

        # Demand proxy = bytes / estimated cycles for that level.
        for lvl_name, lvl in levels.items():
            bw = self._bandwidth_for_level(lvl_name, hardware_spec)
            if bw > 0:
                lvl.bandwidth_demand_bytes_per_cycle = min(float(lvl.total_bytes), bw)

        return levels

    def estimate_compute(
        self,
        tree: MappingTree,
        hardware_spec: PIMNodeArchSpec,
        workload: Optional[WorkloadDAG] = None,
    ) -> ComputeSummary:
        summary = ComputeSummary()
        macs_per_cycle = max(1, hardware_spec.pe.total_mac_per_cycle)

        for op_node in (n for n in tree.walk() if isinstance(n, OpNode)):
            op_spec = workload.get_op(op_node.op_id) if workload is not None else None
            mac_count = self._estimate_op_work(op_node, op_spec, workload)
            cycles = mac_count / macs_per_cycle if mac_count > 0 else 0.0
            summary.op_count += 1
            summary.mac_count += int(mac_count)
            summary.compute_cycles += cycles
            summary.op_cycles[op_node.op_id] = cycles
            op_node.stats = OpStats(compute_cycles=int(ceil(cycles)), mac_count=int(mac_count), energy=None)

        if summary.compute_cycles > 0:
            ideal = summary.mac_count / macs_per_cycle
            summary.pe_utilization = min(1.0, ideal / max(summary.compute_cycles, 1e-9))
        else:
            summary.pe_utilization = 0.0
        return summary

    def estimate_pipeline_timing(
        self,
        tree: MappingTree,
        hardware_spec: PIMNodeArchSpec,
        workload: Optional[WorkloadDAG] = None,
    ) -> Dict[str, PipelineTimingSummary]:
        out: Dict[str, PipelineTimingSummary] = {}
        for scope in tree.collect_scopes():
            if scope.scope_type not in {ScopeType.Pipeline, ScopeType.Pipelined, ScopeType.Sequential, ScopeType.Group, ScopeType.Phase}:
                continue
            timing = self._estimate_scope_timing(scope, hardware_spec, workload)
            out[scope.name] = timing
        return out

    def estimate_dram_cycles(self, dram_bytes: int, hardware_spec: PIMNodeArchSpec) -> float:
        return float(dram_bytes) / max(hardware_spec.dram.total_bandwidth_bytes_per_cycle, 1e-9)

    def compose_onchip_cycles(
        self,
        compute: ComputeSummary,
        pipeline: Mapping[str, PipelineTimingSummary],
    ) -> float:
        if self.config.enable_pipeline_correction:
            # Prefer the root / largest scope estimate if present.
            pipeline_candidates = [p.overlapped_cycles for p in pipeline.values() if p.overlapped_cycles > 0]
            if pipeline_candidates:
                return max(pipeline_candidates)
        return compute.compute_cycles

    def compose_total_cycles(self, onchip_cycles_est: float, dram_cycles_est: float) -> float:
        if self.config.overlap_onchip_with_dram:
            return max(onchip_cycles_est, dram_cycles_est)
        return onchip_cycles_est + dram_cycles_est

    def estimate_energy(
        self,
        per_level: Mapping[str, LevelTraffic],
        compute: ComputeSummary,
    ) -> float:
        if not self.config.enable_energy_estimation:
            return 0.0
        e = 0.0
        for level_name, traffic in per_level.items():
            coeff = self.config.energy_per_byte_by_level.get(level_name.upper(), 1.0)
            e += coeff * float(traffic.total_bytes)
        e += self.config.energy_per_mac * float(compute.mac_count)
        return e

    # -------------------------- internal helpers --------------------------
    def _estimate_scope_timing(
        self,
        scope: ScopeNode,
        hardware_spec: PIMNodeArchSpec,
        workload: Optional[WorkloadDAG],
    ) -> PipelineTimingSummary:
        actions = [n for n in scope.walk() if isinstance(n, ActionNode)]
        op_nodes = [n for n in scope.walk() if isinstance(n, OpNode)]

        load_bytes = 0
        store_bytes = 0
        for action in actions:
            b = self._estimate_action_bytes(action, workload)
            if action.kind in {ActionType.LOAD, ActionType.PREFETCH}:
                load_bytes += b
            elif action.kind in {
                ActionType.STORE,
                ActionType.WRITEBACK,
                ActionType.WRITEBACK_DRAM,
                ActionType.WRITEBACK_SRAM,
                ActionType.EVICT,
            }:
                store_bytes += b

        mac_count = 0
        for op_node in op_nodes:
            op_spec = workload.get_op(op_node.op_id) if workload is not None else None
            mac_count += self._estimate_op_work(op_node, op_spec, workload)

        load_cycles = load_bytes / max(hardware_spec.onchip_total_feed_bandwidth_bytes_per_cycle, 1e-9)
        compute_cycles = mac_count / max(hardware_spec.pe.total_mac_per_cycle, 1e-9)
        store_bw = hardware_spec.pe.output_bandwidth_bytes_per_cycle or hardware_spec.de.bus_bandwidth_bytes_per_cycle
        store_cycles = store_bytes / max(store_bw, 1e-9)
        seq = load_cycles + compute_cycles + store_cycles

        pipeline_enabled = False
        overlapped = seq
        startup = 0.0
        steady = seq
        drain = 0.0
        notes: List[str] = []
        tile_count = max(1, self._infer_scope_tile_count(scope, workload))

        ps = scope.pipeline_spec
        if (
            self.config.enable_pipeline_correction
            and scope.scope_type in {ScopeType.Pipeline, ScopeType.Pipelined}
            and ps is not None
            and hardware_spec.de.supports_pipeline
            and tile_count > 1
        ):
            pipeline_enabled = True
            inflight = max(1, min(hardware_spec.de.max_inflight_tiles, ps.num_stages))
            can_lc = self.config.overlap_load_compute and ps.allow_interleave and ps.buffering != "single"
            can_cs = self.config.overlap_compute_store and ps.allow_interleave and ps.buffering != "single"

            if can_lc and can_cs:
                steady = max(load_cycles, compute_cycles, store_cycles)
                notes.append("load/compute/store overlapped in steady state")
            elif can_lc:
                steady = max(load_cycles, compute_cycles) + store_cycles
                notes.append("load/compute overlapped; store serialized")
            elif can_cs:
                steady = load_cycles + max(compute_cycles, store_cycles)
                notes.append("compute/store overlapped; load serialized")
            else:
                steady = seq
                notes.append("pipeline scope present but overlap disabled by config/spec")

            startup = load_cycles + (0.25 * compute_cycles if can_lc else 0.0)
            drain = store_cycles + (0.25 * compute_cycles if can_cs else 0.0)
            effective_tiles = max(1, tile_count)
            if inflight > 1:
                effective_tiles = max(1, tile_count - (inflight - 1))
                notes.append(f"inflight_tiles={inflight} reduces steady-state exposed tiles")
            overlapped = startup + max(0, effective_tiles - 1) * steady + drain
        else:
            if scope.scope_type in {ScopeType.Pipeline, ScopeType.Pipelined} and ps is not None and tile_count <= 1:
                notes.append("pipeline scope has <=1 exposed tile; no steady-state overlap realized")
            elif scope.scope_type in {ScopeType.Pipeline, ScopeType.Pipelined} and not hardware_spec.de.supports_pipeline:
                notes.append("hardware DE disables pipeline support")
            elif scope.scope_type in {ScopeType.Pipeline, ScopeType.Pipelined}:
                notes.append("pipeline correction fell back to sequential timing")

        timing = PipelineTimingSummary(
            pipeline_enabled=pipeline_enabled,
            scope_name=scope.name,
            tile_count=tile_count,
            load_cycles_per_tile=load_cycles / max(tile_count, 1),
            compute_cycles_per_tile=compute_cycles / max(tile_count, 1),
            store_cycles_per_tile=store_cycles / max(tile_count, 1),
            sequential_cycles=seq,
            startup_cycles=startup,
            steady_state_cycles=steady,
            drain_cycles=drain,
            overlapped_cycles=overlapped,
            notes=notes,
        )
        scope.stats = ScopeStats(total_cycles=int(ceil(overlapped)), overlap_ratio=(seq / overlapped) if overlapped > 0 else None)
        return timing

    def _infer_scope_tile_count(self, scope: ScopeNode, workload: Optional[WorkloadDAG]) -> int:
        """Infer how many tiles are exposed to a pipeline scope.

        Priority:
        1. LoopNode.loop_count matching pipeline_dim
        2. TileNode.loop_count matching pipeline_dim
        3. Workload global dimension for pipeline_dim
        4. Legacy LoopNode.extent / tile_extent fallback
        5. Count / max size of SRAM/PE tiles
        """
        ps = scope.pipeline_spec
        if ps and ps.pipeline_dim:
            target_dim = ps.pipeline_dim

            # 1) explicit loop_count on LoopNode
            for node in scope.walk():
                if isinstance(node, LoopNode) and node.iter_dim == target_dim:
                    if node.loop_count is not None and int(node.loop_count) > 0:
                        return int(node.loop_count)

            # 2) explicit loop_count on TileNode
            for node in scope.walk():
                if isinstance(node, TileNode):
                    lc = getattr(node, "loop_count", None) or {}
                    if target_dim in lc and int(lc[target_dim]) > 0:
                        return int(lc[target_dim])

            # 3) workload global dimension fallback
            if workload is not None and workload.spec is not None:
                spec = workload.spec
                dim_defaults = {
                    "b": getattr(spec, "B", None),
                    "h": getattr(spec, "Hq", None),
                    "head": getattr(spec, "Hq", None),
                    "m": getattr(spec, "Q_len", None),
                    "q": getattr(spec, "Q_len", None),
                    "n": getattr(spec, "KV_total", None) if getattr(spec, "mode", "").lower() == "decode" else getattr(spec, "KV_len", None),
                    "kv": getattr(spec, "KV_total", None) if getattr(spec, "mode", "").lower() == "decode" else getattr(spec, "KV_len", None),
                    "d": getattr(spec, "Dh", None),
                }
                guess = dim_defaults.get(target_dim)
                if guess is not None and int(guess) > 0:
                    return int(guess)

            # 4) legacy fallback to extent / tile_extent
            for node in scope.walk():
                iter_dim = getattr(node, "iter_dim", None)
                if iter_dim != target_dim:
                    continue
                loop_count = getattr(node, "loop_count", None)
                if loop_count is not None and int(loop_count) > 0:
                    return int(loop_count)
                tile_extent = getattr(node, "tile_extent", None)
                if tile_extent is not None and int(tile_extent) > 0:
                    return int(tile_extent)
                extent = getattr(node, "extent", None)
                if extent is not None and int(extent) > 0:
                    return int(extent)

        # 5) fallback: number/size of SRAM or PE tiles in scope
        tiles = [n for n in scope.walk() if isinstance(n, TileNode) and n.mem_level.upper() in {"SRAM", "PE"}]
        if tiles:
            sizes = []
            for tile in tiles:
                if getattr(tile, "loop_count", None):
                    vals = [int(v) for v in tile.loop_count.values() if int(v) > 0]
                    if vals:
                        sizes.append(max(vals))
                        continue
                if tile.tile_size:
                    sizes.append(max(tile.tile_size.values()))
            if sizes:
                return max(1, max(sizes))
            return max(1, len(tiles))
        return 1

    def _estimate_action_bytes(self, action: ActionNode, workload: Optional[WorkloadDAG]) -> int:
        total = 0
        nearest_op = self._nearest_ancestor(action, OpNode)
        nearest_tile = self._nearest_ancestor(action, TileNode)
        op_spec = workload.get_op(nearest_op.op_id) if (workload is not None and nearest_op is not None) else None

        for tensor_name in action.tensors:
            total += self._estimate_tensor_bytes_for_tile(
                tensor_name=tensor_name,
                tile=nearest_tile,
                workload=workload,
                op_spec=op_spec,
            )
        return total

    def _estimate_tile_live_bytes(
        self,
        tile: TileNode,
        workload: Optional[WorkloadDAG],
        *,
        prefer_resident: bool,
    ) -> int:
        if workload is None:
            elems = prod(tile.tile_size.values()) if tile.tile_size else 0
            return int(elems * self.config.bytes_per_element_default)

        tensor_names: Sequence[str]
        resident = list(tile.resident_tensors())
        boundary = list(tile.prefetch_tensors()) + list(tile.writeback_tensors()) + list(tile.evict_tensors())
        if prefer_resident and resident:
            tensor_names = resident
        elif resident or boundary:
            tensor_names = resident + boundary
        else:
            nearest_op = self._nearest_descendant(tile, OpNode)
            tensor_names = list((nearest_op.inputs + nearest_op.outputs) if nearest_op is not None else [])

        nearest_op = self._nearest_descendant(tile, OpNode)
        op_spec = workload.get_op(nearest_op.op_id) if nearest_op is not None else None
        return sum(
            self._estimate_tensor_bytes_for_tile(tn, tile, workload, op_spec)
            for tn in dict.fromkeys(tensor_names)
        )

    def _estimate_tensor_bytes_for_tile(
        self,
        tensor_name: str,
        tile: Optional[TileNode],
        workload: Optional[WorkloadDAG],
        op_spec: Optional[OpSpec],
    ) -> int:
        if workload is None or tensor_name not in workload.tensors:
            elems = prod(tile.tile_size.values()) if (tile and tile.tile_size) else 1
            return int(elems * self.config.bytes_per_element_default)

        tensor = workload.tensors[tensor_name]
        elem_bytes = self._tensor_bytes_per_element(tensor, workload)
        if tile is None or not tile.tile_size:
            return int(prod(tensor.shape) * elem_bytes)

        # Best-effort TileFlow-like projection: map symbolic dims to tile extents.
        if op_spec is not None and tensor_name in op_spec.tensor_index:
            symbolic_axes = op_spec.tensor_index[tensor_name]
            axes_to_shape = {sym: dim for sym, dim in zip(symbolic_axes, tensor.shape)}
            elems = 1
            for sym, full_dim in axes_to_shape.items():
                tile_extent = tile.tile_size.get(sym, full_dim)
                elems *= max(1, min(int(tile_extent), int(full_dim)))
            return int(elems * elem_bytes)

        # Fallback: clip tensor shape prefix by sorted tile extents.
        extents = list(tile.tile_size.values())
        elems = 1
        for i, full_dim in enumerate(tensor.shape):
            extent = extents[i] if i < len(extents) else full_dim
            elems *= max(1, min(int(extent), int(full_dim)))
        return int(elems * elem_bytes)

    def _estimate_op_work(
        self,
        op_node: OpNode,
        op_spec: Optional[OpSpec],
        workload: Optional[WorkloadDAG],
    ) -> int:
        tile = self._nearest_ancestor(op_node, TileNode) or self._nearest_descendant(op_node, TileNode)
        if op_spec is None:
            if tile and tile.tile_size:
                return int(prod(tile.tile_size.values()))
            return 0

        op_type = (op_spec.op_type or "").lower()
        tile_map = dict(tile.tile_size) if (tile and tile.tile_size) else {}

        def extent(var: str, default: int = 1) -> int:
            if var in tile_map:
                return max(1, int(tile_map[var]))
            if workload is not None and workload.spec is not None:
                spec = workload.spec
                defaults = {
                    "b": spec.B,
                    "h": spec.Hq,
                    "m": spec.Q_len,
                    "n": spec.KV_total if spec.mode.lower() == "decode" else spec.KV_len,
                    "d": spec.Dh,
                }
                return int(defaults.get(var, default))
            return default

        if op_type == "matmul":
            out_name = op_spec.outputs[0] if op_spec.outputs else None
            reduction_dim = str(op_spec.attrs.get("reduction_dim", ""))
            output_elems = 1
            if out_name and out_name in op_spec.tensor_index:
                for sym in op_spec.tensor_index[out_name]:
                    output_elems *= extent(sym)
            else:
                output_elems = int(prod(tile_map.values())) if tile_map else 1
            red_extent = extent(reduction_dim, 1) if reduction_dim else 1
            return int(output_elems * red_extent)

        if op_type == "softmax":
            in_name = op_spec.inputs[0] if op_spec.inputs else None
            elems = 1
            if in_name and in_name in op_spec.tensor_index:
                for sym in op_spec.tensor_index[in_name]:
                    elems *= extent(sym)
            else:
                elems = int(prod(tile_map.values())) if tile_map else 1
            return int(5 * elems)

        if op_type in {"identity", "kvappend"}:
            if self.config.assume_identity_compute_is_free:
                return 0
            out_name = op_spec.outputs[0] if op_spec.outputs else None
            elems = 1
            if out_name and out_name in op_spec.tensor_index:
                for sym in op_spec.tensor_index[out_name]:
                    elems *= extent(sym)
            else:
                elems = int(prod(tile_map.values())) if tile_map else 1
            return int(elems)

        if op_spec.index_vars:
            return int(prod(extent(v) for v in op_spec.index_vars))
        return int(prod(tile_map.values())) if tile_map else 1

    def _tensor_bytes_per_element(self, tensor: TensorSpec, workload: WorkloadDAG) -> int:
        dtype = (tensor.dtype or "").lower()
        table = {
            "fp16": 2,
            "float16": 2,
            "bf16": 2,
            "float32": 4,
            "fp32": 4,
            "int8": 1,
            "uint8": 1,
            "int16": 2,
            "int32": 4,
        }
        if dtype in table:
            return table[dtype]
        if workload.spec is not None:
            return int(workload.spec.dtype_bytes)
        return int(self.config.bytes_per_element_default)

    def _bandwidth_for_level(self, level_name: str, hardware_spec: PIMNodeArchSpec) -> float:
        upper = level_name.upper()
        if upper == "DRAM":
            return float(hardware_spec.dram.total_bandwidth_bytes_per_cycle)
        if upper == "SRAM":
            return float(hardware_spec.sram.total_bandwidth_bytes_per_cycle)
        if upper == "PE":
            return float(hardware_spec.pe.input_bandwidth_bytes_per_cycle or hardware_spec.onchip_total_feed_bandwidth_bytes_per_cycle)
        return 0.0

    def _nearest_ancestor(self, node: Node, cls: type) -> Optional[Any]:
        cur = node.parent
        while cur is not None:
            if isinstance(cur, cls):
                return cur
            cur = cur.parent
        return None

    def _nearest_descendant(self, node: Node, cls: type) -> Optional[Any]:
        for child in node.children:
            if isinstance(child, cls):
                return child
            found = self._nearest_descendant(child, cls)
            if found is not None:
                return found
        return None

    def _annotate_tree(self, tree: MappingTree, summary: AnalysisSummary) -> None:
        # Per-tile rough stats.
        for tile in tree.collect_tiles():
            tile.stats = TileStats(
                active_tensors=list(dict.fromkeys(tile.resident_tensors() + tile.prefetch_tensors() + tile.writeback_tensors() + tile.evict_tensors())),
                working_set_bytes={"live": self._estimate_tile_live_bytes(tile, None, prefer_resident=False)},
            )

        # Scope stats already set during pipeline estimation.
        root_scope = tree.root if isinstance(tree.root, ScopeNode) else None
        if root_scope is not None:
            if root_scope.stats is None:
                root_scope.stats = ScopeStats(total_cycles=int(ceil(summary.total_cycles_est)))
            else:
                root_scope.stats.total_cycles = int(ceil(summary.total_cycles_est))


__all__ = [
    "AnalyzerConfig",
    "LevelTraffic",
    "ComputeSummary",
    "FeasibilityReport",
    "PipelineTimingSummary",
    "AnalysisSummary",
    "Analyzer",
]
