from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from pimnode_dse.hardware.arch_spec import HardwareSpec
from pimnode_dse.mapping.tree.mapping_tree import (
    MappingTree, MapNode, Move, OpNode, ScopeNode, TileNode,
)
from pimnode_dse.mapping.workload.workload import WorkloadDAG

from .types import AnalyticalMetrics, Candidate, ObjectiveMetrics


# --------------------------------------------------
# Internal estimate data structures
# --------------------------------------------------

@dataclass(frozen=True)
class TensorTileStat:
    tensor: str
    level: str
    role: str
    tile_elements: int
    tile_bytes: int
    traffic_bytes: int
    reads: int
    writes: int
    reuse_factor: float


@dataclass
class LevelCost:
    level: str
    bandwidth: float = 1.0
    read_bytes: int = 0
    write_bytes: int = 0
    movement_bytes: int = 0
    accesses: int = 0
    movement_cycles: float = 0.0
    peak_live_bytes: int = 0


@dataclass(frozen=True)
class OpCost:
    op_id: str
    kind: str
    flops: float
    compute_cycles: float
    reduction_loops: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionProfile:
    """Analytical execution profile for one subtree.

    total_cycles = fill_cycles + iterations * steady_cycles + drain_cycles

    fill/drain semantics
    --------------------
    fill_cycles  : cycles consumed before the pipeline reaches steady state
                   (e.g. cold-load of the first tile, pipeline warm-up).
    steady_cycles: cycles per repeating unit in steady state.
    drain_cycles : cycles after the last unit (pipeline drain, final write-back).

    For a seq-bind scope these are the fill of the FIRST child and the drain
    of the LAST child, preserving the shape for outer pipe nodes.
    """

    node_id: str
    node_type: str
    level: str
    iterations: int = 1

    local_compute_cycles: float = 0.0
    local_movement_cycles: float = 0.0

    compute_bound_cycles: float = 0.0
    movement_bound_cycles: float = 0.0
    overlappable_cycles: float = 0.0
    non_overlappable_cycles: float = 0.0

    fill_cycles: float = 0.0
    steady_cycles: float = 0.0
    drain_cycles: float = 0.0

    @property
    def total_cycles(self) -> float:
        return float(self.fill_cycles + self.iterations * self.steady_cycles + self.drain_cycles)


@dataclass(frozen=True)
class TreeEstimate:
    compute_ops: float
    compute_cycles: float
    level_costs: Dict[str, LevelCost]
    op_costs: Tuple[OpCost, ...]
    tensor_stats: Tuple[TensorTileStat, ...]
    movement_bytes: int
    dram_bytes: int
    sram_peak_bytes: int
    trace_request_count: int
    latency_proxy: float
    root_profile: ExecutionProfile


@dataclass(frozen=True)
class EstimateContext:
    workload: WorkloadDAG
    hardware: HardwareSpec
    tensor_dims: Dict[str, Tuple[str, ...]]
    tensor_sizes: Dict[str, int]
    tensor_elements: Dict[str, int]
    loop_extent: Dict[str, int]


# --------------------------------------------------
# Public API
# --------------------------------------------------

def estimate_candidate(
    candidate: Candidate,
    workload: WorkloadDAG,
    hw: HardwareSpec,
) -> AnalyticalMetrics:
    tree_est = estimate_tree(candidate.mapping.tree, workload, hw)
    return AnalyticalMetrics(
        dram_bytes=int(tree_est.dram_bytes),
        movement_bytes=int(tree_est.movement_bytes),
        sram_peak_bytes=int(tree_est.sram_peak_bytes),
        est_compute_cycles=float(tree_est.latency_proxy),
        trace_request_count=int(tree_est.trace_request_count),
        est_power_mw=None,  # filled only after DRAMPower simulation
    )


def objectives_from_analytical(metrics: AnalyticalMetrics) -> ObjectiveMetrics:
    return ObjectiveMetrics(
        latency=float(metrics.est_compute_cycles),
        dram_cost=float(metrics.dram_bytes),
        movement=float(metrics.movement_bytes),
        power_mw=None,  # no power at analytical stage; populated post-simulation
    )


def estimate_tree(
    tree: MappingTree,
    workload: WorkloadDAG,
    hw: HardwareSpec,
) -> TreeEstimate:
    """Single-pass post-order traversal that both collects static accounting
    and builds the ExecutionProfile bottom-up.

    Previously the tree was walked twice:
      1. tree.walk() for static accounting (scope/tile/op passes)
      2. recursive estimate_node_profile for the profile

    Now both happen in one post-order walk via _postorder_estimate,
    eliminating the redundant traversal.
    """
    ctx = build_estimate_context(workload, hw)
    level_costs = init_level_costs(hw)
    pinned_keep: Dict[str, Set[str]] = {}
    tensor_stats: List[TensorTileStat] = []
    op_costs: List[OpCost] = []

    root_profile = _postorder_estimate(
        tree.root, ctx, level_costs, pinned_keep, tensor_stats, op_costs
    )

    movement_bytes = sum(v.movement_bytes for v in level_costs.values())
    dram_bytes = level_costs.get("dram", LevelCost(level="dram")).movement_bytes
    sram_peak_bytes = level_costs.get("sram", LevelCost(level="sram")).peak_live_bytes
    trace_request_count = _trace_requests_from_level_cost(
        level_costs.get("dram", LevelCost(level="dram"))
    )

    return TreeEstimate(
        compute_ops=sum(x.flops for x in op_costs),
        compute_cycles=float(sum(x.compute_cycles for x in op_costs)),
        level_costs=level_costs,
        op_costs=tuple(op_costs),
        tensor_stats=tuple(tensor_stats),
        movement_bytes=int(movement_bytes),
        dram_bytes=int(dram_bytes),
        sram_peak_bytes=int(sram_peak_bytes),
        trace_request_count=int(trace_request_count),
        latency_proxy=float(root_profile.total_cycles),
        root_profile=root_profile,
    )


# --------------------------------------------------
# Single-pass post-order estimator
# --------------------------------------------------

def _postorder_estimate(
    node: MapNode,
    ctx: EstimateContext,
    level_costs: Dict[str, LevelCost],
    pinned_keep: Dict[str, Set[str]],
    tensor_stats: List[TensorTileStat],
    op_costs: List[OpCost],
) -> ExecutionProfile:
    """Recurse depth-first; on the way back up, do accounting + profile."""

    if isinstance(node, OpNode):
        oc = estimate_op_cost(node, ctx)
        op_costs.append(oc)
        cycles = oc.compute_cycles
        return ExecutionProfile(
            node_id=node.id,
            node_type="op",
            level="pe",
            iterations=1,
            local_compute_cycles=cycles,
            local_movement_cycles=0.0,
            compute_bound_cycles=cycles,
            movement_bound_cycles=0.0,
            overlappable_cycles=0.0,
            non_overlappable_cycles=cycles,
            fill_cycles=0.0,
            steady_cycles=cycles,
            drain_cycles=0.0,
        )

    if isinstance(node, TileNode):
        child_profile = (
            _postorder_estimate(node.kid, ctx, level_costs, pinned_keep, tensor_stats, op_costs)
            if node.kid is not None
            else zero_profile(node.id, "tile", norm_level(node.attrs.get("level", "unknown")))
        )
        # Tile accounting (peak bytes only; traffic is owned by parent scope entry/exit)
        _apply_tile_peak(node, ctx, level_costs, tensor_stats)
        return _build_tile_profile(node, ctx, child_profile)

    if isinstance(node, ScopeNode):
        child_profiles = [
            _postorder_estimate(k, ctx, level_costs, pinned_keep, tensor_stats, op_costs)
            for k in node.kids
        ]
        # Scope accounting: act-classified traffic + pinned-keep dedup peak
        _apply_scope_accounting(node, ctx, level_costs, pinned_keep)
        return _build_scope_profile(node, ctx, child_profiles)

    return zero_profile("unknown", "unknown", "unknown")


# --------------------------------------------------
# Context building
# --------------------------------------------------

def build_estimate_context(
    workload: WorkloadDAG,
    hw: HardwareSpec,
) -> EstimateContext:
    tensor_dims: Dict[str, Tuple[str, ...]] = {}
    loop_extent: Dict[str, int] = {}
    tensor_sizes: Dict[str, int] = {}
    tensor_elements: Dict[str, int] = {}

    for tname, tspec in workload.tensors().items():
        tensor_sizes[tname] = int(tspec.size_bytes())
        elems = 1
        for d in tspec.shape:
            elems *= int(d)
        tensor_elements[tname] = max(1, elems)

    for _, op in workload.ops().items():
        for ten, dims in getattr(op, "tensor_dims", {}).items():
            if ten not in tensor_dims:
                tensor_dims[str(ten)] = tuple(str(x).strip().lower() for x in dims)
        for dim, val in getattr(op, "dim_constraints", {}).items():
            k = str(dim).strip().lower()
            loop_extent[k] = max(loop_extent.get(k, 1), int(val))
        for dim in getattr(op, "iter_dims", ()):
            k = str(dim).strip().lower()
            loop_extent.setdefault(k, 1)

    return EstimateContext(
        workload=workload,
        hardware=hw,
        tensor_dims=tensor_dims,
        tensor_sizes=tensor_sizes,
        tensor_elements=tensor_elements,
        loop_extent=loop_extent,
    )


def init_level_costs(hw: HardwareSpec) -> Dict[str, LevelCost]:
    return {
        "dram": LevelCost(level="dram", bandwidth=max(1.0, float(hw.dram.bw_hint))),
        "sram": LevelCost(level="sram", bandwidth=max(1.0, float(hw.sram.bw))),
        "pe":   LevelCost(level="pe",   bandwidth=max(1.0, float(hw.pe.mac_per_cycle))),
    }


# --------------------------------------------------
# Static accounting helpers
# --------------------------------------------------

_READ_ACTS  = frozenset({"load", "prefetch"})
_WRITE_ACTS = frozenset({"store", "evict", "writeback"})


def _apply_scope_accounting(
    scope: ScopeNode,
    ctx: EstimateContext,
    level_costs: Dict[str, LevelCost],
    pinned_keep: Dict[str, Set[str]],
) -> None:
    """Act-classified read/write accounting + pinned-keep dedup peak.

    Fix 1 — act-classified bytes:
      entry moves with act in {load, prefetch}      → read_bytes
      entry/exit moves with act in {store, evict, writeback} → write_bytes
      Unknown act defaults conservatively to read on entry, write on exit.

    Fix 3 — in-place tensor dedup:
      Tensors appearing in BOTH live_in AND live_out with a matching
      writeback/store act (classic in-place update pattern, e.g. softmax)
      are counted only once — as if they occupy a single buffer slot.

    Fix 2 (peak) — pinned keep dedup:
      scope.keep tensors are resident for the whole tree at this level.
      We track which (level, tensor) pairs have been counted; new keep
      tensors contribute to peak only once.

      Transient tensors (live_in | live_out) \ keep are counted in full
      each time a scope is visited because each scope iteration may have
      a different resident set.
    """
    level = norm_level(scope.mem)
    lc = level_costs.setdefault(
        level, LevelCost(level=level, bandwidth=level_bandwidth(level, ctx.hardware))
    )

    read_bytes = 0
    write_bytes = 0

    # Identify in-place tensors: appear in both live_in and live_out,
    # and have a writeback/store on exit (in-place update, not copy-out).
    inplace_tens: Set[str] = set()
    exit_wb_tens = {m.tens for m in scope.exit if m.act in _WRITE_ACTS}
    if scope.live_in and scope.live_out:
        inplace_tens = set(scope.live_in) & set(scope.live_out) & exit_wb_tens

    for m in scope.entry:
        act = m.act
        sz = _move_bytes(m, ctx)
        if act in _READ_ACTS:
            read_bytes += sz
        elif act in _WRITE_ACTS:
            write_bytes += sz
        else:
            read_bytes += sz   # unknown → conservative read

    for m in scope.exit:
        act = m.act
        sz = _move_bytes(m, ctx)
        if act in _WRITE_ACTS:
            # In-place tensors: the write-back reuses the same buffer that was
            # loaded on entry; no additional traffic for the exit movement.
            if m.tens in inplace_tens:
                continue
            write_bytes += sz
        elif act in _READ_ACTS:
            read_bytes += sz
        else:
            write_bytes += sz  # unknown exit → conservative write

    lc.read_bytes  += int(read_bytes)
    lc.write_bytes += int(write_bytes)
    lc.movement_bytes += int(read_bytes + write_bytes)
    lc.accesses += len(scope.entry) + len(scope.exit)
    lc.movement_cycles += float(read_bytes + write_bytes) / max(1.0, lc.bandwidth)

    # --- peak live bytes ---
    seen_at_level: Set[str] = pinned_keep.setdefault(level, set())

    # Transient tensors: full size each scope visit.
    # Fix 3: in-place tensors counted once (live_in side only).
    transient = (set(scope.live_in) | set(scope.live_out)) - set(scope.keep) - inplace_tens
    inplace_transient = inplace_tens - set(scope.keep)
    transient_peak = (
        sum(ctx.tensor_sizes.get(t, 64) for t in transient)
        + sum(ctx.tensor_sizes.get(t, 64) for t in inplace_transient)
    )

    # Keep tensors: counted only once per (level, tensor).
    new_keep = set(scope.keep) - seen_at_level
    seen_at_level.update(new_keep)
    keep_peak = sum(ctx.tensor_sizes.get(t, 64) for t in new_keep)

    lc.peak_live_bytes = max(lc.peak_live_bytes, int(transient_peak + keep_peak))


def _apply_tile_peak(
    tile: TileNode,
    ctx: EstimateContext,
    level_costs: Dict[str, LevelCost],
    tensor_stats: List[TensorTileStat],
) -> None:
    """Record per-tile peak SRAM usage and tensor stats.

    Fix 2 — double-buffer peak:
      Only *streamed* (non-keep) tensors need a second buffer slot.
      Pinned/weight tensors that stay resident do not.
      We identify streamed tensors as those in active_tensors whose
      inferred role is NOT "state" — state tensors are written in-place
      and stay in a single buffer.

    Traffic is NOT accumulated here; it is owned by the parent ScopeNode's
    entry/exit Moves to avoid double-counting at the same level.
    """
    level = norm_level(tile.attrs.get("level", "unknown"))
    lc = level_costs.setdefault(
        level, LevelCost(level=level, bandwidth=level_bandwidth(level, ctx.hardware))
    )

    active_tensors = infer_tile_tensors(tile)
    buf_mode = str(tile.attrs.get("buf_mode", "single")).strip().lower()

    streamed_bytes = 0
    static_bytes   = 0

    for ten in active_tensors:
        elems = estimate_tensor_tile_elements(ten, tile.size, ctx)
        total_elems = max(1, ctx.tensor_elements.get(ten, 1))
        total_bytes = ctx.tensor_sizes.get(ten, 64)
        bytes_per_elem = max(1, total_bytes // total_elems)
        tile_bytes = int(elems * bytes_per_elem)

        role = infer_tensor_role(ten)
        reuse = estimate_reuse_factor(ten, tile, ctx)
        traffic_bytes = int(max(1.0, tile_bytes / max(1.0, reuse)))

        reads  = 1
        writes = 1 if role in {"output", "state"} else 0

        tensor_stats.append(TensorTileStat(
            tensor=ten,
            level=level,
            role=role,
            tile_elements=int(elems),
            tile_bytes=int(tile_bytes),
            traffic_bytes=int(traffic_bytes),
            reads=int(reads),
            writes=int(writes),
            reuse_factor=float(reuse),
        ))

        # "state" tensors are in-place accumulators — single buffer, no double-buf
        if role == "state":
            static_bytes += tile_bytes
        else:
            streamed_bytes += tile_bytes

    # Fix 2: only streamed tensors doubled; state/pinned tensors single.
    if buf_mode == "double":
        peak_live = streamed_bytes * 2 + static_bytes
    else:
        peak_live = streamed_bytes + static_bytes

    lc.peak_live_bytes = max(lc.peak_live_bytes, int(peak_live))


# --------------------------------------------------
# Profile builders
# --------------------------------------------------

def _build_tile_profile(
    node: TileNode,
    ctx: EstimateContext,
    child: ExecutionProfile,
) -> ExecutionProfile:
    level = norm_level(node.attrs.get("level", "unknown"))
    local_move = tile_local_movement_cycles(node, ctx)
    buf_mode = str(node.attrs.get("buf_mode", "single")).strip().lower()
    overlap_on = tile_overlap_enabled(node) or (buf_mode == "double")

    if overlap_on:
        # Steady state: compute and prefetch overlap → bottleneck is the max.
        # Fill: cold-load of the very first tile (not hidden by overlap).
        # Drain: final compute after the last prefetch has completed.
        steady      = max(child.steady_cycles, local_move)
        fill        = child.fill_cycles + local_move
        drain       = child.drain_cycles
        overlappable = child.overlappable_cycles + min(child.steady_cycles, local_move)
        non_over     = child.non_overlappable_cycles + max(child.steady_cycles, local_move)
    else:
        steady      = child.steady_cycles + local_move
        fill        = child.fill_cycles
        drain       = child.drain_cycles
        overlappable = child.overlappable_cycles
        non_over     = child.non_overlappable_cycles + local_move

    repeat     = tile_repeat_count(node, ctx)
    iterations = max(1, child.iterations * repeat)

    return ExecutionProfile(
        node_id=node.id,
        node_type="tile",
        level=level,
        iterations=iterations,
        local_compute_cycles=0.0,
        local_movement_cycles=float(local_move),
        compute_bound_cycles=child.compute_bound_cycles,
        movement_bound_cycles=child.movement_bound_cycles + float(local_move),
        overlappable_cycles=float(overlappable),
        non_overlappable_cycles=float(non_over),
        fill_cycles=float(fill),
        steady_cycles=float(steady),
        drain_cycles=float(drain),
    )


def _build_scope_profile(
    node: ScopeNode,
    ctx: EstimateContext,
    child_profiles: Sequence[ExecutionProfile],
) -> ExecutionProfile:
    local_move = scope_local_movement_cycles(node, ctx)
    bind = str(node.bind).strip().lower()
    combined = combine_children_by_bind(bind, child_profiles)

    return ExecutionProfile(
        node_id=node.id,
        node_type="scope",
        level=norm_level(node.mem),
        iterations=1,
        local_compute_cycles=0.0,
        local_movement_cycles=float(local_move),
        compute_bound_cycles=combined.compute_bound_cycles,
        movement_bound_cycles=combined.movement_bound_cycles + float(local_move),
        overlappable_cycles=combined.overlappable_cycles,
        non_overlappable_cycles=combined.non_overlappable_cycles + float(local_move),
        fill_cycles=float(combined.fill_cycles),
        steady_cycles=float(combined.steady_cycles + local_move),
        drain_cycles=float(combined.drain_cycles),
    )


def combine_children_by_bind(
    bind: str,
    child_profiles: Sequence[ExecutionProfile],
) -> ExecutionProfile:
    if not child_profiles:
        return zero_profile("empty", "scope", "unknown")

    vals = list(child_profiles)

    if bind == "seq":
        # Fix 1: seq preserves the first child's fill and the last child's drain
        # so that an enclosing pipe node sees the correct pipeline shape.
        # Steady = sum of all children's total_cycles MINUS the first fill and
        # last drain (those are already captured separately).
        first, last = vals[0], vals[-1]
        mid_cycles = sum(x.total_cycles for x in vals)
        return ExecutionProfile(
            node_id="seq",
            node_type="scope",
            level="unknown",
            iterations=1,
            local_compute_cycles=0.0,
            local_movement_cycles=0.0,
            compute_bound_cycles=sum(x.compute_bound_cycles for x in vals),
            movement_bound_cycles=sum(x.movement_bound_cycles for x in vals),
            overlappable_cycles=sum(x.overlappable_cycles for x in vals),
            non_overlappable_cycles=sum(x.non_overlappable_cycles for x in vals),
            fill_cycles=float(first.fill_cycles),
            steady_cycles=float(
                mid_cycles - first.fill_cycles - last.drain_cycles
            ),
            drain_cycles=float(last.drain_cycles),
        )

    if bind == "par":
        total = max(x.total_cycles for x in vals)
        return ExecutionProfile(
            node_id="par",
            node_type="scope",
            level="unknown",
            iterations=1,
            local_compute_cycles=0.0,
            local_movement_cycles=0.0,
            compute_bound_cycles=max(x.compute_bound_cycles for x in vals),
            movement_bound_cycles=max(x.movement_bound_cycles for x in vals),
            overlappable_cycles=max(x.overlappable_cycles for x in vals),
            non_overlappable_cycles=max(x.non_overlappable_cycles for x in vals),
            fill_cycles=max(x.fill_cycles for x in vals),
            steady_cycles=float(total),
            drain_cycles=max(x.drain_cycles for x in vals),
        )

    if bind == "pipe":
        # Pipeline: steady throughput = bottleneck stage; fill = sum of all
        # stage totals minus steady (pipeline warm-up); drain = 0 (drain is
        # implicitly the last steady slot).
        steady    = max(x.steady_cycles for x in vals)
        one_pass  = sum(x.total_cycles for x in vals)
        fill      = max(0.0, one_pass - steady)
        return ExecutionProfile(
            node_id="pipe",
            node_type="scope",
            level="unknown",
            iterations=1,
            local_compute_cycles=0.0,
            local_movement_cycles=0.0,
            compute_bound_cycles=max(x.compute_bound_cycles for x in vals),
            movement_bound_cycles=max(x.movement_bound_cycles for x in vals),
            overlappable_cycles=sum(x.overlappable_cycles for x in vals),
            non_overlappable_cycles=max(x.non_overlappable_cycles for x in vals),
            fill_cycles=float(fill),
            steady_cycles=float(steady),
            drain_cycles=0.0,
        )

    # share / fallback: sequential semantics, same fill/drain treatment
    first, last = vals[0], vals[-1]
    mid_cycles = sum(x.total_cycles for x in vals)
    return ExecutionProfile(
        node_id="share",
        node_type="scope",
        level="unknown",
        iterations=1,
        local_compute_cycles=0.0,
        local_movement_cycles=0.0,
        compute_bound_cycles=sum(x.compute_bound_cycles for x in vals),
        movement_bound_cycles=sum(x.movement_bound_cycles for x in vals),
        overlappable_cycles=sum(x.overlappable_cycles for x in vals),
        non_overlappable_cycles=sum(x.non_overlappable_cycles for x in vals),
        fill_cycles=float(first.fill_cycles),
        steady_cycles=float(mid_cycles - first.fill_cycles - last.drain_cycles),
        drain_cycles=float(last.drain_cycles),
    )


def zero_profile(node_id: str, node_type: str, level: str) -> ExecutionProfile:
    return ExecutionProfile(
        node_id=node_id, node_type=node_type, level=level,
        iterations=1,
        local_compute_cycles=0.0, local_movement_cycles=0.0,
        compute_bound_cycles=0.0, movement_bound_cycles=0.0,
        overlappable_cycles=0.0, non_overlappable_cycles=0.0,
        fill_cycles=0.0, steady_cycles=0.0, drain_cycles=0.0,
    )


# --------------------------------------------------
# Local movement models
# --------------------------------------------------

def _move_bytes(m: Move, ctx: EstimateContext) -> int:
    """Bytes for a single Move, preferring the explicit field if set."""
    if m.bytes > 0:
        return m.bytes
    return ctx.tensor_sizes.get(m.tens, 64)


def scope_local_movement_cycles(scope: ScopeNode, ctx: EstimateContext) -> float:
    level = norm_level(scope.mem)
    bw = level_bandwidth(level, ctx.hardware)
    total = sum(_move_bytes(m, ctx) for m in scope.entry + scope.exit)
    return float(total) / max(1.0, bw)


def tile_local_movement_cycles(tile: TileNode, ctx: EstimateContext) -> float:
    """Movement cycles attributed to this tile's data transfers.

    Uses reuse-adjusted tile bytes (same formula as TensorTileStat.traffic_bytes)
    so that the profile-level movement matches the static accounting.
    """
    level = norm_level(tile.attrs.get("level", "unknown"))
    bw = level_bandwidth(level, ctx.hardware)
    total = 0.0
    for ten in infer_tile_tensors(tile):
        elems = estimate_tensor_tile_elements(ten, tile.size, ctx)
        total_elems = max(1, ctx.tensor_elements.get(ten, 1))
        total_bytes_all = ctx.tensor_sizes.get(ten, 64)
        bytes_per_elem = max(1, total_bytes_all // total_elems)
        tile_bytes = elems * bytes_per_elem
        reuse = estimate_reuse_factor(ten, tile, ctx)
        total += float(tile_bytes) / max(1.0, reuse)
    return float(total) / max(1.0, bw)


def tile_repeat_count(tile: TileNode, ctx: EstimateContext) -> int:
    if str(tile.mode).strip().lower() != "temp":
        return 1
    repeat = 1
    for dim in tile.loops:
        k = str(dim).strip().lower()
        full = max(1, int(ctx.loop_extent.get(k, 1)))
        part = max(1, int(tile.size.get(k, full)))
        repeat *= int(ceil(float(full) / float(part)))
    return max(1, repeat)


def tile_overlap_enabled(tile: TileNode) -> bool:
    return bool(tile.attrs.get("rw_overlap", False))


# --------------------------------------------------
# Tile tensor analysis
# --------------------------------------------------

def infer_tile_tensors(tile: TileNode) -> Tuple[str, ...]:
    out: List[str] = []
    seen: Set[str] = set()

    def _walk(node: MapNode) -> None:
        if isinstance(node, OpNode):
            for ten in list(node.ins) + list(node.outs):
                name = str(ten).strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
        elif isinstance(node, ScopeNode):
            for kid in node.kids:
                _walk(kid)
        elif isinstance(node, TileNode) and node.kid is not None:
            _walk(node.kid)

    _walk(tile)
    return tuple(out)


def estimate_tensor_tile_elements(
    tensor_name: str,
    tile_size: Mapping[str, int],
    ctx: EstimateContext,
) -> int:
    dims = ctx.tensor_dims.get(tensor_name)
    if not dims:
        return 1
    elems = 1
    for d in dims:
        full = int(ctx.loop_extent.get(d, 1))
        take = int(tile_size.get(d, full))
        elems *= max(1, min(take, full))
    return max(1, elems)


def estimate_reuse_factor(
    tensor_name: str,
    tile: TileNode,
    ctx: EstimateContext,
) -> float:
    """Compute reuse of *tensor_name* across all ancestor TileNodes within
    the same ScopeNode boundary.

    Reuse = Π over every loop dimension NOT in tensor_dims, across all
    ancestor TileNodes up to (but not crossing) the nearest ScopeNode.

    Why stop at a ScopeNode boundary:
      - ScopeNode entry/exit Moves in _apply_scope_accounting already account
        for cross-scope data transfers at full tensor granularity.
      - estimate_reuse_factor only models reuse *within* a scope — i.e., how
        many times the tile-level load is amortised by iteration over outer
        loops that are still inside the same memory scope.
      - Crossing a ScopeNode would double-count: the outer scope already paid
        for the full tensor load; counting batch/group iterations above the
        scope boundary would artificially inflate the reuse factor.

    Why walk multiple TileNodes (not just the immediate tile):
      In GQA/MHA, the K/V tile is often reused across Q-head iterations that
      sit in an outer TileNode *within the same scope*.  The old single-tile
      code missed this cross-tile reuse.  The fix: accumulate trip counts from
      all temp TileNodes on the path from *tile* up to the nearest ScopeNode.

    Example (GQA decode, all nodes in the same group scope):
      TileNode(hq=32, temp)      ← K/V has no hq dim → 32× reuse here
        TileNode(n=64, temp)     ← K/V has no n dim  → 64/tile_n reuse here
          TileNode(d=128, temp)  ← *tile* (starting point)
            OpNode(matmul)
    Total K/V reuse = 32 × (n/tile_n) × (d/tile_d) ← only within-scope tiles
    """
    dims = set(ctx.tensor_dims.get(tensor_name, ()))
    if not dims:
        return 1.0

    reuse = 1.0

    # Start from the current tile's own loops first
    cur: object = tile
    while cur is not None:
        if isinstance(cur, ScopeNode):
            # Stop: crossed a scope boundary
            break
        if isinstance(cur, TileNode) and str(cur.mode).strip().lower() == "temp":
            for dim in cur.loops:
                k = str(dim).strip().lower()
                if k not in dims:
                    full = max(1, int(ctx.loop_extent.get(k, 1)))
                    part = max(1, int(cur.size.get(k, full)))
                    reuse *= max(1.0, float(ceil(float(full) / float(part))))
        cur = getattr(cur, "parent", None)

    return max(1.0, reuse)


def infer_tensor_role(name: str) -> str:
    text = str(name).strip().upper()
    if text in {"O"}:
        return "output"
    if "CACHE" in text or "CTX" in text:
        return "state"
    if text in {"Q", "K", "V", "K_APPEND", "V_APPEND"}:
        return "input"
    return "temp"


# --------------------------------------------------
# Op-level compute analysis
# --------------------------------------------------

def estimate_op_cost(
    op_node: OpNode,
    ctx: EstimateContext,
) -> OpCost:
    op   = ctx.workload.op(op_node.id)
    kind = str(getattr(op, "op_type", "unknown")).lower()

    tile = nearest_tile_for_op(op_node)
    tile_extent = dict(tile.size) if tile is not None else dict(ctx.loop_extent)

    if kind == "matmul":
        flops = estimate_matmul_flops(op, tile_extent)
    elif kind == "softmax":
        flops = estimate_softmax_flops(op, tile_extent)
    elif kind in {"transpose", "headbroadcast", "identity", "kvappend"}:
        flops = estimate_lightweight_flops(op, tile_extent)
    else:
        flops = estimate_generic_flops(op, tile_extent)

    pe_parallel   = estimate_pe_parallelism(tile, ctx.hardware)
    macs          = max(1.0, float(ctx.hardware.pe.macs))
    compute_cycles = float(flops) / max(1.0, float(pe_parallel) * macs)

    red_loops = tuple(str(x).strip().lower() for x in getattr(op, "reduce_dims", ()))
    return OpCost(
        op_id=op_node.id,
        kind=kind,
        flops=float(flops),
        compute_cycles=float(max(compute_cycles, 1.0)),
        reduction_loops=red_loops,
    )


def nearest_tile_for_op(op_node: OpNode) -> Optional[TileNode]:
    cur = getattr(op_node, "parent", None)
    while cur is not None:
        if isinstance(cur, TileNode):
            return cur
        cur = getattr(cur, "parent", None)
    return None


def estimate_pe_parallelism(tile: Optional[TileNode], hw: HardwareSpec) -> int:
    if tile is None:
        return max(1, int(hw.pe.count))
    if str(tile.mode).strip().lower() == "spat":
        spatial_extent = 1
        for val in tile.size.values():
            spatial_extent *= max(1, int(val))
        return max(1, min(int(hw.pe.count), spatial_extent))
    return 1


def estimate_matmul_flops(op, tile_extent: Mapping[str, int]) -> float:
    b = tile_extent.get("b", 1)
    h = tile_extent.get("hq", tile_extent.get("hkv", 1))
    m = tile_extent.get("m", 1)
    n = tile_extent.get("n", 1)
    d = tile_extent.get("d", 1)
    return float(2 * b * h * m * n * d)


def estimate_softmax_flops(op, tile_extent: Mapping[str, int]) -> float:
    """Estimate flops for softmax on the PE array.

    Standard online softmax (numerically stable) requires per element:
      1 cmp   (max reduction, passed over the N dimension)
      1 sub   (subtract max)
      1 exp   (approximated on PE; see softmax_ops_per_elem)
      1 add   (sum reduction)
      1 div   (normalise)
    = 5 baseline ops.

    If the PE has no dedicated exp unit (polynomial approximation),
    exp costs ~4 extra MACs, giving ~9 ops/element.  The default is 8
    (midpoint), matching an exp implementation with CORDIC or 3rd-order
    Horner evaluation.

    The op's attrs["softmax_ops_per_elem"] field allows callers to
    override this per-op if the hardware has a dedicated exp unit (set 5)
    or a slow software exp (set 12+).
    """
    ops_per_elem = int(getattr(op, "attrs", {}).get("softmax_ops_per_elem", 8))
    b = tile_extent.get("b", 1)
    h = tile_extent.get("hq", tile_extent.get("hkv", 1))
    m = tile_extent.get("m", 1)
    n = tile_extent.get("n", 1)
    return float(ops_per_elem * b * h * m * n)


def estimate_lightweight_flops(op, tile_extent: Mapping[str, int]) -> float:
    total = 1
    for val in tile_extent.values():
        total *= max(1, int(val))
    return float(max(1, total))


def estimate_generic_flops(op, tile_extent: Mapping[str, int]) -> float:
    total = 1
    for val in tile_extent.values():
        total *= max(1, int(val))
    red = max(1, len(getattr(op, "reduce_dims", ())))
    return float(total * red)


# --------------------------------------------------
# Trace request estimation (DRAM-level only)
# --------------------------------------------------

def _trace_requests_from_level_cost(lc: LevelCost, burst_bytes: int = 64) -> int:
    """Derive trace request count from the already-accumulated DRAM bytes."""
    total = lc.movement_bytes
    if total <= 0:
        return 0
    q, r = divmod(int(total), int(burst_bytes))
    return q + (1 if r else 0)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def level_bandwidth(level: str, hw: HardwareSpec) -> float:
    lv = norm_level(level)
    if lv == "dram":
        return max(1.0, float(hw.dram.bw_hint))
    if lv == "sram":
        return max(1.0, float(hw.sram.bw))
    if lv == "pe":
        return max(1.0, float(hw.pe.mac_per_cycle))
    return 1.0


def norm_level(raw) -> str:
    return str(raw).strip().lower()


# --------------------------------------------------
# Export
# --------------------------------------------------

__all__ = [
    "EstimateContext",
    "ExecutionProfile",
    "LevelCost",
    "OpCost",
    "TensorTileStat",
    "TreeEstimate",
    "estimate_candidate",
    "estimate_tree",
    "objectives_from_analytical",
]
