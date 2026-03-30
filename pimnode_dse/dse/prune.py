from __future__ import annotations

from typing import Optional, Sequence

from pimnode_dse.hardware.arch_spec import HardwareSpec
from pimnode_dse.mapping.placement.dataflow import FlowBucket
from pimnode_dse.mapping.placement.node import GroupDP
from pimnode_dse.mapping.tilling.tilling_gene import GroupTilingSpec
from pimnode_dse.mapping.workload.workload import WorkloadDAG

from .types import AnalyticalMetrics, Candidate, ObjectiveMetrics
from .pareto import (
    eps_dominance_prune,
    hypervolume_contributions_3d,
    nondominated_sort_3d,
    _hv_reference,
)


# ---------------------------------------------------------------------------
# Placement feasibility
# ---------------------------------------------------------------------------

def placement_fits_sram(
    place: GroupDP,
    workload: WorkloadDAG,
    *,
    sram_cap: int,
    tiling: Optional[GroupTilingSpec] = None,
) -> bool:
    """Check that the combined SRAM footprint of *place* (and optional *tiling*)
    fits within *sram_cap* bytes.

    Residence-aware accounting:
      'evict'  → 0 bytes  (transient; not held across iterations)
      'double' → 2× bytes (double-buffer: two physical slots alternate)
      'pinned' / 'single' → 1× bytes

    Joint placement + tiling check:
      If *tiling* is supplied, the SRAM tile footprint is added so that the
      combined usage is validated in one shot.
    """
    placement_bytes = 0
    for node in place.nodes:
        if str(node.level).strip().lower() != "sram":
            continue
        residence = str(getattr(node, "residence", "single")).strip().lower()
        if residence == "evict":
            continue
        node_bytes = sum(_safe_tensor_bytes(t, workload) for t in node.tens)
        if residence == "double":
            node_bytes *= 2
        placement_bytes += node_bytes

    tiling_bytes = 0
    if tiling is not None:
        sram_tile = tiling.tier_tiles.get("sram")
        if sram_tile is not None:
            buf_mult = 2 if getattr(sram_tile, "buf_mode", "single") == "double" else 1
            tile_elems = 1
            for v in sram_tile.tile_size.values():
                tile_elems *= max(1, int(v))
            # 2 bytes/element (bfloat16 default); conservative overestimate
            tiling_bytes = tile_elems * 2 * buf_mult

    return (placement_bytes + tiling_bytes) <= int(sram_cap)


def _safe_tensor_bytes(tensor_name: str, workload: WorkloadDAG) -> int:
    try:
        return int(workload.tensor(tensor_name).size_bytes())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Option trimming
# ---------------------------------------------------------------------------

def trim_group_options(rows: Sequence[object], limit: int) -> tuple[object, ...]:
    if limit <= 0:
        return tuple(rows)
    return tuple(rows[:limit])


def trim_tiling_options(
    rows: Sequence[GroupTilingSpec],
    limit: int,
    hw: Optional[HardwareSpec] = None,
) -> tuple[GroupTilingSpec, ...]:
    """Prune tiling options via ε-dominance on a roofline proxy, then cap.

    Proxy objectives (all minimise):
      - split_red_count  : fewer reduction splits → simpler pipeline
      - acc_scope_cost   : 0=local (psum stays in PE), 1=sram (psum off-chip)
      - roofline_cycles  : max(compute_cycles, traffic_cycles)

    Roofline model (same formula as estimate.py):
      flops          = 2 × Π(tile_dim)          # 2-op matmul (mul+add)
      compute_cycles = flops / (pe_count × macs_per_cycle)
      traffic_cycles = sram_tile_bytes / hw.sram.bw
                     + dram_tile_bytes / hw.dram.bw_hint
      roofline_cycles = max(compute_cycles, traffic_cycles)

    Why roofline instead of traffic_cycles alone:
      traffic_cycles ignores the compute bound.  In compute-bound regimes a
      larger tile (higher traffic) can be faster overall because it reduces
      the repeat count and amortises the per-tile compute latency.  The
      roofline bottleneck correctly captures this trade-off without any
      tuning constant — all values come from hw spec.

    If *hw* is None, falls back to traffic_cycles only (equal weight,
    no compute model).
    """
    if limit <= 0:
        return tuple(rows)
    if len(rows) <= limit:
        return tuple(rows)

    pe_throughput = max(1.0, float(hw.pe.mac_per_cycle)) if hw is not None else 1.0
    sram_bw       = max(1.0, float(hw.sram.bw))          if hw is not None else 1.0
    dram_bw       = max(1.0, float(hw.dram.bw_hint))     if hw is not None else 1.0

    def _level_bytes(spec: GroupTilingSpec, level: str) -> float:
        tile = spec.tier_tiles.get(level)
        if tile is None:
            return 0.0
        elems = 1
        for v in tile.tile_size.values():
            elems *= max(1, int(v))
        buf = 2 if getattr(tile, "buf_mode", "single") == "double" else 1
        return float(elems * 2 * buf)   # 2 bytes/elem (bfloat16)

    def key(spec: GroupTilingSpec) -> tuple[float, ...]:
        # Estimate flops from the SRAM tile (innermost data volume × 2 for MAC)
        sram_tile = spec.tier_tiles.get("sram")
        tile_elems = 1
        if sram_tile is not None:
            for v in sram_tile.tile_size.values():
                tile_elems *= max(1, int(v))
        flops = float(2 * tile_elems)                   # 2 ops per element (matmul)
        compute_cycles = flops / pe_throughput

        sram_bytes = _level_bytes(spec, "sram")
        dram_bytes = _level_bytes(spec, "dram")
        traffic_cycles = sram_bytes / sram_bw + dram_bytes / dram_bw

        roofline = max(compute_cycles, traffic_cycles)

        return (
            float(len(spec.split_red)),
            float(0 if spec.acc_scope == "local" else 1),
            roofline,
        )

    return tuple(eps_dominance_prune(list(rows), key=key, eps=0.05, cap=limit))


def trim_flow_buckets(rows: Sequence[FlowBucket], limit: int) -> tuple[FlowBucket, ...]:
    """Prune dataflow buckets via ε-dominance, then optional hard cap.

    Proxy objectives (all minimise):
      - group_get_count  : tensors fetched into group scope
      - group_out_count  : tensors written back out of group scope
      - reuse_out_count  : tensors flagged for reuse across groups

    eps=0.10 because these are small integer counts (difference of 1 ≈ 10%).
    """
    if limit <= 0:
        return tuple(rows)
    if len(rows) <= limit:
        return tuple(rows)

    def key(fb: FlowBucket) -> tuple[float, ...]:
        return (
            float(len(fb.out.group_get)),
            float(len(fb.out.group_out)),
            float(len(fb.out.reuse_out)),
        )

    return tuple(eps_dominance_prune(list(rows), key=key, eps=0.10, cap=limit))


# ---------------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------------

def shortlist_candidates(
    rows: Sequence[tuple[Candidate, AnalyticalMetrics]],
    *,
    top_k: int,
    eps: float = 0.05,
) -> list[tuple[Candidate, AnalyticalMetrics]]:
    """ε-dominance shortlist with layered Pareto + hypervolume-contribution ranking.

    Step 1 — ε-dominance archive:
      Map each candidate to a 3-D ε-box on (latency, dram_bytes, movement_bytes).
      Keep one representative per box.  Quality loss ≤ ε on every objective.

    Step 2 — layered Pareto shells (only if archive > top_k):
      Use nondominated_sort_3d (adaptive O(n²) / O(kn log n)).
      Within each shell rank by hypervolume contribution so the kept set
      spans the objective space evenly instead of clustering around one axis.
    """
    if not rows:
        return []

    def obj_key(pair: tuple[Candidate, AnalyticalMetrics]) -> tuple[float, ...]:
        _, m = pair
        return (float(m.est_compute_cycles), float(m.dram_bytes), float(m.movement_bytes))

    archive = eps_dominance_prune(list(rows), key=obj_key, eps=eps, cap=None)

    if top_k <= 0 or len(archive) <= top_k:
        archive.sort(key=obj_key)
        return archive

    scored = [
        ((cand, metrics), ObjectiveMetrics(
            latency=float(metrics.est_compute_cycles),
            dram_cost=float(metrics.dram_bytes),
            movement=float(metrics.movement_bytes),
        ))
        for cand, metrics in archive
    ]

    keep: list[tuple[Candidate, AnalyticalMetrics]] = []
    for shell in nondominated_sort_3d(scored):
        if len(keep) >= top_k:
            break
        shell_pts = [(m.latency, m.dram_cost, m.movement) for _, m in shell]
        if len(shell_pts) > 1:
            ref = _hv_reference(shell_pts)
            contribs = hypervolume_contributions_3d(shell_pts, ref)
            order = sorted(range(len(shell)), key=lambda i: -contribs[i])
        else:
            order = [0]
        need = top_k - len(keep)
        for idx in order[:need]:
            keep.append(shell[idx][0])

    return keep


__all__ = [
    "placement_fits_sram",
    "shortlist_candidates",
    "trim_flow_buckets",
    "trim_group_options",
    "trim_tiling_options",
]


# ---------------------------------------------------------------------------
# DRAM-to-SRAM traffic cost ratio used as a weighting factor in the tiling
# proxy objective.  DRAM access is typically 10-50× more expensive than SRAM.
# 20 is a conservative midpoint suitable for DDR4-class systems.
# ---------------------------------------------------------------------------
_DRAM_SRAM_WEIGHT_RATIO: float = 20.0


# ---------------------------------------------------------------------------
# Placement feasibility
# ---------------------------------------------------------------------------

def placement_fits_sram(
    place: GroupDP,
    workload: WorkloadDAG,
    *,
    sram_cap: int,
    tiling: Optional[GroupTilingSpec] = None,
) -> bool:
    """Check that the combined SRAM footprint of *place* (and optional *tiling*)
    fits within *sram_cap* bytes.

    Fix A — residence-aware accounting:
      StoreNode.residence == 'double'  → 2× tensor bytes (double-buffer).
      StoreNode.residence == 'pinned'  → 1× tensor bytes (always resident).
      StoreNode.residence == 'evict'   → 0  bytes (transient, not held across iters).
      StoreNode.residence == 'single'  → 1× tensor bytes (default).

    Fix B — joint placement + tiling check:
      If *tiling* is supplied the SRAM tile footprint is added to the placement
      footprint so that the combined usage is validated in one pass.
      This prevents tilings whose data + tile buffers together overflow SRAM even
      when each component individually fits.
    """
    placement_bytes = 0
    for node in place.nodes:
        if str(node.level).strip().lower() != "sram":
            continue
        residence = str(getattr(node, "residence", "single")).strip().lower()
        if residence == "evict":
            continue
        node_bytes = sum(_safe_tensor_bytes(t, workload) for t in node.tens)
        if residence == "double":
            node_bytes *= 2
        placement_bytes += node_bytes

    tiling_bytes = 0
    if tiling is not None:
        sram_tile = tiling.tier_tiles.get("sram")
        if sram_tile is not None:
            buf_mult = 2 if getattr(sram_tile, "buf_mode", "single") == "double" else 1
            tile_elems = 1
            for v in sram_tile.tile_size.values():
                tile_elems *= max(1, int(v))
            # 2 bytes/element (bfloat16 default); conservative overestimate
            tiling_bytes = tile_elems * 2 * buf_mult

    return (placement_bytes + tiling_bytes) <= int(sram_cap)


def _safe_tensor_bytes(tensor_name: str, workload: WorkloadDAG) -> int:
    try:
        return int(workload.tensor(tensor_name).size_bytes())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Option trimming
# ---------------------------------------------------------------------------

def trim_group_options(rows: Sequence[object], limit: int) -> tuple[object, ...]:
    if limit <= 0:
        return tuple(rows)
    return tuple(rows[:limit])


def trim_tiling_options(
    rows: Sequence[GroupTilingSpec],
    limit: int,
    *,
    dram_weight: float = _DRAM_SRAM_WEIGHT_RATIO,
) -> tuple[GroupTilingSpec, ...]:
    """Prune tiling options via ε-dominance, then optional hard cap.

    Proxy objectives (all minimise):
      - split_red_count  : fewer reduction splits → simpler pipeline
      - acc_scope_cost   : 0=local (psum in PE), 1=sram (psum off-chip)
      - weighted_traffic : sram_vol + dram_weight × dram_vol
                           Penalises DRAM-heavy tilings proportionally to the
                           actual cost ratio between DRAM and SRAM access.

    Fix: old proxy summed SRAM and DRAM volumes equally, biasing towards large
    SRAM / small DRAM even when DRAM cost dominated.  The weighted form now
    correctly reflects hardware cost asymmetry.
    """
    if limit <= 0:
        return tuple(rows)
    if len(rows) <= limit:
        return tuple(rows)

    def _level_vol(spec: GroupTilingSpec, level: str) -> int:
        tile = spec.tier_tiles.get(level)
        if tile is None:
            return 0
        vol = 1
        for v in tile.tile_size.values():
            vol *= max(1, int(v))
        return vol

    def key(spec: GroupTilingSpec) -> tuple[float, ...]:
        sram_vol = float(_level_vol(spec, "sram"))
        dram_vol = float(_level_vol(spec, "dram"))
        return (
            float(len(spec.split_red)),
            float(0 if spec.acc_scope == "local" else 1),
            sram_vol + dram_weight * dram_vol,
        )

    return tuple(eps_dominance_prune(list(rows), key=key, eps=0.05, cap=limit))


def trim_flow_buckets(rows: Sequence[FlowBucket], limit: int) -> tuple[FlowBucket, ...]:
    """Prune dataflow buckets via ε-dominance, then optional hard cap.

    Proxy objectives (all minimise):
      - group_get_count  : tensors fetched into group scope
      - group_out_count  : tensors written back out of group scope
      - reuse_out_count  : tensors flagged for reuse across groups

    eps=0.10 because these are small integer counts (difference of 1 ≈ 10%).
    """
    if limit <= 0:
        return tuple(rows)
    if len(rows) <= limit:
        return tuple(rows)

    def key(fb: FlowBucket) -> tuple[float, ...]:
        return (
            float(len(fb.out.group_get)),
            float(len(fb.out.group_out)),
            float(len(fb.out.reuse_out)),
        )

    return tuple(eps_dominance_prune(list(rows), key=key, eps=0.10, cap=limit))


# ---------------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------------

def shortlist_candidates(
    rows: Sequence[tuple[Candidate, AnalyticalMetrics]],
    *,
    top_k: int,
    eps: float = 0.05,
) -> list[tuple[Candidate, AnalyticalMetrics]]:
    """ε-dominance shortlist with layered Pareto + hypervolume-contribution ranking.

    Step 1 — ε-dominance archive:
      Map each candidate to a 3-D ε-box on (latency, dram_bytes, movement_bytes).
      Keep one representative per box.  Quality loss ≤ ε on every objective.

    Step 2 — layered Pareto shells (only if archive > top_k):
      Use nondominated_sort_3d (adaptive O(n²) / O(kn log n)).
      Within each shell rank by hypervolume contribution so the kept set
      spans the objective space evenly instead of clustering around one axis.
    """
    if not rows:
        return []

    def obj_key(pair: tuple[Candidate, AnalyticalMetrics]) -> tuple[float, ...]:
        _, m = pair
        return (float(m.est_compute_cycles), float(m.dram_bytes), float(m.movement_bytes))

    archive = eps_dominance_prune(list(rows), key=obj_key, eps=eps, cap=None)

    if top_k <= 0 or len(archive) <= top_k:
        archive.sort(key=obj_key)
        return archive

    scored = [
        ((cand, metrics), ObjectiveMetrics(
            latency=float(metrics.est_compute_cycles),
            dram_cost=float(metrics.dram_bytes),
            movement=float(metrics.movement_bytes),
        ))
        for cand, metrics in archive
    ]

    keep: list[tuple[Candidate, AnalyticalMetrics]] = []
    for shell in nondominated_sort_3d(scored):
        if len(keep) >= top_k:
            break
        shell_pts = [(m.latency, m.dram_cost, m.movement) for _, m in shell]
        if len(shell_pts) > 1:
            ref = _hv_reference(shell_pts)
            contribs = hypervolume_contributions_3d(shell_pts, ref)
            order = sorted(range(len(shell)), key=lambda i: -contribs[i])
        else:
            order = [0]
        need = top_k - len(keep)
        for idx in order[:need]:
            keep.append(shell[idx][0])

    return keep


__all__ = [
    "placement_fits_sram",
    "shortlist_candidates",
    "trim_flow_buckets",
    "trim_group_options",
    "trim_tiling_options",
]
