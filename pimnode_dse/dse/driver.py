from __future__ import annotations

from dataclasses import dataclass
from itertools import product as itertools_product
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
import uuid

from dram_sim.dram_trace_generator import generate_trace_and_cfg
from dram_sim.run_ramulator_sim import run_dram_simulation
from pimnode_dse.dse.types import DramSimMetrics
from pimnode_dse.hardware.hw_space import HWSpace
from pimnode_dse.mapping.fusion.fusion_space import FusionSpace, FusionSpaceConfig
from pimnode_dse.mapping.placement import dataflow, placement
from pimnode_dse.mapping.placement.node import PlacementPlan
from pimnode_dse.mapping.tilling.tilling_gene import GroupTilingSpec, enum_tilings
from pimnode_dse.mapping.tree.mapping_builder import build_mapping_tree
from pimnode_dse.mapping.workload.workload import AttentionWorkloadSpec, WorkloadDAG, build_attention_dag

from .estimate import estimate_candidate, estimate_tree, objectives_from_analytical
from .pareto import pareto_front
from .prune import placement_fits_sram, shortlist_candidates, trim_flow_buckets, trim_tiling_options
from .store import append_record, candidate_artifact_dir, ensure_run_dir, write_manifest, write_pareto, write_tree_text
from .types import Candidate, DSEConfig, EvalResult, MappingChoice, RunRecord, WorkloadCase


@dataclass(frozen=True)
class BatchRunSummary:
    run_dir: Path
    total_candidates: int
    shortlisted: int
    simulated: int
    pareto_size: int


def run_batch(
    workloads: Sequence[AttentionWorkloadSpec],
    hw_space: HWSpace,
    cfg: DSEConfig,
) -> BatchRunSummary:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    run_dir = ensure_run_dir(Path(cfg.out_dir), run_id)
    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "workload_count": len(workloads),
            "hardware_count": hw_space.size(),
            "config": {
                "max_fusion_genes": cfg.max_fusion_genes,
                "max_places_per_group": cfg.max_places_per_group,
                "max_tilings_per_group": cfg.max_tilings_per_group,
                "max_bucket_per_group": cfg.max_bucket_per_group,
                "max_candidates_per_hw": cfg.max_candidates_per_hw,
                "shortlist_top_k": cfg.shortlist_top_k,
                "simulate_top_k": cfg.simulate_top_k,
                "run_dram_sim": cfg.run_dram_sim,
            },
        },
    )

    all_rows: List[Tuple[Candidate, EvalResult]] = []
    total_candidates = 0
    total_shortlisted = 0
    total_simulated = 0

    cases = [build_workload_case(spec) for spec in workloads]
    for case in cases:
        for hw_idx, hw in enumerate(hw_space.iter_specs()):
            pairs = _enumerate_candidates_for_case(case, hw, cfg)
            total_candidates += len(pairs)

            short = shortlist_candidates(pairs, top_k=cfg.shortlist_top_k)
            total_shortlisted += len(short)

            evaluated = _evaluate_shortlist(short, run_dir=run_dir, cfg=cfg)
            total_simulated += sum(1 for _, res in evaluated if res.dram_sim is not None)
            all_rows.extend(evaluated)

    front = pareto_front([(cand, res.objectives) for cand, res in all_rows])
    front_rows = []
    front_ids = {cand.candidate_id for cand, _ in front}
    for cand, res in all_rows:
        if cand.candidate_id in front_ids:
            front_rows.append((cand, res))
    write_pareto(run_dir, front_rows)

    return BatchRunSummary(
        run_dir=run_dir,
        total_candidates=total_candidates,
        shortlisted=total_shortlisted,
        simulated=total_simulated,
        pareto_size=len(front_rows),
    )


def build_workload_case(spec: AttentionWorkloadSpec, *, workload_id: Optional[str] = None) -> WorkloadCase:
    dag = build_attention_dag(spec)
    wid = workload_id or spec.fingerprint()
    return WorkloadCase(workload_id=wid, spec=spec, dag=dag)


def _enumerate_candidates_for_case(
    case: WorkloadCase,
    hw,
    cfg: DSEConfig,
) -> List[Tuple[Candidate, EvalResult]]:
    dag = case.dag
    fusion_space = FusionSpace(
        dag=dag,
        config=FusionSpaceConfig(max_out=cfg.max_fusion_genes),
    )
    genes = fusion_space.enumerate_genes()[: max(1, int(cfg.max_fusion_genes))]

    rows: List[Tuple[Candidate, EvalResult]] = []
    candidate_count = 0

    for gene_idx, gene in enumerate(genes):
        # --- Placement: residence-aware SRAM filter ---
        place_all_raw = placement.enum_dp(workload=dag, fusion=gene, hw=hw)
        place_all: Dict[str, tuple] = {}
        for gid, vals in place_all_raw.items():
            feasible = [p for p in vals if placement_fits_sram(p, dag, sram_cap=hw.sram.cap)]
            place_all[gid] = tuple(feasible[: cfg.max_places_per_group] if cfg.max_places_per_group > 0 else feasible)
        if any(not v for v in place_all.values()):
            continue

        # --- Tiling: enumerate with joint SRAM check at source ---
        # Build a sram_bytes_fn that estimates tiling footprint so that
        # enum_tilings can filter infeasible tilings before they enter the
        # search space entirely — not just after ε-dominance pruning.
        sram_bytes_fn = _make_sram_bytes_fn(dag, hw.sram.cap)

        tiling_all_raw = enum_tilings(
            fusion=gene,
            workload=dag,
            hw=hw,
            sram_bytes_fn=sram_bytes_fn,
        )
        tiling_all: Dict[str, tuple] = {}
        for gid, vals in tiling_all_raw.items():
            tiling_all[gid] = trim_tiling_options(vals, cfg.max_tilings_per_group, hw=hw)
        if any(not v for v in tiling_all.values()):
            continue

        # --- Dataflow: ε-dominance prune ---
        ctxs = dataflow.build_ctxs(fusion=gene, workload=dag)
        flow_all = dataflow.enum_dp(places=place_all, tilings=tiling_all, ctxs=ctxs, hw=hw)
        flow_buckets_raw = dataflow.enum_out(
            dp_map=flow_all, fusion=gene, ctxs=ctxs, hw=hw,
            max_per_bucket=cfg.max_bucket_per_group,
        )
        flow_buckets: Dict[str, tuple] = {}
        for gid, vals in flow_buckets_raw.items():
            flow_buckets[gid] = trim_flow_buckets(vals, cfg.max_bucket_per_group)
        if any(not v for v in flow_buckets.values()):
            continue

        # --- Cartesian product: best-first ranked enumeration ---
        sorted_gids = sorted(place_all.keys())

        # Sort each per-group option list by a simple proxy score so that
        # _cartesian_budget encounters higher-quality combinations first.
        # This matters when the budget is tight: we'd rather evaluate a few
        # good candidates than many mediocre ones.
        place_choices  = [_rank_placements(list(place_all[gid]),   dag)       for gid in sorted_gids]
        tiling_choices = [_rank_tilings(list(tiling_all[gid]),     hw)        for gid in sorted_gids]
        bucket_choices = [list(flow_buckets[gid])                              for gid in sorted_gids]

        for combo in _cartesian_budget(
            place_choices, tiling_choices, bucket_choices,
            budget=cfg.max_candidates_per_hw - len(rows),
        ):
            place_combo, tiling_combo, bucket_combo = combo

            placement_plan = PlacementPlan(
                id=f"place-{case.workload_id}-{gene_idx}-{candidate_count}",
                groups=tuple(place_combo),
            )
            tiling_map = {gid: tiling_combo[i] for i, gid in enumerate(sorted_gids)}
            bucket_map = {gid: (bucket_combo[i],) for i, gid in enumerate(sorted_gids)}

            tree = build_mapping_tree(gene, placement_plan, tiling_map, bucket_map, workload=dag)

            candidate_id = f"{case.workload_id}-hw{hw.name}-g{gene_idx}-{candidate_count}"
            candidate_count += 1

            cand = Candidate(
                candidate_id=candidate_id,
                workload=case,
                hardware=hw,
                fusion_gene=gene,
                mapping=MappingChoice(
                    placement_plan=placement_plan,
                    tiling_map=tiling_map,
                    flow_buckets=bucket_map,
                    tree=tree,
                ),
                group_signatures={gid: bucket_combo[i].eq_sig() for i, gid in enumerate(sorted_gids)},
                meta={"gene_id": gene.gene_id},
            )

            analytical = estimate_candidate(cand, dag, hw)
            result = EvalResult(
                candidate_id=candidate_id,
                analytical=analytical,
                objectives=objectives_from_analytical(analytical),
                dram_sim=None,
                stage="analytical",
            )
            rows.append((cand, result))

            if cfg.max_candidates_per_hw > 0 and len(rows) >= cfg.max_candidates_per_hw:
                return rows

    return rows


# ---------------------------------------------------------------------------
# Ranking helpers for best-first Cartesian enumeration
# ---------------------------------------------------------------------------

def _rank_placements(feasible: list, workload: WorkloadDAG) -> list:
    """Sort placements by estimated SRAM data volume ascending.

    Placements with a smaller SRAM footprint leave more room for tile
    buffers and are therefore more likely to remain feasible as tiling
    options are explored.  We use actual tensor byte sizes from the
    workload — the same source used in placement_fits_sram — so the
    ranking is consistent with the feasibility check.

    'evict' nodes are excluded (they hold 0 bytes at steady state);
    'double' nodes are counted twice (two physical slots).
    """
    def _sram_bytes(p) -> int:
        total = 0
        for node in p.nodes:
            if str(node.level).strip().lower() != "sram":
                continue
            residence = str(getattr(node, "residence", "single")).strip().lower()
            if residence == "evict":
                continue
            mult = 2 if residence == "double" else 1
            for t in node.tens:
                try:
                    total += int(workload.tensor(t).size_bytes()) * mult
                except Exception:
                    pass
        return total

    return sorted(feasible, key=_sram_bytes)


def _rank_tilings(tilings: list, hw: HardwareSpec) -> list:
    """Sort tilings by roofline bottleneck proxy ascending.

    proxy = max(compute_cycles, traffic_cycles)
      compute_cycles = 2 × Π(sram_tile_dim) / (pe_count × macs)
      traffic_cycles = sram_tile_bytes / sram_bw + dram_tile_bytes / dram_bw

    This is the same formula used in trim_tiling_options, so the first
    element of the ranked list is the tiling that the ε-archive would pick
    as its primary representative.  _cartesian_budget therefore evaluates
    the roofline-optimal combination first.

    Why roofline, not traffic_cycles alone:
      In compute-bound configurations a larger tile transfers more data but
      repeats fewer times, reducing total latency.  traffic_cycles would
      rank such tilings last; the roofline proxy correctly ranks them first.
    """
    pe_throughput = max(1.0, float(hw.pe.mac_per_cycle))
    sram_bw       = max(1.0, float(hw.sram.bw))
    dram_bw       = max(1.0, float(hw.dram.bw_hint))

    def _tile_bytes(spec, level: str) -> float:
        tile = spec.tier_tiles.get(level)
        if tile is None:
            return 0.0
        elems = 1
        for v in tile.tile_size.values():
            elems *= max(1, int(v))
        buf = 2 if getattr(tile, "buf_mode", "single") == "double" else 1
        return float(elems * 2 * buf)

    def _roofline(spec) -> tuple:
        sram_tile = spec.tier_tiles.get("sram")
        tile_elems = 1
        if sram_tile is not None:
            for v in sram_tile.tile_size.values():
                tile_elems *= max(1, int(v))
        flops = float(2 * tile_elems)
        compute_cycles = flops / pe_throughput
        traffic_cycles = (_tile_bytes(spec, "sram") / sram_bw
                         + _tile_bytes(spec, "dram") / dram_bw)
        return (
            len(spec.split_red),
            0 if spec.acc_scope == "local" else 1,
            max(compute_cycles, traffic_cycles),
        )

    return sorted(tilings, key=_roofline)


def _make_sram_bytes_fn(
    workload: WorkloadDAG,
    sram_cap: int,
) -> Callable[[GroupTilingSpec], Optional[int]]:
    """Return a sram_bytes_fn suitable for enum_tilings.

    The function estimates the SRAM tile footprint for a GroupTilingSpec
    by multiplying together the SRAM tile extents and scaling by 2 bytes/elem.
    It returns the footprint if it fits within sram_cap (so enum_tilings can
    reject overflowing tilings early), or sram_cap + 1 to signal infeasibility.

    This hooks into enum_tilings's validate_group_tiling check at line 261-267
    of tilling_gene.py, which rejects specs where sram_bytes > hw.sram.cap.
    """
    def _fn(spec: GroupTilingSpec) -> Optional[int]:
        sram_tile = spec.tier_tiles.get("sram")
        if sram_tile is None:
            return None   # no check
        elems = 1
        for v in sram_tile.tile_size.values():
            elems *= max(1, int(v))
        buf_mult = 2 if getattr(sram_tile, "buf_mode", "single") == "double" else 1
        return elems * 2 * buf_mult   # 2 bytes/elem (bfloat16)
    return _fn


def _cartesian_budget(
    place_choices: list,
    tiling_choices: list,
    bucket_choices: list,
    budget: int,
) -> Iterator[Tuple[tuple, tuple, tuple]]:
    """Yield (place_combo, tiling_combo, bucket_combo) tuples up to *budget*.

    Iterates the Cartesian product in ranked order (each per-group list is
    pre-sorted by quality, so earlier combinations are better).  Stops as
    soon as *budget* tuples have been yielded or the product is exhausted.

    The nested-loop order (place outer, tiling middle, bucket inner) ensures
    that for the same placement, all tiling variants are explored before the
    next placement is tried — matching the typical cost hierarchy where
    placement is the coarsest choice and bucket is the finest.
    """
    count = 0
    for place_combo in itertools_product(*place_choices):
        for tiling_combo in itertools_product(*tiling_choices):
            for bucket_combo in itertools_product(*bucket_choices):
                yield place_combo, tiling_combo, bucket_combo
                count += 1
                if budget > 0 and count >= budget:
                    return


def _evaluate_shortlist(
    rows: Sequence[Tuple[Candidate, EvalResult]],
    *,
    run_dir: Path,
    cfg: DSEConfig,
) -> List[Tuple[Candidate, EvalResult]]:
    out: List[Tuple[Candidate, EvalResult]] = []

    for idx, (cand, res) in enumerate(rows):
        artifact_dir = candidate_artifact_dir(run_dir, cand.candidate_id)
        if cfg.persist_tree_text:
            write_tree_text(artifact_dir, cand.mapping.tree.display())

        final = res
        if cfg.run_dram_sim and idx < cfg.simulate_top_k:
            # Reuse the analytical model results already stored in res.analytical
            # to extract per-op compute cycles — avoids re-running estimate_tree.
            # estimate_candidate stores op_costs in the TreeEstimate that produces
            # analytical metrics; we need to re-run estimate_tree here to get
            # op_costs because EvalResult only stores AnalyticalMetrics (no op_costs).
            tree_est = estimate_tree(cand.mapping.tree, cand.workload.dag, cand.hardware)
            op_cycles = {oc.op_id: oc.compute_cycles for oc in tree_est.op_costs}

            trace_art, cfg_path = generate_trace_and_cfg(
                cand.mapping.tree,
                cand.workload.dag,
                cand.hardware,
                artifact_dir,
                op_cycles=op_cycles,
                emit_debug_csv=cfg.emit_trace_debug_csv,
            )
            sim = run_dram_simulation(
                trace_file=str(trace_art.trace_path),
                output_dir=str(artifact_dir),
                mapping_policy=cand.hardware.dram.map,
                number_cores=1,
                banks=cand.hardware.dram.banks,
                channels=cand.hardware.dram.channels,
                ranks=cand.hardware.dram.ranks,
            )
            dram_metrics = DramSimMetrics(
                success=sim.success,
                returncode=sim.returncode,
                dram_cycles=sim.dram_cycles,
                total_read_req=sim.total_read_req,
                total_write_req=sim.total_write_req,
                average_ipc=sim.average_ipc,
                total_instructions=sim.total_instructions,
                noc_hops_avg=sim.noc_hops_avg,
                total_energy_pj=sim.total_energy_pj,
                average_power_mw=sim.average_power_mw,
                cfg_path=sim.cfg_path or str(cfg_path),
                trace_path=str(trace_art.trace_path),
                stdout_path=sim.stdout_path,
                stderr_path=sim.stderr_path,
                bank_access=dict(sim.bank_access),
            )
            latency = float(sim.dram_cycles) if sim.dram_cycles is not None else res.objectives.latency
            final = EvalResult(
                candidate_id=res.candidate_id,
                analytical=res.analytical,
                objectives=res.objectives.__class__(
                    latency=latency,
                    dram_cost=res.objectives.dram_cost,
                    movement=res.objectives.movement,
                    power_mw=dram_metrics.average_power_mw,
                ),
                dram_sim=dram_metrics,
                stage="dram_sim",
            )

        append_record(run_dir, RunRecord(candidate=cand, result=final, artifact_dir=artifact_dir))
        out.append((cand, final))

    return out


__all__ = [
    "BatchRunSummary",
    "build_workload_case",
    "run_batch",
]
