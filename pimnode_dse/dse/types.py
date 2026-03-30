from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from pimnode_dse.hardware.arch_spec import HardwareSpec
from pimnode_dse.mapping.fusion.fusion_gene import FusionGene
from pimnode_dse.mapping.placement.dataflow import FlowBucket
from pimnode_dse.mapping.placement.node import GroupDP, PlacementPlan
from pimnode_dse.mapping.tilling.tilling_gene import GroupTilingSpec
from pimnode_dse.mapping.tree.mapping_tree import MappingTree
from pimnode_dse.mapping.workload.workload import AttentionWorkloadSpec, WorkloadDAG


@dataclass(frozen=True)
class WorkloadCase:
    workload_id: str
    spec: AttentionWorkloadSpec
    dag: WorkloadDAG
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MappingChoice:
    placement_plan: PlacementPlan
    tiling_map: Mapping[str, GroupTilingSpec]
    flow_buckets: Mapping[str, tuple[FlowBucket, ...]]
    tree: MappingTree


@dataclass(frozen=True)
class AnalyticalMetrics:
    dram_bytes: int = 0
    movement_bytes: int = 0
    sram_peak_bytes: int = 0
    est_compute_cycles: float = 0.0
    trace_request_count: int = 0
    est_power_mw: Optional[float] = None   # reserved for DRAMPower integration (P1-B)


@dataclass(frozen=True)
class DramSimMetrics:
    success: bool = False
    returncode: int = 0
    dram_cycles: Optional[float] = None
    total_read_req: Optional[int] = None
    total_write_req: Optional[int] = None
    average_ipc: Optional[float] = None
    total_instructions: Optional[int] = None
    noc_hops_avg: Optional[float] = None
    total_energy_pj: Optional[float] = None   # sum of per-bank DRAMPower energy (pJ)
    average_power_mw: Optional[float] = None  # total_energy_pj / sim_time_ns * 1e-3
    cfg_path: Optional[str] = None
    trace_path: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    bank_access: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveMetrics:
    latency: float
    dram_cost: float
    movement: float
    power_mw: Optional[float] = None   # populated from DramSimMetrics.average_power_mw when sim runs


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    workload: WorkloadCase
    hardware: HardwareSpec
    fusion_gene: FusionGene
    mapping: MappingChoice
    group_signatures: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    candidate_id: str
    analytical: AnalyticalMetrics
    objectives: ObjectiveMetrics
    dram_sim: Optional[DramSimMetrics] = None
    stage: str = "analytical"


@dataclass(frozen=True)
class TraceArtifact:
    trace_path: Path
    debug_csv_path: Optional[Path] = None
    request_count: int = 0


@dataclass(frozen=True)
class RunRecord:
    candidate: Candidate
    result: EvalResult
    artifact_dir: Path


@dataclass(frozen=True)
class DSEConfig:
    out_dir: Path
    max_fusion_genes: int = 4
    max_places_per_group: int = 4
    max_tilings_per_group: int = 8
    max_bucket_per_group: int = 2
    max_candidates_per_hw: int = 64
    shortlist_top_k: int = 16
    simulate_top_k: int = 8
    run_dram_sim: bool = False
    persist_tree_text: bool = True
    emit_trace_debug_csv: bool = True
    random_seed: int = 0


__all__ = [
    "AnalyticalMetrics",
    "Candidate",
    "DramSimMetrics",
    "DSEConfig",
    "EvalResult",
    "MappingChoice",
    "ObjectiveMetrics",
    "RunRecord",
    "TraceArtifact",
    "WorkloadCase",
]
