from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from pimnode_dse.analysis.analyzer import AnalysisSummary
except Exception:
    AnalysisSummary = Any  # type: ignore


@dataclass
class RamulatorSummary:
    """Normalized DRAM-side simulation result."""
    dram_cycles: Optional[float] = None
    dram_stall: Optional[float] = None
    dram_bytes: Optional[int] = None
    dram_energy: Optional[float] = None
    bandwidth_util: Optional[float] = None
    row_util: Optional[float] = None
    bank_util: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveVector:
    """Three-objective result used for dominance / Pareto comparison."""
    latency: float
    dram_bytes: float
    energy: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (float(self.latency), float(self.dram_bytes), float(self.energy))


@dataclass
class CandidateResult:
    """Unified candidate result after analyzer-only or analyzer+Ramulator merge."""
    candidate_id: str
    feasible: bool
    infeasible_reasons: List[str] = field(default_factory=list)

    objectives: Optional[ObjectiveVector] = None

    score_fast: Optional[float] = None
    score_final: Optional[float] = None
    edp: Optional[float] = None

    onchip_cycles: Optional[float] = None
    dram_cycles: Optional[float] = None
    dram_stall: Optional[float] = None
    total_latency: Optional[float] = None
    dram_bytes: Optional[float] = None
    energy: Optional[float] = None
    peak_sram_bytes: Optional[int] = None

    analyzer_summary: Any = None
    ramulator_summary: Optional[RamulatorSummary] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dominance_key(self) -> Optional[Tuple[float, float, float]]:
        return None if self.objectives is None else self.objectives.as_tuple()


@dataclass
class EvaluatorConfig:
    """Config for unified evaluation and multi-objective search support."""
    score_mode_fast: str = "weighted_sum"
    score_mode_final: str = "weighted_sum"

    weight_latency: float = 1.0
    weight_dram_bytes: float = 0.0
    weight_energy: float = 0.0

    norm_latency: float = 1.0
    norm_dram_bytes: float = 1.0
    norm_energy: float = 1.0

    overlap_threshold_tile_count: int = 2
    sync_overhead_cycles: float = 0.0
    stall_penalty_beta: float = 0.0

    prefer_analyzer_total_if_no_ramulator: bool = True
    infeasible_penalty: float = 1e30
    keep_raw_metrics: bool = True


class Evaluator:
    """Unified evaluator for search/rerank/Pareto update.

    Responsibilities:
    1. convert analyzer output into standardized objectives
    2. merge on-chip analyzer and Ramulator DRAM results into final latency
    3. produce helper scalar scores for beam keep / top-K
    4. provide dominance / Pareto utilities
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.cfg = config or EvaluatorConfig()

    def evaluate_fast(
        self,
        candidate_id: str,
        analyzer_summary: Any,
    ) -> CandidateResult:
        """Build a fast candidate result from analyzer only."""
        feasible, reasons = self._extract_feasibility(analyzer_summary)
        onchip_cycles = self._get(analyzer_summary, "onchip_cycles_est")
        dram_cycles_est = self._get(analyzer_summary, "dram_cycles_est")
        total_cycles_est = self._get(analyzer_summary, "total_cycles_est")
        dram_bytes = self._get(analyzer_summary, "dram_bytes")
        energy = self._get(analyzer_summary, "energy_est")
        peak_sram_bytes = self._get(analyzer_summary, "peak_sram_bytes")

        if self.cfg.prefer_analyzer_total_if_no_ramulator and total_cycles_est is not None:
            latency = float(total_cycles_est)
        else:
            onchip = float(onchip_cycles or 0.0)
            dram = float(dram_cycles_est or 0.0)
            latency = max(onchip, dram)

        objectives = ObjectiveVector(
            latency=float(latency),
            dram_bytes=float(dram_bytes or 0.0),
            energy=float(energy or 0.0),
        )

        score_fast = self._compute_scalar_score(
            objectives=objectives,
            feasible=feasible,
            mode=self.cfg.score_mode_fast,
        )
        edp = objectives.latency * objectives.energy

        result = CandidateResult(
            candidate_id=candidate_id,
            feasible=feasible,
            infeasible_reasons=reasons,
            objectives=objectives,
            score_fast=score_fast,
            score_final=None,
            edp=edp,
            onchip_cycles=float(onchip_cycles) if onchip_cycles is not None else None,
            dram_cycles=float(dram_cycles_est) if dram_cycles_est is not None else None,
            dram_stall=None,
            total_latency=float(latency),
            dram_bytes=float(dram_bytes or 0.0),
            energy=float(energy or 0.0),
            peak_sram_bytes=int(peak_sram_bytes) if peak_sram_bytes is not None else None,
            analyzer_summary=analyzer_summary,
            ramulator_summary=None,
            metadata={},
        )

        if self.cfg.keep_raw_metrics:
            result.metadata["source"] = "analyzer_only"
            result.metadata["analyzer_total_cycles_est"] = total_cycles_est
        return result

    def evaluate_final(
        self,
        candidate_id: str,
        analyzer_summary: Any,
        ramulator_summary: Optional[RamulatorSummary | Mapping[str, Any]] = None,
    ) -> CandidateResult:
        """Build final candidate result by merging analyzer + Ramulator."""
        feasible, reasons = self._extract_feasibility(analyzer_summary)
        onchip_cycles = float(self._get(analyzer_summary, "onchip_cycles_est") or 0.0)
        analyzer_dram_cycles = self._get(analyzer_summary, "dram_cycles_est")
        analyzer_dram_bytes = self._get(analyzer_summary, "dram_bytes")
        analyzer_energy = self._get(analyzer_summary, "energy_est")
        peak_sram_bytes = self._get(analyzer_summary, "peak_sram_bytes")

        ram = self._normalize_ramulator_summary(ramulator_summary)
        dram_cycles = (
            float(ram.dram_cycles)
            if ram is not None and ram.dram_cycles is not None
            else float(analyzer_dram_cycles or 0.0)
        )
        dram_stall = (
            float(ram.dram_stall)
            if ram is not None and ram.dram_stall is not None
            else 0.0
        )
        dram_bytes = (
            float(ram.dram_bytes)
            if ram is not None and ram.dram_bytes is not None
            else float(analyzer_dram_bytes or 0.0)
        )

        total_latency = self.merge_total_latency(
            analyzer_summary=analyzer_summary,
            onchip_cycles=onchip_cycles,
            dram_cycles=dram_cycles,
            dram_stall=dram_stall,
        )

        total_energy = self.merge_total_energy(
            analyzer_energy=float(analyzer_energy or 0.0),
            ramulator_summary=ram,
        )

        objectives = ObjectiveVector(
            latency=float(total_latency),
            dram_bytes=float(dram_bytes),
            energy=float(total_energy),
        )

        score_final = self._compute_scalar_score(
            objectives=objectives,
            feasible=feasible,
            mode=self.cfg.score_mode_final,
        )
        edp = objectives.latency * objectives.energy

        result = CandidateResult(
            candidate_id=candidate_id,
            feasible=feasible,
            infeasible_reasons=reasons,
            objectives=objectives,
            score_fast=None,
            score_final=score_final,
            edp=edp,
            onchip_cycles=onchip_cycles,
            dram_cycles=dram_cycles,
            dram_stall=dram_stall,
            total_latency=total_latency,
            dram_bytes=dram_bytes,
            energy=total_energy,
            peak_sram_bytes=int(peak_sram_bytes) if peak_sram_bytes is not None else None,
            analyzer_summary=analyzer_summary,
            ramulator_summary=ram,
            metadata={},
        )

        if self.cfg.keep_raw_metrics:
            result.metadata["source"] = "analyzer_plus_ramulator" if ram is not None else "analyzer_only_fallback"
            result.metadata["latency_merge_mode"] = self._infer_latency_merge_mode(analyzer_summary, dram_stall)
        return result

    def merge_total_latency(
        self,
        analyzer_summary: Any,
        onchip_cycles: float,
        dram_cycles: float,
        dram_stall: float = 0.0,
    ) -> float:
        """Unified total latency merge.

        Overlap mode:
            total ~= max(onchip, dram) + sync + beta * dram_stall

        Serial mode:
            total ~= onchip + dram + sync
        """
        can_overlap = self._can_overlap_dram_with_onchip(analyzer_summary)
        sync = float(self.cfg.sync_overhead_cycles)

        if can_overlap:
            return max(onchip_cycles, dram_cycles) + sync + self.cfg.stall_penalty_beta * float(dram_stall)
        return onchip_cycles + dram_cycles + sync

    def merge_total_energy(
        self,
        analyzer_energy: float,
        ramulator_summary: Optional[RamulatorSummary],
    ) -> float:
        if ramulator_summary is None or ramulator_summary.dram_energy is None:
            return float(analyzer_energy)
        return float(analyzer_energy) + float(ramulator_summary.dram_energy)

    def dominates(self, a: CandidateResult, b: CandidateResult) -> bool:
        """Pareto dominance: all objectives <= and at least one strictly <."""
        if a.objectives is None or b.objectives is None:
            return False
        xa = a.objectives.as_tuple()
        xb = b.objectives.as_tuple()
        all_no_worse = all(va <= vb for va, vb in zip(xa, xb))
        any_strict_better = any(va < vb for va, vb in zip(xa, xb))
        return all_no_worse and any_strict_better

    def pareto_insert(
        self,
        frontier: Sequence[CandidateResult],
        candidate: CandidateResult,
    ) -> List[CandidateResult]:
        """Insert candidate into Pareto frontier and return updated frontier."""
        if candidate.objectives is None:
            return list(frontier)

        new_frontier: List[CandidateResult] = []
        dominated_by_existing = False

        for existing in frontier:
            if existing.objectives is None:
                new_frontier.append(existing)
                continue
            if self.dominates(existing, candidate):
                dominated_by_existing = True
                new_frontier.append(existing)
            elif self.dominates(candidate, existing):
                continue
            else:
                new_frontier.append(existing)

        if not dominated_by_existing:
            new_frontier.append(candidate)
        return new_frontier

    def pareto_filter(self, candidates: Sequence[CandidateResult]) -> List[CandidateResult]:
        frontier: List[CandidateResult] = []
        for cand in candidates:
            frontier = self.pareto_insert(frontier, cand)
        return frontier

    def _compute_scalar_score(
        self,
        objectives: ObjectiveVector,
        feasible: bool,
        mode: str,
    ) -> float:
        if not feasible:
            return float(self.cfg.infeasible_penalty)

        mode = str(mode).lower()
        if mode == "latency_only":
            return float(objectives.latency)
        if mode == "energy_only":
            return float(objectives.energy)
        if mode == "edp":
            return float(objectives.latency * objectives.energy)
        if mode == "weighted_sum":
            return self.weighted_sum(objectives)
        raise ValueError(f"Unsupported score mode: {mode}")

    def weighted_sum(self, objectives: ObjectiveVector) -> float:
        nl = float(objectives.latency) / max(float(self.cfg.norm_latency), 1e-12)
        nd = float(objectives.dram_bytes) / max(float(self.cfg.norm_dram_bytes), 1e-12)
        ne = float(objectives.energy) / max(float(self.cfg.norm_energy), 1e-12)
        return (
            self.cfg.weight_latency * nl
            + self.cfg.weight_dram_bytes * nd
            + self.cfg.weight_energy * ne
        )

    def sort_key_fast(self, result: CandidateResult) -> float:
        if result.score_fast is None:
            return float(self.cfg.infeasible_penalty)
        return float(result.score_fast)

    def sort_key_final(self, result: CandidateResult) -> float:
        if result.score_final is None:
            return float(self.cfg.infeasible_penalty)
        return float(result.score_final)

    def _extract_feasibility(self, analyzer_summary: Any) -> Tuple[bool, List[str]]:
        feasibility = self._get(analyzer_summary, "feasibility")
        if feasibility is None:
            return True, []
        feasible = bool(self._get(feasibility, "feasible", default=True))
        reasons = list(self._get(feasibility, "reasons", default=[]))
        return feasible, reasons

    def _normalize_ramulator_summary(
        self,
        ram: Optional[RamulatorSummary | Mapping[str, Any]],
    ) -> Optional[RamulatorSummary]:
        if ram is None:
            return None
        if isinstance(ram, RamulatorSummary):
            return ram
        if isinstance(ram, Mapping):
            return RamulatorSummary(
                dram_cycles=ram.get("dram_cycles"),
                dram_stall=ram.get("dram_stall"),
                dram_bytes=ram.get("dram_bytes"),
                dram_energy=ram.get("dram_energy"),
                bandwidth_util=ram.get("bandwidth_util"),
                row_util=ram.get("row_util"),
                bank_util=ram.get("bank_util"),
                metadata=dict(ram.get("metadata", {}) or {}),
            )
        raise TypeError("ramulator_summary must be RamulatorSummary, Mapping, or None")

    def _can_overlap_dram_with_onchip(self, analyzer_summary: Any) -> bool:
        pipeline = self._get(analyzer_summary, "pipeline")
        if pipeline is None:
            return False

        if isinstance(pipeline, Mapping):
            for _, timing in pipeline.items():
                enabled = bool(self._get(timing, "pipeline_enabled", default=False))
                tile_count = int(self._get(timing, "tile_count", default=1) or 1)
                if enabled and tile_count >= self.cfg.overlap_threshold_tile_count:
                    return True
            return False

        enabled = bool(self._get(pipeline, "pipeline_enabled", default=False))
        tile_count = int(self._get(pipeline, "tile_count", default=1) or 1)
        return enabled and tile_count >= self.cfg.overlap_threshold_tile_count

    def _infer_latency_merge_mode(self, analyzer_summary: Any, dram_stall: float) -> str:
        can_overlap = self._can_overlap_dram_with_onchip(analyzer_summary)
        if can_overlap and dram_stall > 0:
            return "overlap_with_stall_penalty"
        if can_overlap:
            return "overlap"
        return "serial"

    def _get(self, obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)


__all__ = [
    "RamulatorSummary",
    "ObjectiveVector",
    "CandidateResult",
    "EvaluatorConfig",
    "Evaluator",
]
