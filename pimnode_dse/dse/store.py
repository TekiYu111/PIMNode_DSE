from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .types import Candidate, EvalResult, RunRecord


def ensure_run_dir(root: Path, run_id: str) -> Path:
    out = Path(root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    return out


def candidate_artifact_dir(run_dir: Path, candidate_id: str) -> Path:
    out = run_dir / "artifacts" / candidate_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def append_record(run_dir: Path, record: RunRecord) -> Path:
    path = run_dir / "records.jsonl"
    payload = _record_to_dict(record)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def write_pareto(run_dir: Path, rows: list[tuple[Candidate, EvalResult]]) -> Path:
    path = run_dir / "pareto.json"
    payload = [
        {
            "candidate_id": cand.candidate_id,
            "workload_id": cand.workload.workload_id,
            "hardware": _hardware_to_dict(cand.hardware),
            "objectives": {
                "latency": res.objectives.latency,
                "dram_cost": res.objectives.dram_cost,
                "movement": res.objectives.movement,
            },
            "stage": res.stage,
        }
        for cand, res in rows
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_tree_text(artifact_dir: Path, text: str) -> Path:
    path = artifact_dir / "tree.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _record_to_dict(record: RunRecord) -> dict[str, Any]:
    cand = record.candidate
    res = record.result
    return {
        "candidate_id": cand.candidate_id,
        "workload_id": cand.workload.workload_id,
        "hardware": _hardware_to_dict(cand.hardware),
        "fusion_gene": cand.fusion_gene.gene_id,
        "group_signatures": cand.group_signatures,
        "meta": cand.meta,
        "analytical": {
            "dram_bytes": res.analytical.dram_bytes,
            "movement_bytes": res.analytical.movement_bytes,
            "sram_peak_bytes": res.analytical.sram_peak_bytes,
            "est_compute_cycles": res.analytical.est_compute_cycles,
            "trace_request_count": res.analytical.trace_request_count,
        },
        "objectives": {
            "latency": res.objectives.latency,
            "dram_cost": res.objectives.dram_cost,
            "movement": res.objectives.movement,
        },
        "dram_sim": None if res.dram_sim is None else {
            "success": res.dram_sim.success,
            "returncode": res.dram_sim.returncode,
            "dram_cycles": res.dram_sim.dram_cycles,
            "total_read_req": res.dram_sim.total_read_req,
            "total_write_req": res.dram_sim.total_write_req,
            "average_ipc": res.dram_sim.average_ipc,
            "total_instructions": res.dram_sim.total_instructions,
            "noc_hops_avg": res.dram_sim.noc_hops_avg,
            "cfg_path": res.dram_sim.cfg_path,
            "trace_path": res.dram_sim.trace_path,
            "stdout_path": res.dram_sim.stdout_path,
            "stderr_path": res.dram_sim.stderr_path,
            "bank_access": res.dram_sim.bank_access,
        },
        "stage": res.stage,
        "artifact_dir": str(record.artifact_dir),
    }


def _hardware_to_dict(hw) -> dict[str, Any]:
    return {
        "name": hw.name,
        "dram": {
            "standard": hw.dram.standard,
            "speed": hw.dram.speed,
            "org": hw.dram.org,
            "channels": hw.dram.channels,
            "ranks": hw.dram.ranks,
            "banks": hw.dram.banks,
            "map": hw.dram.map,
        },
        "sram": {
            "cap": hw.sram.cap,
            "bw": hw.sram.bw,
            "concurrency": hw.sram.concurrency,
        },
        "pe": {
            "rows": hw.pe.rows,
            "cols": hw.pe.cols,
            "macs": hw.pe.macs,
            "concurrency": hw.pe.concurrency,
            "has_acc": hw.pe.has_acc,
        },
    }


__all__ = [
    "append_record",
    "candidate_artifact_dir",
    "ensure_run_dir",
    "write_manifest",
    "write_pareto",
    "write_tree_text",
]
