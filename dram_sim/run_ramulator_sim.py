from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


DEFAULT_PINOS_BINARY = "/app/pinos/pinos_release"
DEFAULT_NETWORK_CONFIG = "/app/pinos/examples/mesh11_lat"
DEFAULT_DRAM_CFG_TEMPLATE = "/app/pinos/dramcfg/dram.cfg"
DEFAULT_OUTPUT_DIR = "/app/PIMNode_DSE_o1/experiments/out_ramulator"

_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class RamulatorRunResult:
    success: bool
    returncode: int
    dram_cycles: Optional[float] = None
    total_read_req: Optional[int] = None
    total_write_req: Optional[int] = None
    bank_access: Dict[int, int] = field(default_factory=dict)
    average_ipc: Optional[float] = None
    total_instructions: Optional[int] = None
    noc_hops_avg: Optional[float] = None
    total_energy_pj: Optional[float] = None   # sum of Bank[N] Total Trace Energy (pJ)
    average_power_mw: Optional[float] = None  # derived from total_energy_pj / sim_time_ns * 1e6
    raw_stdout: str = ""
    raw_stderr: str = ""
    cfg_path: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None


def _normalize_trace_file_for_split_trace(trace_file: str) -> str:
    """Prefer the main trace file when split_trace=true.

    If the caller accidentally passes ``xxx.out.0`` while ``xxx.out`` also exists,
    return the main trace path so PINOS does not append another ``.0`` and look
    for ``xxx.out.0.0``.
    """
    trace_path = Path(trace_file).resolve()
    if trace_path.name.endswith('.0'):
        main_candidate = trace_path.with_name(trace_path.name[:-2])
        if main_candidate.is_file():
            return str(main_candidate)
    return str(trace_path)


def _patch_cfg_text(
    cfg_text: str,
    *,
    trace_file: str,
    mapping_policy: str,
    number_cores: int,
    banks: int,
    channels: int,
    ranks: int,
    enable_drampower: bool = True,
) -> str:
    replacements = {
        "trace": trace_file,
        "mapping_policy": mapping_policy,
        "number_cores": str(number_cores),
        "banks": str(banks),
        "channels": str(channels),
        "ranks": str(ranks),
        "drampower_SIM": "brief" if enable_drampower else "off",
    }

    for key, value in replacements.items():
        pattern = rf"(^\s*{re.escape(key)}\s*=\s*)([^;]+)(;)"
        if re.search(pattern, cfg_text, flags=re.MULTILINE):
            cfg_text = re.sub(
                pattern,
                rf"\g<1>{value}\g<3>",
                cfg_text,
                flags=re.MULTILINE,
            )
        else:
            cfg_text += f"\n{key} = {value};\n"

    return cfg_text


def _extract_optional_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None

    raw = m.group(1).strip()
    if raw in {"", "-", "--", "N/A", "n/a", "None", "none"}:
        return None

    try:
        return float(raw)
    except ValueError:
        return None


def _extract_optional_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None

    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_pinos_output(stdout: str, stderr: str, returncode: int, cfg_path: str) -> RamulatorRunResult:
    result = RamulatorRunResult(
        success=(returncode == 0),
        returncode=returncode,
        raw_stdout=stdout,
        raw_stderr=stderr,
        cfg_path=cfg_path,
    )

    merged = stdout + "\n" + stderr
    result.dram_cycles = _extract_optional_float(rf"->\s*cycles:\s*({_NUMBER_RE})", merged)

    m = re.search(r"===>\s*Total\s+total_read_req:\s*(\d+)\s*,\s*total_write_req:\s*(\d+)", merged)
    if m:
        result.total_read_req = int(m.group(1))
        result.total_write_req = int(m.group(2))

    for bank_id, access in re.findall(r"Bank\[\s*(\d+)\]\s*Total Access:\s*(\d+)", merged):
        result.bank_access[int(bank_id)] = int(access)

    result.average_ipc = _extract_optional_float(rf"->\s*average_ipc:\s*({_NUMBER_RE})", merged)
    result.total_instructions = _extract_optional_int(r"->\s*total instructions:\s*(\d+)", merged)
    result.noc_hops_avg = _extract_optional_float(
        rf"Hops average\s*=\s*({_NUMBER_RE}|-|--|N/A|n/a)",
        merged,
    )

    # DRAMPower energy: sum all "Bank[N] Total Trace Energy: <X> pJ" lines.
    # When drampower_SIM = brief/on in the cfg, pinos prints one line per bank.
    bank_energies = re.findall(
        rf"Bank\[\s*\d+\]\s*Total Trace Energy:\s*({_NUMBER_RE})\s*pJ",
        merged,
    )
    if bank_energies:
        total_pj = sum(float(e) for e in bank_energies)
        result.total_energy_pj = total_pj
        # Derive average power from sim time if available.
        # total time line: "-> total time: <X>ns"
        sim_time_ns = _extract_optional_float(rf"->\s*total time:\s*({_NUMBER_RE})\s*ns", merged)
        if sim_time_ns and sim_time_ns > 0:
            # power (mW) = energy (pJ) / time (ns) * 1e-3 [pJ/ns = mW]
            result.average_power_mw = total_pj / sim_time_ns * 1e-3

    return result


def run_dram_simulation(
    trace_file: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    pinos_binary: str = DEFAULT_PINOS_BINARY,
    network_config: str = DEFAULT_NETWORK_CONFIG,
    dram_cfg_template: str = DEFAULT_DRAM_CFG_TEMPLATE,
    mapping_policy: str = "ChRaBaRoCo",
    number_cores: int = 1,
    banks: int = 1,
    channels: int = 1,
    ranks: int = 1,
    enable_drampower: bool = True,
    timeout_sec: int = 300,
) -> RamulatorRunResult:
    """Run pinos with a patched DRAM config and return parsed simulation results.

    When *enable_drampower* is True (the default), the config is patched with
    ``drampower_SIM = brief`` so that pinos prints per-bank energy lines
    (``Bank[N] Total Trace Energy: X pJ``), which are parsed into
    ``total_energy_pj`` and ``average_power_mw`` on the returned result.
    """

    pinos_binary = str(Path(pinos_binary).resolve())
    network_config = str(Path(network_config).resolve())
    dram_cfg_template = str(Path(dram_cfg_template).resolve())
    trace_file = _normalize_trace_file_for_split_trace(trace_file)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(pinos_binary):
        raise FileNotFoundError(f"pinos binary not found: {pinos_binary}")
    if not os.path.isfile(network_config):
        raise FileNotFoundError(f"network config not found: {network_config}")
    if not os.path.isfile(dram_cfg_template):
        raise FileNotFoundError(f"dram cfg template not found: {dram_cfg_template}")
    if not os.path.isfile(trace_file):
        raise FileNotFoundError(f"trace file not found: {trace_file}")

    cfg_text = Path(dram_cfg_template).read_text(encoding="utf-8", errors="ignore")
    patched = _patch_cfg_text(
        cfg_text,
        trace_file=trace_file,
        mapping_policy=mapping_policy,
        number_cores=number_cores,
        banks=banks,
        channels=channels,
        ranks=ranks,
        enable_drampower=enable_drampower,
    )

    patched_cfg_path = output_dir / "dram_patched.cfg"
    patched_cfg_path.write_text(patched, encoding="utf-8")

    cmd = [pinos_binary, network_config, str(patched_cfg_path)]
    stdout_path = output_dir / "pinos_stdout.log"
    stderr_path = output_dir / "pinos_stderr.log"

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        result = _parse_pinos_output(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            cfg_path=str(patched_cfg_path),
        )
    except subprocess.TimeoutExpired as exc:
        result = RamulatorRunResult(
            success=False,
            returncode=-9,
            raw_stdout=exc.stdout or "",
            raw_stderr=exc.stderr or "",
            cfg_path=str(patched_cfg_path),
        )

    stdout_path.write_text(result.raw_stdout, encoding="utf-8", errors="ignore")
    stderr_path.write_text(result.raw_stderr, encoding="utf-8", errors="ignore")
    result.stdout_path = str(stdout_path)
    result.stderr_path = str(stderr_path)
    return result


__all__ = [
    "DEFAULT_PINOS_BINARY",
    "DEFAULT_NETWORK_CONFIG",
    "DEFAULT_DRAM_CFG_TEMPLATE",
    "DEFAULT_OUTPUT_DIR",
    "RamulatorRunResult",
    "run_dram_simulation",
]
