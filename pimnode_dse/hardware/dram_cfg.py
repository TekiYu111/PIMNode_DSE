from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .arch_spec import HardwareSpec


class DramCfgError(ValueError):
    """Raised when a PINOS / Ramulator cfg cannot be rendered."""


@dataclass(frozen=True)
class PinosRunSpec:
    """Run-time inputs for one PINOS / Ramulator simulation.

    These fields are not hardware truth. They belong to one run and are kept
    separate from HardwareSpec so that arch_spec.py stays hardware-only.
    """

    trace: str
    trace_format: str = "zsim"
    split_trace: bool = True
    disable_per_scheduling: bool = True
    core_org: str = "pimLogic"
    core_num: int = 1
    cache: str = "L1L2"
    cpu_frequency: int = 800
    early_exit: str = "off"
    expected_limit_insts: int = 0
    pim_mode: str = "pim"
    translation: str = "Random"
    page_size: int = 10
    unlimit_bandwidth: bool = False
    no_dram_latency: bool = False
    node_buf: int = 1000
    record_cmd_trace: str = "off"
    print_cmd_trace: str = "off"
    guest_cpu: bool = False
    guest_trace: str = ""
    drampower_sim: str = "off"
    drampower_memspecs: str = "./memSpecs/nDRAM_800MHz_8bit_A.xml"
    stats_file: str = "pim.stats"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace:
            raise DramCfgError("trace is required")
        if self.core_num <= 0:
            raise DramCfgError("core_num must be > 0")
        if self.cpu_frequency <= 0:
            raise DramCfgError("cpu_frequency must be > 0")
        if self.page_size <= 0:
            raise DramCfgError("page_size must be > 0")
        if self.node_buf <= 0:
            raise DramCfgError("node_buf must be > 0")


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _line(key: str, value: Any) -> str:
    return f"{key} = {value};"


def _cfg_value(value: Any) -> Any:
    if isinstance(value, bool):
        return _flag(value)
    return value


def render_pinos_cfg(hw: HardwareSpec, run: PinosRunSpec) -> str:
    """Render one parser-friendly dram.cfg in standard PINOS style."""

    lines = [
        "// PINOS Configuration File.",
        "// Auto-generated. Keep the standard format: key = value;",
        "",
        "//========= Trace Config =========//",
        "// The trace file that should be loaded.",
        _line("trace", run.trace),
        "// Defines the source of the memory traces.",
        "// Use zsim for traces generated with the modified ZSim flow.",
        _line("trace_format", run.trace_format),
        "// Use one trace split by core id when true.",
        _line("split_trace", _cfg_value(run.split_trace)),
        "",
        "//========= PIM CORE Config =========//",
        _line("disable_per_scheduling", _cfg_value(run.disable_per_scheduling)),
        _line("core_org", run.core_org),
        _line("number_cores", run.core_num),
        _line("banks", hw.dram.banks),
        _line("cache", run.cache),
        _line("cpu_frequency", run.cpu_frequency),
        _line("early_exit", run.early_exit),
        _line("expected_limit_insts", run.expected_limit_insts),
        _line("pim_mode", run.pim_mode),
        _line("translation", run.translation),
        _line("pageSize", run.page_size),
        "",
        "//========= DRAM Related Config =========//",
        _line("standard", hw.dram.standard),
        _line("unlimit_bandwidth", _cfg_value(run.unlimit_bandwidth)),
        _line("no_DRAM_latency", _cfg_value(run.no_dram_latency)),
        _line("channels", hw.dram.channels),
        _line("ranks", hw.dram.ranks),
        _line("nodeBufSize", run.node_buf),
        _line("dram_speed", hw.dram.speed),
        _line("dram_org", hw.dram.org),
        _line("record_cmd_trace", run.record_cmd_trace),
        _line("print_cmd_trace", run.print_cmd_trace),
        "// address mapping policy",
        _line("mapping_policy", hw.dram.map),
    ]

    if hw.dram.extra:
        for key, value in hw.dram.extra.items():
            lines.append(_line(key, _cfg_value(value)))

    lines.extend([
        "",
        "//========= Guest CPU ===========//",
        _line("guestCPU", _cfg_value(run.guest_cpu)),
    ])

    if run.guest_trace:
        lines.append(_line("guestCPU_trace", run.guest_trace))

    lines.extend([
        "",
        "//========= DRAM-POWER =========//",
        _line("drampower_SIM", run.drampower_sim),
        _line("drampower_memspecs", run.drampower_memspecs),
        "",
        "//========= Simulation output =========//",
        _line("stats_file", run.stats_file),
    ])

    if run.extra:
        lines.append("")
        for key, value in run.extra.items():
            lines.append(_line(key, _cfg_value(value)))

    return "\n".join(lines) + "\n"


def write_pinos_cfg(out: str | Path, hw: HardwareSpec, run: PinosRunSpec) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_pinos_cfg(hw, run), encoding="utf-8")
    return out_path


__all__ = [
    "DramCfgError",
    "PinosRunSpec",
    "render_pinos_cfg",
    "write_pinos_cfg",
]
