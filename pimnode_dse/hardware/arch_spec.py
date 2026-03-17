from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class ArchSpecError(ValueError):
    """Raised when an architecture specification is invalid."""


@dataclass(frozen=True)
class DRAMSpec:
    """DRAM spec parsed from a Ramulator dram.cfg file.

    This object is intentionally a read-only mirror of the DRAM configuration
    that will be used at runtime. It is not the search-space source of truth.
    """

    standard: str
    dram_speed: str
    dram_org: str
    channels: int
    ranks: int
    banks: int
    mapping_policy: str
    unlimit_bandwidth: bool = False
    no_DRAM_latency: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.standard:
            raise ArchSpecError("DRAM standard must be a non-empty string")
        if not self.dram_speed:
            raise ArchSpecError("DRAM dram_speed must be a non-empty string")
        if not self.dram_org:
            raise ArchSpecError("DRAM dram_org must be a non-empty string")
        if self.channels <= 0:
            raise ArchSpecError("DRAM channels must be > 0")
        if self.ranks <= 0:
            raise ArchSpecError("DRAM ranks must be > 0")
        if self.banks <= 0:
            raise ArchSpecError("DRAM banks must be > 0")
        if not self.mapping_policy:
            raise ArchSpecError("DRAM mapping_policy must be a non-empty string")


@dataclass(frozen=True)
class SRAMSpec:
    """Simplified multi-bank SRAM model: capacity + aggregate bandwidth."""

    total_capacity_bytes: int
    total_bandwidth_bytes_per_cycle: float
    bank_count: int = 1
    read_write_shared_bandwidth: bool = True

    def __post_init__(self) -> None:
        if self.total_capacity_bytes <= 0:
            raise ArchSpecError("SRAM capacity must be > 0 bytes")
        if self.total_bandwidth_bytes_per_cycle <= 0:
            raise ArchSpecError("SRAM total_bandwidth_bytes_per_cycle must be > 0")
        if self.bank_count <= 0:
            raise ArchSpecError("SRAM bank_count must be > 0")

    @property
    def coarse_per_bank_bandwidth_bytes_per_cycle(self) -> float:
        return self.total_bandwidth_bytes_per_cycle / float(self.bank_count)


@dataclass(frozen=True)
class DESpec:
    """Data Engine (DE): the fine-grained scheduling / dispatch center."""

    bus_bandwidth_bytes_per_cycle: float
    supports_pipeline: bool = True
    max_inflight_tiles: int = 1
    supports_direct_dram_to_pe: bool = False
    supports_direct_pe_to_sram_writeback: bool = True

    def __post_init__(self) -> None:
        if self.bus_bandwidth_bytes_per_cycle <= 0:
            raise ArchSpecError("DE bus_bandwidth_bytes_per_cycle must be > 0")
        if self.max_inflight_tiles <= 0:
            raise ArchSpecError("DE max_inflight_tiles must be > 0")


@dataclass(frozen=True)
class PEArraySpec:
    """Black-box compute array model."""

    rows: int
    cols: int
    macs_per_pe_per_cycle: int = 1
    input_bandwidth_bytes_per_cycle: Optional[float] = None
    output_bandwidth_bytes_per_cycle: Optional[float] = None
    supports_accumulate_in_place: bool = True

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ArchSpecError("PE array rows and cols must both be > 0")
        if self.macs_per_pe_per_cycle <= 0:
            raise ArchSpecError("macs_per_pe_per_cycle must be > 0")
        if self.input_bandwidth_bytes_per_cycle is not None and self.input_bandwidth_bytes_per_cycle <= 0:
            raise ArchSpecError("PE input_bandwidth_bytes_per_cycle must be > 0 when provided")
        if self.output_bandwidth_bytes_per_cycle is not None and self.output_bandwidth_bytes_per_cycle <= 0:
            raise ArchSpecError("PE output_bandwidth_bytes_per_cycle must be > 0 when provided")

    @property
    def total_pes(self) -> int:
        return self.rows * self.cols

    @property
    def total_mac_per_cycle(self) -> int:
        return self.total_pes * self.macs_per_pe_per_cycle


@dataclass(frozen=True)
class PIMNodeArchSpec:
    """Canonical hardware description for a single PIM-node storage subsystem."""

    dram: DRAMSpec
    sram: SRAMSpec
    de: DESpec
    pe: PEArraySpec
    clock_hz: Optional[float] = None
    bytes_per_element_default: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.clock_hz is not None and self.clock_hz <= 0:
            raise ArchSpecError("clock_hz must be > 0 when provided")
        if self.bytes_per_element_default <= 0:
            raise ArchSpecError("bytes_per_element_default must be > 0")

    @property
    def memory_levels(self) -> tuple[str, ...]:
        return ("DRAM", "SRAM", "PE")

    @property
    def onchip_total_feed_bandwidth_bytes_per_cycle(self) -> float:
        candidates = [self.sram.total_bandwidth_bytes_per_cycle, self.de.bus_bandwidth_bytes_per_cycle]
        if self.pe.input_bandwidth_bytes_per_cycle is not None:
            candidates.append(self.pe.input_bandwidth_bytes_per_cycle)
        return min(candidates)

    def validate_transfer_path(self, src_level: str, dst_level: str) -> bool:
        src = src_level.upper()
        dst = dst_level.upper()
        if src == dst:
            return True
        if src == "DRAM" and dst == "PE":
            return self.de.supports_direct_dram_to_pe
        if src == "PE" and dst == "DRAM":
            return False
        if {src, dst}.issubset({"DRAM", "SRAM", "PE"}):
            if src == "PE" and dst == "SRAM":
                return self.de.supports_direct_pe_to_sram_writeback
            return True
        return False


def kb(x: int | float) -> int:
    return int(x * 1024)


def mb(x: int | float) -> int:
    return int(x * 1024 * 1024)


def gb(x: int | float) -> int:
    return int(x * 1024 * 1024 * 1024)


def _parse_cfg_text(cfg_text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw_line in cfg_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        for marker in ("#", "//"):
            if marker in line:
                line = line.split(marker, 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ArchSpecError(f"Cannot parse boolean value from {v!r}")


def _maybe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    return int(v)


def _maybe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    return float(v)


def dram_spec_from_cfg(cfg_path: str | Path) -> DRAMSpec:
    cfg_path = Path(cfg_path)
    if not cfg_path.is_file():
        raise ArchSpecError(f"DRAM cfg file not found: {cfg_path}")

    raw = _parse_cfg_text(cfg_path.read_text(encoding="utf-8"))
    required = ["standard", "dram_speed", "dram_org", "channels", "ranks", "banks", "mapping_policy"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ArchSpecError(f"Missing required DRAM cfg keys: {', '.join(missing)}")

    consumed = set(required) | {"unlimit_bandwidth", "no_DRAM_latency", "number_cores"}
    extra = {k: v for k, v in raw.items() if k not in consumed}

    return DRAMSpec(
        standard=str(raw["standard"]),
        dram_speed=str(raw["dram_speed"]),
        dram_org=str(raw["dram_org"]),
        channels=int(raw["channels"]),
        ranks=int(raw["ranks"]),
        banks=int(raw["banks"]),
        mapping_policy=str(raw["mapping_policy"]),
        unlimit_bandwidth=_parse_bool(raw.get("unlimit_bandwidth", "false")),
        no_DRAM_latency=_parse_bool(raw.get("no_DRAM_latency", "false")),
        extra=extra,
    )


def arch_from_mapping(mapping: Mapping[str, Any]) -> PIMNodeArchSpec:
    """Build a canonical PIMNodeArchSpec from a plain mapping/dict.

    DRAM must come from a dram.cfg path. SRAM/DE/PE still come from the mapping.
    """

    dram_cfg_path = mapping.get("dram_cfg_path")
    if not dram_cfg_path:
        raise ArchSpecError("Missing required field: dram_cfg_path")

    sram_cfg = dict(mapping.get("sram", {}))
    de_cfg = dict(mapping.get("de", {}))
    pe_cfg = dict(mapping.get("pe", {}))

    dram = dram_spec_from_cfg(str(dram_cfg_path))
    sram = SRAMSpec(
        total_capacity_bytes=int(sram_cfg["capacity_bytes"]),
        total_bandwidth_bytes_per_cycle=float(sram_cfg["bandwidth_bpc"]),
        bank_count=int(sram_cfg.get("bank_count", 1)),
        read_write_shared_bandwidth=bool(sram_cfg.get("read_write_shared_bandwidth", True)),
    )
    de = DESpec(
        bus_bandwidth_bytes_per_cycle=float(de_cfg["bus_bandwidth_bpc"]),
        supports_pipeline=bool(de_cfg.get("supports_pipeline", True)),
        max_inflight_tiles=int(de_cfg.get("max_inflight_tiles", 1)),
        supports_direct_dram_to_pe=bool(de_cfg.get("supports_direct_dram_to_pe", False)),
        supports_direct_pe_to_sram_writeback=bool(de_cfg.get("supports_direct_pe_to_sram_writeback", True)),
    )
    pe = PEArraySpec(
        rows=int(pe_cfg["rows"]),
        cols=int(pe_cfg["cols"]),
        macs_per_pe_per_cycle=int(pe_cfg.get("macs_per_pe_per_cycle", 1)),
        input_bandwidth_bytes_per_cycle=_maybe_float(pe_cfg.get("input_bandwidth_bpc")),
        output_bandwidth_bytes_per_cycle=_maybe_float(pe_cfg.get("output_bandwidth_bpc")),
        supports_accumulate_in_place=bool(pe_cfg.get("supports_accumulate_in_place", True)),
    )
    return PIMNodeArchSpec(
        dram=dram,
        sram=sram,
        de=de,
        pe=pe,
        clock_hz=_maybe_float(mapping.get("clock_hz")),
        bytes_per_element_default=int(mapping.get("bytes_per_element_default", 2)),
        extra=dict(mapping.get("extra", {})),
    )


def arch_from_legacy_hardware_gene(hw: Any, dram_cfg_path: str | Path) -> PIMNodeArchSpec:
    """Convert a legacy HardwareGene-like object into the canonical spec.

    DRAM runtime truth is always loaded from dram_cfg_path.
    SRAM/DE/PE fields are still sourced from the hardware gene.
    """

    def _get(name: str, default: Any = None) -> Any:
        return getattr(hw, name, default)

    dram = dram_spec_from_cfg(dram_cfg_path)

    sram_capacity_kb = float(_get("sram_capacity_kb"))
    sram_bw = float(_get("sram_bandwidth"))
    sram_banks = int(_get("sram_bank_count", 1))
    pe_m = int(_get("pe_array_m"))
    pe_n = int(_get("pe_array_n"))
    pe_mac_num = int(_get("pe_mac_num", 1))

    de_bus_bw = _maybe_float(_get("de_bus_bandwidth"))
    if de_bus_bw is None:
        de_bus_bw = sram_bw

    pe_input_bw = _maybe_float(_get("pe_input_bandwidth"))
    pe_output_bw = _maybe_float(_get("pe_output_bandwidth"))
    extra = dict(_get("extra_params", {}) or {})

    return PIMNodeArchSpec(
        dram=dram,
        sram=SRAMSpec(
            total_capacity_bytes=kb(sram_capacity_kb),
            total_bandwidth_bytes_per_cycle=sram_bw,
            bank_count=sram_banks,
        ),
        de=DESpec(
            bus_bandwidth_bytes_per_cycle=float(de_bus_bw),
            supports_pipeline=bool(_get("supports_pipeline", True)),
            max_inflight_tiles=int(_get("max_inflight_tiles", 1)),
            supports_direct_dram_to_pe=bool(_get("supports_direct_dram_to_pe", False)),
            supports_direct_pe_to_sram_writeback=bool(_get("supports_direct_pe_to_sram_writeback", True)),
        ),
        pe=PEArraySpec(
            rows=pe_m,
            cols=pe_n,
            macs_per_pe_per_cycle=pe_mac_num,
            input_bandwidth_bytes_per_cycle=pe_input_bw,
            output_bandwidth_bytes_per_cycle=pe_output_bw,
            supports_accumulate_in_place=bool(_get("supports_accumulate_in_place", True)),
        ),
        clock_hz=_maybe_float(_get("clock_hz")),
        bytes_per_element_default=int(_get("bytes_per_element_default", 2)),
        extra=extra,
    )


__all__ = [
    "ArchSpecError",
    "DRAMSpec",
    "SRAMSpec",
    "DESpec",
    "PEArraySpec",
    "PIMNodeArchSpec",
    "kb",
    "mb",
    "gb",
    "dram_spec_from_cfg",
    "arch_from_mapping",
    "arch_from_legacy_hardware_gene",
]
