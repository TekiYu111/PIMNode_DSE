from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping


class ArchSpecError(ValueError):
    """Raised when a hardware spec is invalid."""


@dataclass(frozen=True)
class DRAMSpec:
    """Runtime DRAM truth mirrored from a standard PINOS cfg."""

    standard: str
    speed: str
    org: str
    channels: int
    ranks: int
    banks: int
    map: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.standard:
            raise ArchSpecError("dram standard is required")
        if not self.speed:
            raise ArchSpecError("dram speed is required")
        if not self.org:
            raise ArchSpecError("dram org is required")
        if self.channels <= 0:
            raise ArchSpecError("dram channels must be > 0")
        if self.ranks <= 0:
            raise ArchSpecError("dram ranks must be > 0")
        if self.banks <= 0:
            raise ArchSpecError("dram banks must be > 0")
        if not self.map:
            raise ArchSpecError("dram map is required")

    @property
    def bw_hint(self) -> float:
        """Coarse DRAM throughput hint for early ranking only."""
        return float(self.channels)


@dataclass(frozen=True)
class SRAMSpec:
    """On-chip SRAM boundary for placement and tiling."""

    cap: int
    bw: float
    concurrency: int = 1

    def __post_init__(self) -> None:
        if self.cap <= 0:
            raise ArchSpecError("sram cap must be > 0")
        if self.bw <= 0:
            raise ArchSpecError("sram bw must be > 0")
        if self.concurrency <= 0:
            raise ArchSpecError("sram concurrency must be > 0")


@dataclass(frozen=True)
class PESpec:
    """Black-box PE array for architecture-level modeling."""

    rows: int
    cols: int
    macs: int = 1
    concurrency: int = 1
    has_acc: bool = True

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ArchSpecError("pe rows must be > 0")
        if self.cols <= 0:
            raise ArchSpecError("pe cols must be > 0")
        if self.macs <= 0:
            raise ArchSpecError("pe macs must be > 0")
        if self.concurrency <= 0:
            raise ArchSpecError("pe concurrency must be > 0")

    @property
    def count(self) -> int:
        return self.rows * self.cols

    @property
    def mac_per_cycle(self) -> int:
        return self.count * self.macs


@dataclass(frozen=True)
class HardwareSpec:
    """Canonical single-node hardware spec.

    This object stores hardware truth only:
    - DRAM
    - SRAM
    - PE array

    Placement, tiling, trace generation, and performance modeling should
    consume this object, but their derived constraints should not be stored
    here.
    """

    dram: DRAMSpec
    sram: SRAMSpec
    pe: PESpec
    name: str = "pim-node"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def levels(self) -> tuple[str, ...]:
        return ("DRAM", "SRAM", "PE")

    @property
    def feed_bw(self) -> float:
        return self.sram.bw

    def can_move(self, src: str, dst: str) -> bool:
        edge = (src.upper(), dst.upper())
        if edge[0] == edge[1]:
            return True
        return edge in {
            ("DRAM", "SRAM"),
            ("SRAM", "DRAM"),
            ("SRAM", "PE"),
            ("PE", "SRAM"),
        }


def kb(x: int | float) -> int:
    return int(x * 1024)


def mb(x: int | float) -> int:
    return int(x * 1024 * 1024)


def gb(x: int | float) -> int:
    return int(x * 1024 * 1024 * 1024)


def _clean_cfg_value(value: str) -> str:
    text = value.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def _parse_cfg_text(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        for mark in ("//", "#"):
            if mark in line:
                line = line.split(mark, 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = _clean_cfg_value(value)
    return out


def load_dram(cfg_path: str | Path) -> DRAMSpec:
    cfg_file = Path(cfg_path)
    if not cfg_file.is_file():
        raise ArchSpecError(f"dram cfg not found: {cfg_file}")

    raw = _parse_cfg_text(cfg_file.read_text(encoding="utf-8"))
    need = [
        "standard",
        "dram_speed",
        "dram_org",
        "channels",
        "ranks",
        "banks",
        "mapping_policy",
    ]
    miss = [key for key in need if key not in raw]
    if miss:
        raise ArchSpecError(f"missing dram cfg keys: {', '.join(miss)}")

    keep = set(need) | {"unlimit_bandwidth", "no_DRAM_latency", "number_cores"}
    extra = {key: value for key, value in raw.items() if key not in keep}

    return DRAMSpec(
        standard=str(raw["standard"]),
        speed=str(raw["dram_speed"]),
        org=str(raw["dram_org"]),
        channels=int(raw["channels"]),
        ranks=int(raw["ranks"]),
        banks=int(raw["banks"]),
        map=str(raw["mapping_policy"]),
        extra=extra,
    )


def hw_from_dict(data: Mapping[str, Any]) -> HardwareSpec:
    """Build a HardwareSpec from a plain Python dict.

    Expected shape:
    {
        "name": "node-a",
        "dram_cfg": "path/to/dram.cfg",
        "sram": {"cap": 262144, "bw": 64, "concurrency": 2},
        "pe": {
            "rows": 16,
            "cols": 16,
            "macs": 1,
            "concurrency": 1,
            "has_acc": True,
        }
    }
    """

    dram_cfg = data.get("dram_cfg")
    if not dram_cfg:
        raise ArchSpecError("dram_cfg is required")

    sram_raw = dict(data.get("sram", {}))
    pe_raw = dict(data.get("pe", {}))

    return HardwareSpec(
        name=str(data.get("name", "pim-node")),
        dram=load_dram(str(dram_cfg)),
        sram=SRAMSpec(
            cap=int(sram_raw["cap"]),
            bw=float(sram_raw["bw"]),
            concurrency=int(sram_raw.get("concurrency", 1)),
        ),
        pe=PESpec(
            rows=int(pe_raw["rows"]),
            cols=int(pe_raw["cols"]),
            macs=int(pe_raw.get("macs", 1)),
            concurrency=int(pe_raw.get("concurrency", 1)),
            has_acc=bool(pe_raw.get("has_acc", True)),
        ),
        extra=dict(data.get("extra", {})),
    )


__all__ = [
    "ArchSpecError",
    "DRAMSpec",
    "SRAMSpec",
    "PESpec",
    "HardwareSpec",
    "kb",
    "mb",
    "gb",
    "load_dram",
    "hw_from_dict",
]
