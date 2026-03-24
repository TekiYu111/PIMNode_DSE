from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Sequence

from .arch_spec import DRAMSpec, HardwareSpec, PESpec, SRAMSpec


class HWSpaceError(ValueError):
    """Raised when a hardware design space is invalid."""


@dataclass(frozen=True)
class DRAMSpace:
    standard: Sequence[str]
    speed: Sequence[str]
    org: Sequence[str]
    map: Sequence[str]
    channels: Sequence[int] = (1,)
    ranks: Sequence[int] = (1,)
    banks: Sequence[int] = (1,)

    def __post_init__(self) -> None:
        if not self.standard:
            raise HWSpaceError("dram standard space is empty")
        if not self.speed:
            raise HWSpaceError("dram speed space is empty")
        if not self.org:
            raise HWSpaceError("dram org space is empty")
        if not self.map:
            raise HWSpaceError("dram map space is empty")
        if not self.channels:
            raise HWSpaceError("dram channels space is empty")
        if not self.ranks:
            raise HWSpaceError("dram ranks space is empty")
        if not self.banks:
            raise HWSpaceError("dram banks space is empty")

    def size(self) -> int:
        return (
            len(self.standard)
            * len(self.speed)
            * len(self.org)
            * len(self.map)
            * len(self.channels)
            * len(self.ranks)
            * len(self.banks)
        )

    def iter_specs(self) -> Iterator[DRAMSpec]:
        for standard, speed, org, map_name, channels, ranks, banks in product(
            self.standard,
            self.speed,
            self.org,
            self.map,
            self.channels,
            self.ranks,
            self.banks,
        ):
            yield DRAMSpec(
                standard=standard,
                speed=speed,
                org=org,
                map=map_name,
                channels=int(channels),
                ranks=int(ranks),
                banks=int(banks),
            )


@dataclass(frozen=True)
class SRAMSpace:
    cap: Sequence[int]
    bw: Sequence[float]
    concurrency: Sequence[int]

    def __post_init__(self) -> None:
        if not self.cap:
            raise HWSpaceError("sram cap space is empty")
        if not self.bw:
            raise HWSpaceError("sram bw space is empty")
        if not self.concurrency:
            raise HWSpaceError("sram concurrency space is empty")

    def size(self) -> int:
        return len(self.cap) * len(self.bw) * len(self.concurrency)

    def iter_specs(self) -> Iterator[SRAMSpec]:
        for cap, bw, concurrency in product(self.cap, self.bw, self.concurrency):
            yield SRAMSpec(cap=int(cap), bw=float(bw), concurrency=int(concurrency))


@dataclass(frozen=True)
class PESpace:
    rows: Sequence[int]
    cols: Sequence[int]
    macs: Sequence[int] = (1,)
    concurrency: Sequence[int] = (1,)
    has_acc: Sequence[bool] = (True,)

    def __post_init__(self) -> None:
        if not self.rows:
            raise HWSpaceError("pe rows space is empty")
        if not self.cols:
            raise HWSpaceError("pe cols space is empty")
        if not self.macs:
            raise HWSpaceError("pe macs space is empty")
        if not self.concurrency:
            raise HWSpaceError("pe concurrency space is empty")
        if not self.has_acc:
            raise HWSpaceError("pe has_acc space is empty")

    def size(self) -> int:
        return (
            len(self.rows)
            * len(self.cols)
            * len(self.macs)
            * len(self.concurrency)
            * len(self.has_acc)
        )

    def iter_specs(self) -> Iterator[PESpec]:
        for rows, cols, macs, concurrency, has_acc in product(
            self.rows,
            self.cols,
            self.macs,
            self.concurrency,
            self.has_acc,
        ):
            yield PESpec(
                rows=int(rows),
                cols=int(cols),
                macs=int(macs),
                concurrency=int(concurrency),
                has_acc=bool(has_acc),
            )


@dataclass(frozen=True)
class HWSpace:
    dram: DRAMSpace
    sram: SRAMSpace
    pe: PESpace
    name: str = "pim-node"

    def size(self) -> int:
        return self.dram.size() * self.sram.size() * self.pe.size()

    def iter_specs(self) -> Iterator[HardwareSpec]:
        for dram, sram, pe in product(
            self.dram.iter_specs(),
            self.sram.iter_specs(),
            self.pe.iter_specs(),
        ):
            yield HardwareSpec(dram=dram, sram=sram, pe=pe, name=self.name)


def build_space(
    *,
    dram_standard: Sequence[str],
    dram_speed: Sequence[str],
    dram_org: Sequence[str],
    dram_map: Sequence[str],
    sram_cap: Sequence[int],
    sram_bw: Sequence[float],
    sram_concurrency: Sequence[int],
    pe_rows: Sequence[int],
    pe_cols: Sequence[int],
    pe_macs: Sequence[int] = (1,),
    pe_concurrency: Sequence[int] = (1,),
    pe_has_acc: Sequence[bool] = (True,),
    channels: Sequence[int] = (1,),
    ranks: Sequence[int] = (1,),
    banks: Sequence[int] = (1,),
    name: str = "pim-node",
) -> HWSpace:
    return HWSpace(
        name=name,
        dram=DRAMSpace(
            standard=dram_standard,
            speed=dram_speed,
            org=dram_org,
            map=dram_map,
            channels=channels,
            ranks=ranks,
            banks=banks,
        ),
        sram=SRAMSpace(
            cap=sram_cap,
            bw=sram_bw,
            concurrency=sram_concurrency,
        ),
        pe=PESpace(
            rows=pe_rows,
            cols=pe_cols,
            macs=pe_macs,
            concurrency=pe_concurrency,
            has_acc=pe_has_acc,
        ),
    )


DEFAULT_SPACE = build_space(
    name="pim-node",
    dram_standard=("nDRAM",),
    dram_speed=("nDRAM_800D", "nDRAM_1600H", "nDRAM_1600K"),
    dram_org=("nDRAM_512Mb_x4", "nDRAM_512Mb_x8", "nDRAM_512Mb_x16"),
    dram_map=("ChRaBaRoCo", "RoBaRaCoCh"),
    channels=(1,),
    ranks=(1,),
    banks=(1,),
    sram_cap=(256 * 1024, 512 * 1024),
    sram_bw=(64.0, 128.0),
    sram_concurrency=(1, 2),
    pe_rows=(8, 16),
    pe_cols=(8, 16),
    pe_macs=(1, 2),
    pe_concurrency=(1, 2),
    pe_has_acc=(True, False),
)


__all__ = [
    "HWSpaceError",
    "DRAMSpace",
    "SRAMSpace",
    "PESpace",
    "HWSpace",
    "build_space",
    "DEFAULT_SPACE",
]
