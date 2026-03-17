from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional


@dataclass
class HardwareGene:
    """Legacy hardware gene container aligned with current arch_spec.py.

    Notes
    -----
    - Runtime DRAM truth now comes from ``dram_cfg_path`` in ``arch_spec.py``.
    - This class still carries DRAM-search related knobs so upper layers can
      generate / override a dram.cfg when needed.
    - ``dram_bandwidth`` and ``dram_bank_count`` are kept as compatibility
      accessors for older call sites.
    - All hardware knobs must be passed explicitly when constructing an
      instance; no implicit defaults are provided here.
    """

    # DRAM search / cfg-related knobs
    dram_speed: str
    dram_org: str
    channels: int
    ranks: int
    banks: int
    mapping_policy: str

    # On-chip hierarchy knobs used by arch_from_legacy_hardware_gene()
    sram_capacity_kb: int
    sram_bandwidth: float
    sram_bank_count: int
    pe_array_m: int
    pe_array_n: int
    pe_mac_num: int

    # Optional knobs consumed by current arch_spec.py legacy adapter
    de_bus_bandwidth: Optional[float] = None
    pe_input_bandwidth: Optional[float] = None
    pe_output_bandwidth: Optional[float] = None
    supports_pipeline: bool = True
    max_inflight_tiles: int = 1
    supports_direct_dram_to_pe: bool = False
    supports_direct_pe_to_sram_writeback: bool = True
    supports_accumulate_in_place: bool = True
    clock_hz: Optional[float] = None
    bytes_per_element_default: int = 2
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def get_total_compute_power(self) -> float:
        """Return total PE-array throughput in MACs/cycle."""
        return self.pe_array_m * self.pe_array_n * self.pe_mac_num

    def get_compute_to_memory_ratio(self) -> float:
        """A coarse compute-to-memory ratio for ranking / heuristics."""
        dram_bw_equiv = self.dram_bandwidth
        if dram_bw_equiv <= 0:
            raise ValueError("dram bandwidth must be > 0")
        return self.get_total_compute_power() / dram_bw_equiv

    def to_dram_cfg_overrides(self) -> Dict[str, Any]:
        """Return fields that can be directly written into dram.cfg."""
        return {
            "dram_speed": self.dram_speed,
            "dram_org": self.dram_org,
            "channels": int(self.channels),
            "ranks": int(self.ranks),
            "banks": int(self.banks),
            "mapping_policy": self.mapping_policy,
        }

    def copy(self) -> "HardwareGene":
        """Return a full dataclass copy without accidentally dropping fields."""
        return replace(self)

    @property
    def dram_bank_count(self) -> int:
        """Backward-compatible alias for older code."""
        return int(self.banks)

    @dram_bank_count.setter
    def dram_bank_count(self, value: int) -> None:
        self.banks = int(value)

    @property
    def dram_bandwidth(self) -> float:
        """Compatibility placeholder for older heuristics.

        Current arch_spec.py no longer consumes this field directly; DRAM runtime
        truth is parsed from dram.cfg. We expose a stable numeric proxy so old
        ranking code can still run.
        """
        return float(self.channels)
