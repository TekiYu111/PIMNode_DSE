from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import csv

from pimnode_dse.hardware.arch_spec import HardwareSpec
from pimnode_dse.hardware.dram_cfg import PinosRunSpec, write_pinos_cfg
from pimnode_dse.mapping.tree.mapping_tree import MappingTree, Move, ScopeNode, TileNode, OpNode
from pimnode_dse.mapping.workload.workload import WorkloadDAG

# Late import of TraceArtifact avoids the circular import:
#   dram_trace_generator → pimnode_dse.dse.types → pimnode_dse.dse.__init__
#   → driver.py → dram_trace_generator
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pimnode_dse.dse.types import TraceArtifact


# ---------------------------------------------------------------------------
# PINOS zsim trace format (reverse-engineered from base.trace):
#
#   field 1: thread_id    (always 0, single-core PIM)
#   field 2: processor_id (always 0)
#   field 3: issue_cycle  (0 for L/S/B; compute cycles for I)
#   field 4: op_type      L | S | B | I
#   field 5: address      byte address (0 for B and I)
#   field 6: size         bytes for L/S; 0 for B and I
#
# Address layout (reverse-engineered from base.trace):
#   [0x00000000, 0x40000000)  load region 0  – Q, O and other non-KV inputs
#   [0x40000000, 0x80000000)  load region 1  – K, V, cache tensors
#   [0x80000000, 0xC0000000)  store region 0 – primary outputs (O)
#   [0xC0000000, ...)         store region 1 – cache/state outputs
#
# Per-tile access pattern (verified against base.trace):
#   Load tile:   L addr size → B 0 0 → I compute_cycles 0 0 → B 0 0
#   Store tile:  S addr size → B 0 0
#
# Key design decisions:
#   1. compute_cycles for I comes from the pre-computed op_cycles dict
#      (OpCost.compute_cycles keyed by op_id), NOT re-computed here.
#   2. Move size comes from mv.bytes (tile granularity) when mv.bytes > 0;
#      falls back to full tensor size only when mv.bytes == 0.
#   3. repeat_hint on a Move drives replication of the tile access sequence,
#      corresponding to the tiling iteration count for that scope level.
# ---------------------------------------------------------------------------

DEFAULT_BURST_BYTES = 64

_LOAD_BASE_0  = 0x00000000
_LOAD_BASE_1  = 0x40000000
_STORE_BASE_0 = 0x80000000
_STORE_BASE_1 = 0xC0000000


def generate_trace(
    tree: MappingTree,
    workload: WorkloadDAG,
    hw: HardwareSpec,
    out_dir: Path,
    *,
    op_cycles: Optional[Dict[str, float]] = None,
    emit_debug_csv: bool = True,
    burst_bytes: int = DEFAULT_BURST_BYTES,
) -> "TraceArtifact":
    """Generate a PINOS zsim trace file from a mapping tree.

    Parameters
    ----------
    tree        : The mapping tree for one candidate.
    workload    : WorkloadDAG (for tensor byte sizes when mv.bytes == 0).
    hw          : HardwareSpec (unused here but kept for API symmetry).
    out_dir     : Directory where trace.0 and optional trace_debug.csv are written.
    op_cycles   : Dict[op_id → compute_cycles] from TreeEstimate.op_costs.
                  When provided, I instructions use these pre-computed values.
                  When None, compute cycles default to 1 (no idle time modelled).
    emit_debug_csv : Write a human-readable trace_debug.csv alongside trace.0.
    burst_bytes : DRAM burst granularity in bytes (default 64).
    """
    from pimnode_dse.dse.types import TraceArtifact

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if op_cycles is None:
        op_cycles = {}

    # Build a scope_id → compute_cycles map from the tree structure.
    # For each ScopeNode, sum the compute_cycles of all OpNodes it directly contains.
    scope_compute = _build_scope_compute_map(tree, op_cycles)

    annotated = _collect_annotated_moves(tree, workload, scope_compute)
    rows = _expand_to_trace_rows(annotated, workload, burst_bytes=burst_bytes)

    trace_path = out_dir / "trace.0"
    with trace_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                f"{r['thread_id']} {r['processor_id']} {r['issue_cycle']} "
                f"{r['op_type']} {r['address']} {r['size']}\n"
            )

    debug_csv_path = None
    if emit_debug_csv:
        debug_csv_path = out_dir / "trace_debug.csv"
        with debug_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "thread_id", "processor_id", "issue_cycle", "op_type",
                "address", "size", "tensor", "scope_id", "act", "src", "dst", "repeat",
            ])
            writer.writeheader()
            writer.writerows(rows)

    request_count = sum(1 for r in rows if r["op_type"] in {"L", "S"})
    return TraceArtifact(
        trace_path=trace_path,
        debug_csv_path=debug_csv_path,
        request_count=request_count,
    )


def generate_trace_and_cfg(
    tree: MappingTree,
    workload: WorkloadDAG,
    hw: HardwareSpec,
    out_dir: Path,
    *,
    op_cycles: Optional[Dict[str, float]] = None,
    core_num: int = 1,
    emit_debug_csv: bool = True,
) -> "tuple[TraceArtifact, Path]":
    """Generate trace + PINOS cfg for one candidate.

    op_cycles should come from TreeEstimate.op_costs:
        op_cycles = {oc.op_id: oc.compute_cycles for oc in tree_est.op_costs}
    """
    artifact = generate_trace(
        tree, workload, hw, out_dir,
        op_cycles=op_cycles,
        emit_debug_csv=emit_debug_csv,
    )
    cfg_path = write_pinos_cfg(
        out=Path(out_dir) / "dram.cfg",
        hw=hw,
        run=PinosRunSpec(trace=str(artifact.trace_path), core_num=int(core_num)),
    )
    return artifact, cfg_path


# ---------------------------------------------------------------------------
# Internal: scope compute map
# ---------------------------------------------------------------------------

def _build_scope_compute_map(
    tree: MappingTree,
    op_cycles: Dict[str, float],
) -> Dict[str, float]:
    """Map scope_id → total compute cycles for OpNodes directly in that scope.

    'Directly in' means the OpNode's nearest ScopeNode ancestor is this scope.
    We do NOT include compute from nested scopes so that each load-barrier-compute
    sequence in the trace reflects only the work done within that memory scope level.

    Algorithm: walk the tree; for each OpNode, find its nearest ScopeNode parent
    (skipping TileNode ancestors) and accumulate op_cycles[op_id] there.
    """
    result: Dict[str, float] = {}

    for node in tree.walk():
        if not isinstance(node, OpNode):
            continue
        cyc = op_cycles.get(node.id, 1.0)

        # Walk up to find nearest ScopeNode
        cur = getattr(node, "parent", None)
        while cur is not None:
            if isinstance(cur, ScopeNode):
                result[cur.id] = result.get(cur.id, 0.0) + cyc
                break
            cur = getattr(cur, "parent", None)

    return result


# ---------------------------------------------------------------------------
# Internal: move collection and expansion
# ---------------------------------------------------------------------------

def _collect_annotated_moves(
    tree: MappingTree,
    workload: WorkloadDAG,
    scope_compute: Dict[str, float],
) -> List[tuple[str, Move, float, int]]:
    """Collect (scope_id, Move, compute_cycles, move_bytes) for every Move.

    move_bytes uses mv.bytes (tile granularity) when > 0; falls back to the
    full tensor size only when mv.bytes == 0 (unset by the mapping builder).

    compute_cycles is the PE compute time for the scope that issues this load.
    Stores get 0 compute cycles (no I instruction after a store).
    """
    out: List[tuple[str, Move, float, int]] = []
    for node in tree.walk():
        if not isinstance(node, ScopeNode):
            continue
        scope_cyc = scope_compute.get(node.id, 0.0)

        for mv in node.entry:
            mb = mv.bytes if mv.bytes > 0 else _tensor_size_bytes(workload, mv.tens)
            out.append((node.id, mv, scope_cyc, mb))

        for mv in node.exit:
            mb = mv.bytes if mv.bytes > 0 else _tensor_size_bytes(workload, mv.tens)
            out.append((node.id, mv, 0.0, mb))   # stores carry no compute

    return out


def _tensor_size_bytes(workload: WorkloadDAG, name: str) -> int:
    try:
        return int(workload.tensor(name).size_bytes())
    except Exception:
        return DEFAULT_BURST_BYTES


# ---------------------------------------------------------------------------
# Address allocator
# ---------------------------------------------------------------------------

class _AddrAllocator:
    """Per-tensor monotone address allocator with region separation.

    Load and store regions are in disjoint DRAM address spaces so that
    address-mapping policies (ChRaBaRoCo vs RoBaRaCoCh) are exercised
    independently for reads and writes — matching base.trace conventions.

    Each tensor gets a dedicated sub-region within its class so that
    bank-access statistics reflect per-tensor access patterns.
    """

    # 256 MB per tensor — large enough for any workload, avoids inter-tensor aliasing
    _TENSOR_REGION_BYTES = 256 * 1024 * 1024

    def __init__(self, burst: int) -> None:
        self._burst = burst
        self._ptrs = {
            "load0":  _LOAD_BASE_0,
            "load1":  _LOAD_BASE_1,
            "store0": _STORE_BASE_0,
            "store1": _STORE_BASE_1,
        }
        self._tensor_load_base:    Dict[str, int] = {}
        self._tensor_store_base:   Dict[str, int] = {}
        self._tensor_load_offset:  Dict[str, int] = {}
        self._tensor_store_offset: Dict[str, int] = {}

    def _load_region(self, tensor: str) -> str:
        upper = tensor.strip().upper()
        if any(k in upper for k in ("K", "V", "CACHE", "CTX")):
            return "load1"
        return "load0"

    def _store_region(self, tensor: str) -> str:
        upper = tensor.strip().upper()
        if any(k in upper for k in ("CACHE", "CTX", "STATE")):
            return "store1"
        return "store0"

    def _ensure_tensor_load(self, tensor: str) -> None:
        if tensor not in self._tensor_load_base:
            region = self._load_region(tensor)
            self._tensor_load_base[tensor] = self._ptrs[region]
            self._tensor_load_offset[tensor] = 0
            self._ptrs[region] += self._TENSOR_REGION_BYTES

    def _ensure_tensor_store(self, tensor: str) -> None:
        if tensor not in self._tensor_store_base:
            region = self._store_region(tensor)
            self._tensor_store_base[tensor] = self._ptrs[region]
            self._tensor_store_offset[tensor] = 0
            self._ptrs[region] += self._TENSOR_REGION_BYTES

    def next_load_addr(self, tensor: str, size: int) -> int:
        self._ensure_tensor_load(tensor)
        addr = self._tensor_load_base[tensor] + self._tensor_load_offset[tensor]
        self._tensor_load_offset[tensor] += size
        return addr

    def next_store_addr(self, tensor: str, size: int) -> int:
        self._ensure_tensor_store(tensor)
        addr = self._tensor_store_base[tensor] + self._tensor_store_offset[tensor]
        self._tensor_store_offset[tensor] += size
        return addr


# ---------------------------------------------------------------------------
# Trace row expansion
# ---------------------------------------------------------------------------

_BARRIER = {
    "thread_id": 0, "processor_id": 0, "issue_cycle": 0,
    "op_type": "B", "address": 0, "size": 0,
    "tensor": "", "scope_id": "", "act": "", "src": "", "dst": "", "repeat": 1,
}


def _expand_to_trace_rows(
    annotated: List[tuple[str, Move, float, int]],
    workload: WorkloadDAG,
    *,
    burst_bytes: int,
) -> List[dict]:
    """Expand annotated Moves into flat trace rows.

    Load pattern per burst:   L → B → I(compute_cycles) → B
    Store pattern per burst:  S → B

    Compute cycles for I:
      - scope_cyc is the total PE compute time for the scope.
      - Distributed evenly across all bursts in the tile load:
          cyc_per_burst = ceil(scope_cyc / num_bursts_in_tile)
      - This matches the base.trace convention where I.issue_cycle = 16 or 64
        (16 cycles for a 4096-byte load at 256 bytes/cycle, 64 for partial sums).

    repeat_hint drives replication: a Move with repeat_hint=N means the tile
    is accessed N times (one per tiling iteration at that scope level), so
    the full L/B/I/B sequence is emitted N times.
    """
    rows: List[dict] = []
    alloc = _AddrAllocator(burst_bytes)

    for scope_id, mv, scope_cyc, move_bytes in annotated:
        is_load = mv.act in {"load", "prefetch"}
        op_type = "L" if is_load else "S"
        repeat  = max(1, int(mv.repeat_hint))

        bursts = max(1, (move_bytes + burst_bytes - 1) // burst_bytes)

        # Compute cycles distributed evenly across bursts.
        # Use ceiling so that even a single-burst load has at least 1 compute cycle
        # (avoids IPC=inf edge case in the simulator).
        cyc_per_burst = (
            max(1, -(-int(scope_cyc) // bursts))   # ceiling division
            if (is_load and scope_cyc > 0)
            else 0
        )

        for _ in range(repeat):
            for b in range(bursts):
                req_size = (
                    burst_bytes if b < bursts - 1
                    else max(1, move_bytes - (bursts - 1) * burst_bytes)
                )

                addr = (
                    alloc.next_load_addr(mv.tens, req_size)
                    if is_load
                    else alloc.next_store_addr(mv.tens, req_size)
                )

                rows.append({
                    "thread_id": 0, "processor_id": 0, "issue_cycle": 0,
                    "op_type": op_type, "address": addr, "size": req_size,
                    "tensor": mv.tens, "scope_id": scope_id,
                    "act": mv.act, "src": mv.src, "dst": mv.dst, "repeat": repeat,
                })
                rows.append(dict(_BARRIER))

                if is_load and cyc_per_burst > 0:
                    rows.append({
                        "thread_id": 0, "processor_id": 0,
                        "issue_cycle": cyc_per_burst, "op_type": "I",
                        "address": 0, "size": 0,
                        "tensor": mv.tens, "scope_id": scope_id,
                        "act": "compute", "src": "", "dst": "", "repeat": repeat,
                    })
                    rows.append(dict(_BARRIER))

    return rows


__all__ = [
    "generate_trace",
    "generate_trace_and_cfg",
]
