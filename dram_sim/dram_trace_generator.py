from __future__ import annotations

"""Generate PINOS-compatible zsim traces from MappingTree.

Design goals for this first practical version:
- match current PINOS config: trace_format=zsim, split_trace=true
- emit a *single* trace file containing processor/core id per request
- support the current PIMNode_DSE_o1 tree/action structure
- focus on phased + burst-grouped traces
- start from the user's current modeling stage: single node == single local bank

This module does **not** try to reconstruct a cycle-accurate issue schedule.
Instead it emits a stable logical issue order (`cycle` column) that PINOS can
consume, while preserving action grouping and phase hints.
"""

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple
import csv

from pimnode_dse.mapping.mapping_tree import ActionNode, ActionType, MappingTree, Node, OpNode, ScopeNode, TileNode
from pimnode_dse.workload.workload import OpSpec, TensorSpec, WorkloadDAG


TraceStyle = Literal["single_file_zsim"]
AddressRadix = Literal["dec", "hex"]
MappingPolicy = Literal["ChRaBaRoCo", "RoBaRaCoCh"]


@dataclass(frozen=True)
class AddressTuple:
    channel: int
    rank: int
    bank: int
    row: int
    col: int


@dataclass
class TraceGenConfig:
    # PINOS trace interface
    trace_style: TraceStyle = "single_file_zsim"
    address_radix: AddressRadix = "dec"  # PINOS zsim parser expects decimal addr

    # current modeling stage: one active node / one local bank by default
    num_cores: int = 1
    default_core_id: int = 0
    local_bank_id: int = 0
    default_channel_id: int = 0
    default_rank_id: int = 0
    allow_remote_bank: bool = False

    # address mapping policy must match PINOS Config.h
    mapping_policy: MappingPolicy = "ChRaBaRoCo"

    # address geometry (must match current dram cfg / nDRAM organization)
    addr_bits_channel: int = 0
    addr_bits_rank: int = 0
    addr_bits_bank: int = 0
    addr_bits_row: int = 16
    addr_bits_col: int = 10

    # traffic granularity
    burst_bytes: int = 64
    addr_alignment: int = 64
    bytes_per_element_default: int = 2

    # region planning in row space so bank selection stays controlled
    tensor_region_row_gap: int = 4
    tensor_row_bases: Dict[str, int] = field(default_factory=dict)

    # issue order / cycle generation
    base_cycle_step: int = 1
    inter_action_cycle_gap: int = 1

    # emit helpful metadata
    emit_csv_debug: bool = True

    def cols_per_row(self) -> int:
        return 1 << max(int(self.addr_bits_col), 0)


@dataclass
class DRAMActionRecord:
    action_index: int
    action_type: str
    op_type: Literal["L", "S"]
    tensor_name: str
    phase: str
    node_name: str
    action_anchor: str
    core_id: int
    estimated_bytes: int
    burst_count: int


@dataclass
class DRAMRequest:
    thread_id: int
    processor_id: int
    instr_num: int
    issue_cycle: int
    op_type: Literal["L", "S"]
    tensor_name: str
    address: int
    size: int
    channel: int
    rank: int
    bank: int
    row: int
    col: int
    phase: str
    action_type: str
    action_anchor: str
    node_name: str
    action_index: int
    burst_index: int


class AddressEncoder:
    """Encode/decode address tuples using PINOS-supported mapping policies.

    Important note:
    This encoder is designed to be *policy-consistent* and debuggable. It does
    not try to mirror every internal translation detail inside PINOS. For the
    current single-node/local-bank stage, we use it to guarantee that requests
    stay in the intended bank and produce a stable linear address.
    """

    def __init__(self, cfg: TraceGenConfig):
        self.cfg = cfg
        self.ch_bits = int(cfg.addr_bits_channel)
        self.ra_bits = int(cfg.addr_bits_rank)
        self.ba_bits = int(cfg.addr_bits_bank)
        self.ro_bits = int(cfg.addr_bits_row)
        self.co_bits = int(cfg.addr_bits_col)

    def encode(self, at: AddressTuple) -> int:
        if self.cfg.mapping_policy == "ChRaBaRoCo":
            return self._encode_chra_baroco(at)
        if self.cfg.mapping_policy == "RoBaRaCoCh":
            return self._encode_roba_racoch(at)
        raise ValueError(f"Unsupported mapping_policy: {self.cfg.mapping_policy}")

    def decode(self, addr: int) -> AddressTuple:
        if self.cfg.mapping_policy == "ChRaBaRoCo":
            return self._decode_chra_baroco(addr)
        if self.cfg.mapping_policy == "RoBaRaCoCh":
            return self._decode_roba_racoch(addr)
        raise ValueError(f"Unsupported mapping_policy: {self.cfg.mapping_policy}")

    def _mask(self, bits: int) -> int:
        return (1 << bits) - 1 if bits > 0 else 0

    # Low-to-high packing order chosen to mirror the policy naming.
    # For single-node/local-bank use, the exact high-bit interpretation matters
    # less than internal consistency between encode/decode.
    def _encode_chra_baroco(self, at: AddressTuple) -> int:
        shift = 0
        addr = 0
        addr |= (at.col & self._mask(self.co_bits)) << shift
        shift += self.co_bits
        addr |= (at.row & self._mask(self.ro_bits)) << shift
        shift += self.ro_bits
        if self.ba_bits > 0:
            addr |= (at.bank & self._mask(self.ba_bits)) << shift
            shift += self.ba_bits
        if self.ra_bits > 0:
            addr |= (at.rank & self._mask(self.ra_bits)) << shift
            shift += self.ra_bits
        if self.ch_bits > 0:
            addr |= (at.channel & self._mask(self.ch_bits)) << shift
        return addr

    def _decode_chra_baroco(self, addr: int) -> AddressTuple:
        x = int(addr)
        col = x & self._mask(self.co_bits)
        x >>= self.co_bits
        row = x & self._mask(self.ro_bits)
        x >>= self.ro_bits
        bank = x & self._mask(self.ba_bits) if self.ba_bits > 0 else 0
        x >>= self.ba_bits
        rank = x & self._mask(self.ra_bits) if self.ra_bits > 0 else 0
        x >>= self.ra_bits
        channel = x & self._mask(self.ch_bits) if self.ch_bits > 0 else 0
        return AddressTuple(channel=channel, rank=rank, bank=bank, row=row, col=col)

    def _encode_roba_racoch(self, at: AddressTuple) -> int:
        shift = 0
        addr = 0
        if self.ch_bits > 0:
            addr |= (at.channel & self._mask(self.ch_bits)) << shift
            shift += self.ch_bits
        addr |= (at.col & self._mask(self.co_bits)) << shift
        shift += self.co_bits
        if self.ra_bits > 0:
            addr |= (at.rank & self._mask(self.ra_bits)) << shift
            shift += self.ra_bits
        if self.ba_bits > 0:
            addr |= (at.bank & self._mask(self.ba_bits)) << shift
            shift += self.ba_bits
        addr |= (at.row & self._mask(self.ro_bits)) << shift
        return addr

    def _decode_roba_racoch(self, addr: int) -> AddressTuple:
        x = int(addr)
        channel = x & self._mask(self.ch_bits) if self.ch_bits > 0 else 0
        x >>= self.ch_bits
        col = x & self._mask(self.co_bits)
        x >>= self.co_bits
        rank = x & self._mask(self.ra_bits) if self.ra_bits > 0 else 0
        x >>= self.ra_bits
        bank = x & self._mask(self.ba_bits) if self.ba_bits > 0 else 0
        x >>= self.ba_bits
        row = x & self._mask(self.ro_bits)
        return AddressTuple(channel=channel, rank=rank, bank=bank, row=row, col=col)


class DRAMTraceGenerator:
    def __init__(self, config: Optional[TraceGenConfig] = None) -> None:
        self.cfg = config or TraceGenConfig()
        self.encoder = AddressEncoder(self.cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def allocate_tensor_regions(self, workload: WorkloadDAG) -> None:
        """Assign each tensor a row-base region while keeping bank/channel fixed.

        This is more robust than assigning arbitrary raw base addresses because
        it preserves the intended bank selection under the chosen mapping policy.
        """
        if self.cfg.tensor_row_bases:
            return

        cur_row = 0
        cols_per_row = self.cfg.cols_per_row()
        burst_elems = max(1, self.cfg.burst_bytes)
        for tname, tensor in workload.tensors.items():
            total_bytes = self._tensor_total_bytes(tensor, workload)
            bursts = max(1, ceil(total_bytes / burst_elems))
            rows_needed = max(1, ceil(bursts / cols_per_row))
            self.cfg.tensor_row_bases[tname] = cur_row
            cur_row += rows_needed + int(self.cfg.tensor_region_row_gap)

    def collect_dram_actions(self, tree: MappingTree) -> List[ActionNode]:
        out: List[ActionNode] = []
        for node in tree.walk():
            if not isinstance(node, ActionNode):
                continue
            if self._is_dram_action(node):
                out.append(node)
        return out

    def generate_requests(
        self,
        tree: MappingTree,
        workload: WorkloadDAG,
        analyzer_summary: Optional[Any] = None,
    ) -> List[DRAMRequest]:
        del analyzer_summary  # reserved for future use
        self.allocate_tensor_regions(workload)

        requests: List[DRAMRequest] = []
        issue_cycle = 0
        instr_num = 0
        dram_actions = self.collect_dram_actions(tree)

        for action_index, action in enumerate(dram_actions):
            phase = self._infer_phase(action)
            core_id = self._infer_core_id(action)
            op_type = self._action_to_trace_op(action)
            node_name = self._nearest_tile_or_scope_name(action)
            anchor = getattr(action, "anchor", "") or ""

            for tensor_name in action.tensors:
                total_bytes = self._estimate_action_tensor_bytes(action, tensor_name, workload)
                burst_count = max(1, ceil(total_bytes / self.cfg.burst_bytes))

                for burst_idx in range(burst_count):
                    size = self.cfg.burst_bytes
                    last = burst_idx == burst_count - 1
                    if last:
                        remaining = total_bytes - burst_idx * self.cfg.burst_bytes
                        size = max(1, remaining)
                    addr_tuple = self._make_address_tuple(tensor_name, burst_idx)
                    addr = self.encoder.encode(addr_tuple)
                    req = DRAMRequest(
                        thread_id=core_id,
                        processor_id=core_id,
                        instr_num=instr_num,
                        issue_cycle=issue_cycle,
                        op_type=op_type,
                        tensor_name=tensor_name,
                        address=addr,
                        size=size,
                        channel=addr_tuple.channel,
                        rank=addr_tuple.rank,
                        bank=addr_tuple.bank,
                        row=addr_tuple.row,
                        col=addr_tuple.col,
                        phase=phase,
                        action_type=action.action_type,
                        action_anchor=anchor,
                        node_name=node_name,
                        action_index=action_index,
                        burst_index=burst_idx,
                    )
                    requests.append(req)
                    instr_num += 1
                    issue_cycle += self.cfg.base_cycle_step

                issue_cycle += self.cfg.inter_action_cycle_gap

        return requests

    def write_trace(self, requests: Sequence[DRAMRequest], trace_path: str | Path) -> Path:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for req in requests:
                addr_str = str(req.address) if self.cfg.address_radix == "dec" else hex(req.address)
                # PINOS zsim parser effectively consumes:
                # thread_id processor_id cycle type addr size
                f.write(
                    f"{req.thread_id} {req.processor_id} {req.issue_cycle} "
                    f"{req.op_type} {addr_str} {req.size}\n"
                )
        return path

    def export_requests_csv(self, requests: Sequence[DRAMRequest], csv_path: str | Path) -> Path:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "thread_id", "processor_id", "instr_num", "issue_cycle", "op_type",
                "tensor_name", "address", "size", "channel", "rank", "bank", "row", "col",
                "phase", "action_type", "action_anchor", "node_name", "action_index", "burst_index",
            ])
            for r in requests:
                writer.writerow([
                    r.thread_id, r.processor_id, r.instr_num, r.issue_cycle, r.op_type,
                    r.tensor_name, r.address, r.size, r.channel, r.rank, r.bank, r.row, r.col,
                    r.phase, r.action_type, r.action_anchor, r.node_name, r.action_index, r.burst_index,
                ])
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _is_dram_action(self, action: ActionNode) -> bool:
        src = (action.src_level or "").upper()
        dst = (action.dst_level or "").upper()
        kind = action.kind
        if kind in {ActionType.LOAD, ActionType.PREFETCH}:
            return src == "DRAM"
        if kind in {ActionType.STORE, ActionType.WRITEBACK, ActionType.WRITEBACK_DRAM, ActionType.EVICT}:
            return dst == "DRAM"
        return False

    def _action_to_trace_op(self, action: ActionNode) -> Literal["L", "S"]:
        kind = action.kind
        if kind in {ActionType.LOAD, ActionType.PREFETCH}:
            return "L"
        if kind in {ActionType.STORE, ActionType.WRITEBACK, ActionType.WRITEBACK_DRAM, ActionType.EVICT}:
            return "S"
        raise ValueError(f"Unsupported DRAM action kind for trace emission: {action.kind}")

    def _infer_phase(self, node: Node) -> str:
        cur: Optional[Node] = node
        while cur is not None:
            phase = getattr(cur, "phase", None)
            if phase:
                return str(phase)
            cur = cur.parent
        kind = getattr(node, "action_type", "")
        if kind in {"LOAD", "PREFETCH"}:
            return "load"
        if kind in {"STORE", "WRITEBACK", "WRITEBACK_DRAM", "EVICT"}:
            return "writeback"
        return "compute_support"

    def _infer_core_id(self, node: Node) -> int:
        # current project stage: single active node/core unless explicitly annotated
        attrs = getattr(node, "attrs", {}) or {}
        for key in ("processor_id", "core_id", "node_id"):
            if key in attrs:
                return int(attrs[key])
        return int(self.cfg.default_core_id)

    def _nearest_tile_or_scope_name(self, node: Node) -> str:
        cur: Optional[Node] = node.parent
        while cur is not None:
            if isinstance(cur, (TileNode, ScopeNode)):
                return cur.name
            cur = cur.parent
        return node.name

    def _nearest_ancestor(self, node: Node, cls: type) -> Optional[Any]:
        cur = node.parent
        while cur is not None:
            if isinstance(cur, cls):
                return cur
            cur = cur.parent
        return None

    def _nearest_descendant(self, node: Node, cls: type) -> Optional[Any]:
        for child in getattr(node, "children", []):
            if isinstance(child, cls):
                return child
            found = self._nearest_descendant(child, cls)
            if found is not None:
                return found
        return None

    def _estimate_action_tensor_bytes(self, action: ActionNode, tensor_name: str, workload: WorkloadDAG) -> int:
        tile = self._nearest_ancestor(action, TileNode)
        op_node = self._nearest_descendant(action, OpNode) or self._nearest_ancestor(action, OpNode)
        op_spec = workload.get_op(op_node.op_id) if op_node is not None else None

        if tensor_name not in workload.tensors:
            elems = 1
            if tile is not None and getattr(tile, "tile_size", None):
                elems = 1
                for v in tile.tile_size.values():
                    elems *= max(1, int(v))
            return elems * int(self.cfg.bytes_per_element_default)

        tensor = workload.tensors[tensor_name]
        elem_bytes = self._tensor_bytes_per_element(tensor, workload)

        if tile is None or not getattr(tile, "tile_size", None):
            return self._tensor_total_bytes(tensor, workload)

        tile_map = dict(tile.tile_size or {})
        if op_spec is not None and tensor_name in op_spec.tensor_index:
            syms = op_spec.tensor_index[tensor_name]
            elems = 1
            for sym, full_dim in zip(syms, tensor.shape):
                extent = tile_map.get(sym, full_dim)
                elems *= max(1, min(int(extent), int(full_dim)))
            return int(elems * elem_bytes)

        elems = 1
        extents = list(tile_map.values())
        for i, full_dim in enumerate(tensor.shape):
            extent = extents[i] if i < len(extents) else full_dim
            elems *= max(1, min(int(extent), int(full_dim)))
        return int(elems * elem_bytes)

    def _tensor_total_bytes(self, tensor: TensorSpec, workload: WorkloadDAG) -> int:
        total_elems = 1
        for d in tensor.shape:
            total_elems *= int(d)
        return total_elems * self._tensor_bytes_per_element(tensor, workload)

    def _tensor_bytes_per_element(self, tensor: TensorSpec, workload: WorkloadDAG) -> int:
        dtype = (tensor.dtype or "").lower()
        table = {
            "fp16": 2,
            "float16": 2,
            "bf16": 2,
            "float32": 4,
            "fp32": 4,
            "int8": 1,
            "uint8": 1,
            "int16": 2,
            "int32": 4,
        }
        if dtype in table:
            return table[dtype]
        if workload.spec is not None:
            return int(workload.spec.dtype_bytes)
        return int(self.cfg.bytes_per_element_default)

    def _make_address_tuple(self, tensor_name: str, burst_idx: int) -> AddressTuple:
        row_base = int(self.cfg.tensor_row_bases.get(tensor_name, 0))
        cols_per_row = self.cfg.cols_per_row()
        row_delta = burst_idx // cols_per_row
        col = burst_idx % cols_per_row
        bank = int(self.cfg.local_bank_id)
        if self.cfg.allow_remote_bank:
            bank = int(self.cfg.local_bank_id)
        return AddressTuple(
            channel=int(self.cfg.default_channel_id),
            rank=int(self.cfg.default_rank_id),
            bank=bank,
            row=row_base + row_delta,
            col=col,
        )


__all__ = [
    "TraceGenConfig",
    "AddressTuple",
    "DRAMActionRecord",
    "DRAMRequest",
    "AddressEncoder",
    "DRAMTraceGenerator",
]
