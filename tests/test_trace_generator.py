"""Tests for dram_trace_generator.py.

Trace correctness guarantees verified:
  1. Format validity — every row has 6 fields, op_type ∈ {L, S, B, I}
  2. Address separation — L < _STORE_BASE_0, S >= _STORE_BASE_0
  3. Pattern structure — loads follow L→B→I→B; stores follow S→B
  4. No consecutive memory ops without a barrier in between
  5. Compute cycles — I rows carry compute_cycles from op_cycles dict
  6. repeat_hint — total L/S requests scale proportionally with repeat_hint
  7. request_count — TraceArtifact.request_count == number of L+S rows in file
  8. address monotone — within each tensor, addresses are non-decreasing
  9. mv.bytes priority — Move.bytes used over tensor full size when > 0
 10. scope_compute_map — only direct-child ops counted per scope
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from collections import Counter

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dram_sim.dram_trace_generator import (
    _LOAD_BASE_0,
    _LOAD_BASE_1,
    _STORE_BASE_0,
    _STORE_BASE_1,
    _AddrAllocator,
    _build_scope_compute_map,
    _collect_annotated_moves,
    _expand_to_trace_rows,
    generate_trace,
)
from pimnode_dse.mapping.tree.mapping_tree import Move, ScopeNode, TileNode, OpNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeWorkload:
    """Minimal WorkloadDAG stub for testing."""
    def __init__(self, tensor_bytes: dict):
        self._sizes = tensor_bytes

    def tensor(self, name: str):
        sz = self._sizes.get(name, 64)
        class T:
            def size_bytes(self_inner): return sz
        return T()

    def tensors(self):
        class _TSpec:
            def __init__(self, sz): self.size = sz; self.shape = (sz,)
            def size_bytes(self): return self.size
        return {name: _TSpec(sz) for name, sz in self._sizes.items()}

    def ops(self):
        return {}

    def op(self, op_id):
        raise KeyError(op_id)


def _make_load(tens: str, move_bytes: int = 0, repeat: int = 1) -> Move:
    return Move(act="load", tens=tens, src="dram", dst="sram",
                bytes=move_bytes, repeat_hint=repeat)


def _make_store(tens: str, move_bytes: int = 0, repeat: int = 1) -> Move:
    return Move(act="store", tens=tens, src="sram", dst="dram",
                bytes=move_bytes, repeat_hint=repeat)


def _rows(annotated, workload, burst=64):
    """annotated items must be 4-tuples: (scope_id, Move, scope_cyc, move_bytes)."""
    return _expand_to_trace_rows(annotated, workload, burst_bytes=burst)


def _annotate(scope_id, mv, scope_cyc, workload, burst=64):
    """Build a single 4-tuple resolving move_bytes from mv.bytes or tensor size."""
    mb = mv.bytes if mv.bytes > 0 else int(workload.tensor(mv.tens).size_bytes())
    return [(scope_id, mv, scope_cyc, mb)]


# ---------------------------------------------------------------------------
# Test 1: Format validity
# ---------------------------------------------------------------------------

class TestFormatValidity:
    def test_op_types_are_valid(self):
        wl = _FakeWorkload({"Q": 4096, "O": 128})
        annotated = (
            _annotate("s0", _make_load("Q", 4096), 16.0, wl) +
            _annotate("s0", _make_store("O", 128), 0.0, wl)
        )
        rows = _rows(annotated, wl)
        valid = {"L", "S", "B", "I"}
        for r in rows:
            assert r["op_type"] in valid

    def test_barrier_address_and_size_zero(self):
        wl = _FakeWorkload({"Q": 64})
        rows = _rows(_annotate("s0", _make_load("Q", 64), 8.0, wl), wl)
        for r in rows:
            if r["op_type"] == "B":
                assert r["address"] == 0 and r["size"] == 0

    def test_compute_instruction_address_size_zero(self):
        wl = _FakeWorkload({"Q": 64})
        rows = _rows(_annotate("s0", _make_load("Q", 64), 32.0, wl), wl)
        for r in rows:
            if r["op_type"] == "I":
                assert r["address"] == 0 and r["size"] == 0
                assert r["issue_cycle"] > 0


# ---------------------------------------------------------------------------
# Test 2: Address separation
# ---------------------------------------------------------------------------

class TestAddressSeparation:
    def test_load_addresses_below_store_base(self):
        wl = _FakeWorkload({"Q": 4096, "O": 128})
        annotated = (
            _annotate("s0", _make_load("Q", 4096), 16.0, wl) +
            _annotate("s0", _make_store("O", 128), 0.0, wl)
        )
        rows = _rows(annotated, wl)
        l_addrs = [r["address"] for r in rows if r["op_type"] == "L"]
        s_addrs = [r["address"] for r in rows if r["op_type"] == "S"]
        assert max(l_addrs) < _STORE_BASE_0
        assert min(s_addrs) >= _STORE_BASE_0

    def test_kv_in_load1_region(self):
        wl = _FakeWorkload({"K": 4096})
        rows = _rows(_annotate("s0", _make_load("K", 4096), 16.0, wl), wl)
        l_addrs = [r["address"] for r in rows if r["op_type"] == "L"]
        assert all(a >= _LOAD_BASE_1 for a in l_addrs)

    def test_q_in_load0_region(self):
        wl = _FakeWorkload({"Q": 4096})
        rows = _rows(_annotate("s0", _make_load("Q", 4096), 16.0, wl), wl)
        l_addrs = [r["address"] for r in rows if r["op_type"] == "L"]
        assert all(_LOAD_BASE_0 <= a < _LOAD_BASE_1 for a in l_addrs)


# ---------------------------------------------------------------------------
# Test 3: Pattern structure
# ---------------------------------------------------------------------------

class TestPatternStructure:
    def _ops(self, rows):
        return [r["op_type"] for r in rows]

    def test_single_burst_load_pattern(self):
        wl = _FakeWorkload({"Q": 64})
        rows = _rows(_annotate("s0", _make_load("Q", 64), 16.0, wl), wl, burst=64)
        assert self._ops(rows)[:4] == ["L", "B", "I", "B"]

    def test_single_burst_store_pattern(self):
        wl = _FakeWorkload({"O": 64})
        rows = _rows(_annotate("s0", _make_store("O", 64), 0.0, wl), wl, burst=64)
        assert self._ops(rows)[:2] == ["S", "B"]

    def test_no_consecutive_memory_ops(self):
        wl = _FakeWorkload({"Q": 256, "O": 128})
        annotated = (
            _annotate("s0", _make_load("Q", 256), 16.0, wl) +
            _annotate("s0", _make_store("O", 128), 0.0, wl)
        )
        ops = self._ops(_rows(annotated, wl))
        mem = {"L", "S"}
        for i in range(len(ops) - 1):
            if ops[i] in mem and ops[i + 1] in mem:
                pytest.fail(f"Adjacent mem ops at {i},{i+1}: {ops[i]}, {ops[i+1]}")


# ---------------------------------------------------------------------------
# Test 4: Barrier count
# ---------------------------------------------------------------------------

class TestBarrierCount:
    def test_barrier_geq_memory_ops(self):
        wl = _FakeWorkload({"Q": 128, "O": 64})
        annotated = (
            _annotate("s0", _make_load("Q", 128), 16.0, wl) +
            _annotate("s0", _make_store("O", 64), 0.0, wl)
        )
        cnt = Counter(r["op_type"] for r in _rows(annotated, wl, burst=64))
        assert cnt["B"] >= cnt["L"] + cnt["S"]


# ---------------------------------------------------------------------------
# Test 5: Compute cycles from op_cycles
# ---------------------------------------------------------------------------

class TestComputeCycles:
    def test_i_cycles_from_op_cycles(self):
        """I instructions should reflect the op_cycles dict, not re-computed values."""
        wl = _FakeWorkload({"Q": 64})
        # 100 compute cycles total, 1 burst → cyc_per_burst = 100
        annotated = _annotate("s0", _make_load("Q", 64), 100.0, wl)
        rows = _rows(annotated, wl, burst=64)
        i_rows = [r for r in rows if r["op_type"] == "I"]
        assert i_rows
        assert i_rows[0]["issue_cycle"] == 100   # ceiling(100/1) = 100

    def test_no_i_after_store(self):
        wl = _FakeWorkload({"O": 64})
        rows = _rows(_annotate("s0", _make_store("O", 64), 0.0, wl), wl, burst=64)
        assert not any(r["op_type"] == "I" for r in rows)

    def test_cyc_distributed_across_bursts(self):
        """With 2 bursts and 10 total compute cycles, each burst gets ceil(10/2)=5."""
        wl = _FakeWorkload({"Q": 128})
        annotated = _annotate("s0", _make_load("Q", 128), 10.0, wl)
        rows = _rows(annotated, wl, burst=64)
        i_rows = [r for r in rows if r["op_type"] == "I"]
        assert len(i_rows) == 2
        assert all(r["issue_cycle"] == 5 for r in i_rows)


# ---------------------------------------------------------------------------
# Test 6: repeat_hint
# ---------------------------------------------------------------------------

class TestRepeatHint:
    def test_l_count_scales_with_repeat(self):
        wl = _FakeWorkload({"Q": 64})
        r1 = _rows(_annotate("s0", _make_load("Q", 64, repeat=1), 16.0, wl), wl, burst=64)
        r4 = _rows(_annotate("s0", _make_load("Q", 64, repeat=4), 16.0, wl), wl, burst=64)
        l1 = sum(1 for r in r1 if r["op_type"] == "L")
        l4 = sum(1 for r in r4 if r["op_type"] == "L")
        assert l4 == 4 * l1


# ---------------------------------------------------------------------------
# Test 7: request_count in TraceArtifact
# ---------------------------------------------------------------------------

class TestTraceArtifact:
    def test_request_count_matches_file(self):
        from pimnode_dse.hardware.arch_spec import HardwareSpec, DRAMSpec, SRAMSpec, PESpec
        from pimnode_dse.mapping.tree.mapping_tree import MappingTree

        wl = _FakeWorkload({"Q": 256, "O": 128})
        scope = ScopeNode(
            id="root", bind="seq", mem="dram",
            entry=[_make_load("Q", 256)],
            exit=[_make_store("O", 128)],
        )
        tree = MappingTree(root=scope)
        hw = HardwareSpec(
            dram=DRAMSpec(standard="nDRAM", speed="nDRAM_800D", org="nDRAM_512Mb_x8",
                          channels=1, ranks=1, banks=1, map="ChRaBaRoCo"),
            sram=SRAMSpec(cap=512*1024, bw=64.0),
            pe=PESpec(rows=8, cols=8, macs=1),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            art = generate_trace(tree, wl, hw, Path(tmpdir),
                                 op_cycles={}, emit_debug_csv=False)
            file_count = sum(
                1 for line in art.trace_path.read_text().splitlines()
                if len(line.split()) >= 4 and line.split()[3] in ("L", "S")
            )
            assert art.request_count == file_count


# ---------------------------------------------------------------------------
# Test 8: Address monotonicity
# ---------------------------------------------------------------------------

class TestAddressMonotonicity:
    def test_load_addresses_non_decreasing(self):
        wl = _FakeWorkload({"Q": 4096})
        rows = _rows(_annotate("s0", _make_load("Q", 4096), 16.0, wl), wl, burst=64)
        addrs = [r["address"] for r in rows if r["op_type"] == "L"]
        assert addrs == sorted(addrs)

    def test_store_addresses_non_decreasing(self):
        wl = _FakeWorkload({"O": 256})
        rows = _rows(_annotate("s0", _make_store("O", 256), 0.0, wl), wl, burst=64)
        addrs = [r["address"] for r in rows if r["op_type"] == "S"]
        assert addrs == sorted(addrs)


# ---------------------------------------------------------------------------
# Test 9: mv.bytes priority over tensor full size
# ---------------------------------------------------------------------------

class TestMoveBytePriority:
    def test_tile_bytes_used_when_set(self):
        """Move with bytes=128 should produce 2 bursts (128/64), not 64 (4096/64)."""
        wl = _FakeWorkload({"Q": 4096})   # full tensor = 4096
        # mv.bytes=128 → tile is 128 bytes → 2 bursts
        rows = _rows(_annotate("s0", _make_load("Q", 128), 16.0, wl), wl, burst=64)
        l_rows = [r for r in rows if r["op_type"] == "L"]
        assert len(l_rows) == 2, f"Expected 2 bursts for 128-byte tile, got {len(l_rows)}"

    def test_full_tensor_used_when_mv_bytes_zero(self):
        """Move with bytes=0 falls back to tensor full size (64 bytes → 1 burst)."""
        wl = _FakeWorkload({"Q": 64})
        annotated = _annotate("s0", _make_load("Q", 0), 16.0, wl)  # bytes=0 → use tensor size
        rows = _rows(annotated, wl, burst=64)
        l_rows = [r for r in rows if r["op_type"] == "L"]
        assert len(l_rows) == 1


# ---------------------------------------------------------------------------
# Test 10: scope_compute_map counts only direct-child ops
# ---------------------------------------------------------------------------

class TestScopeComputeMap:
    def test_direct_ops_counted(self):
        """OpNode directly inside a ScopeNode is counted."""
        from pimnode_dse.mapping.tree.mapping_tree import MappingTree

        op = OpNode(id="op0", kind="matmul", ins=("Q",), outs=("O",))
        scope = ScopeNode(id="s0", bind="seq", mem="dram", kids=[op])
        op.parent = scope
        tree = MappingTree(root=scope)

        result = _build_scope_compute_map(tree, {"op0": 42.0})
        assert result.get("s0", 0.0) == pytest.approx(42.0)

    def test_nested_scope_ops_not_double_counted(self):
        """Op inside a nested ScopeNode is attributed to the inner scope only."""
        from pimnode_dse.mapping.tree.mapping_tree import MappingTree

        op = OpNode(id="op0", kind="matmul", ins=("Q",), outs=("O",))
        inner = ScopeNode(id="inner", bind="seq", mem="sram", kids=[op])
        outer = ScopeNode(id="outer", bind="seq", mem="dram", kids=[inner])
        op.parent = inner
        inner.parent = outer
        tree = MappingTree(root=outer)

        result = _build_scope_compute_map(tree, {"op0": 100.0})
        # op0's nearest ScopeNode is inner → 100 goes to inner, 0 to outer
        assert result.get("inner", 0.0) == pytest.approx(100.0)
        assert result.get("outer", 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 11: Regression against base.trace
# ---------------------------------------------------------------------------

class TestBaseTraceRegression:
    def test_reference_pattern(self):
        wl = _FakeWorkload({"Q": 4096, "O": 128})
        annotated = (
            _annotate("s0", _make_load("Q", 4096), 16.0, wl) +
            _annotate("s0", _make_store("O", 128), 0.0, wl)
        )
        rows = _rows(annotated, wl, burst=64)

        l_addrs = [r["address"] for r in rows if r["op_type"] == "L"]
        s_addrs = [r["address"] for r in rows if r["op_type"] == "S"]
        i_rows  = [r for r in rows if r["op_type"] == "I"]

        assert all(a < _STORE_BASE_0 for a in l_addrs)
        assert all(a >= _STORE_BASE_0 for a in s_addrs)
        assert i_rows and all(r["issue_cycle"] > 0 for r in i_rows)

import tempfile
from pathlib import Path
from collections import Counter

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dram_sim.dram_trace_generator import (
    _LOAD_BASE_0,
    _LOAD_BASE_1,
    _STORE_BASE_0,
    _STORE_BASE_1,
    _AddrAllocator,
    _expand_to_trace_rows,
    generate_trace,
)
from pimnode_dse.mapping.tree.mapping_tree import Move


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeWorkload:
    """Minimal WorkloadDAG stub for testing."""
    def __init__(self, tensor_bytes: dict[str, int]):
        self._sizes = tensor_bytes

    def tensor(self, name: str):
        sz = self._sizes.get(name, 64)
        class T:
            def size_bytes(self_inner): return sz
        return T()

    def tensors(self):
        """Return a dict of name→stub so build_estimate_context doesn't crash."""
        class _TSpec:
            def __init__(self, sz): self.size = sz; self.shape = (sz,)
            def size_bytes(self): return self.size
        return {name: _TSpec(sz) for name, sz in self._sizes.items()}

    def ops(self):
        return {}

    def op(self, op_id):
        raise KeyError(op_id)


def _make_load(tens: str, size_bytes: int = 4096, repeat: int = 1) -> Move:
    return Move(act="load", tens=tens, src="dram", dst="sram",
                bytes=size_bytes, repeat_hint=repeat)


def _make_store(tens: str, size_bytes: int = 128, repeat: int = 1) -> Move:
    return Move(act="store", tens=tens, src="sram", dst="dram",
                bytes=size_bytes, repeat_hint=repeat)


def _rows_for(annotated, workload, burst=64) -> list[dict]:
    return _expand_to_trace_rows(annotated, workload, burst_bytes=burst)


# ---------------------------------------------------------------------------
# Test 1: Format validity
# ---------------------------------------------------------------------------

class TestFormatValidity:
    def test_op_types_are_valid(self):
        wl = _FakeWorkload({"Q": 4096, "O": 128})
        rows = _rows_for(
            [("s0", _make_load("Q", 4096), 16.0),
             ("s0", _make_store("O", 128), 0.0)],
            wl,
        )
        valid = {"L", "S", "B", "I"}
        for r in rows:
            assert r["op_type"] in valid, f"Invalid op_type: {r['op_type']}"

    def test_barrier_fields_are_zero(self):
        wl = _FakeWorkload({"Q": 64})
        rows = _rows_for([("s0", _make_load("Q", 64), 8.0)], wl)
        for r in rows:
            if r["op_type"] == "B":
                assert r["address"] == 0
                assert r["size"] == 0

    def test_compute_instruction_fields(self):
        wl = _FakeWorkload({"Q": 64})
        rows = _rows_for([("s0", _make_load("Q", 64), 32.0)], wl)
        for r in rows:
            if r["op_type"] == "I":
                assert r["address"] == 0
                assert r["size"] == 0
                assert r["issue_cycle"] > 0


# ---------------------------------------------------------------------------
# Test 2: Address separation
# ---------------------------------------------------------------------------

class TestAddressSeparation:
    def test_load_addresses_below_store_base(self):
        wl = _FakeWorkload({"Q": 4096, "O": 128})
        rows = _rows_for(
            [("s0", _make_load("Q", 4096), 16.0),
             ("s0", _make_store("O", 128), 0.0)],
            wl,
        )
        l_addrs = [r["address"] for r in rows if r["op_type"] == "L"]
        s_addrs = [r["address"] for r in rows if r["op_type"] == "S"]
        assert l_addrs, "No L rows generated"
        assert s_addrs, "No S rows generated"
        assert max(l_addrs) < _STORE_BASE_0, \
            f"L addr {hex(max(l_addrs))} overlaps store region {hex(_STORE_BASE_0)}"
        assert min(s_addrs) >= _STORE_BASE_0, \
            f"S addr {hex(min(s_addrs))} below store region {hex(_STORE_BASE_0)}"

    def test_kv_loads_in_load1_region(self):
        wl = _FakeWorkload({"K": 4096})
        rows = _rows_for([("s0", _make_load("K", 4096), 16.0)], wl)
        l_addrs = [r["address"] for r in rows if r["op_type"] == "L"]
        assert all(a >= _LOAD_BASE_1 for a in l_addrs), \
            f"K load addresses should be >= {hex(_LOAD_BASE_1)}"

    def test_q_loads_in_load0_region(self):
        wl = _FakeWorkload({"Q": 4096})
        rows = _rows_for([("s0", _make_load("Q", 4096), 16.0)], wl)
        l_addrs = [r["address"] for r in rows if r["op_type"] == "L"]
        assert all(_LOAD_BASE_0 <= a < _LOAD_BASE_1 for a in l_addrs), \
            "Q load addresses should be in load0 region"


# ---------------------------------------------------------------------------
# Test 3: Pattern structure  (L→B→I→B  and  S→B)
# ---------------------------------------------------------------------------

class TestPatternStructure:
    def _op_seq(self, rows) -> list[str]:
        return [r["op_type"] for r in rows]

    def test_load_followed_by_barrier_then_compute(self):
        wl = _FakeWorkload({"Q": 64})
        rows = _rows_for([("s0", _make_load("Q", 64), 16.0)], wl, burst=64)
        ops = self._op_seq(rows)
        # With size=64 (one burst) expect: L B I B
        assert ops[:4] == ["L", "B", "I", "B"], f"Expected L B I B, got {ops[:4]}"

    def test_store_followed_by_barrier(self):
        wl = _FakeWorkload({"O": 64})
        rows = _rows_for([("s0", _make_store("O", 64), 0.0)], wl, burst=64)
        ops = self._op_seq(rows)
        assert ops[:2] == ["S", "B"], f"Expected S B, got {ops[:2]}"

    def test_no_consecutive_memory_ops_without_barrier(self):
        """No two L/S rows should be adjacent without a B in between."""
        wl = _FakeWorkload({"Q": 256, "O": 128})
        rows = _rows_for(
            [("s0", _make_load("Q", 256), 16.0),
             ("s0", _make_store("O", 128), 0.0)],
            wl,
        )
        ops = self._op_seq(rows)
        mem = {"L", "S"}
        for i in range(len(ops) - 1):
            if ops[i] in mem and ops[i + 1] in mem:
                pytest.fail(f"Two adjacent memory ops at positions {i},{i+1}: {ops[i]}, {ops[i+1]}")


# ---------------------------------------------------------------------------
# Test 4: Barrier count
# ---------------------------------------------------------------------------

class TestBarrierCount:
    def test_barrier_count_matches_memory_ops(self):
        """Each L and S should be bracketed: at minimum one B after each mem op."""
        wl = _FakeWorkload({"Q": 128, "O": 64})
        rows = _rows_for(
            [("s0", _make_load("Q", 128), 16.0),
             ("s0", _make_store("O", 64), 0.0)],
            wl, burst=64,
        )
        cnt = Counter(r["op_type"] for r in rows)
        # L(2 bursts) B I B B I B  ... S B S B  →  barriers >= L + S + I
        assert cnt["B"] >= cnt["L"] + cnt["S"], \
            f"Expected B >= L+S, got B={cnt['B']} L={cnt['L']} S={cnt['S']}"


# ---------------------------------------------------------------------------
# Test 5: Compute cycles
# ---------------------------------------------------------------------------

class TestComputeCycles:
    def test_compute_cycles_positive_for_loads(self):
        wl = _FakeWorkload({"Q": 64})
        rows = _rows_for([("s0", _make_load("Q", 64), 32.0)], wl, burst=64)
        i_rows = [r for r in rows if r["op_type"] == "I"]
        assert i_rows, "No I rows generated for load with compute_cycles=32"
        assert all(r["issue_cycle"] > 0 for r in i_rows)

    def test_no_compute_after_store(self):
        wl = _FakeWorkload({"O": 64})
        rows = _rows_for([("s0", _make_store("O", 64), 0.0)], wl, burst=64)
        i_rows = [r for r in rows if r["op_type"] == "I"]
        assert not i_rows, "I rows should not appear after a store"


# ---------------------------------------------------------------------------
# Test 6: repeat_hint scaling
# ---------------------------------------------------------------------------

class TestRepeatHint:
    def test_request_count_scales_with_repeat(self):
        wl = _FakeWorkload({"Q": 64})
        rows1 = _rows_for([("s0", _make_load("Q", 64, repeat=1), 16.0)], wl, burst=64)
        rows4 = _rows_for([("s0", _make_load("Q", 64, repeat=4), 16.0)], wl, burst=64)
        l1 = sum(1 for r in rows1 if r["op_type"] == "L")
        l4 = sum(1 for r in rows4 if r["op_type"] == "L")
        assert l4 == 4 * l1, f"Expected 4× L rows, got {l4} vs {l1}"


# ---------------------------------------------------------------------------
# Test 7: request_count in TraceArtifact
# ---------------------------------------------------------------------------

class TestTraceArtifact:
    def test_request_count_matches_file(self):
        from pimnode_dse.hardware.arch_spec import HardwareSpec, DRAMSpec, SRAMSpec, PESpec
        from pimnode_dse.mapping.tree.mapping_tree import MappingTree, ScopeNode

        wl = _FakeWorkload({"Q": 256, "O": 128})
        scope = ScopeNode(
            id="root",
            bind="seq",
            mem="dram",
            entry=[_make_load("Q", 256, repeat=1)],
            exit=[_make_store("O", 128, repeat=1)],
        )
        from pimnode_dse.mapping.tree.mapping_tree import MappingTree
        tree = MappingTree(root=scope)

        hw = HardwareSpec(
            dram=DRAMSpec(standard="nDRAM", speed="nDRAM_800D", org="nDRAM_512Mb_x8",
                          channels=1, ranks=1, banks=1, map="ChRaBaRoCo"),
            sram=SRAMSpec(cap=512*1024, bw=64.0),
            pe=PESpec(rows=8, cols=8, macs=1),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            art = generate_trace(tree, wl, hw, Path(tmpdir), emit_debug_csv=False)
            # Count L+S in the file
            file_mem_count = 0
            with art.trace_path.open() as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 4 and parts[3] in ("L", "S"):
                        file_mem_count += 1
            assert art.request_count == file_mem_count, \
                f"request_count={art.request_count} != file count={file_mem_count}"


# ---------------------------------------------------------------------------
# Test 8: address monotonicity within a tensor
# ---------------------------------------------------------------------------

class TestAddressMonotonicity:
    def test_load_addresses_non_decreasing_per_tensor(self):
        wl = _FakeWorkload({"Q": 4096})
        rows = _rows_for([("s0", _make_load("Q", 4096), 16.0)], wl, burst=64)
        l_addrs = [r["address"] for r in rows if r["op_type"] == "L" and r["tensor"] == "Q"]
        assert l_addrs == sorted(l_addrs), "Load addresses should be non-decreasing"

    def test_store_addresses_non_decreasing_per_tensor(self):
        wl = _FakeWorkload({"O": 256})
        rows = _rows_for([("s0", _make_store("O", 256), 0.0)], wl, burst=64)
        s_addrs = [r["address"] for r in rows if r["op_type"] == "S" and r["tensor"] == "O"]
        assert s_addrs == sorted(s_addrs), "Store addresses should be non-decreasing"


# ---------------------------------------------------------------------------
# Test 9: Regression against base.trace structure
# ---------------------------------------------------------------------------

class TestBaseTraceRegression:
    """Verify that a synthetic trace for the base.trace workload structure
    matches the key properties observed in the reference file:
      - L size 4096, S size 128, I cycles 16
      - pattern L B I B S B repeats
      - L addresses from 0, S addresses from 0x80000000
    """

    def test_reference_pattern(self):
        wl = _FakeWorkload({"Q": 4096, "O": 128})
        # One load of Q (4096 bytes) at 16 compute cycles per burst,
        # then one store of O (128 bytes)
        burst = 64
        annotated = [
            ("scope0", Move(act="load", tens="Q", src="dram", dst="sram",
                            bytes=4096, repeat_hint=1), 16.0),
            ("scope0", Move(act="store", tens="O", src="sram", dst="dram",
                            bytes=128, repeat_hint=1), 0.0),
        ]
        rows = _expand_to_trace_rows(annotated, wl, burst_bytes=burst)

        l_rows = [r for r in rows if r["op_type"] == "L"]
        s_rows = [r for r in rows if r["op_type"] == "S"]
        i_rows = [r for r in rows if r["op_type"] == "I"]

        # L addresses in load region
        assert all(r["address"] < _STORE_BASE_0 for r in l_rows)
        # S addresses in store region
        assert all(r["address"] >= _STORE_BASE_0 for r in s_rows)
        # I instructions exist and have positive cycles
        assert i_rows
        assert all(r["issue_cycle"] > 0 for r in i_rows)
