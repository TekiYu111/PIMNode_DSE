from __future__ import annotations

import bisect
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from .types import ObjectiveMetrics

T = TypeVar("T")

# Relative tolerance for float dominance comparisons.
# Two values within _REL_TOL of each other are treated as equal to avoid
# spurious dominance from floating-point rounding.
_REL_TOL: float = 1e-9


# --------------------------------------------------
# ObjectiveMetrics helpers
# --------------------------------------------------

def _obj_to_vec(m: ObjectiveMetrics) -> Tuple[float, ...]:
    """Convert ObjectiveMetrics to a float vector for dominance checks.

    When power_mw is not None (i.e. DRAMPower simulation has run), the
    vector is 4-D (latency, dram_cost, movement, power_mw).  Otherwise
    it is 3-D.  All downstream functions use this vector, so they
    automatically adapt without any caller changes.
    """
    if m.power_mw is not None:
        return (m.latency, m.dram_cost, m.movement, m.power_mw)
    return (m.latency, m.dram_cost, m.movement)


def dominates(a: ObjectiveMetrics, b: ObjectiveMetrics) -> bool:
    """True iff *a* strictly Pareto-dominates *b* (all minimise).

    Automatically uses 3-D or 4-D depending on whether power_mw is set.
    Mixed comparisons (one has power_mw, other does not) fall back to 3-D
    to avoid comparing incompatible vectors.
    """
    av = _obj_to_vec(a)
    bv = _obj_to_vec(b)
    # Mixed-dimension guard: compare only the shared prefix (3-D).
    if len(av) != len(bv):
        av = av[:3]
        bv = bv[:3]
    return _vec_dominates(av, bv)


def _vec_dominates(av: Tuple[float, ...], bv: Tuple[float, ...]) -> bool:
    """Float-safe dominance: a ≼ b on all dims, a < b on at least one."""
    leq_all = True
    lt_one = False
    for a, b in zip(av, bv):
        tol = _REL_TOL * max(abs(a), abs(b), 1.0)
        if a > b + tol:
            return False
        if a < b - tol:
            lt_one = True
    return lt_one


def rank_key(metric: ObjectiveMetrics) -> tuple:
    return _obj_to_vec(metric)


# --------------------------------------------------
# 3-D Pareto front — O(n log n)  Sort-and-Sweep
#
# Reference: Kung, Luccio, Preparata (1975)
# "On Finding the Maxima of a Set of Vectors"
# Adapted for minimisation (all objectives minimised).
#
# Algorithm:
#   1. Sort by d0 ascending (ties broken by d1, d2).
#   2. Sweep; maintain a 2-D staircase on (d1, d2).
#      Staircase invariant: d1 strictly increasing, d2 strictly decreasing.
#      A point is dominated iff ∃ entry with e1 ≤ d1 AND e2 ≤ d2.
#   3. _Staircase uses two parallel lists + bisect → O(log r) lookup,
#      O(r) amortised in-place insert/prune.
# --------------------------------------------------

class _Staircase:
    """2-D non-dominated staircase (d1 strictly inc, d2 strictly dec).

    Parallel lists allow bisect on _d1 directly; mutations are in-place.
    """
    __slots__ = ("_d1", "_d2")

    def __init__(self) -> None:
        self._d1: List[float] = []
        self._d2: List[float] = []

    def dominates_point(self, d1: float, d2: float) -> bool:
        """True iff ∃ entry with e1 ≤ d1 AND e2 ≤ d2 (with _REL_TOL)."""
        k = bisect.bisect_right(self._d1, d1 + _REL_TOL * max(abs(d1), 1.0)) - 1
        if k < 0:
            return False
        tol2 = _REL_TOL * max(abs(d2), abs(self._d2[k]), 1.0)
        return self._d2[k] <= d2 + tol2

    def insert(self, d1: float, d2: float) -> None:
        """Insert (d1, d2); prune dominated entries to the right."""
        p = bisect.bisect_right(self._d1, d1)
        end = p
        tol2 = _REL_TOL * max(abs(d2), 1.0)
        while end < len(self._d2) and self._d2[end] >= d2 - tol2:
            end += 1
        self._d1[p:end] = [d1]
        self._d2[p:end] = [d2]


def _process_equal_d0_batch(
    sc: _Staircase,
    batch: List[Tuple[int, float, float]],   # (orig_idx, d1, d2)
    out_indices: List[int],
) -> None:
    """Handle a batch of equal-d0 points; update sc and out_indices."""
    survivors: List[Tuple[int, float, float]] = [
        t for t in batch if not sc.dominates_point(t[1], t[2])
    ]
    if not survivors:
        return

    survivors.sort(key=lambda x: (x[1], x[2]))
    inner = _Staircase()
    for orig_i, d1, d2 in survivors:
        if not inner.dominates_point(d1, d2):
            out_indices.append(orig_i)
            inner.insert(d1, d2)
            if not sc.dominates_point(d1, d2):
                sc.insert(d1, d2)


def _pareto_front_3d(
    pts: List[Tuple[float, float, float]],
    indices: List[int],
) -> List[int]:
    """Return indices of 3-D Pareto front (minimise all) in O(n log n)."""
    if not pts:
        return []
    n = len(pts)
    order = sorted(range(n), key=lambda i: (pts[i][0], pts[i][1], pts[i][2]))
    sc = _Staircase()
    front_indices: List[int] = []
    i = 0
    while i < n:
        d0 = pts[order[i]][0]
        batch: List[Tuple[int, float, float]] = []
        j = i
        tol0 = _REL_TOL * max(abs(d0), 1.0)
        while j < n and pts[order[j]][0] <= d0 + tol0:
            idx = order[j]
            batch.append((indices[idx], pts[idx][1], pts[idx][2]))
            j += 1
        _process_equal_d0_batch(sc, batch, front_indices)
        i = j
    return front_indices


def pareto_front_fast(
    items: Sequence[tuple[object, ObjectiveMetrics]],
) -> list[tuple[object, ObjectiveMetrics]]:
    """Pareto front for ObjectiveMetrics (minimise all), 3-D or 4-D.

    When all items have power_mw set, operates in 4-D using the generic
    O(n²) filter (_strict_pareto_filter_generic).  When power_mw is None
    for all items, uses the O(n log n) 3-D sort-and-sweep.  Mixed sets
    (some with power_mw, some without) use 3-D only.
    """
    if not items:
        return []
    # Determine effective dimensionality from the first item.
    sample_vec = _obj_to_vec(items[0][1])
    all_same_dim = all(len(_obj_to_vec(m)) == len(sample_vec) for _, m in items)
    ndim = len(sample_vec) if all_same_dim else 3

    if ndim == 3:
        pts = [(m.latency, m.dram_cost, m.movement) for _, m in items]
        idx_list = list(range(len(items)))
        front_idx = set(_pareto_front_3d(pts, idx_list))
        return [items[i] for i in range(len(items)) if i in front_idx]
    else:
        # 4-D: use generic O(n²) filter via _strict_pareto_filter_generic.
        def _key(item: tuple) -> Tuple[float, ...]:
            return _obj_to_vec(item[1])
        return _strict_pareto_filter_generic(list(items), _key)


def pareto_front(
    items: Sequence[tuple[object, ObjectiveMetrics]],
) -> list[tuple[object, ObjectiveMetrics]]:
    """Pareto front — auto-selects 3-D (O(n log n)) or 4-D (O(n²)) algorithm."""
    return pareto_front_fast(items)


# --------------------------------------------------
# Non-dominated sort — adaptive algorithm selection
#
# For n ≤ _NSORT_THRESHOLD  : NSGA-II dominance-count  O(n²)
#   — one pass builds all dom_count / dom_set arrays; shell extraction
#     is O(n) BFS.  Constant factor is very small (3 float comparisons
#     per pair, integer increments, no sort).
#
# For n > _NSORT_THRESHOLD  : repeated O(n log n) Pareto-front peeling
#   — each call to pareto_front_fast is O(n log n); we peel k shells,
#     total O(k n log n).  Beats O(n²) when n is large.
#
# Threshold determined empirically: crossover ≈ n=300 for pure Python.
# --------------------------------------------------

_NSORT_THRESHOLD: int = 300


def nondominated_sort_3d(
    items: Sequence[tuple[object, ObjectiveMetrics]],
) -> List[List[tuple[object, ObjectiveMetrics]]]:
    """Partition *items* into Pareto shells with adaptive algorithm.

    Name kept for backwards compatibility; actually supports N-D:
    - 3-D with power_mw=None: NSGA-II count sort (O(n²)) for n ≤ 300,
      repeated front-peeling (O(k n log n)) for n > 300.
    - 4-D with power_mw set: uses generic _strict_pareto_filter_generic
      via repeated peeling (always O(n²) per shell).

    shells[0] = strict Pareto front, shells[1] = next rank, …
    """
    n = len(items)
    if n == 0:
        return []
    if n <= _NSORT_THRESHOLD:
        return _nondominated_sort_count(items)
    return _nondominated_sort_peel(items)


def _nondominated_sort_count(
    items: Sequence[tuple[object, ObjectiveMetrics]],
) -> List[List[tuple[object, ObjectiveMetrics]]]:
    """NSGA-II dominance-count sort — O(n²), best for small n.  Works N-D."""
    n = len(items)
    objs: List[Tuple[float, ...]] = [_obj_to_vec(m) for _, m in items]
    dom_count: List[int] = [0] * n
    dom_set: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        ai = objs[i]
        for j in range(i + 1, n):
            aj = objs[j]
            # Mixed-dim guard: compare 3-D prefix if dimensions differ.
            av, bv = (ai, aj) if len(ai) == len(aj) else (ai[:3], aj[:3])
            if _vec_dominates(av, bv):
                dom_set[i].append(j)
                dom_count[j] += 1
            elif _vec_dominates(bv, av):
                dom_set[j].append(i)
                dom_count[i] += 1

    shells: List[List[tuple[object, ObjectiveMetrics]]] = []
    current: List[int] = [i for i in range(n) if dom_count[i] == 0]
    while current:
        shells.append([items[i] for i in current])
        nxt: List[int] = []
        for i in current:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return shells


def _nondominated_sort_peel(
    items: Sequence[tuple[object, ObjectiveMetrics]],
) -> List[List[tuple[object, ObjectiveMetrics]]]:
    """Repeated Pareto-front peeling — O(k n log n), best for large n."""
    remaining = list(items)
    shells: List[List[tuple[object, ObjectiveMetrics]]] = []
    while remaining:
        front = pareto_front_fast(remaining)
        shells.append(front)
        front_set = {id(x) for x in front}
        remaining = [x for x in remaining if id(x) not in front_set]
    return shells


# --------------------------------------------------
# Hypervolume contribution — O(n log n) for 3-D Pareto front ranking
#
# Reference: Beume, Naujoks, Emmerich (2007)
# "SMS-EMOA: Multiobjective selection based on dominated hypervolume"
# The 3-D hypervolume contribution of point p w.r.t. reference point r
# is the hypervolume of the region dominated by p but not by any other
# point in the front.
#
# We use a sweep-line algorithm over the sorted front:
#   Sort by d0; maintain a 1-D "active" structure on d1 values.
#   For each successive d0 slice, compute the hypervolume slab contributed
#   by points in that slice before the next d0 value.
#
# Used in shortlist_candidates to rank within a Pareto shell by
# hypervolume contribution (largest contribution = most "useful" point).
# --------------------------------------------------

def hypervolume_3d(
    pts: List[Tuple[float, float, float]],
    ref: Tuple[float, float, float],
) -> float:
    """Compute exact 3-D hypervolume dominated by *pts* below reference *ref*.

    Algorithm: sort by d0 ascending; sweep d0 slabs; for each slab maintain
    a 2-D staircase on (d1, d2) and compute the 2-D hypervolume of the
    staircase × the slab thickness in d0.  O(n log n) total.

    All objectives are minimised; ref must be ≥ all points on all dims.
    Points that do not dominate ref on any axis are ignored.
    """
    # Filter to points that are inside the reference box
    valid = [(p0, p1, p2) for p0, p1, p2 in pts
             if p0 < ref[0] and p1 < ref[1] and p2 < ref[2]]
    if not valid:
        return 0.0

    # Sort by d0 ascending
    valid.sort(key=lambda x: x[0])

    # We sweep d0 from min to ref[0].
    # At each distinct d0 value we have a set of points that "appear".
    # Between d0_prev and d0_cur the staircase stays the same; we add
    # the 2-D hypervolume of the staircase × (d0_cur - d0_prev).
    # After processing all distinct d0s, add final slab up to ref[0].

    hv = 0.0
    # staircase: (d1, d2) pairs, d1 strictly increasing, d2 strictly decreasing
    # We also track the staircase "clipped" to ref on d1 and d2.
    sc_d1: List[float] = []
    sc_d2: List[float] = []

    def _sc_area() -> float:
        """2-D hypervolume of current staircase w.r.t. (ref[1], ref[2])."""
        if not sc_d1:
            return 0.0
        area = 0.0
        prev_d1 = sc_d1[0]
        for k in range(len(sc_d1)):
            next_d1 = sc_d1[k + 1] if k + 1 < len(sc_d1) else ref[1]
            area += (next_d1 - sc_d1[k]) * (ref[2] - sc_d2[k])
        return area

    def _sc_insert(d1: float, d2: float) -> None:
        """Insert (d1, d2) into 2-D staircase, pruning dominated entries."""
        p = bisect.bisect_left(sc_d1, d1)
        # Remove dominated entries to the right (d1 >= d1_new, d2 >= d2_new)
        end = p
        while end < len(sc_d2) and sc_d2[end] >= d2:
            end += 1
        sc_d1[p:end] = [d1]
        sc_d2[p:end] = [d2]

    prev_d0 = valid[0][0]
    i = 0
    while i < len(valid):
        cur_d0 = valid[i][0]
        # Add slab from prev_d0 to cur_d0
        hv += (cur_d0 - prev_d0) * _sc_area()
        # Insert all points at cur_d0 into staircase
        j = i
        while j < len(valid) and valid[j][0] == cur_d0:
            _, d1, d2 = valid[j]
            if d1 < ref[1] and d2 < ref[2]:
                _sc_insert(d1, d2)
            j += 1
        prev_d0 = cur_d0
        i = j

    # Final slab from last d0 to ref[0]
    hv += (ref[0] - prev_d0) * _sc_area()
    return hv


def hypervolume_contributions_3d(
    pts: List[Tuple[float, float, float]],
    ref: Tuple[float, float, float],
) -> List[float]:
    """Compute hypervolume contribution of each point in *pts* in O(n² log n).

    HV contribution of point p = HV(pts) - HV(pts \\ {p}).

    This brute-force approach (remove one, recompute) is correct and simple.
    For n ≤ 50 (typical Pareto shell size after ε-pruning) the cost is
    negligible.  A true O(n log n) algorithm (Emmerich et al. 2006) exists
    but is considerably more complex to implement correctly.
    """
    total_hv = hypervolume_3d(pts, ref)
    contribs: List[float] = []
    for i in range(len(pts)):
        rest = pts[:i] + pts[i + 1:]
        contribs.append(total_hv - hypervolume_3d(rest, ref))
    return contribs


def _hv_reference(pts: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Compute a nadir reference point = max on each axis × 1.1 padding."""
    if not pts:
        return (1.0, 1.0, 1.0)
    r0 = max(p[0] for p in pts) * 1.1 + 1.0
    r1 = max(p[1] for p in pts) * 1.1 + 1.0
    r2 = max(p[2] for p in pts) * 1.1 + 1.0
    return (r0, r1, r2)


# --------------------------------------------------
# ε-dominance pruning — O(n log n) via log-grid boxing
# --------------------------------------------------

def _box_key(v: Tuple[float, ...], log1pe: float) -> Tuple[int, ...]:
    """ε-box index: box[d] = floor(ln(v[d]) / ln(1+ε)).

    Two points in the same box differ by ≤ (1+ε) on every dimension.
    v[d] ≤ 0 maps to sentinel -(1<<30).
    """
    _NEGINF = -(1 << 30)
    return tuple(
        int(math.floor(math.log(x) / log1pe)) if x > 0.0 else _NEGINF
        for x in v
    )


def _box_best_representative(
    existing_v: Tuple[float, ...],
    new_v: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Choose the better representative for an ε-box.

    Returns the point with the smaller value on each dimension independently
    (component-wise minimum).  This is Pareto-optimal within the box: the
    resulting representative is guaranteed to be at least as good as either
    input on every objective.  It may not correspond to an actual point,
    but since we only use it for dominance comparison (not as a final
    candidate), this is safe for filtering purposes.

    For the actual returned *item* we keep whichever original point has
    the smaller Pareto rank (i.e., is not dominated by the other).
    """
    return tuple(min(a, b) for a, b in zip(existing_v, new_v))


def eps_dominance_prune(
    items: Sequence[T],
    key: Callable[[T], Tuple[float, ...]],
    eps: float = 0.05,
    cap: Optional[int] = None,
) -> List[T]:
    """Prune *items* using ε-dominance (Laumanns et al. 2002), then hard-cap.

    Point *a* ε-dominates *b* iff ∀d: a[d] ≤ (1+ε)·b[d], ∃d: a[d] < b[d].

    Improvements over naive implementation:
      1. Log-grid ε-boxing: O(n) boxing via hash table.
      2. Within each box: keep the item whose key vector is *Pareto-non-
         dominated* by the other.  If neither dominates the other, keep
         the one with smaller sum of normalised objectives (proxy tiebreak).
      3. Cross-box strict Pareto filter via O(r log n) 3-D sort-and-sweep.
      4. Float-safe dominance comparisons with _REL_TOL guard.
      5. Optional hard cap (sort by rank_key + slice).

    Relative error guarantee: any dropped point's true objectives differ
    from its box representative by at most ε on every axis.
    """
    if not items:
        return []
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    log1pe = math.log(1.0 + eps)

    # Step 1: ε-boxing — O(n)
    # box_repr[box] = (item, v) — the current best representative
    box_repr: Dict[Tuple[int, ...], Tuple[T, Tuple[float, ...]]] = {}
    for item in items:
        v = key(item)
        bk = _box_key(v, log1pe)
        if bk not in box_repr:
            box_repr[bk] = (item, v)
        else:
            _, ev = box_repr[bk]
            # Choose representative: prefer Pareto-non-dominated item.
            # If one dominates the other, keep the dominator.
            # If incomparable, keep the one with smaller objective sum
            # (normalized by existing values to avoid scale bias).
            if _vec_dominates(v, ev):
                box_repr[bk] = (item, v)
            elif not _vec_dominates(ev, v):
                # Incomparable: tiebreak by sum of ratios v[d]/ev[d]
                ratio_new = sum(a / max(b, 1e-12) for a, b in zip(v, ev))
                if ratio_new < len(v):   # new is better on average
                    box_repr[bk] = (item, v)

    reps_items = [item for item, _ in box_repr.values()]
    reps_vecs  = [v    for _, v   in box_repr.values()]

    # Step 2: cross-box Pareto filter — O(r log r) for 3-D, O(r²) for 4-D
    if len(reps_items) > 1:
        ndim = len(reps_vecs[0]) if reps_vecs else 0
        if ndim == 3:
            pts3 = [(v[0], v[1], v[2]) for v in reps_vecs]
            idx_list = list(range(len(reps_items)))
            front_idx = set(_pareto_front_3d(pts3, idx_list))
            reps_items = [reps_items[i] for i in range(len(reps_items)) if i in front_idx]
        else:
            reps_items = _strict_pareto_filter_generic(reps_items, key)

    # Step 3: hard cap — prefer by HV contribution if ndim==3, else rank_key
    if cap is not None and cap > 0 and len(reps_items) > cap:
        reps_vecs_filtered = [key(x) for x in reps_items]
        if len(reps_vecs_filtered) > 0 and len(reps_vecs_filtered[0]) == 3:
            pts3 = [(v[0], v[1], v[2]) for v in reps_vecs_filtered]
            ref = _hv_reference(pts3)
            contribs = hypervolume_contributions_3d(pts3, ref)
            # Sort by contribution descending: highest-contribution points kept
            ranked = sorted(range(len(reps_items)), key=lambda i: -contribs[i])
            reps_items = [reps_items[i] for i in ranked[:cap]]
        else:
            reps_items.sort(key=key)
            reps_items = reps_items[:cap]

    return reps_items


def _strict_pareto_filter_generic(
    items: List[T],
    key: Callable[[T], Tuple[float, ...]],
) -> List[T]:
    """O(n²) strict Pareto filter for arbitrary dimension (non-3D fallback)."""
    front: List[T] = []
    for cand in items:
        cv = key(cand)
        dominated = False
        survivors: List[T] = []
        for existing in front:
            ev = key(existing)
            if _vec_dominates(ev, cv):
                dominated = True
                survivors.append(existing)
                continue
            if _vec_dominates(cv, ev):
                continue
            survivors.append(existing)
        if not dominated:
            survivors.append(cand)
        front = survivors
    return front


# --------------------------------------------------
# Legacy shim
# --------------------------------------------------

def skyline_and_cap(
    items: Sequence[T],
    key: Callable[[T], Tuple[float, ...]],
    cap: int,
) -> List[T]:
    """Pareto skyline then hard cap.  Prefer eps_dominance_prune for new code."""
    return eps_dominance_prune(items, key=key, eps=0.05, cap=cap if cap > 0 else None)


__all__ = [
    "dominates",
    "eps_dominance_prune",
    "hypervolume_3d",
    "hypervolume_contributions_3d",
    "nondominated_sort_3d",
    "pareto_front",
    "pareto_front_fast",
    "rank_key",
    "skyline_and_cap",
    "_obj_to_vec",
]
