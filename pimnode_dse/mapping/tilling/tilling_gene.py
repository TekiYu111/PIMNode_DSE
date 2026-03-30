from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations, permutations, product
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pimnode_dse.hardware.arch_spec import ArchSpecError, HardwareSpec

LevelName = str
LoopName = str
AccScope = str
BufMode = str

VALID_ACC_SCOPE: Tuple[AccScope, ...] = ("local", "sram")
VALID_BUF_MODE: Tuple[BufMode, ...] = ("single", "double")

# 分层归约的 partial-sum 生命周期选项
# "inner"  : partial sum 只在最内层 tile 内累加，不跨 tile 溢出
# "sram"   : partial sum 驻留 SRAM，跨 PE-tile 累加后写回
# "global" : partial sum 跨多个 SRAM-tile（跨节点或跨 epoch）累加
VALID_ACC_LIFETIME: Tuple[str, ...] = ("inner", "sram", "global")

# 归约在不同层次累加深度的表达
# 1 表示不分层（传统单层归约），>1 表示分 acc_depth 层 partial-sum
_DEFAULT_ACC_DEPTH: int = 1


class TilingSpecError(ValueError):
    """Raised when a tiling spec or tiling search config is invalid."""


@dataclass(frozen=True)
class MemTileSpec:
    tile_size: Mapping[LoopName, int]
    loop_order: Sequence[LoopName]
    buf_mode: BufMode = "single"
    rw_overlap: bool = False

    def __post_init__(self) -> None:
        tile_size = _norm_tile(self.tile_size)
        loop_order = _norm_order(self.loop_order)
        _check_order_cover(order=loop_order, tile=tile_size)
        buf_mode = str(self.buf_mode).strip().lower()
        if buf_mode not in VALID_BUF_MODE:
            raise TilingSpecError(f"invalid buf_mode: {self.buf_mode!r}")
        object.__setattr__(self, "tile_size", tile_size)
        object.__setattr__(self, "loop_order", loop_order)
        object.__setattr__(self, "buf_mode", buf_mode)
        object.__setattr__(self, "rw_overlap", bool(self.rw_overlap))


@dataclass(frozen=True)
class GroupTilingSpec:
    group_id: str
    tier_tiles: Mapping[LevelName, MemTileSpec]
    split_red: Sequence[LoopName] = ()
    acc_scope: AccScope = "sram"

    def __post_init__(self) -> None:
        group_id = str(self.group_id).strip()
        if not group_id:
            raise TilingSpecError("group_id must not be empty")

        tier_tiles = {_norm_level(level): spec for level, spec in dict(self.tier_tiles).items()}
        if not tier_tiles:
            raise TilingSpecError("tier_tiles must not be empty")
        for level, spec in tier_tiles.items():
            if not isinstance(spec, MemTileSpec):
                raise TypeError(f"tier_tiles[{level!r}] must be MemTileSpec")

        split_red = tuple(_norm_name(name) for name in self.split_red if str(name).strip())
        if len(set(split_red)) != len(split_red):
            raise TilingSpecError("split_red must not contain duplicates")

        acc_scope = str(self.acc_scope).strip().lower()
        if acc_scope not in VALID_ACC_SCOPE:
            raise TilingSpecError(f"invalid acc_scope: {self.acc_scope!r}")

        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "tier_tiles", tier_tiles)
        object.__setattr__(self, "split_red", split_red)
        object.__setattr__(self, "acc_scope", acc_scope)

    def get(self, level: str) -> MemTileSpec:
        key = _norm_level(level)
        try:
            return self.tier_tiles[key]
        except KeyError as exc:
            raise KeyError(f"missing tier {key!r}") from exc


@dataclass(frozen=True)
class GroupSpace:
    group_id: str
    loops: Tuple[LoopName, ...]
    extent: Dict[LoopName, int]
    red_loops: Tuple[LoopName, ...] = ()

    def __post_init__(self) -> None:
        group_id = str(self.group_id).strip()
        loops = tuple(_norm_name(name) for name in self.loops if str(name).strip())
        if not group_id:
            raise TilingSpecError("group_id must not be empty")
        if not loops:
            raise TilingSpecError("loops must not be empty")
        if len(set(loops)) != len(loops):
            raise TilingSpecError("loops must not contain duplicates")

        extent = {_norm_name(name): int(size) for name, size in dict(self.extent).items()}
        for name in loops:
            if name not in extent:
                raise TilingSpecError(f"missing extent for loop {name!r}")
            if extent[name] <= 0:
                raise TilingSpecError(f"extent[{name!r}] must be > 0")

        red_loops = tuple(_norm_name(name) for name in self.red_loops if str(name).strip())
        for name in red_loops:
            if name not in extent:
                raise TilingSpecError(f"unknown red loop {name!r}")

        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "loops", loops)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "red_loops", red_loops)


@dataclass(frozen=True)
class EnumCfg:
    tile_vals: Mapping[LevelName, Mapping[LoopName, Sequence[int]]] = field(default_factory=dict)
    order_mode: str = "heuristic"
    order_limit: Optional[int] = None
    split_red_mode: str = "all"
    acc_scopes: Sequence[AccScope] = VALID_ACC_SCOPE
    sram_modes: Sequence[BufMode] = VALID_BUF_MODE
    sram_overlap: Sequence[bool] = (False, True)

    # 扩展 tile 候选生成：除数之外的“近似 tile”以及硬件对齐友好候选。
    # - add_approx: 允许非整除尾块（extent != k*tile 的情况）通过覆盖率阈值过滤
    # - coverage_min: 近似 tile 覆盖率阈值（ceil(extent/tile)*tile / extent 的倒数）
    # - add_power2: 加入 2 的幂候选（便于对齐 / 简化地址生成）
    # - add_aligns: 加入若干对齐粒度（burst、cacheline、bank 等）
    add_approx: bool = True
    coverage_min: float = 0.85
    add_power2: bool = True
    add_aligns: Sequence[int] = ()

    def __post_init__(self) -> None:
        tile_vals = {
            _norm_level(level): {
                _norm_name(loop): tuple(int(v) for v in vals)
                for loop, vals in dict(raw).items()
            }
            for level, raw in dict(self.tile_vals).items()
        }
        order_mode = str(self.order_mode).strip().lower()
        if order_mode not in {"heuristic", "exhaustive"}:
            raise TilingSpecError(f"invalid order_mode: {self.order_mode!r}")
        split_red_mode = str(self.split_red_mode).strip().lower()
        if split_red_mode not in {"none", "all", "single", "any"}:
            raise TilingSpecError(f"invalid split_red_mode: {self.split_red_mode!r}")
        acc_scopes = tuple(str(x).strip().lower() for x in self.acc_scopes)
        sram_modes = tuple(str(x).strip().lower() for x in self.sram_modes)
        sram_overlap = tuple(bool(x) for x in self.sram_overlap)
        if not sram_overlap:
            raise TilingSpecError("sram_overlap must not be empty")
        for scope in acc_scopes:
            if scope not in VALID_ACC_SCOPE:
                raise TilingSpecError(f"invalid acc_scope candidate: {scope!r}")
        for mode in sram_modes:
            if mode not in VALID_BUF_MODE:
                raise TilingSpecError(f"invalid sram mode candidate: {mode!r}")

        add_approx = bool(self.add_approx)
        try:
            coverage_min = float(self.coverage_min)
        except (TypeError, ValueError) as exc:
            raise TilingSpecError("coverage_min must be a float") from exc
        if not (0.0 < coverage_min <= 1.0):
            raise TilingSpecError("coverage_min must be in (0, 1]")

        add_power2 = bool(self.add_power2)
        add_aligns = tuple(sorted({int(x) for x in self.add_aligns if int(x) > 0}))

        object.__setattr__(self, "tile_vals", tile_vals)
        object.__setattr__(self, "order_mode", order_mode)
        object.__setattr__(self, "split_red_mode", split_red_mode)
        object.__setattr__(self, "acc_scopes", acc_scopes)
        object.__setattr__(self, "sram_modes", sram_modes)
        object.__setattr__(self, "sram_overlap", sram_overlap)
        object.__setattr__(self, "add_approx", add_approx)
        object.__setattr__(self, "coverage_min", coverage_min)
        object.__setattr__(self, "add_power2", add_power2)
        object.__setattr__(self, "add_aligns", add_aligns)


def validate_group_tiling(
    spec: GroupTilingSpec,
    hw: HardwareSpec,
    loops: Sequence[str],
    extent: Mapping[str, int],
    red_loops: Sequence[str] = (),
    sram_bytes: Optional[int] = None,
) -> None:
    loops_n = tuple(_norm_name(name) for name in loops if str(name).strip())
    if not loops_n:
        raise TilingSpecError("loops must not be empty")

    extent_n = {_norm_name(name): int(size) for name, size in dict(extent).items()}
    red_n = tuple(_norm_name(name) for name in red_loops if str(name).strip())
    hw_levels = tuple(_norm_level(name) for name in hw.levels)
    if set(spec.tier_tiles) != set(hw_levels):
        raise TilingSpecError(f"levels mismatch: spec={tuple(spec.tier_tiles)} hw={hw_levels}")

    for level in hw_levels:
        tile = spec.tier_tiles[level]
        if set(tile.tile_size) != set(loops_n):
            miss = set(loops_n) - set(tile.tile_size)
            extra = set(tile.tile_size) - set(loops_n)
            raise TilingSpecError(
                f"{level}: tile dims mismatch, missing={sorted(miss)}, extra={sorted(extra)}"
            )
        if set(tile.loop_order) != set(loops_n):
            raise TilingSpecError(f"{level}: loop_order must cover all loops exactly once")
        for name in loops_n:
            val = tile.tile_size[name]
            lim = extent_n[name]
            if val <= 0 or val > lim:
                raise TilingSpecError(f"{level}: tile_size[{name!r}]={val} exceeds extent {lim}")

    for name in loops_n:
        pe_val = spec.tier_tiles["pe"].tile_size[name]
        sram_val = spec.tier_tiles["sram"].tile_size[name]
        dram_val = spec.tier_tiles["dram"].tile_size[name]
        if not (pe_val <= sram_val <= dram_val):
            raise TilingSpecError(
                f"tile monotonicity failed at loop {name!r}: pe={pe_val}, sram={sram_val}, dram={dram_val}"
            )

    split_red = set(spec.split_red)
    if not split_red.issubset(set(red_n)):
        bad = sorted(split_red - set(red_n))
        raise TilingSpecError(f"split_red must be subset of red_loops, bad={bad}")

    if spec.acc_scope == "local" and not hw.pe.has_acc:
        raise TilingSpecError("acc_scope='local' requires hw.pe.has_acc")

    sram = spec.tier_tiles["sram"]
    if sram.rw_overlap and hw.sram.concurrency < 2:
        raise TilingSpecError("sram.rw_overlap=True requires hw.sram.concurrency >= 2")
    if sram.buf_mode not in VALID_BUF_MODE:
        raise TilingSpecError(f"invalid sram buf_mode: {sram.buf_mode!r}")

    dram = spec.tier_tiles["dram"]
    pe = spec.tier_tiles["pe"]
    if dram.buf_mode != "single" or dram.rw_overlap:
        raise TilingSpecError("dram tier must keep buf_mode='single' and rw_overlap=False")
    if pe.buf_mode != "single" or pe.rw_overlap:
        raise TilingSpecError("pe tier must keep buf_mode='single' and rw_overlap=False")

    if sram_bytes is not None:
        want = int(sram_bytes)
        if want < 0:
            raise TilingSpecError("sram_bytes must be >= 0")
        if want > int(hw.sram.cap):
            raise TilingSpecError(f"sram footprint {want} exceeds cap {int(hw.sram.cap)}")


def enum_level_orders(
    loops: Sequence[str],
    red_loops: Sequence[str],
    mode: str = "heuristic",
    limit: Optional[int] = None,
) -> Tuple[Tuple[str, ...], ...]:
    vals = tuple(_norm_name(name) for name in loops if str(name).strip())
    if len(set(vals)) != len(vals):
        raise TilingSpecError("loops must not contain duplicates")
    red_set = set(_norm_name(name) for name in red_loops if str(name).strip())
    red = tuple(name for name in vals if name in red_set)
    keep = tuple(name for name in vals if name not in red_set)
    mode_n = str(mode).strip().lower()

    if mode_n == "exhaustive":
        out = tuple(tuple(order) for order in permutations(vals))
    elif mode_n == "heuristic":
        seeds = [
            vals,
            tuple(reversed(vals)),
            keep + red if keep else vals,
            red + keep if red else vals,
        ]
        out = _dedup_orders(seeds)
    else:
        raise TilingSpecError(f"invalid order mode: {mode!r}")

    if limit is not None:
        if int(limit) <= 0:
            raise TilingSpecError("limit must be > 0")
        out = out[: int(limit)]
    return out


def enum_level_tiles_pruned(
    level: str,
    space: GroupSpace,
    hw: HardwareSpec,
    cfg: EnumCfg,
) -> Tuple[MemTileSpec, ...]:
    level_n = _norm_level(level)
    tile_cands: List[List[int]] = []
    for loop in space.loops:
        vals = _tile_vals_for(level=level_n, loop=loop, extent=space.extent[loop], cfg=cfg)
        tile_cands.append(list(vals))

    orders = enum_level_orders(
        loops=space.loops,
        red_loops=space.red_loops,
        mode=cfg.order_mode,
        limit=cfg.order_limit,
    )
    if level_n == "sram":
        policy_cands = tuple(product(cfg.sram_modes, cfg.sram_overlap))
    else:
        policy_cands = (("single", False),)

    out: List[MemTileSpec] = []
    seen: Set[Tuple[Any, ...]] = set()
    for vals in product(*tile_cands):
        tile = {loop: int(val) for loop, val in zip(space.loops, vals)}
        if not _structural_tile_ok(level_n, tile, space.extent):
            continue
        for order in orders:
            for buf_mode, rw_overlap in policy_cands:
                spec = MemTileSpec(
                    tile_size=tile,
                    loop_order=order,
                    buf_mode=buf_mode,
                    rw_overlap=rw_overlap,
                )
                key = (
                    level_n,
                    tuple(spec.loop_order),
                    tuple(sorted(spec.tile_size.items())),
                    spec.buf_mode,
                    spec.rw_overlap,
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(spec)
    return tuple(out)


def build_tiling_key(spec: GroupTilingSpec) -> Tuple[Any, ...]:
    return (
        spec.acc_scope,
        tuple(spec.split_red),
        tuple(
            (
                level,
                tuple(tile.loop_order),
                tuple(sorted(tile.tile_size.items())),
                tile.buf_mode,
                tile.rw_overlap,
            )
            for level, tile in sorted(spec.tier_tiles.items())
        ),
    )


def enum_group_tilings(
    hw: HardwareSpec,
    *,
    group: Optional[Any] = None,
    workload: Optional[Any] = None,
    group_id: Optional[str] = None,
    loops: Optional[Sequence[str]] = None,
    extent: Optional[Mapping[str, int]] = None,
    red_loops: Optional[Sequence[str]] = None,
    cfg: Optional[EnumCfg] = None,
    sram_bytes_fn: Optional[Callable[[GroupTilingSpec], Optional[int]]] = None,
    early_prune: bool = True,
) -> Tuple[GroupTilingSpec, ...]:
    space = _resolve_group_space(
        group=group,
        workload=workload,
        group_id=group_id,
        loops=loops,
        extent=extent,
        red_loops=red_loops,
    )
    cfg = cfg or EnumCfg()

    level_specs = {
        level: enum_level_tiles_pruned(level=level, space=space, hw=hw, cfg=cfg)
        for level in (_norm_level(name) for name in hw.levels)
    }

    split_sets = _enum_split_red(space.red_loops, cfg.split_red_mode)
    out: List[GroupTilingSpec] = []
    seen: Set[Tuple[Any, ...]] = set()

    for dram_spec in level_specs["dram"]:
        for sram_spec in level_specs["sram"]:
            if early_prune and not _monotonic_pair(sram_spec, dram_spec, space.loops):
                continue
            for pe_spec in level_specs["pe"]:
                if early_prune and not _monotonic_pair(pe_spec, sram_spec, space.loops):
                    continue
                tier_tiles = {"dram": dram_spec, "sram": sram_spec, "pe": pe_spec}
                for split_red in split_sets:
                    scopes = cfg.acc_scopes if split_red else ("sram",)
                    for acc_scope in scopes:
                        spec = GroupTilingSpec(
                            group_id=space.group_id,
                            tier_tiles=tier_tiles,
                            split_red=split_red,
                            acc_scope=acc_scope,
                        )
                        sram_bytes = sram_bytes_fn(spec) if sram_bytes_fn is not None else None
                        try:
                            validate_group_tiling(
                                spec=spec,
                                hw=hw,
                                loops=space.loops,
                                extent=space.extent,
                                red_loops=space.red_loops,
                                sram_bytes=sram_bytes,
                            )
                        except (TilingSpecError, ArchSpecError, ValueError, TypeError):
                            continue
                        key = build_tiling_key(spec)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(spec)
    return tuple(out)


def enum_tilings(
    fusion: Any,
    workload: Any,
    hw: HardwareSpec,
    *,
    cfg: Optional[EnumCfg] = None,
    sram_bytes_fn: Optional[Callable[[GroupTilingSpec], Optional[int]]] = None,
    early_prune: bool = True,
) -> Dict[str, Tuple[GroupTilingSpec, ...]]:
    groups = getattr(fusion, "groups", None)
    if groups is None:
        raise TilingSpecError("fusion must provide groups")
    out: Dict[str, Tuple[GroupTilingSpec, ...]] = {}
    for group in groups:
        gid = _pick_group_id(group)
        out[gid] = enum_group_tilings(
            hw,
            group=group,
            workload=workload,
            cfg=cfg,
            sram_bytes_fn=sram_bytes_fn,
            early_prune=early_prune,
        )
    return out


def _resolve_group_space(
    *,
    group: Optional[Any],
    workload: Optional[Any],
    group_id: Optional[str],
    loops: Optional[Sequence[str]],
    extent: Optional[Mapping[str, int]],
    red_loops: Optional[Sequence[str]],
) -> GroupSpace:
    if loops is not None and extent is not None:
        gid = str(group_id or _pick_group_id(group) or "group").strip()
        return GroupSpace(
            group_id=gid,
            loops=tuple(loops),
            extent=dict(extent),
            red_loops=tuple(red_loops or ()),
        )
    if group is None or workload is None:
        raise TilingSpecError(
            "enum_group_tilings needs either (group + workload) or (group_id + loops + extent)"
        )
    return _group_space_from_workload(group, workload)


def _group_space_from_workload(group: Any, workload: Any) -> GroupSpace:
    gid = _pick_group_id(group)
    op_ids = getattr(group, "ops", None)
    if op_ids is None:
        raise TilingSpecError("group must provide ops")

    seen: List[str] = []
    used: Set[str] = set()
    extent: Dict[str, int] = {}
    red: List[str] = []

    for op_id in op_ids:
        op = _workload_op(workload, op_id)
        dims = _pick_iter_dims(op)
        for name in dims:
            if name not in used:
                used.add(name)
                seen.append(name)
        for name, size in _pick_dim_extent(op).items():
            cur = extent.get(name)
            if cur is None or size > cur:
                extent[name] = size
        for name in _pick_red_dims(op):
            if name not in red:
                red.append(name)

    if not seen:
        raise TilingSpecError(f"group {gid!r} resolved no loops from workload")
    for name in seen:
        extent.setdefault(name, 1)

    return GroupSpace(group_id=gid, loops=tuple(seen), extent=extent, red_loops=tuple(red))


def _tile_vals_for(level: str, loop: str, extent: int, cfg: EnumCfg) -> Tuple[int, ...]:
    raw = cfg.tile_vals.get(level, {}).get(loop)
    if raw:
        vals = tuple(sorted({int(v) for v in raw if 0 < int(v) <= extent}))
        if vals:
            return vals

    # 默认：既保留“整除因子”（利于搜索/精确分块），也加入一批
    # 与硬件友好的近似 tile 候选（允许尾块、padding 后对齐、burst/bank 对齐等）。
    out: Set[int] = set(_divisors(extent))

    # 2 的幂候选：更容易对齐/地址生成/向量化
    if cfg.add_power2:
        p = 1
        while p <= extent:
            out.add(p)
            p <<= 1

    # 对齐粒度候选（burst、cacheline、bank 等），将这些粒度的倍数加入候选
    for a in cfg.add_aligns:
        if a <= 0:
            continue
        k = 1
        while k * a <= extent:
            out.add(k * a)
            k += 1

    # 近似 tile：从一组“常见”粒度出发枚举，并用覆盖率过滤
    # coverage = extent / ceil(extent / t) / t  ?（等价于 extent / padded）
    if cfg.add_approx:
        # 经验候选：因子 + 幂次 + 对齐倍数之后，补充一些分段采样
        # 这里不枚举 1..extent（避免爆炸），而是以步长采样 + 对齐修正。
        step = max(1, int(math.sqrt(extent)))
        for t in range(1, extent + 1, step):
            out.add(t)
        # 也加入接近 extent 的几个候选
        for t in (extent, max(1, extent // 2), max(1, (2 * extent) // 3)):
            out.add(int(t))

        filtered: Set[int] = set()
        for t in out:
            if t <= 0 or t > extent:
                continue
            padded = int(math.ceil(extent / t)) * t
            if padded <= 0:
                continue
            coverage = float(extent) / float(padded)
            if coverage >= cfg.coverage_min:
                filtered.add(int(t))
        out = filtered

    return tuple(sorted(out))


def _enum_split_red(red_loops: Sequence[str], mode: str) -> Tuple[Tuple[str, ...], ...]:
    vals = tuple(red_loops)
    if not vals or mode == "none":
        return ((),)
    if mode == "single":
        return ((),) + tuple((name,) for name in vals)
    if mode == "any":
        out: List[Tuple[str, ...]] = [()]
        for size in range(1, len(vals) + 1):
            out.extend(tuple(combo) for combo in combinations(vals, size))
        return tuple(out)
    return ((), vals)


def _monotonic_pair(
    lower: MemTileSpec,
    upper: MemTileSpec,
    loops: Sequence[str],
) -> bool:
    for name in loops:
        if lower.tile_size[name] > upper.tile_size[name]:
            return False
    return True


def _structural_tile_ok(
    level: str,
    tile: Mapping[str, int],
    extent: Mapping[str, int],
) -> bool:
    if not tile:
        return False
    active = sum(1 for name, val in tile.items() if int(val) > 1)
    if active == 0 and any(int(extent[name]) > 1 for name in tile):
        return False
    if level == "dram":
        return True
    if level == "sram":
        return True
    return True


def _workload_op(workload: Any, op_id: Any) -> Any:
    if hasattr(workload, "op"):
        return workload.op(op_id)
    ops = getattr(workload, "ops", None)
    if isinstance(ops, Mapping):
        return ops[op_id]
    raise TilingSpecError("workload must provide op(op_id) or ops mapping")


def _pick_group_id(group: Optional[Any]) -> str:
    if group is None:
        return "group"
    for name in ("group_id", "gid", "id", "name"):
        if hasattr(group, name):
            value = getattr(group, name)
            if str(value).strip():
                return str(value).strip()
    return "group"


def _pick_iter_dims(op: Any) -> Tuple[str, ...]:
    raw = getattr(op, "iter_dims", ())
    out: List[str] = []
    seen: Set[str] = set()
    for dim in raw:
        name = _dim_name(dim)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _pick_red_dims(op: Any) -> Tuple[str, ...]:
    raw = getattr(op, "reduce_dims", ())
    out: List[str] = []
    seen: Set[str] = set()
    for dim in raw:
        name = _dim_name(dim)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _pick_dim_extent(op: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, value in getattr(op, "dim_constraints", {}).items():
        name = _dim_name(key)
        size = _dim_size(value)
        if name and size is not None and size > 0:
            out[name] = max(int(size), int(out.get(name, 0)))
    attrs = getattr(op, "attrs", {})
    if isinstance(attrs, Mapping):
        for raw in attrs.get("dims", ()):
            name = _dim_name(raw)
            size = _dim_size(raw)
            if name and size is not None and size > 0:
                out[name] = max(int(size), int(out.get(name, 0)))
    return out


def _dim_name(item: Any) -> str:
    if isinstance(item, str):
        return _norm_name(item)
    for name in ("name", "id", "dim"):
        if hasattr(item, name):
            val = getattr(item, name)
            if str(val).strip():
                return _norm_name(val)
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and item:
        return _norm_name(item[0])
    return ""


def _dim_size(item: Any) -> Optional[int]:
    if isinstance(item, int):
        return int(item)
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
        try:
            return int(item[1])
        except (TypeError, ValueError):
            return None
    for name in ("size", "extent", "value"):
        if hasattr(item, name):
            raw = getattr(item, name)
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
    return None


def _norm_tile(tile: Mapping[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, value in dict(tile).items():
        name = _norm_name(key)
        val = int(value)
        if not name:
            raise TilingSpecError("tile_size contains empty loop name")
        if val <= 0:
            raise TilingSpecError(f"tile_size[{name!r}] must be > 0")
        out[name] = val
    if not out:
        raise TilingSpecError("tile_size must not be empty")
    return out


def _norm_order(order: Sequence[str]) -> Tuple[str, ...]:
    out = tuple(_norm_name(name) for name in order if str(name).strip())
    if not out:
        raise TilingSpecError("loop_order must not be empty")
    if len(set(out)) != len(out):
        raise TilingSpecError("loop_order must not contain duplicates")
    return out


def _check_order_cover(*, order: Sequence[str], tile: Mapping[str, int]) -> None:
    if set(order) != set(tile):
        miss = set(tile) - set(order)
        extra = set(order) - set(tile)
        raise TilingSpecError(
            f"loop_order/tile_size mismatch, missing={sorted(miss)}, extra={sorted(extra)}"
        )


def _norm_level(level: Any) -> str:
    text = str(level).strip().lower()
    if text.endswith("_tier"):
        text = text[:-5]
    if not text:
        raise TilingSpecError("level must not be empty")
    return text


def _norm_name(name: Any) -> str:
    return str(name).strip().lower()


def _dedup_orders(items: Sequence[Sequence[str]]) -> Tuple[Tuple[str, ...], ...]:
    out: List[Tuple[str, ...]] = []
    seen: Set[Tuple[str, ...]] = set()
    for item in items:
        row = tuple(_norm_name(name) for name in item if str(name).strip())
        if not row or row in seen:
            continue
        seen.add(row)
        out.append(row)
    return tuple(out)


def _divisors(extent: int) -> Tuple[int, ...]:
    vals: List[int] = []
    for cand in range(1, int(extent) + 1):
        if int(extent) % cand == 0:
            vals.append(cand)
    return tuple(vals)


__all__ = [
    "TilingSpecError",
    "MemTileSpec",
    "GroupTilingSpec",
    "GroupSpace",
    "EnumCfg",
    "validate_group_tiling",
    "enum_level_orders",
    "enum_level_tiles_pruned",
    "build_tiling_key",
    "enum_group_tilings",
    "enum_tilings",
]
