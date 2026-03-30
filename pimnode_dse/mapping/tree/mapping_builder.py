from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pimnode_dse.mapping.fusion.fusion_gene import FusionGene, FusionGroup
from pimnode_dse.mapping.placement.dataflow import FlowBucket, FlowContract
from pimnode_dse.mapping.placement.node import GroupDP, PlacementPlan, StoreNode
from pimnode_dse.mapping.tilling.tilling_gene import GroupTilingSpec
from pimnode_dse.mapping.tree.mapping_tree import MappingTree, Move, OpNode, ScopeNode, TileNode


@dataclass(frozen=True)
class GroupChoice:
    group_id: str
    place: GroupDP
    tiling: GroupTilingSpec
    contract: Optional[FlowContract] = None
    bucket: Optional[FlowBucket] = None


def build_mapping_tree(
    fusion_gene: FusionGene,
    placement_plan: Union[PlacementPlan, Mapping[str, GroupDP]],
    tilings: Mapping[str, Union[GroupTilingSpec, Sequence[GroupTilingSpec]]],
    *,
    flow_contracts: Optional[Mapping[str, Sequence[FlowContract]]] = None,
    flow_buckets: Optional[Mapping[str, Sequence[FlowBucket]]] = None,
    workload: Optional[Any] = None,
    group_bind_map: Optional[Mapping[str, str]] = None,
) -> MappingTree:
    root = ScopeNode(
        id="root",
        bind="seq",
        mem="dram",
        stage_kind="root",
        repeat_hint=1,
        overlap_policy="none",
        resource_domain="dram",
        attrs={"level": "dram"},
    )

    place_map = _placement_map(placement_plan)
    bind_map = {str(k): str(v).strip().lower() for k, v in dict(group_bind_map or {}).items()}
    contract_map = {str(k): tuple(v) for k, v in dict(flow_contracts or {}).items()}
    bucket_map = {str(k): tuple(v) for k, v in dict(flow_buckets or {}).items()}

    for group in fusion_gene.groups:
        gid = group.group_id
        place = place_map.get(gid)
        if place is None:
            continue

        tiling = _pick_tiling(tilings.get(gid), gid)
        if tiling is None:
            continue

        contract = _pick_contract(contract_map.get(gid, ()))
        bucket = _pick_bucket(bucket_map.get(gid, ()))
        bind = bind_map.get(gid, "seq")

        group_scope = _build_group_scope(
            group=group,
            place=place,
            tiling=tiling,
            bind=bind,
            contract=contract,
            bucket=bucket,
            workload=workload,
        )
        root.add_kid(group_scope)

    tree = MappingTree(root=root)
    tree.validate()
    return tree


# --------------------------------------------------
# Group scope
# --------------------------------------------------

def _build_group_scope(
    group: FusionGroup,
    place: GroupDP,
    tiling: GroupTilingSpec,
    *,
    bind: str,
    contract: Optional[FlowContract],
    bucket: Optional[FlowBucket],
    workload: Optional[Any],
) -> ScopeNode:
    group_mem = _group_home_mem(place)

    repeat_hint = _group_repeat_hint(contract=contract, tiling=tiling, group=group, workload=workload)
    overlap_policy = _group_overlap_policy(contract=contract, tiling=tiling)
    resource_domain = group_mem

    scope = ScopeNode(
        id=f"group::{group.group_id}",
        bind=bind,
        mem=group_mem,
        stage_kind="group",
        repeat_hint=repeat_hint,
        overlap_policy=overlap_policy,
        resource_domain=resource_domain,
        attrs={
            "group_id": group.group_id,
            "fusion_ops": tuple(group.ops),
            "placement_levels": tuple(place.levels()),
            "group_mem": group_mem,
            "tiling_group_id": getattr(tiling, "group_id", group.group_id),
        },
    )

    if contract is not None:
        _apply_flow_contract(
            scope=scope,
            contract=contract,
            place=place,
            group_mem=group_mem,
            repeat_hint=repeat_hint,
            workload=workload,
        )
        tile_chain = _build_tile_chain_from_contract(
            group=group,
            tiling=tiling,
            contract=contract,
            workload=workload,
        )
    else:
        if bucket is not None:
            _apply_flow_bucket(
                scope=scope,
                bucket=bucket,
                place=place,
                group_mem=group_mem,
                repeat_hint=repeat_hint,
                workload=workload,
            )
        tile_chain = _build_tile_chain_from_tiling(
            group=group,
            tiling=tiling,
            workload=workload,
        )

    scope.add_kid(tile_chain)
    return scope


# --------------------------------------------------
# Tile chain
# --------------------------------------------------

def _build_tile_chain_from_contract(
    group: FusionGroup,
    tiling: GroupTilingSpec,
    contract: FlowContract,
    workload: Optional[Any],
) -> TileNode:
    block_map = {blk.level: blk for blk in contract.level_blocks}
    level_order = ["dram", "sram", "pe"]

    head: Optional[TileNode] = None
    prev: Optional[TileNode] = None

    for level in level_order:
        if level not in tiling.tier_tiles:
            continue

        spec = tiling.tier_tiles[level]
        blk = block_map.get(level)

        loops = list(blk.loops) if blk is not None else list(spec.loop_order)
        size = dict(blk.tile_size) if blk is not None else dict(spec.tile_size)
        temporal = tuple(blk.temporal_loops) if blk is not None else _default_temporal_loops(level, loops)
        spatial = tuple(blk.spatial_loops) if blk is not None else _default_spatial_loops(level, loops)
        repeat_hint = int(blk.repeat_hint) if blk is not None else _repeat_hint_from_tiling(level, spec, group, workload)
        replication_hint = int(blk.replication_hint) if blk is not None else _replication_hint_from_tiling(level, spec)
        overlap_hint = bool(blk.overlap_hint) if blk is not None else bool(getattr(spec, "rw_overlap", False))

        mode = "spat" if spatial else ("spat" if level == "pe" else "temp")

        tile = TileNode(
            id=f"tile::{group.group_id}::{level}",
            mode=mode,
            loops=loops,
            size=size,
            order=loops,
            attrs={
                "level": level,
                "temporal_loops": temporal,
                "spatial_loops": spatial,
                "buf_mode": getattr(spec, "buf_mode", "single"),
                "rw_overlap": overlap_hint,
                "repeat_hint": repeat_hint,
                "replication_hint": replication_hint,
                "acc_scope": getattr(tiling, "acc_scope", "sram"),
                "split_red": tuple(getattr(tiling, "split_red", ())),
            },
        )

        if head is None:
            head = tile
        if prev is not None:
            prev.set_kid(tile)
        prev = tile

    op_scope = _build_op_scope(group, workload)
    assert prev is not None
    prev.set_kid(op_scope)
    assert head is not None
    return head


def _build_tile_chain_from_tiling(
    group: FusionGroup,
    tiling: GroupTilingSpec,
    workload: Optional[Any],
) -> TileNode:
    level_order = ["dram", "sram", "pe"]

    head: Optional[TileNode] = None
    prev: Optional[TileNode] = None

    for level in level_order:
        if level not in tiling.tier_tiles:
            continue

        spec = tiling.tier_tiles[level]
        loops = list(spec.loop_order)
        temporal = _default_temporal_loops(level, loops)
        spatial = _default_spatial_loops(level, loops)
        repeat_hint = _repeat_hint_from_tiling(level, spec, group, workload)
        replication_hint = _replication_hint_from_tiling(level, spec)
        overlap_hint = bool(getattr(spec, "rw_overlap", False))

        mode = "spat" if spatial else ("spat" if level == "pe" else "temp")

        tile = TileNode(
            id=f"tile::{group.group_id}::{level}",
            mode=mode,
            loops=loops,
            size=dict(spec.tile_size),
            order=list(spec.loop_order),
            attrs={
                "level": level,
                "temporal_loops": temporal,
                "spatial_loops": spatial,
                "buf_mode": getattr(spec, "buf_mode", "single"),
                "rw_overlap": overlap_hint,
                "repeat_hint": repeat_hint,
                "replication_hint": replication_hint,
                "acc_scope": getattr(tiling, "acc_scope", "sram"),
                "split_red": tuple(getattr(tiling, "split_red", ())),
            },
        )

        if head is None:
            head = tile
        if prev is not None:
            prev.set_kid(tile)
        prev = tile

    op_scope = _build_op_scope(group, workload)
    assert prev is not None
    prev.set_kid(op_scope)
    assert head is not None
    return head


def _build_op_scope(group: FusionGroup, workload: Optional[Any]) -> ScopeNode:
    scope = ScopeNode(
        id=f"ops::{group.group_id}",
        bind="seq",
        mem="pe",
        stage_kind="ops",
        repeat_hint=1,
        overlap_policy="none",
        resource_domain="pe",
        attrs={"group_id": group.group_id, "ops": tuple(group.ops)},
    )

    # 用 op 的 I/O 粗略填充 active tensor class
    read_tensors = set()
    update_tensors = set()
    for op_id in group.ops:
        op_node = _build_op_node(op_id, workload)
        scope.add_kid(op_node)
        read_tensors.update(op_node.ins)
        update_tensors.update(op_node.outs)

    scope.read_tensors = read_tensors
    scope.update_tensors = update_tensors
    scope.live_in = set(read_tensors)
    scope.live_out = set(update_tensors)
    return scope


# --------------------------------------------------
# Op nodes
# --------------------------------------------------

def _build_op_node(op_id: str, workload: Optional[Any]) -> OpNode:
    if workload is None:
        return OpNode(
            id=op_id,
            kind="unknown",
            ins=(),
            outs=(),
            attrs={},
        )

    op = workload.op(op_id) if hasattr(workload, "op") else None
    if op is None:
        return OpNode(
            id=op_id,
            kind="unknown",
            ins=(),
            outs=(),
            attrs={},
        )

    kind = str(getattr(op, "op_type", "unknown"))
    ins = tuple(str(x) for x in getattr(op, "inputs", ()))
    outs = tuple(str(x) for x in getattr(op, "outputs", ()))

    return OpNode(
        id=op_id,
        kind=kind,
        ins=ins,
        outs=outs,
        attrs={
            "op_type": kind,
            "iter_dims": tuple(str(x).strip().lower() for x in getattr(op, "iter_dims", ())),
            "reduce_dims": tuple(str(x).strip().lower() for x in getattr(op, "reduce_dims", ())),
            "tensor_dims": {
                str(k): tuple(str(x).strip().lower() for x in v)
                for k, v in getattr(op, "tensor_dims", {}).items()
            },
        },
    )


# --------------------------------------------------
# Flow semantic injection
# --------------------------------------------------

def _apply_flow_contract(
    scope: ScopeNode,
    contract: FlowContract,
    *,
    place: GroupDP,
    group_mem: str,
    repeat_hint: int,
    workload: Optional[Any],
) -> None:
    scope.need = set(contract.group_get)
    scope.keep = set(contract.reuse_out)
    scope.live_in = set(contract.group_get)
    scope.live_out = set(contract.reuse_out)

    scope.fill_tensors = set(contract.group_get)
    scope.read_tensors = set(contract.group_get) | set(contract.reuse_out) | set(contract.edge_tensors)
    scope.update_tensors = set(contract.reuse_out) | set(contract.hold_tensors)
    scope.wb_tensors = set(contract.group_out)

    # pinned/evict semantics (group-scope only):
    # - pinned: keep it alive in this scope, but DO NOT skip boundary load/store; we stay conservative.
    # - evict: ensure a store if written and not already emitted.
    pinned = _tensors_with_residence(place, group_mem, "pinned")
    evict = _tensors_with_residence(place, group_mem, "evict")

    scope.keep |= pinned

    scope.entry = [
        Move(
            act="load",
            tens=str(ten),
            src="dram",
            dst=group_mem,
            bytes=_tensor_size_bytes(workload, ten),
            role=_tensor_role(workload, ten),
            repeat_hint=repeat_hint,
            scope_id=scope.id,
        )
        for ten in contract.group_get
    ]

    scope.exit = [
        Move(
            act="writeback" if ten in set(contract.reuse_out) else "store",
            tens=str(ten),
            src=group_mem,
            dst="dram",
            bytes=_tensor_size_bytes(workload, ten),
            role=_tensor_role(workload, ten),
            repeat_hint=repeat_hint,
            scope_id=scope.id,
        )
        for ten in contract.group_out
    ]

    have_exit = {mv.tens for mv in scope.exit}
    for ten in sorted(evict):
        if ten in set(contract.group_out) and ten not in have_exit:
            scope.exit.append(
                Move(
                    act="store",
                    tens=str(ten),
                    src=group_mem,
                    dst="dram",
                    bytes=_tensor_size_bytes(workload, ten),
                    role=_tensor_role(workload, ten),
                    repeat_hint=repeat_hint,
                    scope_id=scope.id,
                )
            )

    scope.attrs["contract_edge_tensors"] = tuple(contract.edge_tensors)
    scope.attrs["contract_hold_tensors"] = tuple(contract.hold_tensors)
    scope.attrs["level_blocks"] = tuple(
        {
            "level": blk.level,
            "loops": tuple(blk.loops),
            "tile_size": dict(blk.tile_size),
            "temporal_loops": tuple(blk.temporal_loops),
            "spatial_loops": tuple(blk.spatial_loops),
            "repeat_hint": blk.repeat_hint,
            "replication_hint": blk.replication_hint,
            "overlap_hint": blk.overlap_hint,
        }
        for blk in contract.level_blocks
    )


def _apply_flow_bucket(
    scope: ScopeNode,
    bucket: FlowBucket,
    *,
    place: GroupDP,
    group_mem: str,
    repeat_hint: int,
    workload: Optional[Any],
) -> None:
    out = bucket.out

    scope.need = set(out.group_get)
    scope.keep = set(out.reuse_out)
    scope.live_in = set(out.group_get)
    scope.live_out = set(out.reuse_out)

    scope.fill_tensors = set(out.group_get)
    scope.read_tensors = set(out.group_get) | set(out.reuse_out)
    scope.update_tensors = set(out.reuse_out)
    scope.wb_tensors = set(out.group_out)

    # pinned/evict semantics (group-scope only)
    pinned = _tensors_with_residence(place, group_mem, "pinned")
    evict = _tensors_with_residence(place, group_mem, "evict")

    scope.keep |= pinned

    scope.entry = [
        Move(
            act="load",
            tens=str(ten),
            src="dram",
            dst=group_mem,
            bytes=_tensor_size_bytes(workload, ten),
            role=_tensor_role(workload, ten),
            repeat_hint=repeat_hint,
            scope_id=scope.id,
        )
        for ten in out.group_get
        if ten not in pinned
    ]

    scope.exit = [
        Move(
            act="writeback" if ten in set(out.reuse_out) else "store",
            tens=str(ten),
            src=group_mem,
            dst="dram",
            bytes=_tensor_size_bytes(workload, ten),
            role=_tensor_role(workload, ten),
            repeat_hint=repeat_hint,
            scope_id=scope.id,
        )
        for ten in out.group_out
        if ten not in pinned
    ]

    have_exit = {mv.tens for mv in scope.exit}
    for ten in sorted(evict):
        if ten in set(out.group_out) and ten not in have_exit and ten not in pinned:
            scope.exit.append(
                Move(
                    act="store",
                    tens=str(ten),
                    src=group_mem,
                    dst="dram",
                    bytes=_tensor_size_bytes(workload, ten),
                    role=_tensor_role(workload, ten),
                    repeat_hint=repeat_hint,
                    scope_id=scope.id,
                )
            )


def _tensors_with_residence(place: GroupDP, group_mem: str, want: str) -> set[str]:
    lv = str(group_mem).strip().lower()
    mode = str(want).strip().lower()
    out: set[str] = set()
    for node in place.nodes_at(lv):
        if str(getattr(node, "residence", "single")).strip().lower() != mode:
            continue
        for ten in node.tens:
            name = str(ten).strip()
            if name:
                out.add(name)
    return out


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _group_home_mem(place: GroupDP) -> str:
    levels = place.levels()
    return levels[-1] if levels else "dram"


def _pick_tiling(
    raw: Optional[Union[GroupTilingSpec, Sequence[GroupTilingSpec]]],
    gid: str,
) -> Optional[GroupTilingSpec]:
    del gid
    if raw is None:
        return None
    if isinstance(raw, GroupTilingSpec):
        return raw

    rows = tuple(raw)
    if not rows:
        return None

    rows = sorted(
        rows,
        key=lambda x: (
            len(getattr(x, "split_red", ())),
            getattr(x, "acc_scope", "sram") != "local",
            str(getattr(x, "tier_tiles", {})),
        ),
    )
    return rows[0]


def _pick_contract(rows: Sequence[FlowContract]) -> Optional[FlowContract]:
    if not rows:
        return None
    return rows[0]


def _pick_bucket(rows: Sequence[FlowBucket]) -> Optional[FlowBucket]:
    if not rows:
        return None
    return rows[0]


def _placement_map(src: Union[PlacementPlan, Mapping[str, GroupDP]]) -> Dict[str, GroupDP]:
    if isinstance(src, PlacementPlan):
        return src.group_map()
    return {str(k): v for k, v in dict(src).items()}


def _group_loop_extents(group: FusionGroup, workload: Optional[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if workload is None:
        return out

    for op_id in group.ops:
        op = workload.op(op_id) if hasattr(workload, "op") else None
        if op is None:
            continue

        for dim, val in getattr(op, "dim_constraints", {}).items():
            key = str(dim).strip().lower()
            out[key] = max(out.get(key, 1), int(val))

        for dim in getattr(op, "iter_dims", ()):
            key = str(dim).strip().lower()
            out.setdefault(key, 1)

    return out


def _default_temporal_loops(level: str, loops: Sequence[str]) -> Tuple[str, ...]:
    if str(level).strip().lower() == "pe":
        return ()
    return tuple(str(x).strip().lower() for x in loops if str(x).strip())


def _default_spatial_loops(level: str, loops: Sequence[str]) -> Tuple[str, ...]:
    if str(level).strip().lower() == "pe":
        return tuple(str(x).strip().lower() for x in loops if str(x).strip())
    return ()


def _repeat_hint_from_tiling(level: str, spec: Any, group: FusionGroup, workload: Optional[Any]) -> int:
    temporal_loops = _default_temporal_loops(level, spec.loop_order)
    loop_extents = _group_loop_extents(group, workload)
    repeat = 1
    for loop in temporal_loops:
        full = max(1, int(loop_extents.get(loop, 1)))
        part = max(1, int(spec.tile_size.get(loop, full)))
        repeat *= int(ceil(float(full) / float(part)))
    return max(1, repeat)


def _replication_hint_from_tiling(level: str, spec: Any) -> int:
    spatial_loops = _default_spatial_loops(level, spec.loop_order)
    repl = 1
    for loop in spatial_loops:
        repl *= max(1, int(spec.tile_size.get(loop, 1)))
    return max(1, repl)


def _group_repeat_hint(
    *,
    contract: Optional[FlowContract],
    tiling: GroupTilingSpec,
    group: FusionGroup,
    workload: Optional[Any],
) -> int:
    if contract is not None and contract.level_blocks:
        val = 1
        for blk in contract.level_blocks:
            val = max(val, int(blk.repeat_hint))
        return max(1, val)

    total = 1
    for level, spec in tiling.tier_tiles.items():
        total = max(total, _repeat_hint_from_tiling(str(level), spec, group, workload))
    return max(1, total)


def _group_overlap_policy(
    *,
    contract: Optional[FlowContract],
    tiling: GroupTilingSpec,
) -> str:
    if contract is not None:
        for blk in contract.level_blocks:
            if blk.overlap_hint:
                return "rw"

    for _, spec in tiling.tier_tiles.items():
        if bool(getattr(spec, "rw_overlap", False)):
            return "rw"
    return "none"


def _tensor_size_bytes(workload: Optional[Any], tensor_name: str) -> int:
    if workload is None:
        return 0
    try:
        return int(workload.tensor(tensor_name).size_bytes())
    except Exception:
        return 0


def _tensor_role(workload: Optional[Any], tensor_name: str) -> str:
    if workload is not None:
        try:
            tensor = workload.tensor(tensor_name)
            role = getattr(tensor, "role", None)
            if role is not None:
                value = getattr(role, "value", role)
                text = str(value).strip().upper()
                if text == "O":
                    return "output"
                if text in {"Q", "K", "V"}:
                    return "input"
        except Exception:
            pass

    text = str(tensor_name).strip().upper()
    if text == "O":
        return "output"
    if "CACHE" in text or "CTX" in text:
        return "state"
    if text in {"Q", "K", "V", "K_APPEND", "V_APPEND"}:
        return "input"
    return "temp"


__all__ = [
    "GroupChoice",
    "build_mapping_tree",
]
