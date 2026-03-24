from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pimnode_dse.mapping.placement.node import GroupDP
from pimnode_dse.mapping.placement.placement import GroupCtx
from pimnode_dse.mapping.placement.rules import dp_key, legal_dp


@dataclass(frozen=True)
class DpStat:
    nodes: int
    levels: int
    sig: int

    def as_dict(self) -> dict[str, int]:
        return {
            "nodes": self.nodes,
            "levels": self.levels,
            "sig": self.sig,
        }


# -----------------------------
# public
# -----------------------------

def stat_dp(dp: GroupDP) -> DpStat:
    return DpStat(
        nodes=len(dp.nodes),
        levels=len({node.level for node in dp.nodes}),
        sig=len(dp_key(dp)),
    )



def prune_bad(dps: Sequence[GroupDP], ctx: GroupCtx) -> tuple[GroupDP, ...]:
    return tuple(dp for dp in dps if legal_dp(dp, ctx))



def prune_dup(dps: Sequence[GroupDP]) -> tuple[GroupDP, ...]:
    keep: dict[tuple[tuple[object, ...], ...], GroupDP] = {}
    for dp in dps:
        key = dp_key(dp)
        if key not in keep:
            keep[key] = dp
    return tuple(keep.values())



def prune_eq(dps: Sequence[GroupDP]) -> tuple[GroupDP, ...]:
    """Drop structurally equivalent candidates after level/slot normalization.

    This is not heuristic pruning. Two candidates are equivalent only when they
    induce the same normalized storage structure.
    """
    keep: dict[tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...], GroupDP] = {}
    for dp in dps:
        key = _eq_key(dp)
        if key not in keep:
            keep[key] = dp
    return tuple(keep.values())



def prune_dom(dps: Sequence[GroupDP], ctx: GroupCtx | None = None) -> tuple[GroupDP, ...]:
    """No dominance pruning without a proven dominance relation.

    TCM prunes with proofs derived from dataplacement/dataflow/tile-shape
    structure. Until such proofs exist here, this step must stay lossless.
    """
    return tuple(dps)



def sort_dp(dps: Sequence[GroupDP]) -> tuple[GroupDP, ...]:
    return tuple(sorted(dps, key=_sort_key))



def prune_dp(dps: Sequence[GroupDP], ctx: GroupCtx, top: int | None = None) -> tuple[GroupDP, ...]:
    out = tuple(dps)
    out = prune_bad(out, ctx)
    out = prune_dup(out)
    out = prune_eq(out)
    out = prune_dom(out, ctx)
    out = sort_dp(out)
    if top is not None:
        out = out[: max(0, top)]
    return out



def explain_score(dp: GroupDP, ctx: GroupCtx | None = None) -> dict[str, int]:
    return stat_dp(dp).as_dict()


# -----------------------------
# helpers
# -----------------------------

def _eq_key(dp: GroupDP) -> tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (node.level, node.kind, node.tensors, node.axis)
        for node in dp.nodes
    )



def _sort_key(dp: GroupDP) -> tuple[object, ...]:
    stat = stat_dp(dp)
    level_rank = {"pe": 0, "sram": 1, "dram": 2}
    kind_rank = {"acc": 0, "state": 1, "hold": 2, "flow": 3}
    node_sig = tuple(
        (
            level_rank.get(node.level, 9),
            kind_rank.get(node.kind, 9),
            node.slot,
            node.tensors,
            node.axis,
            node.id,
        )
        for node in dp.nodes
    )
    return (stat.nodes, stat.levels, node_sig, dp.group)
