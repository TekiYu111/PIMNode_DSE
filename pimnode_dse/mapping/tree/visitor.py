from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from pimnode_dse.mapping.tree.mapping_tree import MappingTree, Move, OpNode, ScopeNode, TileNode, walk


class Visitor:
    def visit_scope(self, node: ScopeNode) -> None:
        return

    def visit_tile(self, node: TileNode) -> None:
        return

    def visit_op(self, node: OpNode) -> None:
        return

    def run(self, root: ScopeNode) -> None:
        for node in walk(root):
            if isinstance(node, ScopeNode):
                self.visit_scope(node)
            elif isinstance(node, TileNode):
                self.visit_tile(node)
            elif isinstance(node, OpNode):
                self.visit_op(node)


@dataclass
class TreeStats:
    scopes: int = 0
    tiles: int = 0
    ops: int = 0
    entry_moves: int = 0
    exit_moves: int = 0
    levels: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.levels is None:
            self.levels = {}


class StatsVisitor(Visitor):
    def __init__(self) -> None:
        self.stats = TreeStats()

    def visit_scope(self, node: ScopeNode) -> None:
        self.stats.scopes += 1
        self.stats.entry_moves += len(node.entry)
        self.stats.exit_moves += len(node.exit)

    def visit_tile(self, node: TileNode) -> None:
        self.stats.tiles += 1
        level = str(node.attrs.get("level", "unknown")).strip().lower()
        self.stats.levels[level] = self.stats.levels.get(level, 0) + 1

    def visit_op(self, node: OpNode) -> None:
        self.stats.ops += 1


class TensorMoveVisitor(Visitor):
    def __init__(self) -> None:
        self.entry: list[Move] = []
        self.exit: list[Move] = []

    def visit_scope(self, node: ScopeNode) -> None:
        self.entry.extend(node.entry)
        self.exit.extend(node.exit)


class ValidateMovesVisitor(Visitor):
    def visit_scope(self, node: ScopeNode) -> None:
        for mv in list(node.entry) + list(node.exit):
            if not mv.tens:
                raise ValueError(f"empty tensor in move under scope {node.id}")
            if not mv.src or not mv.dst:
                raise ValueError(f"empty src/dst in move under scope {node.id}")


def collect_stats(tree: MappingTree) -> TreeStats:
    vis = StatsVisitor()
    vis.run(tree.root)
    return vis.stats


def collect_moves(tree: MappingTree) -> tuple[tuple[Move, ...], tuple[Move, ...]]:
    vis = TensorMoveVisitor()
    vis.run(tree.root)
    return tuple(vis.entry), tuple(vis.exit)


def validate_moves(tree: MappingTree) -> None:
    vis = ValidateMovesVisitor()
    vis.run(tree.root)


__all__ = [
    "Visitor",
    "TreeStats",
    "StatsVisitor",
    "TensorMoveVisitor",
    "ValidateMovesVisitor",
    "collect_stats",
    "collect_moves",
    "validate_moves",
]
