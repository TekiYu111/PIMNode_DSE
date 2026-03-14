# tests/helpers.py

from pimnode_dse.mapping.mapping_tree import ScopeNode, TileNode, LoopNode, StorageNode, ActionNode, OpNode


def collect_nodes(tree, node_cls):
    return [n for n in tree.walk() if isinstance(n, node_cls)]


def collect_tiles(tree, mem_level=None):
    tiles = collect_nodes(tree, TileNode)
    if mem_level is None:
        return tiles
    return [t for t in tiles if t.mem_level == mem_level]


def collect_ops(tree):
    return collect_nodes(tree, OpNode)


def collect_actions(tree):
    return collect_nodes(tree, ActionNode)


def collect_storage(tree):
    return collect_nodes(tree, StorageNode)


def collect_loops(tree):
    return collect_nodes(tree, LoopNode)


def first_group_scope(tree):
    groups = [
        n for n in tree.walk()
        if isinstance(n, ScopeNode) and n.name.startswith("Group(")
    ]
    assert groups, "No Group scope found"
    return groups[0]


def first_tile_under(node, mem_level):
    for child in node.children:
        if isinstance(child, TileNode) and child.mem_level == mem_level:
            return child
    raise AssertionError(f"No {mem_level} tile found under {node.name}")
