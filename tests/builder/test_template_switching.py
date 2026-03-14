from pimnode_dse.mapping.mapping_builder import MappingBuilder
from pimnode_dse.mapping.mapping_tree import TileNode, StorageNode, ActionNode


def _collect_tiles(tree, mem_level=None):
    tiles = [n for n in tree.walk() if isinstance(n, TileNode)]
    if mem_level is None:
        return tiles
    return [t for t in tiles if t.mem_level == mem_level]


def _collect_storage(tree):
    return [n for n in tree.walk() if isinstance(n, StorageNode)]


def _collect_actions(tree):
    return [n for n in tree.walk() if isinstance(n, ActionNode)]


def _tile_signature(tile: TileNode):
    return {
        "mem_level": tile.mem_level,
        "tile_size": dict(tile.tile_size or {}),
        "loop_order": list(tile.loop_order or []),
        "resident": tuple(tile.resident_tensors()),
        "prefetch": tuple(tile.prefetch_tensors()),
        "writeback": tuple(tile.writeback_tensors()),
        "evict": tuple(tile.evict_tensors()),
    }


def _tree_signature(tree):
    dram_tiles = _collect_tiles(tree, "DRAM")
    sram_tiles = _collect_tiles(tree, "SRAM")

    storage_sig = sorted(
        (s.mem_level, s.tensor, s.keep, s.anchor)
        for s in _collect_storage(tree)
    )

    action_sig = sorted(
        (
            a.action_type,
            tuple(a.tensors),
            a.src_level,
            a.dst_level,
            a.anchor,
        )
        for a in _collect_actions(tree)
    )

    return {
        "dram_tiles": [_tile_signature(t) for t in dram_tiles],
        "sram_tiles": [_tile_signature(t) for t in sram_tiles],
        "storage": storage_sig,
        "actions": action_sig,
    }


def _build_tree(
    *,
    dag,
    fusion,
    tiling,
    placement_templates,
    builder_config,
    template_name: str,
    phase: str = "prefill",
):
    builder = MappingBuilder(
        dag=dag,
        fusion=fusion,
        tiling=tiling,
        placement_plan=placement_templates,
        selected_templates={"g0": template_name},
        config=builder_config,
    )
    result = builder.build(phase=phase)
    print(f"\n=== TEMPLATE: {template_name} ===")
    result.tree.display()
    return result.tree, result.report


def test_template_switching_changes_tree(
    prefill_dag,
    single_group_fusion,
    basic_tiling,
    placement_templates,
    builder_config,
):
    tree_balanced, report_balanced = _build_tree(
        dag=prefill_dag,
        fusion=single_group_fusion,
        tiling=basic_tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="Balanced",
        phase="prefill",
    )

    tree_maxreuse, report_maxreuse = _build_tree(
        dag=prefill_dag,
        fusion=single_group_fusion,
        tiling=basic_tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="MaxReuse",
        phase="prefill",
    )

    sig_balanced = _tree_signature(tree_balanced)
    sig_maxreuse = _tree_signature(tree_maxreuse)

    # 至少树签名之一发生变化
    assert sig_balanced != sig_maxreuse

    # template 选择记录不同
    assert report_balanced.placement_selection["g0"] == "Balanced"
    assert report_maxreuse.placement_selection["g0"] == "MaxReuse"

    # SRAM tile placement 至少有差异
    assert sig_balanced["sram_tiles"] != sig_maxreuse["sram_tiles"]


def test_all_core_templates_build_successfully(
    prefill_dag,
    single_group_fusion,
    basic_tiling,
    placement_templates,
    builder_config,
):
    template_names = ["Balanced", "MaxReuse", "MinReuse", "K-D Fusion"]

    built = {}
    for name in template_names:
        tree, report = _build_tree(
            dag=prefill_dag,
            fusion=single_group_fusion,
            tiling=basic_tiling,
            placement_templates=placement_templates,
            builder_config=builder_config,
            template_name=name,
            phase="prefill",
        )
        built[name] = (_tree_signature(tree), report)

    for name in template_names:
        sig, report = built[name]
        assert report.placement_selection["g0"] == name
        assert len(sig["sram_tiles"]) >= 1
        assert len(sig["dram_tiles"]) >= 1

    # 不要求每两个模板都不同，但至少应该有不止一种树签名
    unique_signatures = {
        repr(built[name][0]) for name in template_names
    }
    assert len(unique_signatures) >= 2


def test_template_switching_changes_storage_or_actions(
    prefill_dag,
    single_group_fusion,
    basic_tiling,
    placement_templates,
    builder_config,
):
    tree_minreuse, _ = _build_tree(
        dag=prefill_dag,
        fusion=single_group_fusion,
        tiling=basic_tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="MinReuse",
        phase="prefill",
    )

    tree_kdfusion, _ = _build_tree(
        dag=prefill_dag,
        fusion=single_group_fusion,
        tiling=basic_tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="K-D Fusion",
        phase="prefill",
    )

    storage_minreuse = sorted(
        (s.mem_level, s.tensor, s.keep, s.anchor)
        for s in _collect_storage(tree_minreuse)
    )
    storage_kdfusion = sorted(
        (s.mem_level, s.tensor, s.keep, s.anchor)
        for s in _collect_storage(tree_kdfusion)
    )

    actions_minreuse = sorted(
        (a.action_type, tuple(a.tensors), a.src_level, a.dst_level, a.anchor)
        for a in _collect_actions(tree_minreuse)
    )
    actions_kdfusion = sorted(
        (a.action_type, tuple(a.tensors), a.src_level, a.dst_level, a.anchor)
        for a in _collect_actions(tree_kdfusion)
    )

    assert (storage_minreuse != storage_kdfusion) or (actions_minreuse != actions_kdfusion)
