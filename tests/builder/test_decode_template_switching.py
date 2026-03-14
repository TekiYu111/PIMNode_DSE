from pimnode_dse.mapping.mapping_builder import MappingBuilder
from pimnode_dse.mapping.mapping_tree import TileNode, StorageNode, ActionNode
from pimnode_dse.mapping.fusion_gene import FusionGene, OpFusionGroup, FusionStyle
from pimnode_dse.mapping.tilling_gene import TilingGene, GroupTilingSpec, MemTileSpec


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

def _make_decode_fusion():
    return FusionGene(
        gene_id="fusion_decode_two_groups",
        groups=[
            OpFusionGroup(
                group_id="g0",
                op_names=["Op_K_Append", "Op_V_Append"],
                fusion_style=FusionStyle.SEQUENTIAL,
                phase="decode",
                special_role="KV_CACHE",
            ),
            OpFusionGroup(
                group_id="g1",
                op_names=["Op_QK", "Op_Softmax", "Op_AV", "Op_O"],
                fusion_style=FusionStyle.SEQUENTIAL,
                phase="decode",
                special_role="KV_CACHE",
            ),
        ],
    )


def _make_decode_tiling():
    tg = TilingGene()
    tg.add_group_spec(
        GroupTilingSpec(
            group_id="g0",
            tiles={
                "DRAM": MemTileSpec(
                    mem_level="DRAM",
                    fixed_tile_size={"Sq": 1, "Skv": 128, "Dh": 64},
                    fixed_loop_order=["Skv", "Dh"],
                    is_spatial=False,
                ),
                "SRAM": MemTileSpec(
                    mem_level="SRAM",
                    fixed_tile_size={"Sq": 1, "Skv": 64, "Dh": 64},
                    fixed_loop_order=["Skv", "Dh"],
                    is_spatial=False,
                ),
            },
            phase="decode",
            special_role="KV_CACHE",
        )
    )
    return tg


def _build_tree(
    *,
    dag,
    fusion,
    tiling,
    placement_templates,
    builder_config,
    template_name: str,
    phase: str = "decode",
    print_tree: bool = False,
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
    if print_tree:
        print(f"\n=== DECODE TEMPLATE: {template_name} ===")
        result.tree.display()
    return result.tree, result.report


def _has_kv_cache_signal(report):
    bindings = report.role_bindings.get("g0", {})
    return "KV_CACHE" in bindings and len(bindings["KV_CACHE"]) > 0


def _find_bad_kv_actions(tree):
    bad = []
    for a in _collect_actions(tree):
        role = a.attrs.get("role")
        if role == "KV_CACHE" and a.action_type in {"WRITEBACK", "WRITEBACK_DRAM", "EVICT"}:
            bad.append(a)
    return bad


def test_decode_templates_build_successfully(
    decode_dag,
    placement_templates,
    builder_config,
):
    fusion = _make_decode_fusion()
    tiling = _make_decode_tiling()

    template_names = ["Balanced", "MaxReuse", "MinReuse", "K-D Fusion"]

    for name in template_names:
        tree, report = _build_tree(
            dag=decode_dag,
            fusion=fusion,
            tiling=tiling,
            placement_templates=placement_templates,
            builder_config=builder_config,
            template_name=name,
            phase="decode",
        )

        assert report.placement_selection["g0"] == name
        assert len(_collect_tiles(tree, "DRAM")) >= 1
        assert len(_collect_tiles(tree, "SRAM")) >= 1
        assert _has_kv_cache_signal(report)


def test_decode_template_switching_changes_tree(
    decode_dag,
    placement_templates,
    builder_config,
):
    fusion = _make_decode_fusion()
    tiling = _make_decode_tiling()

    tree_maxreuse, report_maxreuse = _build_tree(
        dag=decode_dag,
        fusion=fusion,
        tiling=tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="MaxReuse",
        phase="decode",
        print_tree=True,
    )

    tree_kdfusion, report_kdfusion = _build_tree(
        dag=decode_dag,
        fusion=fusion,
        tiling=tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="K-D Fusion",
        phase="decode",
        print_tree=True,
    )

    sig_maxreuse = _tree_signature(tree_maxreuse)
    sig_kdfusion = _tree_signature(tree_kdfusion)

    assert report_maxreuse.placement_selection["g0"] == "MaxReuse"
    assert report_kdfusion.placement_selection["g0"] == "K-D Fusion"

    assert sig_maxreuse != sig_kdfusion
    assert sig_maxreuse["sram_tiles"] != sig_kdfusion["sram_tiles"]


def test_decode_kv_cache_not_written_back_or_evicted(
    decode_dag,
    placement_templates,
    builder_config,
):
    fusion = _make_decode_fusion()
    tiling = _make_decode_tiling()

    for template_name in ["Balanced", "MaxReuse", "MinReuse", "K-D Fusion"]:
        tree, report = _build_tree(
            dag=decode_dag,
            fusion=fusion,
            tiling=tiling,
            placement_templates=placement_templates,
            builder_config=builder_config,
            template_name=template_name,
            phase="decode",
        )

        assert _has_kv_cache_signal(report)

        bad = _find_bad_kv_actions(tree)
        assert not bad, f"{template_name} produced illegal KV_CACHE actions: {bad}"


def test_decode_kd_fusion_prefers_kv_side_residency(
    decode_dag,
    placement_templates,
    builder_config,
):
    fusion = _make_decode_fusion()
    tiling = _make_decode_tiling()

    tree, report = _build_tree(
        dag=decode_dag,
        fusion=fusion,
        tiling=tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="K-D Fusion",
        phase="decode",
        print_tree=True,
    )

    sram_tiles = _collect_tiles(tree, "SRAM")
    assert sram_tiles, "No SRAM tile found in decode K-D Fusion tree"

    # 只检查至少一个 SRAM tile 呈现 KV 偏向
    found_k_or_v = False
    for tile in sram_tiles:
        resident = set(tile.resident_tensors())
        if ("K" in resident) or ("V" in resident):
            found_k_or_v = True
            break

    assert found_k_or_v, "Expected K-D Fusion decode tree to keep K or V resident in SRAM"


def test_decode_templates_produce_nontrivial_action_or_storage_difference(
    decode_dag,
    placement_templates,
    builder_config,
):
    fusion = _make_decode_fusion()
    tiling = _make_decode_tiling()

    tree_balanced, _ = _build_tree(
        dag=decode_dag,
        fusion=fusion,
        tiling=tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="Balanced",
        phase="decode",
    )

    tree_minreuse, _ = _build_tree(
        dag=decode_dag,
        fusion=fusion,
        tiling=tiling,
        placement_templates=placement_templates,
        builder_config=builder_config,
        template_name="MinReuse",
        phase="decode",
    )

    storage_balanced = sorted(
        (s.mem_level, s.tensor, s.keep, s.anchor)
        for s in _collect_storage(tree_balanced)
    )
    storage_minreuse = sorted(
        (s.mem_level, s.tensor, s.keep, s.anchor)
        for s in _collect_storage(tree_minreuse)
    )

    actions_balanced = sorted(
        (a.action_type, tuple(a.tensors), a.src_level, a.dst_level, a.anchor)
        for a in _collect_actions(tree_balanced)
    )
    actions_minreuse = sorted(
        (a.action_type, tuple(a.tensors), a.src_level, a.dst_level, a.anchor)
        for a in _collect_actions(tree_minreuse)
    )

    assert (storage_balanced != storage_minreuse) or (actions_balanced != actions_minreuse)
