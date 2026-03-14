from pimnode_dse.mapping.tilling_gene import TilingGene, GroupTilingSpec, MemTileSpec


def test_tiling_gene_get_group_spec(basic_tiling):
    spec = basic_tiling.get_group_spec("g0")
    assert spec.group_id == "g0"
    assert spec.tiles["DRAM"].tile_size["Sq"] == 128
    assert spec.tiles["SRAM"].tile_size["Sq"] == 64


def test_tiling_gene_phase_role_overlay():
    tg = TilingGene()
    tg.add_group_spec(
        GroupTilingSpec(
            group_id="g0",
            tiles={
                "DRAM": MemTileSpec("DRAM", {"Sq": 128}, ["Sq"]),
                "SRAM": MemTileSpec("SRAM", {"Sq": 64}, ["Sq"]),
            },
            phase="prefill",
        )
    )
    tg.add_phase_role_mapping(
        "g0",
        "decode",
        "KV_CACHE",
        {
            "SRAM": MemTileSpec("SRAM", {"Sq": 1, "Skv": 256}, ["Skv", "Sq"]),
        },
    )

    base = tg.get_group_spec("g0")
    override = tg.get_group_spec("g0", phase="decode", role="KV_CACHE")

    assert base.tiles["SRAM"].tile_size == {"Sq": 64}
    assert override.tiles["SRAM"].tile_size == {"Sq": 1, "Skv": 256}
    assert override.phase == "decode"
    assert override.special_role == "KV_CACHE"
