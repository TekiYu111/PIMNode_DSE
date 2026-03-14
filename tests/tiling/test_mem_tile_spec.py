from pimnode_dse.mapping.tilling_gene import MemTileSpec


def test_mem_tile_spec_basic():
    spec = MemTileSpec(
        mem_level="SRAM",
        fixed_tile_size={"Sq": 64, "Skv": 128},
        fixed_loop_order=["Sq", "Skv"],
        is_spatial=False,
    )
    tile_size, loop_order = spec.get_tile_spec()
    assert tile_size == {"Sq": 64, "Skv": 128}
    assert loop_order == ["Sq", "Skv"]
    assert spec.tile_size == {"Sq": 64, "Skv": 128}
    assert spec.loop_order == ["Sq", "Skv"]


def test_mem_tile_spec_default_loop_order():
    spec = MemTileSpec(
        mem_level="SRAM",
        fixed_tile_size={"Sq": 64, "Skv": 128},
    )
    tile_size, loop_order = spec.get_tile_spec()
    assert tile_size == {"Sq": 64, "Skv": 128}
    assert loop_order == ["Sq", "Skv"]


def test_mem_tile_spec_clone():
    spec = MemTileSpec(
        mem_level="DRAM",
        fixed_tile_size={"Sq": 128},
        fixed_loop_order=["Sq"],
    )
    cloned = spec.clone()
    assert cloned.mem_level == spec.mem_level
    assert cloned.tile_size == spec.tile_size
    assert cloned.loop_order == spec.loop_order
    assert cloned is not spec
