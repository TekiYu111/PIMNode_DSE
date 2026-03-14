# run_design_space_demo.py
from pathlib import Path
from design_space import load_design_space, dump_example_expansion

def main():
    # 1. 读取 design_space.yaml
    yaml_path = Path("design_space.yaml")
    ds = load_design_space(yaml_path)

    # 2. 展开硬件空间（DRAM configs + SRAM/PE独立值）
    hw_candidates = ds.expand_hardware()
    print(f"Hardware candidates (showing first 3): {hw_candidates[:3]}")

    # 3. 展开 fusion / placement / tiling 的独立参数组合
    fusion_candidates = ds.expand_independent_section("fusion")
    placement_candidates = ds.expand_independent_section("placement")
    tiling_candidates = ds.expand_independent_section("tiling")

    print(f"Fusion candidates (first 3): {fusion_candidates[:3]}")
    print(f"Placement candidates (first 3): {placement_candidates[:3]}")
    print(f"Tiling candidates (first 3): {tiling_candidates[:3]}")

    # 4. 快速调试：打印组合空间的示例
    preview = dump_example_expansion(ds, max_items=2)
    print("Preview of combined expanded space:")
    for section, items in preview.items():
        print(f"{section}: {items}")

if __name__ == "__main__":
    main()
