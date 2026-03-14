from __future__ import annotations

import argparse
from pathlib import Path

from pimnode_dse.dse_search.dse_framework import run_dse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a first-pass DSE pipeline.")
    parser.add_argument("--design-space", type=Path, default=Path(__file__).with_name("design_space.yaml"))
    parser.add_argument("--max-hw", type=int, default=1)
    parser.add_argument("--max-fusion", type=int, default=4)
    parser.add_argument("--max-placement", type=int, default=4)
    parser.add_argument("--max-tiling", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--enable-bandwidth-upper-bound", action="store_true")
    args = parser.parse_args()

    result = run_dse(
        args.design_space,
        max_hw=args.max_hw,
        max_fusion=args.max_fusion,
        max_placement=args.max_placement,
        max_tiling=args.max_tiling,
        topk=args.topk,
        enable_bandwidth_upper_bound=args.enable_bandwidth_upper_bound,
    )

    print("===== DSE Summary =====")
    print(f"workload: {result['workload_name']}")
    print(f"tasks: {result['task_count']}")
    print(f"stage_a_runs: {result['stage_a_runs']}")
    print(f"evaluated_candidates: {result['evaluated_count']}")
    print(f"bandwidth_bound_enabled: {result['bandwidth_bound_enabled']}")

    print("\n===== Top Candidates =====")
    for rank, item in enumerate(result["top_candidates"], start=1):
        cand = item.candidate
        tile_sram = cand.tiling_spec.tiles.get("SRAM")
        tile_pe = cand.tiling_spec.tiles.get("PE")
        sram_desc = tile_sram.tile_size if tile_sram is not None else {}
        pe_desc = tile_pe.tile_size if tile_pe is not None else {}
        print(
            f"#{rank} score={item.score:.4f} cycles={item.estimated_cycles:.1f} "
            f"hw={cand.hardware_index} fusion={cand.fusion_index} placement={cand.placement_index} "
            f"tiling={cand.tiling_index} template={cand.template_name} group={cand.group_id} "
            f"SRAM={sram_desc} PE={pe_desc}"
        )


if __name__ == "__main__":
    main()
