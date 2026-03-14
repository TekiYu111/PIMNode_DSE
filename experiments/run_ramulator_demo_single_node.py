from __future__ import annotations
import shutil
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if os.path.basename(PROJECT_ROOT) != "PIMNode_DSE_o1":
    guessed_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(os.path.join(guessed_root, "pimnode_dse")):
        PROJECT_ROOT = guessed_root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hardware.arch_spec import arch_from_mapping
from pimnode_dse.workload.workload import build_attention_dag
from pimnode_dse.mapping.fusion_gene import FusionGene, OpFusionGroup, FusionStyle
from pimnode_dse.mapping.tilling_gene import TilingGene, GroupTilingSpec, MemTileSpec
from pimnode_dse.mapping.mapping_builder import MappingBuilder, BuildConfig
from pimnode_dse.analysis.analyzer import Analyzer
from pimnode_dse.analysis.evaluator import Evaluator
from pimnode_dse.placement.templates import build_core_templates
from dram_sim.dram_trace_generator import DRAMTraceGenerator, TraceGenConfig
from dram_sim.run_ramulator_sim import run_dram_simulation

DEFAULT_TEMPLATE = "Balanced"
DEFAULT_VARIANT = "MHA"
DEFAULT_PINOS_BINARY = "/app/pinos/pinos_release"
DEFAULT_NETWORK_CONFIG = "/app/pinos/examples/mesh11_lat"
DEFAULT_DRAM_CFG_TEMPLATE = "/app/pinos/dramcfg/dram.cfg"


def make_demo_components(
    seq_len: int,
    d_model: int,
    phase: str,
    *,
    batch_size: int,
    num_heads: int,
    variant: str,
    template_name: str,
):
    dag = build_attention_dag(
        name="demo_attn",
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        d_model=d_model,
        phase=phase,
        variant=variant,
    )

    op_ids = dag.topo_order()
    if not op_ids:
        raise RuntimeError("build_attention_dag() returned an empty DAG")

    group = OpFusionGroup(
        group_id="g0",
        op_names=op_ids,
        fusion_style=FusionStyle.SEQUENTIAL,
        phase=phase,
        special_role=None,
    )
    fusion = FusionGene(groups=[group])

    sram_tile = min(seq_len, 64)
    gspec = GroupTilingSpec(
        group_id="g0",
        tiles={
            "DRAM": MemTileSpec(
                mem_level="DRAM",
                tile_size={"m": seq_len, "n": seq_len},
                loop_order=["m", "n"],
                is_spatial=False,
            ),
            "SRAM": MemTileSpec(
                mem_level="SRAM",
                tile_size={"m": sram_tile, "n": sram_tile},
                loop_order=["m", "n"],
                is_spatial=False,
            ),
        },
        op_schedule_style="sequential",
        phase=phase,
        special_role=None,
    )
    tiling = TilingGene(group_tiles={"g0": gspec})

    builder = MappingBuilder(
        dag=dag,
        fusion=fusion,
        tiling=tiling,
        placement_plan=build_core_templates(),
        config=BuildConfig(
            default_template_name=template_name,
            outer_mem_level="DRAM",
            inner_mem_level="SRAM",
            include_storage_nodes=True,
            include_load_for_resident=False,
        ),
    )
    build_result = builder.build(phase=phase)
    return dag, build_result.tree, build_result



def make_hw(mapping_policy: str):
    return arch_from_mapping(
        {
            "dram": {
                "capacity_bytes": 2 * 1024 * 1024 * 1024,
                "bank_count": 1,
                "bandwidth_bpc": 32.0,
                "address_mapping_mode": mapping_policy,
                "mapping_descriptor": {},
            },
            "sram": {
                "capacity_bytes": 512 * 1024,
                "bandwidth_bpc": 64.0,
                "bank_count": 1,
                "read_write_shared_bandwidth": True,
            },
            "de": {
                "bus_bandwidth_bpc": 64.0,
                "supports_pipeline": True,
                "max_inflight_tiles": 2,
                "supports_direct_dram_to_pe": False,
                "supports_direct_pe_to_sram_writeback": True,
            },
            "pe": {
                "rows": 16,
                "cols": 16,
                "macs_per_pe_per_cycle": 1,
                "input_bandwidth_bpc": 64.0,
                "output_bandwidth_bpc": 32.0,
                "supports_accumulate_in_place": True,
            },
            "clock_hz": 1e9,
            "bytes_per_element_default": 2,
            "extra": {},
        }
    )



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--phase", type=str, default="prefill")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--variant", type=str, default=DEFAULT_VARIANT)
    ap.add_argument("--template-name", type=str, default=DEFAULT_TEMPLATE)
    ap.add_argument("--mapping-policy", type=str, default="ChRaBaRoCo")
    ap.add_argument("--address-radix", type=str, default="dec", choices=["dec", "hex"])
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(PROJECT_ROOT) / "experiments" / "out_trace_demo"),
    )
    ap.add_argument("--pinos-binary", type=str, default=DEFAULT_PINOS_BINARY)
    ap.add_argument("--network-config", type=str, default=DEFAULT_NETWORK_CONFIG)
    ap.add_argument("--pinos-dram-cfg-template", type=str, default=DEFAULT_DRAM_CFG_TEMPLATE)
    ap.add_argument("--number-cores", type=int, default=1)
    ap.add_argument("--banks", type=int, default=1)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--ranks", type=int, default=1)
    return ap.parse_args()



def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dag, tree, build_result = make_demo_components(
        seq_len=args.seq_len,
        d_model=args.d_model,
        phase=args.phase,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        variant=args.variant,
        template_name=args.template_name,
    )

    hw = make_hw(args.mapping_policy)

    analyzer = Analyzer()
    analyzer_summary = analyzer.analyze(
        candidate_id="demo_candidate",
        tree=tree,
        hardware_spec=hw,
        workload=dag,
    )

    evaluator = Evaluator()
    fast_result = evaluator.evaluate_fast(
        candidate_id="demo_candidate",
        analyzer_summary=analyzer_summary,
    )

    print("[OK] builder finished")
    print(f"  template_name    = {args.template_name}")
    print(f"  placement_select = {build_result.report.placement_selection}")
    print("[OK] analyzer finished")
    print(f"  total_cycles_est = {analyzer_summary.total_cycles_est}")
    print(f"  dram_bytes       = {analyzer_summary.dram_bytes}")
    print(f"  energy_est       = {analyzer_summary.energy_est}")
    print(f"  fast_score       = {fast_result.score_fast}")

    trace_cfg = TraceGenConfig(
        trace_style="single_file_zsim",
        address_radix=args.address_radix,
        num_cores=args.number_cores,
        default_core_id=0,
        mapping_policy=args.mapping_policy,
        addr_alignment=64,
        burst_bytes=64,
        local_bank_id=0,
        default_channel_id=0,
        default_rank_id=0,
        allow_remote_bank=False,
    )

    generator = DRAMTraceGenerator(trace_cfg)
    requests = generator.generate_requests(
        tree=tree,
        workload=dag,
        analyzer_summary=analyzer_summary,
    )

    trace_path = out_dir / "demo_trace.out"
    trace_core0_path = out_dir / "demo_trace.out.0"
    csv_path = out_dir / "demo_trace_requests.csv"

    generator.write_trace(requests=requests, trace_path=str(trace_path))
    shutil.copyfile(trace_path, trace_core0_path)

    generator.export_requests_csv(requests=requests, csv_path=str(csv_path))


    print(f"[OK] request count   = {len(requests)}")
    print(f"[OK] trace written to: {trace_path}")
    print(f"[OK] csv written to:   {csv_path}")

    if not os.path.isfile(args.pinos_binary):
        print(f"[WARN] pinos binary not found: {args.pinos_binary}")
        return 0
    if not os.path.isfile(args.network_config):
        print(f"[WARN] network config not found: {args.network_config}")
        return 0
    if not os.path.isfile(args.pinos_dram_cfg_template):
        print(f"[WARN] dram cfg template not found: {args.pinos_dram_cfg_template}")
        return 0

    sim_out_dir = out_dir / "pinos_run"
    sim_out_dir.mkdir(parents=True, exist_ok=True)

    sim_result = run_dram_simulation(
        trace_file=str(trace_path),
        output_dir=str(sim_out_dir),
        pinos_binary=args.pinos_binary,
        network_config=args.network_config,
        dram_cfg_template=args.pinos_dram_cfg_template,
        mapping_policy=args.mapping_policy,
        number_cores=args.number_cores,
        banks=args.banks,
        channels=args.channels,
        ranks=args.ranks,
    )

    print("[OK] pinos run finished")
    print(f"  success          = {sim_result.success}")
    print(f"  returncode       = {sim_result.returncode}")
    print(f"  dram_cycles      = {sim_result.dram_cycles}")
    print(f"  total_read_req   = {sim_result.total_read_req}")
    print(f"  total_write_req  = {sim_result.total_write_req}")
    print(f"  bank_access      = {sim_result.bank_access}")
    print(f"  average_ipc      = {sim_result.average_ipc}")
    print(f"  total_inst       = {sim_result.total_instructions}")
    print(f"  noc_hops_avg     = {sim_result.noc_hops_avg}")
    print(f"  patched_cfg      = {sim_result.cfg_path}")
    print(f"  stdout_log       = {sim_result.stdout_path}")
    print(f"  stderr_log       = {sim_result.stderr_path}")

    final_result = evaluator.evaluate_final(
        candidate_id="demo_candidate",
        analyzer_summary=analyzer_summary,
        ramulator_summary={
            "dram_cycles": sim_result.dram_cycles,
            "dram_bytes": analyzer_summary.dram_bytes,
            "dram_energy": 0.0,
        },
    )

    print("[OK] final evaluator finished")
    print(f"  total_latency    = {final_result.total_latency}")
    print(f"  final_score      = {final_result.score_final}")
    print(f"  edp              = {final_result.edp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
