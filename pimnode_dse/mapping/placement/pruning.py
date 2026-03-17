# placement/pruning.py

from typing import List
from pimnode_dse.placement.placement_ir import PlacementPlan, PlacementScope, ResidentSet
from pimnode_dse.placement.rules import derive_actions
from copy import deepcopy

# -------------------------------
# 可行性剪枝
# -------------------------------
def feasibility_prune(plan: PlacementPlan, sram_capacity: int, dram_bank_count: int, phase: str) -> PlacementPlan:
    """
    Remove templates that violate hard memory constraints:
    - SRAM resident tensor count <= sram_capacity
    - DRAM parallel access <= dram_bank_count
    - Phase-specific constraints can be applied
    Returns a new PlacementPlan with feasible scopes only
    """
    new_plan = PlacementPlan(scopes={})
    for scope_name, scope in plan.scopes.items():
        feasible = True

        # SRAM容量检查
        for rs in scope.resident_sets:
            if rs.mem == "SRAM" and len(rs.tensors) > sram_capacity:
                feasible = False
                break

        # DRAM bank 检查（粗略：如果 resident tensor 数超过 bank 数）
        if feasible:
            for rs in scope.resident_sets:
                if rs.mem == "DRAM" and len(rs.tensors) > dram_bank_count:
                    feasible = False
                    break

        # 可加入 phase-specific constraints
        # (这里保留接口，可按需要扩展)

        if feasible:
            new_plan.add_scope(deepcopy(scope))

    return new_plan

# -------------------------------
# 支配剪枝
# -------------------------------
def dominance_prune(plans: List[PlacementPlan], compute_metric_fn) -> List[PlacementPlan]:
    """
    Remove dominated PlacementPlan templates.
    - compute_metric_fn: function(plan) -> dict with keys:
        'data_movement_bytes', 'dram_access_bytes', 'compute_cycles', 'energy'
    """
    survivors = []
    metrics_cache = {}

    # 先计算每个模板指标
    for plan in plans:
        metrics_cache[id(plan)] = compute_metric_fn(plan)

    for i, plan_i in enumerate(plans):
        dominated = False
        for j, plan_j in enumerate(plans):
            if i == j:
                continue
            mi = metrics_cache[id(plan_i)]
            mj = metrics_cache[id(plan_j)]
            # 支配条件示例：数据搬运量和 compute cycles
            if (mi['data_movement_bytes'] >= mj['data_movement_bytes'] and
                mi['compute_cycles'] >= mj['compute_cycles']):
                # 再加 DRAM访问次数和能耗判定
                if (mi['dram_access_bytes'] >= mj['dram_access_bytes'] and
                    mi['energy'] >= mj['energy']):
                    dominated = True
                    break
        if not dominated:
            survivors.append(plan_i)

    return survivors

# -------------------------------
# Example compute_metric_fn
# -------------------------------
def example_metric_fn(plan: PlacementPlan):
    """
    Simple example metric function
    Count resident tensor sizes as data movement
    Assume 1 tensor unit = 1 byte (示例)
    """
    data_movement = 0
    dram_access = 0
    compute_cycles = 0
    energy = 0

    for scope_name, scope in plan.scopes.items():
        # 遍历 ResidentSet
        for rs in scope.resident_sets:
            data_movement += len(rs.tensors)
            if rs.mem == "DRAM":
                dram_access += len(rs.tensors)
        # 遍历 BoundaryAction (writeback/evict)
        for ba in scope.boundary_actions:
            data_movement += len(ba.writeback) + len(ba.evict)
            dram_access += len(ba.writeback)
        # PE 计算简单假设
        for rs in scope.resident_sets:
            if rs.mem == "PE":
                compute_cycles += len(rs.tensors)  # 假设每tensor1 cycle
        # Energy 粗略估算
        energy = data_movement + compute_cycles  # 仅示例

    return {
        'data_movement_bytes': data_movement,
        'dram_access_bytes': dram_access,
        'compute_cycles': compute_cycles,
        'energy': energy
    }
