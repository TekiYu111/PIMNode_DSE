from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pimnode_dse.placement.placement_ir import (
    DEFAULT_SCOPE,
    LifetimeScope,
    RolePlacementTemplate,
    TensorPlacementSpec,
    instantiate_tensor_placement,
)
from pimnode_dse.placement.rules import RuleResult, apply_rules
from pimnode_dse.placement.templates import build_templates_by_phase
from pimnode_dse.mapping.fusion_gene import FusionGene, OpFusionGroup


# ============================================================
# Public data objects
# ============================================================

@dataclass
class GroupPlacementInput:
    """
    fusion -> placement 的显式接口对象。
    """
    group_id: str
    phase: Optional[str]

    group_tensors: List[str]
    inputs: List[str]
    outputs: List[str]
    temps: List[str]
    shared: List[str]

    input_roles: Dict[str, str]
    output_roles: Dict[str, str]

    # tensor_name -> metadata dict
    tensor_meta: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def role_bindings(self) -> Dict[str, List[str]]:
        """
        placement_ir.instantiate_tensor_placement(...) 需要的是:
            role -> [tensor names]
        """
        role_to_tensors: Dict[str, List[str]] = {}

        def _append(role_map: Mapping[str, str]) -> None:
            for tensor_name, role_name in role_map.items():
                role_to_tensors.setdefault(role_name, [])
                if tensor_name not in role_to_tensors[role_name]:
                    role_to_tensors[role_name].append(tensor_name)

        _append(self.input_roles)
        _append(self.output_roles)

        # 对没有显式 role 的张量，不强行分配角色；让模板自己决定是否覆盖
        for role_name in role_to_tensors:
            role_to_tensors[role_name] = sorted(role_to_tensors[role_name])

        return role_to_tensors


@dataclass
class GroupPlacementCandidate:
    """
    单个 group 的一个 placement 候选。
    """
    group_id: str
    template_name: str
    placement: TensorPlacementSpec
    rule_result: RuleResult

    @property
    def is_valid(self) -> bool:
        return self.rule_result.is_valid


@dataclass
class PlacementPlan:
    """
    多个 group 的组合 placement 方案。
    """
    fusion_gene_id: str
    group_candidates: Dict[str, GroupPlacementCandidate]

    @property
    def placements(self) -> Dict[str, TensorPlacementSpec]:
        return {gid: cand.placement for gid, cand in self.group_candidates.items()}


# ============================================================
# Public API
# ============================================================

def build_group_placement_input(
    group: OpFusionGroup,
    dag,
) -> GroupPlacementInput:
    """
    从 fusion group + workload DAG 构造 placement 输入。

    这里负责生成 placement 视角下的 tensor_meta。
    """
    group_tensors = sorted(set(group.inputs) | set(group.outputs) | set(group.temps))

    tensor_meta: Dict[str, Dict[str, object]] = {}
    for tensor_name in group_tensors:
        ts = dag.get_tensor(tensor_name)

        # 这里不叫 spillable，也不叫 has_backing
        # 用更直白的 can_store_offchip
        can_store_offchip = ts.role in {"input", "output", "state"}
        if tensor_name in group.temps:
            can_store_offchip = False

        size_bytes = _tensor_size_bytes(ts, dag)

        tensor_meta[tensor_name] = {
            "size_bytes": size_bytes,
            "can_store_offchip": can_store_offchip,
            "role": ts.role,
            "special_role": ts.special_role,
            "shape": tuple(ts.shape),
            "dtype": ts.dtype,
            "is_shared": tensor_name in set(group.shared),
            "is_temp": tensor_name in set(group.temps),
        }

    return GroupPlacementInput(
        group_id=group.group_id,
        phase=group.phase,
        group_tensors=group_tensors,
        inputs=list(group.inputs),
        outputs=list(group.outputs),
        temps=list(group.temps),
        shared=list(group.shared),
        input_roles=dict(group.in_roles),
        output_roles=dict(group.out_roles),
        tensor_meta=tensor_meta,
    )


def build_group_placements(
    group_input: GroupPlacementInput,
    *,
    scope: LifetimeScope = DEFAULT_SCOPE,
    template_names: Optional[Sequence[str]] = None,
) -> List[GroupPlacementCandidate]:
    """
    对单个 group 生成 placement 候选。

    当前策略：
    - 取 phase-compatible 模板
    - 全部实例化
    - rules 过滤
    """
    library = build_templates_by_phase(group_input.phase or "", scope=scope)

    selected_templates: List[RolePlacementTemplate] = []
    if template_names is None:
        selected_templates = list(library.templates.values())
    else:
        for name in template_names:
            if name not in library.templates:
                raise KeyError(
                    f"Template {name!r} not found for phase={group_input.phase!r}"
                )
            selected_templates.append(library.templates[name])

    role_bindings = group_input.role_bindings()
    tensor_to_role = {}
    tensor_to_role.update(group_input.input_roles)
    tensor_to_role.update(group_input.output_roles)

    candidates: List[GroupPlacementCandidate] = []

    for template in selected_templates:
        placement = instantiate_tensor_placement(
            template,
            scope_name=group_input.group_id,
            role_bindings=role_bindings,
            phase=group_input.phase,
            tensor_universe=group_input.group_tensors,
            extra_metadata={
                "group_id": group_input.group_id,
                "phase": group_input.phase,
            },
        )

        rule_result = apply_rules(
            placement,
            phase=group_input.phase or "",
            tensor_to_role=tensor_to_role,
            tensor_meta=group_input.tensor_meta,
            hardware_limits=None,
            constraints=None,
        )

        candidates.append(
            GroupPlacementCandidate(
                group_id=group_input.group_id,
                template_name=template.template_name,
                placement=placement,
                rule_result=rule_result,
            )
        )

    return candidates


def build_placement_plans(
    fusion_gene: FusionGene,
    dag,
    *,
    scope: LifetimeScope = DEFAULT_SCOPE,
    template_names_by_group: Optional[Mapping[str, Sequence[str]]] = None,
    max_plans: Optional[int] = None,
) -> List[PlacementPlan]:
    """
    先单 group 生成，再跨 group 组合，最后做 shared tensor 一致性检查。
    """
    group_inputs: Dict[str, GroupPlacementInput] = {}
    group_candidates: Dict[str, List[GroupPlacementCandidate]] = {}

    for group in fusion_gene.groups:
        g_input = build_group_placement_input(group, dag)
        group_inputs[group.group_id] = g_input
        group_candidates[group.group_id] = build_group_placements(
            g_input,
            scope=scope,
            template_names=None if template_names_by_group is None
            else template_names_by_group.get(group.group_id),
        )

    group_ids = [g.group_id for g in fusion_gene.groups]
    candidate_lists = [group_candidates[gid] for gid in group_ids]

    plans: List[PlacementPlan] = []
    for combo in product(*candidate_lists):
        plan = PlacementPlan(
            fusion_gene_id=fusion_gene.gene_id,
            group_candidates={cand.group_id: cand for cand in combo},
        )

        if not _check_plan_valid(plan):
            continue
        if not _check_shared_tensor_consistency(plan, fusion_gene, group_inputs):
            continue

        plans.append(plan)
        if max_plans is not None and len(plans) >= max_plans:
            break

    return plans


# ============================================================
# Internal helpers
# ============================================================

def _tensor_size_bytes(ts, dag) -> int:
    dtype_bytes = 4
    if getattr(dag, "spec", None) is not None:
        dtype_bytes = getattr(dag.spec, "dtype_bytes", dtype_bytes)

    total = 1
    for x in ts.shape:
        total *= int(x)
    return total * int(dtype_bytes)


def _check_plan_valid(plan: PlacementPlan) -> bool:
    """
    所有 group placement candidate 都必须先通过自身 rules 检查。
    """
    return all(cand.is_valid for cand in plan.group_candidates.values())


def _check_shared_tensor_consistency(
    plan: PlacementPlan,
    fusion_gene: FusionGene,
    group_inputs: Mapping[str, GroupPlacementInput],
) -> bool:
    """
    第一版最小 shared tensor 一致性检查：

    1. 若 tensor 是 shared，则它必须出现在相邻 group 的至少一侧 shared 中
    2. 若某个 shared tensor 在一个 group 中被标记为不能存外层，
       则不允许下游 group 假设它是“天然可外存回退”的边界对象
       （第一版只做 metadata 一致性，不做站点级复杂检查）
    """
    # 建立 tensor -> groups 映射
    tensor_to_groups: Dict[str, List[str]] = {}
    for gid, g_input in group_inputs.items():
        for t in g_input.shared:
            tensor_to_groups.setdefault(t, []).append(gid)

    # 只检查 FusionGene coarse edges 上相邻 group 的 shared 接口
    for src_gid, dst_gid in fusion_gene.edges:
        src_input = group_inputs[src_gid]
        dst_input = group_inputs[dst_gid]

        edge_shared = (set(src_input.shared) & set(dst_input.shared))
        if not edge_shared:
            # 允许 coarse edge 只靠结构存在，不强制一定声明 shared
            # 但第一版中如果完全没有 shared，可直接跳过
            continue

        for t in edge_shared:
            src_meta = src_input.tensor_meta.get(t, {})
            dst_meta = dst_input.tensor_meta.get(t, {})

            src_can_store = bool(src_meta.get("can_store_offchip", False))
            dst_can_store = bool(dst_meta.get("can_store_offchip", False))

            # 第一版采用保守规则：
            # 如果上游明确不能外存回退，而下游又把它当可外存对象，就视为不一致
            if (not src_can_store) and dst_can_store:
                return False

    return True
