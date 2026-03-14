from pimnode_dse.placement.placement_ir import (
    PlacementScope,
    ResidentSet,
    BoundaryAction,
    RoleResidentSet,
    RoleBoundaryAction,
    PlacementTemplateScope,
    instantiate_template_scope,
    normalize_scope,
)


def test_normalize_scope_fills_missing_levels():
    scope = PlacementScope(
        scope_name="s0",
        resident_sets=[ResidentSet(mem="SRAM", tensors={"Q"})],
        boundary_actions=[],
    )
    out = normalize_scope(scope, memory_levels=("DRAM", "SRAM", "PE"))
    assert len(out.resident_sets) == 3
    assert len(out.boundary_actions) == 3


def test_instantiate_template_scope():
    tmpl = PlacementTemplateScope(
        scope_name="Balanced",
        resident_sets=[RoleResidentSet(mem="SRAM", tensor_roles={"Q", "K"})],
        boundary_actions=[RoleBoundaryAction(mem="SRAM", prefetch_roles={"V"})],
    )
    concrete = instantiate_template_scope(
        tmpl,
        role_bindings={
            "Q": {"Q_real"},
            "K": {"K_real"},
            "V": {"V_real"},
        },
        phase="prefill",
        special_role=None,
    )
    rs = next(x for x in concrete.resident_sets if x.mem == "SRAM")
    ba = next(x for x in concrete.boundary_actions if x.mem == "SRAM")
    assert rs.tensors == {"Q_real", "K_real"}
    assert ba.prefetch == {"V_real"}
