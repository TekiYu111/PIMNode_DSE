from pimnode_dse.placement.templates import build_core_templates, build_templates_by_phase


def test_build_core_templates():
    plan = build_core_templates()
    assert "Balanced" in plan.scopes
    assert "MaxReuse" in plan.scopes
    assert "MinReuse" in plan.scopes
    assert "K-D Fusion" in plan.scopes


def test_build_templates_by_phase_prefill():
    plan = build_templates_by_phase("prefill")
    assert len(plan.scopes) >= 1
    for _, scope in plan.scopes.items():
        if scope.supported_phases:
            assert "prefill" in scope.supported_phases
