from pimnode_dse.placement.placement_ir import PlacementScope, ResidentSet, BoundaryAction
from pimnode_dse.placement.rules import (
    get_tensor_role,
    is_kv_cache,
    apply_hard_rules,
    derive_actions,
)


def test_get_tensor_role_priority():
    role = get_tensor_role(
        "K_cache",
        tensor_to_role={"K_cache": "KV_CACHE"},
        tensor_meta={"K_cache": {"role": "K"}},
    )
    assert role == "KV_CACHE"


def test_apply_hard_rules_decode_kv_cache():
    scope = PlacementScope(
        scope_name="decode_scope",
        resident_sets=[ResidentSet(mem="SRAM", tensors={"K_cache"})],
        boundary_actions=[
            BoundaryAction(
                mem="SRAM",
                prefetch=set(),
                writeback={"K_cache"},
                evict={"K_cache"},
            )
        ],
        phase="Decode",
    )
    out = apply_hard_rules(
        scope,
        phase="Decode",
        tensor_to_role={"K_cache": "KV_CACHE"},
        tensor_meta={"K_cache": {"special_role": "KV_CACHE"}},
    )
    ba = next(x for x in out.boundary_actions if x.mem == "SRAM")
    assert "K_cache" not in ba.writeback
    assert "K_cache" not in ba.evict


def test_derive_actions_basic():
    scope = PlacementScope(
        scope_name="s0",
        resident_sets=[ResidentSet(mem="SRAM", tensors={"Q"})],
        boundary_actions=[BoundaryAction(mem="SRAM", prefetch={"V"}, writeback={"O"})],
        phase="prefill",
    )
    acts = derive_actions(
        scope,
        phase="prefill",
        tensor_to_role={"Q": "Q", "V": "V", "O": "O"},
    )
    assert any(a.action == "PREFETCH" for a in acts)
    assert any(a.action == "WRITEBACK" for a in acts)
