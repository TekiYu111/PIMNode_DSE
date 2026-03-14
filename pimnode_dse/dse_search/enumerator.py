from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

from pimnode_dse.dse_search.stage_a import FeasibleSubspace
from pimnode_dse.mapping.fusion_gene import FusionGene, OpFusionGroup
from pimnode_dse.mapping.tilling_gene import GroupTilingSpec
from pimnode_dse.placement.placement_ir import PlacementTemplateScope


CandidateChoice = Tuple[OpFusionGroup, str, PlacementTemplateScope, GroupTilingSpec]


class StageASubspaceEnumerator:
    """Enumerate final feasible (group, template, tiling) tuples from Stage A.

    The older seed carrier layer was only a thin transport wrapper and has been
    removed. Downstream code should either consume these tuples directly or use
    candidate.build_candidates_from_stage_a_subspace() as the single candidate
    materialization entrypoint.
    """

    def __init__(self, fusion: FusionGene, subspace: FeasibleSubspace):
        self.fusion = fusion
        self.subspace = subspace
        self._group_index: Dict[str, OpFusionGroup] = {g.group_id: g for g in fusion.groups}

    def iter_choices(self) -> Iterator[CandidateChoice]:
        for group_id, template_names in self.subspace.placement_options_by_group.items():
            group = self._group_index.get(group_id)
            if group is None:
                continue
            for template_name in template_names:
                template_scope = self.subspace.placement_templates_by_name[template_name]
                tiling_specs = self.subspace.tiling_subspace_by_key.get((group_id, template_name), [])
                for tiling_spec in tiling_specs:
                    yield (group, template_name, template_scope, tiling_spec)

    def list_choices(self) -> List[CandidateChoice]:
        return list(self.iter_choices())
