from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


GroupEdge = tuple[str, str]
GroupEdgeTensor = tuple[str, str, str]


@dataclass(frozen=True)
class FusionGroup:
    """Pure structural fusion group.

    This layer only describes which workload ops belong to the same coarse
    fusion subgraph, plus the boundary tensors derived from workload facts.
    It does NOT contain placement / tiling / schedule information.
    """

    group_id: str
    ops: tuple[str, ...]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    temps: tuple[str, ...] = ()

    def op_set(self) -> set[str]:
        return set(self.ops)


@dataclass(frozen=True)
class FusionGene:
    """Minimal coarse fusion IR.

    - groups: group-level partition of workload ops
    - group_edges: coarse DAG edges between groups
    - group_edges_tensors: tensor-level connections carried by each group edge
    """

    gene_id: str
    groups: tuple[FusionGroup, ...]
    group_edges: tuple[GroupEdge, ...] = ()
    group_edges_tensors: tuple[GroupEdgeTensor, ...] = ()

    def group_ids(self) -> tuple[str, ...]:
        return tuple(g.group_id for g in self.groups)

    def group_map(self) -> dict[str, FusionGroup]:
        return {g.group_id: g for g in self.groups}

    def op_to_group(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for g in self.groups:
            for op in g.ops:
                out[op] = g.group_id
        return out

    def tensors_for_edge(self, src_group: str, dst_group: str) -> tuple[str, ...]:
        tensors = [
            t
            for s, d, t in self.group_edges_tensors
            if s == src_group and d == dst_group
        ]
        return tuple(sorted(set(tensors)))

    def validate(
        self,
        workload_op_ids: Optional[Iterable[str]] = None,
        graph_adapter: Optional[object] = None,
    ) -> None:
        """Validate minimal structural consistency.

        workload_op_ids:
            Optional complete op-id universe for coverage checking.

        graph_adapter:
            Optional adapter that provides:
              - is_convex(op_set) -> bool
              - boundary(op_set) -> object with inputs/outputs/temps
        """

        self._validate_unique_group_ids()
        self._validate_nonempty_groups()
        self._validate_disjoint_ops()
        self._validate_edges_reference_existing_groups()
        self._validate_edge_tensors_reference_existing_edges()
        self._validate_group_edges_acyclic()

        if workload_op_ids is not None:
            self._validate_full_coverage(set(workload_op_ids))

        if graph_adapter is not None:
            self._validate_convexity(graph_adapter)
            self._validate_boundaries(graph_adapter)

    def _validate_unique_group_ids(self) -> None:
        ids = [g.group_id for g in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate fusion group_id detected.")

    def _validate_nonempty_groups(self) -> None:
        for g in self.groups:
            if not g.ops:
                raise ValueError(f"Fusion group '{g.group_id}' has no ops.")

    def _validate_disjoint_ops(self) -> None:
        seen: dict[str, str] = {}
        for g in self.groups:
            for op in g.ops:
                prev = seen.get(op)
                if prev is not None:
                    raise ValueError(
                        f"Op '{op}' appears in multiple groups: '{prev}' and '{g.group_id}'."
                    )
                seen[op] = g.group_id

    def _validate_full_coverage(self, workload_ops: set[str]) -> None:
        grouped_ops = {op for g in self.groups for op in g.ops}
        if grouped_ops != workload_ops:
            missing = sorted(workload_ops - grouped_ops)
            extra = sorted(grouped_ops - workload_ops)
            raise ValueError(
                "Fusion groups do not cover workload ops exactly. "
                f"missing={missing}, extra={extra}"
            )

    def _validate_edges_reference_existing_groups(self) -> None:
        valid_ids = set(self.group_ids())
        for src, dst in self.group_edges:
            if src not in valid_ids or dst not in valid_ids:
                raise ValueError(
                    f"Invalid group edge ({src!r}, {dst!r}); endpoint group not found."
                )
            if src == dst:
                raise ValueError(f"Invalid self-edge on group '{src}'.")

    def _validate_edge_tensors_reference_existing_edges(self) -> None:
        valid_edges = set(self.group_edges)
        for src, dst, tensor in self.group_edges_tensors:
            if (src, dst) not in valid_edges:
                raise ValueError(
                    "group_edges_tensors contains a tensor connection whose "
                    f"group edge is missing: ({src!r}, {dst!r}, {tensor!r})"
                )

    def _validate_group_edges_acyclic(self) -> None:
        nodes = set(self.group_ids())
        succ: dict[str, set[str]] = {n: set() for n in nodes}
        indeg: dict[str, int] = {n: 0 for n in nodes}

        for src, dst in self.group_edges:
            if dst not in succ[src]:
                succ[src].add(dst)
                indeg[dst] += 1

        ready = sorted([n for n, d in indeg.items() if d == 0])
        visited = 0

        while ready:
            node = ready.pop(0)
            visited += 1
            for nxt in sorted(succ[node]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
                    ready.sort()

        if visited != len(nodes):
            raise ValueError("group_edges must form a DAG.")

    def _validate_convexity(self, graph_adapter: object) -> None:
        if not hasattr(graph_adapter, "is_convex"):
            return
        for g in self.groups:
            if not graph_adapter.is_convex(g.op_set()):
                raise ValueError(
                    f"Fusion group '{g.group_id}' is not convex in workload DAG."
                )

    def _validate_boundaries(self, graph_adapter: object) -> None:
        if not hasattr(graph_adapter, "boundary"):
            return
        for g in self.groups:
            b = graph_adapter.boundary(g.op_set())
            exp_inputs = tuple(sorted(set(getattr(b, "inputs", ()))))
            exp_outputs = tuple(sorted(set(getattr(b, "outputs", ()))))
            exp_temps = tuple(sorted(set(getattr(b, "temps", ()))))

            got_inputs = tuple(sorted(set(g.inputs)))
            got_outputs = tuple(sorted(set(g.outputs)))
            got_temps = tuple(sorted(set(g.temps)))

            if got_inputs != exp_inputs:
                raise ValueError(
                    f"Fusion group '{g.group_id}' inputs mismatch: "
                    f"expected={exp_inputs}, got={got_inputs}"
                )
            if got_outputs != exp_outputs:
                raise ValueError(
                    f"Fusion group '{g.group_id}' outputs mismatch: "
                    f"expected={exp_outputs}, got={got_outputs}"
                )
            if got_temps != exp_temps:
                raise ValueError(
                    f"Fusion group '{g.group_id}' temps mismatch: "
                    f"expected={exp_temps}, got={got_temps}"
                )

    @classmethod
    def from_groups(
        cls,
        gene_id: str,
        groups: Sequence[FusionGroup],
    ) -> "FusionGene":
        """Build a minimal FusionGene from groups' boundary tensors only.

        Assumptions:
        - group.outputs and group.inputs have already been derived from workload.
        - If a tensor is in src.outputs and dst.inputs, it is treated as a
          cross-group tensor connection.
        """
        group_edges_tensors = build_group_edges_tensors(groups)
        group_edges = build_group_edges(group_edges_tensors)
        return cls(
            gene_id=gene_id,
            groups=tuple(groups),
            group_edges=group_edges,
            group_edges_tensors=group_edges_tensors,
        )


def build_group_edges_tensors(
    groups: Sequence[FusionGroup],
) -> tuple[GroupEdgeTensor, ...]:
    """Infer (producer_group, consumer_group, tensor) from group boundaries."""
    out: list[GroupEdgeTensor] = []
    for src in groups:
        src_out = set(src.outputs)
        if not src_out:
            continue
        for dst in groups:
            if src.group_id == dst.group_id:
                continue
            shared = src_out & set(dst.inputs)
            for tensor in sorted(shared):
                out.append((src.group_id, dst.group_id, tensor))
    return tuple(out)


def build_group_edges(
    group_edges_tensors: Sequence[GroupEdgeTensor],
) -> tuple[GroupEdge, ...]:
    """Collapse tensor-level connections into unique coarse group edges."""
    edges = sorted({(src, dst) for src, dst, _ in group_edges_tensors})
    return tuple(edges)
