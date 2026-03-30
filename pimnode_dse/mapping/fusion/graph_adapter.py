from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class FusionDataEdge:
    src_op: str
    dst_op: str
    tensor: str
    shape: Tuple[int, ...]
    dtype: str
    tensor_type: str
    src_dims: Tuple[str, ...]
    dst_dims: Tuple[str, ...]
    #reduction_dims: Tuple[str, ...]


@dataclass(frozen=True)
class FusionBoundary:
    inputs: frozenset[str]
    outputs: frozenset[str]
    temps: frozenset[str]
    shared: frozenset[str]


class WorkloadFusionGraphAdapter:
    """
    Stable structural view over a workload DAG.

    Responsibilities:
    - expose op-level dataflow facts
    - expose tensor producer / consumer facts
    - expose stable boundary and convexity queries

    Non-responsibilities:
    - op or tensor semantic classification
    - fusion-group scoring or legality policy
    - placement / tiling decisions
    """

    def __init__(self, dag: Any) -> None:
        self._dag = dag
        if hasattr(self._dag, "finalize"):
            self._dag.finalize()

        self._edges: Tuple[FusionDataEdge, ...] = tuple(self._build_edges())
        self._out_by_src: Dict[str, List[FusionDataEdge]] = {}
        self._in_by_dst: Dict[str, List[FusionDataEdge]] = {}
        self._between: Dict[Tuple[str, str], List[FusionDataEdge]] = {}

        for edge in self._edges:
            self._out_by_src.setdefault(edge.src_op, []).append(edge)
            self._in_by_dst.setdefault(edge.dst_op, []).append(edge)
            self._between.setdefault((edge.src_op, edge.dst_op), []).append(edge)

    @property
    def dag(self) -> Any:
        return self._dag

    def _build_edges(self) -> List[FusionDataEdge]:
        raw_edges = self._raw_edges()
        out: List[FusionDataEdge] = []

        for raw in raw_edges:
            src_op = self._get_attr(raw, ("src_op", "src"))
            dst_op = self._get_attr(raw, ("dst_op", "dst"))
            tensor = self._get_attr(raw, ("tensor", "name"), default="")
            src_dims = tuple(str(x) for x in self._get_iter_attr(raw, ("src_dims",)))
            dst_dims = tuple(str(x) for x in self._get_iter_attr(raw, ("dst_dims",)))
            #reduce_dims = tuple(str(x) for x in self._get_iter_attr(raw, ("reduce_dims", "reduction_dims")))

            if src_op is None or dst_op is None:
                continue

            ten_name = str(tensor)
            shape, dtype, tensor_type = self._tensor_meta(ten_name)
            raw_shape = self._get_iter_attr(raw, ("shape",))
            if raw_shape:
                shape = tuple(int(x) for x in raw_shape)

            raw_dtype = self._get_attr(raw, ("dtype",), default=None)
            if raw_dtype is not None:
                dtype = str(raw_dtype)

            raw_tensor_type = self._get_attr(raw, ("tensor_type",), default=None)
            if raw_tensor_type is not None:
                tensor_type = self._enum_or_str(raw_tensor_type)

            out.append(
                FusionDataEdge(
                    src_op=str(src_op),
                    dst_op=str(dst_op),
                    tensor=ten_name,
                    shape=shape,
                    dtype=dtype,
                    tensor_type=tensor_type,
                    src_dims=src_dims,
                    dst_dims=dst_dims,
                    #reduction_dims=reduce_dims,
                )
            )

        return out

    def _raw_edges(self) -> Iterable[Any]:
        if hasattr(self._dag, "to_op_flow_edges"):
            return tuple(self._dag.to_op_flow_edges())
        if hasattr(self._dag, "edge_list"):
            return tuple(self._dag.edge_list())
        if hasattr(self._dag, "edges"):
            return tuple(self._dag.edges())
        raise AttributeError(
            "Workload DAG must provide to_op_flow_edges(), edge_list(), or edges()."
        )

    def _tensor_meta(self, tensor: str) -> Tuple[Tuple[int, ...], str, str]:
        shape: Tuple[int, ...] = ()
        dtype = "unknown"
        tensor_type = "unknown"

        if not tensor:
            return shape, dtype, tensor_type

        ten_obj = None
        if hasattr(self._dag, "tensor"):
            try:
                ten_obj = self._dag.tensor(tensor)
            except Exception:
                ten_obj = None

        if ten_obj is None and hasattr(self._dag, "tensors"):
            try:
                all_tens = self._dag.tensors()
                if isinstance(all_tens, dict):
                    ten_obj = all_tens.get(tensor)
            except Exception:
                ten_obj = None

        if ten_obj is None:
            return shape, dtype, tensor_type

        raw_shape = self._get_iter_attr(ten_obj, ("shape",))
        if raw_shape:
            shape = tuple(int(x) for x in raw_shape)

        raw_dtype = self._get_attr(ten_obj, ("dtype",), default=None)
        if raw_dtype is not None:
            dtype = str(raw_dtype)

        raw_tensor_type = self._get_attr(ten_obj, ("tensor_type",), default=None)
        if raw_tensor_type is not None:
            tensor_type = self._enum_or_str(raw_tensor_type)

        return shape, dtype, tensor_type

    @staticmethod
    def _enum_or_str(value: Any) -> str:
        if hasattr(value, "value"):
            return str(getattr(value, "value"))
        return str(value)

    @staticmethod
    def _get_attr(obj: Any, names: Tuple[str, ...], default: Any = None) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    @staticmethod
    def _get_iter_attr(obj: Any, names: Tuple[str, ...]) -> Tuple[Any, ...]:
        value = WorkloadFusionGraphAdapter._get_attr(obj, names, default=())
        if value is None:
            return ()
        return tuple(value)

    def op_names(self) -> frozenset[str]:
        if hasattr(self._dag, "op_ids"):
            return frozenset(str(x) for x in self._dag.op_ids())
        if hasattr(self._dag, "op_names"):
            return frozenset(str(x) for x in self._dag.op_names())
        raise AttributeError("Workload DAG must provide op_ids() or op_names().")

    def topo_order(self) -> Tuple[str, ...]:
        if hasattr(self._dag, "topological_op_order"):
            return tuple(str(x) for x in self._dag.topological_op_order())
        if hasattr(self._dag, "topo_order"):
            return tuple(str(x) for x in self._dag.topo_order())
        return tuple(sorted(self.op_names()))

    def op(self, op_id: str) -> Any:
        if hasattr(self._dag, "op"):
            return self._dag.op(op_id)
        if hasattr(self._dag, "get_op"):
            return self._dag.get_op(op_id)
        raise AttributeError("Workload DAG must provide op() or get_op().")

    def tensor(self, tensor: str) -> Optional[Any]:
        if hasattr(self._dag, "tensor"):
            return self._dag.tensor(tensor)
        return None

    def predecessors(self, op_id: str) -> Tuple[str, ...]:
        if hasattr(self._dag, "predecessors"):
            return tuple(str(x) for x in self._dag.predecessors(op_id))
        return tuple(sorted({edge.src_op for edge in self.incoming_edges(op_id)}))

    def successors(self, op_id: str) -> Tuple[str, ...]:
        if hasattr(self._dag, "successors"):
            return tuple(str(x) for x in self._dag.successors(op_id))
        return tuple(sorted({edge.dst_op for edge in self.outgoing_edges(op_id)}))

    def has_direct_edge(self, src_op: str, dst_op: str, tensor: Optional[str] = None) -> bool:
        edges = self._between.get((src_op, dst_op), [])
        if tensor is None:
            return bool(edges)
        return any(edge.tensor == tensor for edge in edges)

    def edges(self) -> Tuple[FusionDataEdge, ...]:
        return self._edges

    def edges_between(self, src_op: str, dst_op: str) -> Tuple[FusionDataEdge, ...]:
        return tuple(self._between.get((src_op, dst_op), ()))

    def incoming_edges(self, op_id: str) -> Tuple[FusionDataEdge, ...]:
        return tuple(self._in_by_dst.get(op_id, ()))

    def outgoing_edges(self, op_id: str) -> Tuple[FusionDataEdge, ...]:
        return tuple(self._out_by_src.get(op_id, ()))

    def source_tensors(self) -> frozenset[str]:
        if hasattr(self._dag, "source_tensors"):
            return frozenset(str(x) for x in self._dag.source_tensors())
        return frozenset(
            edge.tensor
            for edge in self._edges
            if self.producer_of(edge.tensor) is None
        )

    def sink_tensors(self) -> frozenset[str]:
        if hasattr(self._dag, "sink_tensors"):
            return frozenset(str(x) for x in self._dag.sink_tensors())
        return frozenset(
            edge.tensor
            for edge in self._edges
            if not self.consumers_of(edge.tensor)
        )

    def producer_of(self, tensor: str) -> Optional[str]:
        if hasattr(self._dag, "producer_of"):
            prod = self._dag.producer_of(tensor)
            return None if prod is None else str(prod)
        for edge in self._edges:
            if edge.tensor == tensor:
                return edge.src_op
        return None

    def consumers_of(self, tensor: str) -> Tuple[str, ...]:
        if hasattr(self._dag, "consumers_of"):
            return tuple(str(x) for x in self._dag.consumers_of(tensor))
        return tuple(sorted({edge.dst_op for edge in self._edges if edge.tensor == tensor}))

    def boundary(self, op_set: Set[str]) -> FusionBoundary:
        op_set = set(op_set)
        if hasattr(self._dag, "subgraph_tensors"):
            part = self._dag.subgraph_tensors(op_set)
            inputs = set(part.get("inputs", ()))
            outputs = set(part.get("outputs", ()))
            temps = set(part.get("temps", ()))
        elif hasattr(self._dag, "boundary_tensors"):
            part = self._dag.boundary_tensors(op_set)
            inputs = set(part.get("inputs", ()))
            outputs = set(part.get("outputs", ()))
            temps = set(part.get("temps", ()))
        else:
            inputs, outputs, temps = self._derive_boundary(op_set)

        shared = self._shared_boundary_tensors(op_set, inputs, outputs)
        return FusionBoundary(
            inputs=frozenset(sorted(inputs)),
            outputs=frozenset(sorted(outputs)),
            temps=frozenset(sorted(temps)),
            shared=frozenset(sorted(shared)),
        )

    def _derive_boundary(self, op_set: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
        inputs: Set[str] = set()
        outputs: Set[str] = set()
        temps: Set[str] = set()

        for op_id in op_set:
            op_obj = self.op(op_id)
            for tensor in getattr(op_obj, "inputs", ()):
                prod = self.producer_of(str(tensor))
                if prod not in op_set:
                    inputs.add(str(tensor))
                else:
                    temps.add(str(tensor))
            for tensor in getattr(op_obj, "outputs", ()):
                consumers = set(self.consumers_of(str(tensor)))
                if not consumers or any(dst not in op_set for dst in consumers):
                    outputs.add(str(tensor))
                if consumers and consumers.issubset(op_set):
                    temps.add(str(tensor))

        temps -= inputs
        temps -= outputs
        return inputs, outputs, temps

    def _shared_boundary_tensors(
        self,
        op_set: Set[str],
        inputs: Set[str],
        outputs: Set[str],
    ) -> Set[str]:
        shared: Set[str] = set()
        for tensor in inputs | outputs:
            prod = self.producer_of(tensor)
            consumers = set(self.consumers_of(tensor))
            cross_in = prod is not None and prod not in op_set
            cross_out = any(dst not in op_set for dst in consumers)
            if cross_in or cross_out:
                shared.add(tensor)
        return shared

    def is_convex(self, op_set: Set[str]) -> bool:
        op_set = set(op_set)
        if hasattr(self._dag, "is_convex_subgraph"):
            return bool(self._dag.is_convex_subgraph(op_set))
        return self._is_convex_fallback(op_set)

    def _is_convex_fallback(self, op_set: Set[str]) -> bool:
        for src in op_set:
            for dst in op_set:
                if src == dst:
                    continue
                if self._path_exits_and_reenters(src, dst, op_set):
                    return False
        return True

    def _path_exits_and_reenters(self, src: str, dst: str, op_set: Set[str]) -> bool:
        stack: List[Tuple[str, bool]] = [(src, False)]
        seen: Set[Tuple[str, bool]] = set()

        while stack:
            node, left = stack.pop()
            state = (node, left)
            if state in seen:
                continue
            seen.add(state)

            for nxt in self.successors(node):
                next_left = left or (nxt not in op_set)
                if nxt == dst and next_left:
                    return True
                stack.append((nxt, next_left))
        return False


    # def edge_reduction_dims(
    #     self,
    #     src_op: str,
    #     dst_op: str,
    #     tensor: Optional[str] = None,
    # ) -> frozenset[str]:
    #     dims: Set[str] = set()
    #     for edge in self.edges_between(src_op, dst_op):
    #         if tensor is not None and edge.tensor != tensor:
    #             continue
    #         dims.update(edge.reduction_dims)
    #     return frozenset(sorted(dims))


__all__ = [
    "FusionBoundary",
    "FusionDataEdge",
    "WorkloadFusionGraphAdapter",
]
