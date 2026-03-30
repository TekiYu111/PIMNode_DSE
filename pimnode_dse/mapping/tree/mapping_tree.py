from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

# valid bind: seq / par / pipe / share
# valid mode: temp / spat
# valid act: load / store / evict / prefetch / writeback


@dataclass(frozen=True)
class Move:
    act: str
    tens: str
    src: str
    dst: str
    bytes: int = 0
    role: str = "unknown"
    repeat_hint: int = 1
    scope_id: str = ""

    def __post_init__(self) -> None:
        valid = {"load", "store", "evict", "prefetch", "writeback"}
        act = str(self.act).strip().lower()
        tens = str(self.tens).strip()
        src = str(self.src).strip().lower()
        dst = str(self.dst).strip().lower()
        role = str(self.role).strip().lower() if str(self.role).strip() else "unknown"
        bytes_ = int(self.bytes)
        repeat_hint = int(self.repeat_hint)
        scope_id = str(self.scope_id).strip()
        if act not in valid:
            raise ValueError("invalid move act: {}".format(self.act))
        if not tens:
            raise ValueError("move tens must not be empty")
        if not src:
            raise ValueError("move src must not be empty")
        if not dst:
            raise ValueError("move dst must not be empty")
        if bytes_ < 0:
            raise ValueError("move bytes must be >= 0")
        if repeat_hint <= 0:
            raise ValueError("move repeat_hint must be > 0")
        object.__setattr__(self, "act", act)
        object.__setattr__(self, "tens", tens)
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dst", dst)
        object.__setattr__(self, "bytes", bytes_)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "repeat_hint", repeat_hint)
        object.__setattr__(self, "scope_id", scope_id)


class _NodeBase:
    id: str
    parent: Optional["MapNode"]
    attrs: Dict[str, Any]

    def walk(self) -> Iterator["MapNode"]:
        yield from walk(self)  # type: ignore[arg-type]

    def root(self) -> "MapNode":
        node = self  # type: ignore[assignment]
        while getattr(node, "parent", None) is not None:
            node = node.parent  # type: ignore[assignment]
        return node  # type: ignore[return-value]


@dataclass
class ScopeNode(_NodeBase):
    id: str
    bind: str
    mem: str
    kids: List["MapNode"] = field(default_factory=list)

    need: Set[str] = field(default_factory=set)
    keep: Set[str] = field(default_factory=set)
    live_in: Set[str] = field(default_factory=set)
    live_out: Set[str] = field(default_factory=set)

    read_tensors: Set[str] = field(default_factory=set)
    fill_tensors: Set[str] = field(default_factory=set)
    update_tensors: Set[str] = field(default_factory=set)
    wb_tensors: Set[str] = field(default_factory=set)

    entry: List[Move] = field(default_factory=list)
    exit: List[Move] = field(default_factory=list)

    stage_kind: str = ""
    repeat_hint: int = 1
    overlap_policy: str = "none"
    resource_domain: str = ""

    parent: Optional["MapNode"] = field(default=None, repr=False, compare=False)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"seq", "par", "pipe", "share"}
        bind = str(self.bind).strip().lower()
        mem = str(self.mem).strip().lower()
        stage_kind = str(self.stage_kind).strip().lower()
        overlap_policy = str(self.overlap_policy).strip().lower() if str(self.overlap_policy).strip() else "none"
        resource_domain = str(self.resource_domain).strip().lower()
        repeat_hint = int(self.repeat_hint)
        if bind not in valid:
            raise ValueError("invalid scope bind: {}".format(self.bind))
        if not self.id:
            raise ValueError("scope id must not be empty")
        if not mem:
            raise ValueError("scope mem must not be empty")
        if repeat_hint <= 0:
            raise ValueError("scope repeat_hint must be > 0")
        object.__setattr__(self, "bind", bind)
        object.__setattr__(self, "mem", mem)
        object.__setattr__(self, "stage_kind", stage_kind)
        object.__setattr__(self, "overlap_policy", overlap_policy)
        object.__setattr__(self, "resource_domain", resource_domain)
        object.__setattr__(self, "repeat_hint", repeat_hint)
        for kid in list(self.kids):
            kid.parent = self

    def add_kid(self, kid: "MapNode") -> None:
        kid.parent = self
        self.kids.append(kid)

    def extend_kids(self, kids: Iterable["MapNode"]) -> None:
        for kid in kids:
            self.add_kid(kid)

    def clear_flow(self) -> None:
        self.need.clear()
        self.keep.clear()
        self.live_in.clear()
        self.live_out.clear()
        self.read_tensors.clear()
        self.fill_tensors.clear()
        self.update_tensors.clear()
        self.wb_tensors.clear()
        self.entry.clear()
        self.exit.clear()


@dataclass
class TileNode(_NodeBase):
    id: str
    mode: str
    loops: List[str]
    size: Dict[str, int]
    order: List[str]
    kid: Optional["MapNode"] = None

    parent: Optional["MapNode"] = field(default=None, repr=False, compare=False)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"temp", "spat"}
        mode = str(self.mode).strip().lower()
        loops = [str(x).strip().lower() for x in self.loops if str(x).strip()]
        order = [str(x).strip().lower() for x in self.order if str(x).strip()]
        size = dict((str(k).strip().lower(), int(v)) for k, v in dict(self.size).items())
        attrs = dict(self.attrs)

        if mode not in valid:
            raise ValueError("invalid tile mode: {}".format(self.mode))
        if not self.id:
            raise ValueError("tile id must not be empty")
        if not loops:
            raise ValueError("tile loops must not be empty")
        if not order:
            raise ValueError("tile order must not be empty")
        if set(order) != set(loops):
            raise ValueError("tile order must match tile loops")
        if not size:
            raise ValueError("tile size must not be empty")
        unknown = set(size) - set(loops)
        if unknown:
            raise ValueError("tile size keys not in loops: {}".format(sorted(unknown)))
        for dim, val in size.items():
            if val <= 0:
                raise ValueError("tile size must be > 0: {}={}".format(dim, val))

        level = str(attrs.get("level", "")).strip().lower()
        temporal_loops = tuple(
            str(x).strip().lower() for x in attrs.get("temporal_loops", ()) if str(x).strip()
        )
        spatial_loops = tuple(
            str(x).strip().lower() for x in attrs.get("spatial_loops", ()) if str(x).strip()
        )
        buf_mode = str(attrs.get("buf_mode", "single")).strip().lower()
        rw_overlap = bool(attrs.get("rw_overlap", False))
        repeat_hint = int(attrs.get("repeat_hint", 1))
        replication_hint = int(attrs.get("replication_hint", 1))
        if repeat_hint <= 0:
            raise ValueError("tile repeat_hint must be > 0")
        if replication_hint <= 0:
            raise ValueError("tile replication_hint must be > 0")
        attrs["level"] = level
        attrs["temporal_loops"] = temporal_loops
        attrs["spatial_loops"] = spatial_loops
        attrs["buf_mode"] = buf_mode
        attrs["rw_overlap"] = rw_overlap
        attrs["repeat_hint"] = repeat_hint
        attrs["replication_hint"] = replication_hint

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "loops", loops)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "attrs", attrs)

        if self.kid is not None:
            self.kid.parent = self

    def set_kid(self, kid: "MapNode") -> None:
        if isinstance(kid, ScopeNode):
            kid.parent = self
            self.kid = kid
            return
        kid.parent = self
        self.kid = kid


@dataclass
class OpNode(_NodeBase):
    id: str
    kind: str
    ins: Tuple[str, ...]
    outs: Tuple[str, ...]

    parent: Optional["MapNode"] = field(default=None, repr=False, compare=False)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("op id must not be empty")
        if not self.kind:
            raise ValueError("op kind must not be empty")


MapNode = Union[ScopeNode, TileNode, OpNode]


@dataclass
class MappingTree:
    root: ScopeNode
    attrs: Dict[str, Any] = field(default_factory=dict)

    def walk(self) -> Iterator[MapNode]:
        yield from walk(self.root)

    def validate(self) -> None:
        validate_tree(self.root)

    def collect_scopes(self, bind: Optional[str] = None, mem: Optional[str] = None) -> List[ScopeNode]:
        out = []  # type: List[ScopeNode]
        for node in self.walk():
            if not isinstance(node, ScopeNode):
                continue
            if bind is not None and node.bind != bind:
                continue
            if mem is not None and node.mem != mem:
                continue
            out.append(node)
        return out

    def collect_tiles(self, mode: Optional[str] = None) -> List[TileNode]:
        out = []  # type: List[TileNode]
        for node in self.walk():
            if not isinstance(node, TileNode):
                continue
            if mode is not None and node.mode != mode:
                continue
            out.append(node)
        return out

    def collect_ops(self, kind: Optional[str] = None) -> List[OpNode]:
        out = []  # type: List[OpNode]
        for node in self.walk():
            if not isinstance(node, OpNode):
                continue
            if kind is not None and node.kind != kind:
                continue
            out.append(node)
        return out

    def display(self) -> str:
        lines = []  # type: List[str]
        _render(self.root, lines, 0)
        return "\n".join(lines)


def walk(node: MapNode) -> Iterator[MapNode]:
    yield node
    if isinstance(node, ScopeNode):
        for kid in node.kids:
            yield from walk(kid)
    elif isinstance(node, TileNode) and node.kid is not None:
        yield from walk(node.kid)


def iter_scopes(node: MapNode) -> Iterator[ScopeNode]:
    for cur in walk(node):
        if isinstance(cur, ScopeNode):
            yield cur


def iter_tiles(node: MapNode) -> Iterator[TileNode]:
    for cur in walk(node):
        if isinstance(cur, TileNode):
            yield cur


def iter_ops(node: MapNode) -> Iterator[OpNode]:
    for cur in walk(node):
        if isinstance(cur, OpNode):
            yield cur


def validate_tree(node: MapNode) -> None:
    for cur in walk(node):
        if isinstance(cur, ScopeNode):
            if not cur.kids:
                raise ValueError("scope has no kids: {}".format(cur.id))
            for kid in cur.kids:
                if not isinstance(kid, (ScopeNode, TileNode, OpNode)):
                    raise TypeError(
                        "invalid scope kid type under {}: {}".format(cur.id, type(kid).__name__)
                    )
                if kid.parent is not cur:
                    raise ValueError("kid parent mismatch under scope {}".format(cur.id))
        elif isinstance(cur, TileNode):
            if cur.kid is None:
                raise ValueError("tile has no kid: {}".format(cur.id))
            if not isinstance(cur.kid, (ScopeNode, TileNode, OpNode)):
                raise TypeError(
                    "invalid tile kid type under {}: {}".format(cur.id, type(cur.kid).__name__)
                )
            if cur.kid.parent is not cur:
                raise ValueError("kid parent mismatch under tile {}".format(cur.id))
        elif isinstance(cur, OpNode):
            continue


def clone_scope_head(node: ScopeNode) -> ScopeNode:
    return ScopeNode(
        id=node.id,
        bind=node.bind,
        mem=node.mem,
        need=set(node.need),
        keep=set(node.keep),
        live_in=set(node.live_in),
        live_out=set(node.live_out),
        read_tensors=set(node.read_tensors),
        fill_tensors=set(node.fill_tensors),
        update_tensors=set(node.update_tensors),
        wb_tensors=set(node.wb_tensors),
        entry=list(node.entry),
        exit=list(node.exit),
        stage_kind=node.stage_kind,
        repeat_hint=node.repeat_hint,
        overlap_policy=node.overlap_policy,
        resource_domain=node.resource_domain,
        attrs=dict(node.attrs),
    )


def clone_tile_head(node: TileNode) -> TileNode:
    return TileNode(
        id=node.id,
        mode=node.mode,
        loops=list(node.loops),
        size=dict(node.size),
        order=list(node.order),
        attrs=dict(node.attrs),
    )


def clone_op(node: OpNode) -> OpNode:
    return OpNode(
        id=node.id,
        kind=node.kind,
        ins=tuple(node.ins),
        outs=tuple(node.outs),
        attrs=dict(node.attrs),
    )


def _render(node: MapNode, lines: List[str], depth: int) -> None:
    pad = "  " * depth
    if isinstance(node, ScopeNode):
        extra = ["bind={}".format(node.bind), "mem={}".format(node.mem)]  # type: List[str]
        if node.stage_kind:
            extra.append("stage={}".format(node.stage_kind))
        if node.repeat_hint != 1:
            extra.append("repeat={}".format(node.repeat_hint))
        if node.overlap_policy and node.overlap_policy != "none":
            extra.append("overlap={}".format(node.overlap_policy))
        if node.resource_domain:
            extra.append("resource={}".format(node.resource_domain))
        if node.need:
            extra.append("need={}".format(sorted(node.need)))
        if node.keep:
            extra.append("keep={}".format(sorted(node.keep)))
        if node.live_in:
            extra.append("live_in={}".format(sorted(node.live_in)))
        if node.live_out:
            extra.append("live_out={}".format(sorted(node.live_out)))
        if node.fill_tensors:
            extra.append("fill={}".format(sorted(node.fill_tensors)))
        if node.wb_tensors:
            extra.append("wb={}".format(sorted(node.wb_tensors)))
        if node.entry:
            extra.append("entry={}".format(len(node.entry)))
        if node.exit:
            extra.append("exit={}".format(len(node.exit)))
        lines.append("{}Scope({}: {})".format(pad, node.id, ", ".join(extra)))
        for kid in node.kids:
            _render(kid, lines, depth + 1)
        return
    if isinstance(node, TileNode):
        tloops = node.attrs.get("temporal_loops", ())
        sloops = node.attrs.get("spatial_loops", ())
        repeat_hint = node.attrs.get("repeat_hint", 1)
        replication_hint = node.attrs.get("replication_hint", 1)
        level = node.attrs.get("level", "")
        buf_mode = node.attrs.get("buf_mode", "single")
        rw_overlap = node.attrs.get("rw_overlap", False)
        lines.append(
            "{}Tile({}: mode={}, level={}, loops={}, size={}, order={}, temporal={}, spatial={}, repeat={}, repl={}, buf={}, overlap={})".format(
                pad,
                node.id,
                node.mode,
                level,
                node.loops,
                node.size,
                node.order,
                list(tloops),
                list(sloops),
                repeat_hint,
                replication_hint,
                buf_mode,
                rw_overlap,
            )
        )
        if node.kid is not None:
            _render(node.kid, lines, depth + 1)
        return
    lines.append(
        "{}Op({}: kind={}, ins={}, outs={})".format(
            pad, node.id, node.kind, list(node.ins), list(node.outs)
        )
    )