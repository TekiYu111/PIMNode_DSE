# validators.py
"""
本文件负责：对 WorkloadSpec / WorkloadDAG / TilingGene 做“语义级”合法性检查。

设计原则（避免越写越乱）
────────────────────────
只检查同时满足以下两点的约束：
  1) 很容易被“悄悄写错”，但程序不一定立刻崩溃（会导致结果静默错误）
  2) 不能靠 Python 的 TypeError/KeyError 等轻易捕获

刻意不做（应由别处负责或可自然暴露）：
  - 每个 shape 字段都做 >0 检查（多数情况下 0/负数会在后续自然崩溃）
  - 搜索算法（枚举/变异/选择）相关逻辑
  - 性能/能耗/搬运量计算（属于 analysis）
  - Ramulator trace 生成（属于 trace_gen）
  - 构树（属于 builder）

本文件当前提供：
  - validate_spec(spec): WorkloadSpec 语义检查
  - validate_dag(dag):   WorkloadDAG 语义检查（含 einsum-like 语义一致性检查）
  - validate_tiling_gene(dag, fusion, tiling, arch=None): TilingGene 语义检查
    （工程化：收集所有 violations 一次性抛错；硬件相关检查“有 arch 才做”，并预留硬件字段接口）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    # workload 包内部相对导入（避免包路径差异）
    from .workload import AttentionWorkloadSpec, WorkloadDAG, OpSpec, TensorSpec, DomainSpec


# ─────────────────────────────────────────────────────────────
# 小工具：同时支持 arch 是 dataclass 对象或 dict
# ─────────────────────────────────────────────────────────────

def _get_arch_field(arch: Any, name: str, default: Any = None) -> Any:
    if arch is None:
        return default
    if isinstance(arch, dict):
        return arch.get(name, default)
    return getattr(arch, name, default)


# ─────────────────────────────────────────────────────────────
# DAG 小工具：拓扑排序（避免依赖 WorkloadDAG 必须实现 topo_order）
# ─────────────────────────────────────────────────────────────

def _topo_order_or_raise(nodes: Set[str], edges: List[Tuple[str, str]]) -> List[str]:
    """
    Kahn 拓扑排序。若存在环则抛 ValueError。
    """
    from collections import defaultdict, deque

    indeg: Dict[str, int] = {n: 0 for n in nodes}
    adj: Dict[str, List[str]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] = indeg.get(v, 0) + 1
        if u not in indeg:
            indeg[u] = 0

    q = deque([n for n, d in indeg.items() if d == 0])
    order: List[str] = []
    while q:
        x = q.popleft()
        order.append(x)
        for y in adj.get(x, []):
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)

    if len(order) != len(indeg):
        # 存在环：找出剩余节点用于报错
        remain = [n for n, d in indeg.items() if d > 0]
        raise ValueError(f"DAG 存在环（无法拓扑排序），涉及节点：{remain}")
    return order


# ─────────────────────────────────────────────────────────────
# Spec-level validation
# ─────────────────────────────────────────────────────────────

def validate_spec(spec: "AttentionWorkloadSpec") -> None:
    """检查 AttentionWorkloadSpec 的语义约束。违反则抛 ValueError（中文信息）。"""

    # 1) Head 分组一致性：Hq 必须能被 Hkv 整除
    if spec.Hq % spec.Hkv != 0:
        raise ValueError(
            f"Hq 必须能被 Hkv 整除（GQA/MQA 头映射需要 floor(hq/group) 有定义）。当前 Hq={spec.Hq}, Hkv={spec.Hkv}。"
        )

    # 2) attn_type 与头数比例一致性
    if spec.attn_type == "mha" and spec.Hq != spec.Hkv:
        raise ValueError(
            f"attn_type='mha' 要求 Hq == Hkv。当前 Hq={spec.Hq}, Hkv={spec.Hkv}。如果是分组注意力请用 attn_type='gqa'。"
        )

    if spec.attn_type == "mqa" and spec.Hkv != 1:
        raise ValueError(
            f"attn_type='mqa' 要求 Hkv == 1。当前 Hkv={spec.Hkv}。"
        )

    # 3) decode 必需字段
    if spec.mode == "decode":
        if not spec.kv_cache_enabled:
            raise ValueError(
                "decode 模式必须设置 kv_cache_enabled=True（decode 需要建模 KVCache 状态）。"
            )

        if spec.cache_len_before is None:
            raise ValueError(
                "decode 模式必须提供 cache_len_before（本步之前历史 KV 长度）。"
            )

        if spec.append_len is None:
            raise ValueError(
                "decode 模式必须提供 append_len（本步新增 token 数，通常为 1）。"
            )

        # 4) KV_len 与 cache_len_before 一致（避免 KV_total 语义不清导致静默错误）
        if spec.KV_len != spec.cache_len_before:
            raise ValueError(
                f"decode 模式下要求 KV_len == cache_len_before，以保证 KV_total = cache_len_before + append_len 语义明确。\n"
                f"当前 KV_len={spec.KV_len}, cache_len_before={spec.cache_len_before}。建议设置 KV_len = cache_len_before。"
            )


# ─────────────────────────────────────────────────────────────
# Einsum-like semantic consistency checks
# ─────────────────────────────────────────────────────────────

def _validate_op_einsum_semantics(op: "OpSpec", dag_tensors: Dict[str, "TensorSpec"], violations: List[str]) -> None:
    """
    对单个 OpSpec 做 einsum-like 语义一致性检查。
    仅在 op.has_einsum_semantics() 为 True 时执行强校验；否则不强制（兼容旧 workload）。
    """
    def E(msg: str) -> None:
        violations.append(msg)

    if not getattr(op, "has_einsum_semantics", lambda: False)():
        return

    index_vars = getattr(op, "index_vars", ()) or ()
    tensor_index = getattr(op, "tensor_index", {}) or {}

    if not isinstance(index_vars, tuple) or not all(isinstance(x, str) and x for x in index_vars):
        E(f"[E] op={op.name}: index_vars 必须是非空的字符串元组。当前={index_vars!r}")
        return

    if len(set(index_vars)) != len(index_vars):
        E(f"[E] op={op.name}: index_vars 存在重复变量：{index_vars!r}")

    if not isinstance(tensor_index, dict) or not tensor_index:
        E(f"[E] op={op.name}: tensor_index 必须为非空 dict。当前={tensor_index!r}")
        return

    # tensor_index 中出现的张量必须属于 reads ∪ writes（允许子集；但不允许多余张量名）
    rw = set(getattr(op, "reads", ()) or ()) | set(getattr(op, "writes", ()) or ())
    extra_tensors = sorted(set(tensor_index.keys()) - rw)
    if extra_tensors:
        E(f"[E] op={op.name}: tensor_index 包含未在 reads/writes 声明的张量 {extra_tensors}。")

    # 对 reads / writes，建议都能在 tensor_index 中找到（否则后续 tiler/analysis 语义不完整）
    missing_reads = sorted(set(getattr(op, "reads", ()) or ()) - set(tensor_index.keys()))
    missing_writes = sorted(set(getattr(op, "writes", ()) or ()) - set(tensor_index.keys()))
    if missing_reads:
        E(f"[E] op={op.name}: reads={missing_reads} 未提供 tensor_index 映射。")
    if missing_writes:
        E(f"[E] op={op.name}: writes={missing_writes} 未提供 tensor_index 映射。")

    # 每个 tensor 的索引变量必须是 index_vars 的子集，且不能重复
    iv_set = set(index_vars)
    for tname, tvars in tensor_index.items():
        if not isinstance(tvars, tuple) or not all(isinstance(x, str) and x for x in tvars):
            E(f"[E] op={op.name} tensor={tname}: tensor_index 映射必须是字符串元组，当前={tvars!r}")
            continue
        if len(set(tvars)) != len(tvars):
            E(f"[E] op={op.name} tensor={tname}: tensor_index 存在重复变量：{tvars!r}")
        bad = sorted(set(tvars) - iv_set)
        if bad:
            E(f"[E] op={op.name} tensor={tname}: tensor_index 使用了不在 index_vars 中的变量 {bad}。")

        # 若 dag.tensors 中存在该张量，则 dim_names 数量应等于映射长度（最强语义约束）
        ts = dag_tensors.get(tname)
        if ts is not None:
            if len(getattr(ts, "dim_names", ())) != len(tvars):
                E(
                    f"[E] op={op.name} tensor={tname}: TensorSpec.dim_names 长度={len(ts.dim_names)} "
                    f"但 tensor_index 映射长度={len(tvars)}，不一致。"
                )

    # domain 变量要求：required_vars 必须 ⊆ index_vars
    domain = getattr(op, "domain", None)
    if domain is not None:
        req = set(getattr(domain, "required_vars", lambda: set())() or set())
        bad = sorted(req - iv_set)
        if bad:
            E(f"[E] op={op.name}: domain.kind={getattr(domain, 'kind', None)!r} 需要变量 {bad}，但未出现在 index_vars 中。")

    # legacy loop_dims / reduction_dims 与 einsum 之间的一致性（尽量做“强而不苛刻”的检查）
    loop_dims = getattr(op, "loop_dims", {}) or {}
    red_dims = set(getattr(op, "reduction_dims", set()) or set())

    if loop_dims:
        # 约定：loop_dims 的 key 应是“符号维度名”，不必与 index_vars 相同。
        # 但最常见情况下它们一致；若不一致，仍允许，只做 warning 级别的检查在 dag-level 完成。
        pass

    # reduction_dims 必须是 loop_dims 子集（legacy 规则），且若与 index_vars 共用命名也应是子集
    if red_dims and loop_dims:
        bad = sorted(red_dims - set(loop_dims.keys()))
        if bad:
            E(f"[E] op={op.name}: reduction_dims={sorted(red_dims)} 必须是 loop_dims keys 的子集；非法={bad}。")


# ─────────────────────────────────────────────────────────────
# DAG-level validation
# ─────────────────────────────────────────────────────────────

def validate_dag(dag: "WorkloadDAG") -> None:
    """检查 WorkloadDAG 的语义约束。违反则抛 ValueError（中文信息）。"""

    violations: List[str] = []
    warnings: List[str] = []

    def E(msg: str) -> None:
        violations.append(msg)

    def W(msg: str) -> None:
        warnings.append(msg)

    # 0) Spec 基本一致性
    try:
        dag.spec.validate()
    except Exception as e:
        E(f"[E] spec.validate() 失败：{e}")

    # 1) edge 端点必须存在
    for u, v in dag.edges:
        if u not in dag.ops:
            E(f"[E] 边 ({u!r} -> {v!r}) 的源算子 {u!r} 不存在于 dag.ops。")

        if v not in dag.ops:
            E(f"[E] 边 ({u!r} -> {v!r}) 的目的算子 {v!r} 不存在于 dag.ops。")

    # 2) DAG 必须无环
    try:
        _topo_order_or_raise(set(dag.ops.keys()), dag.edges)
    except Exception as e:
        E(f"[E] DAG 拓扑检查失败：{e}")

    # 3) decode DAG 必须包含 KVRead/KVWrite 且 state_desc 完整
    if dag.spec.mode == "decode":
        op_types = {op.op_type for op in dag.ops.values()}

        if "KVRead" not in op_types:
            E("[E] decode WorkloadDAG 必须包含 KVRead（建模从 DRAM/状态中读取历史 KV）。")

        if "KVWrite" not in op_types:
            E("[E] decode WorkloadDAG 必须包含 KVWrite（建模把更新后的 KVCache 写回 DRAM/状态）。")

        if dag.state_desc.get("state_name") != "KVCache":
            E(f"[E] decode WorkloadDAG 必须满足 state_desc['state_name']=='KVCache'。当前 state_desc={dag.state_desc!r}。")

    # 4) 每条 edge 都必须在 _edge_tensor 注册并指向 dag.tensors 中存在的张量名
    for edge in dag.edges:
        if edge not in dag._edge_tensor:
            u, v = edge
            E(f"[E] 边 ({u!r} -> {v!r}) 没有在 dag._edge_tensor 注册。构图 builder 必须为每条边填充 _edge_tensor。")
            continue

        tname = dag._edge_tensor[edge]
        if tname not in dag.tensors:
            u, v = edge
            E(f"[E] 边 ({u!r} -> {v!r}) 引用张量 {tname!r}，但该张量不在 dag.tensors 中注册。")

    # 5) tensor 名称唯一性与基本自洽（不做过度 shape 检查）
    if len(set(dag.tensors.keys())) != len(dag.tensors):
        E("[E] dag.tensors 存在重复 key（理论上 dict 不会，但若由外部 merge 产生异常对象，应视为错误）。")

    # 6) OpSpec einsum-like 语义一致性
    for op in dag.ops.values():
        _validate_op_einsum_semantics(op, dag.tensors, violations)

        # 若存在 einsum-like 语义，则建议 loop_dims 覆盖这些变量（非强制）
        if getattr(op, "has_einsum_semantics", lambda: False)():
            iv = set(op.index_vars)
            ld = set((op.loop_dims or {}).keys())
            if ld and iv and (iv != ld):
                # 仅 warning：允许 loop_dims 使用不同命名体系
                W(f"[W] op={op.name}: index_vars={list(op.index_vars)} 与 loop_dims keys={sorted(ld)} 不一致。若后续 tiler 使用 index_vars，需保证维度名映射层存在。")

    if violations:
        raise ValueError("\n".join(violations + warnings))

    return


# ─────────────────────────────────────────────────────────────
# TilingGene validation (工程化：收集 violations 一次性抛错)
# ─────────────────────────────────────────────────────────────

def validate_tiling_gene(
    dag: "WorkloadDAG",
    fusion: Any,
    tiling: Any,
    arch: Optional[Any] = None,
) -> None:
    """
    检查 TilingGene 的语义合法性。

    重要约定：
    - 这里的检查是“语义/结构级”，不计算性能
    - 与硬件强相关的检查（如 SRAM 容量/带宽）在 arch 提供时才启用（并预留字段接口）
    - 允许 tile 不能整除维度长度，尾块 remainder 单独处理（因此整除性仅给 warning）

    抛错方式：
    - 收集 violations（错误）与 warnings（建议），最后一次性抛出所有错误，便于调参和搜索剪枝。
    """

    violations: List[str] = []
    warnings: List[str] = []

    def E(msg: str) -> None:
        violations.append(msg)

    def W(msg: str) -> None:
        warnings.append(msg)

    # ---------- 0) 基本空值检查 ----------
    if dag is None:
        E("[E] dag 为空。")

    if fusion is None:
        E("[E] fusion_gene 为空。")

    if tiling is None:
        E("[E] tiling_gene 为空。")

    if violations:
        raise ValueError("\n".join(violations))

    # ---------- 1) 取出 fusion groups / tiling group spec ----------
    groups = getattr(fusion, "groups", None)
    if not groups:
        E("[E] fusion_gene.groups 为空，无法校验 tiling。")

    group_tiles = getattr(tiling, "group_tiles", None)
    if group_tiles is None:
        E("[E] tiling_gene.group_tiles 不存在或为空。")

    gp = getattr(tiling, "global_policy", None)
    if gp is None:
        E("[E] tiling_gene.global_policy 不存在。")

    if violations:
        raise ValueError("\n".join(violations))

    # kv_page_tokens（decode 友好检查）
    kv_page_tokens = getattr(gp, "kv_page_tokens", None)
    if kv_page_tokens is None:
        kv_page_tokens = 256  # 默认值兜底
        W("[W] global_policy.kv_page_tokens 未设置，默认按 256 处理。")

    # ---------- 2) 读取 arch 的预留字段接口（有则用，无则跳过） ----------
    # SRAM 容量（字节）
    Csram_bytes: Optional[int] = None
    sram_kb = _get_arch_field(arch, "sram_capacity_kb", None)
    if sram_kb is not None:
        try:
            Csram_bytes = int(sram_kb) * 1024
        except Exception:
            W(f"[W] arch.sram_capacity_kb={sram_kb!r} 不是合法整数，跳过 SRAM 容量检查。")

    # 预留：DRAM 最小请求粒度
    _ = _get_arch_field(arch, "dram_req_bytes", None)

    # ---------- 3) 对每个 group 做检查 ----------
    for g in groups:
        gid = getattr(g, "group_id", None)
        op_names = getattr(g, "op_names", None) or []

        if not gid:
            E("[E] fusion_gene 存在 group_id 为空的 group。")
            continue

        if gid not in group_tiles:
            E(f"[E] group={gid}: tiling_gene 缺少该 group 的配置（group_tiles 中找不到）。")
            continue

        gspec = group_tiles[gid]

        # 3.1 group 必须有 DRAM/SRAM tile spec
        tiles = getattr(gspec, "tiles", None)
        if not isinstance(tiles, dict):
            E(f"[E] group={gid}: group_tiles[{gid}].tiles 不是 dict。")
            continue

        if "DRAM" not in tiles:
            E(f"[E] group={gid}: 缺少 DRAM 的 tile 配置（tiles['DRAM']）。")
        if "SRAM" not in tiles:
            E(f"[E] group={gid}: 缺少 SRAM 的 tile 配置（tiles['SRAM']）。")

        if "DRAM" not in tiles or "SRAM" not in tiles:
            continue

        dram = tiles["DRAM"]
        sram = tiles["SRAM"]

        # 3.2 计算该 group 的关键维度 + 维度长度（extent）
        key_dims: Set[str] = set()
        extent: Dict[str, int] = {}

        if not op_names:
            W(f"[W] group={gid}: group.op_names 为空，无法推断关键维度覆盖性。将跳过 tile_size 覆盖检查。")
        else:
            for op_name in op_names:
                if op_name not in dag.ops:
                    E(f"[E] group={gid}: group 中包含算子 {op_name!r}，但 dag.ops 不存在该算子。")
                    continue
                op = dag.ops[op_name]
                loop_dims = getattr(op, "loop_dims", None) or {}
                for d, n in loop_dims.items():
                    key_dims.add(d)
                    try:
                        nn = int(n)
                        extent[d] = max(extent.get(d, 0), nn)
                    except Exception:
                        E(f"[E] group={gid} op={op_name}: loop_dims[{d}]={n!r} 不是合法整数。")

        # 3.3 每层：tile_size / loop_order 基础一致性
        def _check_one_level(level: str, spec: Any) -> Tuple[Dict[str, int], List[str]]:
            ts = getattr(spec, "tile_size", None)
            lo = getattr(spec, "loop_order", None)

            if not isinstance(ts, dict):
                E(f"[E] group={gid} level={level}: tile_size 不是 dict。")
                return {}, []

            if not isinstance(lo, list):
                E(f"[E] group={gid} level={level}: loop_order 不是 list。")
                return ts, []

            # loop_order 必须与 tile_size 的维度集合完全一致（不能少不能多）
            if set(lo) != set(ts.keys()):
                missing = sorted(set(ts.keys()) - set(lo))
                extra = sorted(set(lo) - set(ts.keys()))
                if missing:
                    E(f"[E] group={gid} level={level}: loop_order 缺少维度 {missing}（tile_size 已声明）。")
                if extra:
                    E(f"[E] group={gid} level={level}: loop_order 包含未在 tile_size 声明的维度 {extra}。")

            # 重复维度
            if len(set(lo)) != len(lo):
                E(f"[E] group={gid} level={level}: loop_order 出现重复维度：{lo}")

            # tile_size 值必须是正整数（兜底）
            for d, v in ts.items():
                if not isinstance(d, str) or not d:
                    E(f"[E] group={gid} level={level}: tile_size 存在非法维度名：{d!r}")
                    continue
                if not isinstance(v, int) or v <= 0:
                    E(f"[E] group={gid} level={level}: tile_size[{d}] 必须是正整数，当前={v!r}")

            return ts, lo

        dram_ts, _ = _check_one_level("DRAM", dram)
        sram_ts, _ = _check_one_level("SRAM", sram)

        # 3.4 tile_size 必须覆盖关键维度（强制）
        if key_dims:
            for lvl, ts in [("DRAM", dram_ts), ("SRAM", sram_ts)]:
                missing = sorted(key_dims - set(ts.keys()))
                if missing:
                    E(f"[E] group={gid} level={lvl}: tile_size 缺少关键维度 {missing}。")

        # 3.5 SRAM tile 不得大于 DRAM tile（按关键维度）
        for d in sorted(key_dims):
            if d in dram_ts and d in sram_ts:
                if sram_ts[d] > dram_ts[d]:
                    E(f"[E] group={gid}: SRAM tile_size[{d}]={sram_ts[d]} 大于 DRAM tile_size[{d}]={dram_ts[d]}。")

        # 3.6 tile_size 不得超过实际维度长度；不整除给 warning（尾块单独处理）
        for lvl, ts in [("DRAM", dram_ts), ("SRAM", sram_ts)]:
            for d in sorted(key_dims):
                if d not in ts:
                    continue
                if d not in extent or extent[d] <= 0:
                    W(f"[W] group={gid} level={lvl}: 无法获得维度 {d} 的实际长度（extent），跳过上界/尾块检查。")
                    continue

                if ts[d] > extent[d]:
                    E(f"[E] group={gid} level={lvl}: tile_size[{d}]={ts[d]} 超过实际维度长度 extent[{d}]={extent[d]}。")
                else:
                    rem = extent[d] % ts[d]
                    if rem != 0:
                        W(f"[W] group={gid} level={lvl}: 维度 {d} 长度 {extent[d]} 不能被 tile_size {ts[d]} 整除，尾块 remainder={rem} 将单独处理。")

        # 3.7 decode 下 KV page 对齐（先 warning，不判死）
        if getattr(dag.spec, "mode", None) == "decode" and "KV" in key_dims:
            if "KV" in sram_ts and kv_page_tokens:
                if sram_ts["KV"] % int(kv_page_tokens) != 0:
                    W(f"[W] group={gid}: decode 场景下建议 SRAM 的 KV tile_size={sram_ts['KV']} 与 kv_page_tokens={kv_page_tokens} 对齐（整倍数）。")

        # 3.8 IntermediatePolicy 逻辑一致性（跨组必须可写回；声明 DRAM 不能永不写回）
        inter = getattr(gspec, "intermediate", None)
        if inter is not None:
            residency = getattr(inter, "residency", {}) or {}
            default_pol = getattr(inter, "default", None)

            items: List[Tuple[str, Any]] = list(residency.items())
            if default_pol is not None:
                items.append(("<default>", default_pol))

            for k, p in items:
                where = getattr(p, "where", None)
                writeback = getattr(p, "writeback", None)
                lifetime = getattr(p, "lifetime", None)

                if lifetime == "across_group" and writeback == "never":
                    E(f"[E] group={gid} intermediate={k}: lifetime=across_group 但 writeback=never（跨组传递不写回是不可能的）。")

                if where == "DRAM" and writeback == "never":
                    E(f"[E] group={gid} intermediate={k}: where=DRAM 但 writeback=never（声明放 DRAM 却从不写回，自相矛盾）。")

                if lifetime == "across_group" and where == "PE":
                    W(f"[W] group={gid} intermediate={k}: lifetime=across_group 但 where=PE，通常 PE 容量很小，建议改为 SRAM/DRAM 或调整策略。")

        # 3.9 SRAM 预算字段与 arch 容量接口（先做弱检查/预留）
        sram_budget = getattr(gspec, "sram_budget_bytes", None)
        if sram_budget is not None:
            try:
                sb = int(sram_budget)
                if sb <= 0:
                    E(f"[E] group={gid}: sram_budget_bytes 必须为正整数或 None，当前={sram_budget!r}。")
                if Csram_bytes is not None and sb > Csram_bytes:
                    W(f"[W] group={gid}: sram_budget_bytes={sb}B 大于硬件 SRAM 总容量 Csram_bytes={Csram_bytes}B，将按 Csram_bytes 为上限处理。")
            except Exception:
                E(f"[E] group={gid}: sram_budget_bytes={sram_budget!r} 不是合法整数。")

        
        # 3.10 shared_tensors_by_level (dp-first contract)
        shared_by_lvl = getattr(gspec, "shared_tensors_by_level", None) or {}

        if not isinstance(shared_by_lvl, dict):
            E(f"[E] group={gid}: shared_tensors_by_level 必须是 dict(mem_level -> list[tensor])，当前={type(shared_by_lvl)}。")
        else:
            # 当前 builder 只会把 SRAM 的 shared 编译成 Sharing scope
            strong_levels = {"SRAM"}
            weak_levels = set(shared_by_lvl.keys()) - strong_levels

            # 计算本 group 的“可见张量集合”：所有 op 的 reads ∪ writes
            group_visible_tensors: Set[str] = set()
            for op_name in op_names:
                if op_name not in dag.ops:
                    continue
                op = dag.ops[op_name]
                group_visible_tensors |= set(getattr(op, "reads", ()) or ())
                group_visible_tensors |= set(getattr(op, "writes", ()) or ())

            # 弱层级提示（不会强迫你现在就是四级存储）
            for lvl in sorted(weak_levels):
                W(
                    f"[W] group={gid}: shared_tensors_by_level[{lvl}] 被声明。"
                    f" 注意：当前 builder 只会对 SRAM 插入 Sharing 边界；"
                    f" {lvl} 要生效需要未来在树里引入对应层级的 Tile/Scope 编译规则。"
                )

            # 对所有层级做基础类型检查；对 SRAM 做更严格语义检查
            for lvl, names in shared_by_lvl.items():
                if not isinstance(lvl, str) or not lvl:
                    E(f"[E] group={gid}: shared_tensors_by_level 出现非法 mem_level key：{lvl!r}。")
                    continue
                if names is None:
                    continue
                if not isinstance(names, list):
                    E(f"[E] group={gid}: shared_tensors_by_level[{lvl}] 必须是 list，当前={type(names)}。")
                    continue

                if len(names) != len(set(names)):
                    W(f"[W] group={gid}: shared_tensors_by_level[{lvl}] 存在重复 tensor 名，将在 canonicalize 中去重。")

                for t in names:
                    if not isinstance(t, str) or not t:
                        E(f"[E] group={gid}: shared_tensors_by_level[{lvl}] 出现非法 tensor 名：{t!r}。")
                        continue
                    # 只对 SRAM 强制“必须在该 group 可见”
                    if lvl in strong_levels and group_visible_tensors and t not in group_visible_tensors:
                        E(
                            f"[E] group={gid}: shared tensor {t!r} 不在该 group 的 reads/writes 可见集合中。"
                            f" 可见张量={sorted(group_visible_tensors)}。"
                        )

            # end_of_group 必须落在 SRAM Sharing 边界（因为当前只有这一个结构承载）
            inter = getattr(gspec, "intermediate", None)
            if inter is not None:
                residency = getattr(inter, "residency", {}) or {}
                end_of_group_tensors: Set[str] = set()
                for k, pol in residency.items():
                    if getattr(pol, "writeback", None) == "end_of_group":
                        end_of_group_tensors.add(k)

                shared_sram = set(shared_by_lvl.get("SRAM", []) or [])
                missing = sorted(end_of_group_tensors - shared_sram)
                if missing:
                    E(
                        f"[E] group={gid}: intermediate 中声明 writeback=end_of_group 的张量 {missing} "
                        f"必须出现在 shared_tensors_by_level['SRAM'] 中，"
                        f"否则 builder 无法生成 SRAM Sharing 边界（结构缺失，易静默错误）。"
                    )
    
    # ---------- 4) 统一抛错（工程化） ----------
    if violations:
        msg = "\n".join(violations + warnings)
        raise ValueError(msg)

    return
