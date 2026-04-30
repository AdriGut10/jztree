from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .config import PairCountConfig
from .data import InteractionList, Pos, PosLvl, verify_ilist
from .jax_ext import raise_if
from .stats import AllocStats, stats_callback
from .tools import masked_to_dense
from .tree import grouped_dense_interaction_list, zsort_and_tree, zsort_and_tree_multi_type

from jztree_cuda import ffi_paircount

jax.ffi.register_ffi_target("PairCountNode2Node", ffi_paircount.PairCountNode2Node(), platform="CUDA")
jax.ffi.register_ffi_target("PairCountLeaf2LeafR", ffi_paircount.PairCountLeaf2LeafR(), platform="CUDA")
jax.ffi.register_ffi_target("PairCountLeaf2LeafAniso", ffi_paircount.PairCountLeaf2LeafAniso(), platform="CUDA")

__all__ = ["DD", "DDrppi", "DDsmu"]

_NODE_BLOCK_SIZE = 64


@dataclass(unsafe_hash=True, frozen=True)
class _BinSpec:
    edges: tuple[float, ...]
    origin: float
    inv_step: float
    log_bins: bool


@dataclass(unsafe_hash=True, frozen=True)
class _RppiSpec:
    rp: _BinSpec
    pi: _BinSpec


@dataclass(unsafe_hash=True, frozen=True)
class _SmuSpec:
    s: _BinSpec
    mu: _BinSpec


@dataclass(unsafe_hash=True, frozen=True)
class _PairCountRuntimeConfig:
    boxsize: float
    los_axis: int
    same_catalog: bool
    r: _BinSpec | None = None
    rppi: _RppiSpec | None = None
    smu: _SmuSpec | None = None


def _format_bin_spec(
    spec,
    *,
    allow_log: bool,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> _BinSpec:
    if not isinstance(spec, tuple) or len(spec) != 4:
        raise ValueError(f"{name} must be a tuple (lower, upper, nbins, spacing).")

    lower, upper, nbins, spacing = spec
    if spacing not in ("lin", "log"):
        raise ValueError(f"{name} spacing must be 'lin' or 'log'.")
    if spacing == "log" and not allow_log:
        raise ValueError(f"{name} only supports 'lin' spacing.")
    if not isinstance(nbins, (int, np.integer)) or int(nbins) < 1:
        raise ValueError(f"{name} nbins must be a positive integer.")

    lower = float(lower)
    upper = float(upper)
    nbins = int(nbins)
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError(f"{name} bounds must be finite and strictly increasing.")
    if min_value is not None and lower < min_value:
        raise ValueError(f"{name} lower bound must be >= {min_value}.")
    if max_value is not None and upper > max_value:
        raise ValueError(f"{name} upper bound must be <= {max_value}.")
    if spacing == "log" and lower <= 0.0:
        raise ValueError(f"{name} log bins require a positive lower bound.")

    if spacing == "log":
        edges = np.geomspace(lower, upper, nbins + 1, dtype=np.float64)
        origin = float(np.log(lower))
        inv_step = float(nbins / (np.log(upper) - np.log(lower)))
        log_bins = True
    else:
        edges = np.linspace(lower, upper, nbins + 1, dtype=np.float64)
        origin = lower
        inv_step = float(nbins / (upper - lower))
        log_bins = False

    return _BinSpec(
        edges=tuple(float(x) for x in edges),
        origin=origin,
        inv_step=inv_step,
        log_bins=log_bins,
    )


def _compact_points(pos: jax.Array, valid: jax.Array | None, num: jax.Array | int | None) -> tuple[jax.Array, jax.Array]:
    pos = jnp.asarray(pos)
    if valid is None and num is None:
        return pos, jnp.asarray(pos.shape[0], dtype=jnp.int32)

    n = pos.shape[0]
    mask = jnp.ones((n,), dtype=bool)
    if num is not None:
        num = jnp.clip(jnp.asarray(num, dtype=jnp.int32), 0, n)
        mask = mask & (jnp.arange(n, dtype=jnp.int32) < num)
    if valid is not None:
        valid = jnp.asarray(valid, dtype=bool)
        if valid.ndim != 1 or valid.shape[0] != n:
            raise ValueError("valid must be a one-dimensional mask with the same length as pos.")
        mask = mask & valid

    pos_compact, num_compact = masked_to_dense(pos, mask, fill_value=jnp.nan)
    return pos_compact, jnp.asarray(num_compact, dtype=jnp.int32)


def _paircount_node2node_ilist(
    ilist: InteractionList,
    spl_parent: jax.Array,
    node_data: PosLvl,
    node_qcount: jax.Array,
    node_scount: jax.Array,
    *,
    same_catalog: bool,
    use_r: bool,
    rmax: float,
    use_rppi: bool,
    rpmax: float,
    pimax: float,
    use_smu: bool,
    smax: float,
    los_axis: int,
    boxsize: float,
    block_size: int = _NODE_BLOCK_SIZE,
) -> InteractionList:
    assert ilist.ispl.shape[0] == spl_parent.shape[0]
    assert len(node_data.pos) == len(node_qcount) == len(node_scount)
    assert ilist.iother.dtype == ilist.ispl.dtype == spl_parent.dtype == node_qcount.dtype == node_scount.dtype == jnp.int32

    size = len(node_data.pos)
    outputs = (
        jax.ShapeDtypeStruct((size + 1,), jnp.int32),
        jax.ShapeDtypeStruct((ilist.size(),), jnp.int32),
    )

    ispl, iother = jax.ffi.ffi_call("PairCountNode2Node", outputs)(
        ilist.ispl,
        ilist.iother,
        spl_parent,
        node_data.pos_lvl(),
        node_qcount,
        node_scount,
        same_catalog=same_catalog,
        use_r=use_r,
        use_rppi=use_rppi,
        use_smu=use_smu,
        los_axis=np.int32(los_axis),
        boxsize=float(boxsize),
        rmax=float(rmax),
        rpmax=float(rpmax),
        pimax=float(pimax),
        smax=float(smax),
        block_size=np.int32(block_size),
        dim=np.int32(node_data.pos.shape[-1]),
    )

    ispl = ispl + raise_if(
        ispl[-1] > iother.size,
        "The interaction list allocation is too small. (need: {n1} have: {n2})\n"
        "Hint: increase alloc_fac_ilist at least by a factor of {ratio:.1f}",
        n1=ispl[-1],
        n2=iother.size,
        ratio=ispl[-1] / iother.size,
    )

    stats_callback("allocation", AllocStats.record_filled_interactions, ispl[-1], iother.size)

    return verify_ilist(InteractionList(ispl, iother))


_paircount_node2node_ilist.jit = jax.jit(
    _paircount_node2node_ilist,
    static_argnames=(
        "same_catalog",
        "use_r",
        "rmax",
        "use_rppi",
        "rpmax",
        "pimax",
        "use_smu",
        "smax",
        "los_axis",
        "boxsize",
        "block_size",
    ),
)


def _paircount_dual_walk(
    th,
    *,
    ptype_query: int,
    ptype_source: int,
    same_catalog: bool,
    use_r: bool,
    rmax: float,
    use_rppi: bool,
    rpmax: float,
    pimax: float,
    use_smu: bool,
    smax: float,
    los_axis: int,
    boxsize: float,
    alloc_fac_ilist: float,
) -> InteractionList:
    nlevels = th.num_planes()
    size = th.size()

    spl, ilist, nsup = grouped_dense_interaction_list(
        th.lvl.num(nlevels - 1),
        int(th.size_leaves * alloc_fac_ilist),
        ngroup=32,
        size_super=size,
    )

    for i in range(nlevels):
        level = nlevels - i - 1
        parent_spl = spl if i == 0 else th.ispl_n2n.get(level + 1, size + 1)
        node_data = PosLvl(pos=th.geom_cent.get(level, size), lvl=th.lvl.get(level, size))
        node_qcount = th.npart(level, ptype=ptype_query, size=size)
        node_scount = th.npart(level, ptype=ptype_source, size=size)
        valid_node = jnp.arange(size, dtype=jnp.int32) < th.num(level)
        node_qcount = jnp.where(valid_node, node_qcount, 0)
        node_scount = jnp.where(valid_node, node_scount, 0)
        ilist = _paircount_node2node_ilist(
            ilist,
            parent_spl,
            node_data,
            node_qcount,
            node_scount,
            same_catalog=same_catalog,
            use_r=use_r,
            rmax=rmax,
            use_rppi=use_rppi,
            rpmax=rpmax,
            pimax=pimax,
            use_smu=use_smu,
            smax=smax,
            los_axis=los_axis,
            boxsize=boxsize,
        )

    return ilist


_paircount_dual_walk.jit = jax.jit(
    _paircount_dual_walk,
    static_argnames=(
        "ptype_query",
        "ptype_source",
        "same_catalog",
        "use_r",
        "rmax",
        "use_rppi",
        "rpmax",
        "pimax",
        "use_smu",
        "smax",
        "los_axis",
        "boxsize",
        "alloc_fac_ilist",
    ),
)


def _paircount_leaf2leaf_r(
    ilist: InteractionList,
    splT: jax.Array,
    xT: jax.Array,
    *,
    r: _BinSpec,
    same_catalog: bool,
    boxsize: float,
    block_size: int,
    splQ: jax.Array | None = None,
    xQ: jax.Array | None = None,
) -> jax.Array:
    if splQ is None:
        splQ = splT
    if xQ is None:
        xQ = xT

    outputs = (jax.ShapeDtypeStruct((len(r.edges) - 1,), jnp.int32),)
    return jax.ffi.ffi_call("PairCountLeaf2LeafR", outputs)(
        ilist.ispl,
        ilist.iother,
        splT,
        xT,
        splQ,
        xQ,
        same_catalog=same_catalog,
        boxsize=float(boxsize),
        rlower=float(r.edges[0]),
        rupper=float(r.edges[-1]),
        rorigin=float(r.origin),
        rinv_step=float(r.inv_step),
        rlog_bins=r.log_bins,
        block_size=np.int32(block_size),
    )[0]


_paircount_leaf2leaf_r.jit = jax.jit(
    _paircount_leaf2leaf_r,
    static_argnames=("r", "same_catalog", "boxsize", "block_size"),
)


def _paircount_leaf2leaf_aniso(
    ilist: InteractionList,
    splT: jax.Array,
    xT: jax.Array,
    *,
    config: _PairCountRuntimeConfig,
    block_size: int,
    splQ: jax.Array | None = None,
    xQ: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if splQ is None:
        splQ = splT
    if xQ is None:
        xQ = xT

    if config.rppi is None:
        rppi_shape = (0, 0)
        rp_lower = rp_upper = rp_origin = 0.0
        rp_inv_step = 1.0
        rp_log_bins = False
        pi_lower = pi_upper = pi_origin = 0.0
        pi_inv_step = 1.0
    else:
        rppi_shape = (len(config.rppi.rp.edges) - 1, len(config.rppi.pi.edges) - 1)
        rp_lower = float(config.rppi.rp.edges[0])
        rp_upper = float(config.rppi.rp.edges[-1])
        rp_origin = float(config.rppi.rp.origin)
        rp_inv_step = float(config.rppi.rp.inv_step)
        rp_log_bins = config.rppi.rp.log_bins
        pi_lower = float(config.rppi.pi.edges[0])
        pi_upper = float(config.rppi.pi.edges[-1])
        pi_origin = float(config.rppi.pi.origin)
        pi_inv_step = float(config.rppi.pi.inv_step)

    if config.smu is None:
        smu_shape = (0, 0)
        s_lower = s_upper = s_origin = 0.0
        s_inv_step = 1.0
        s_log_bins = False
        mu_lower = mu_upper = mu_origin = 0.0
        mu_inv_step = 1.0
    else:
        smu_shape = (len(config.smu.s.edges) - 1, len(config.smu.mu.edges) - 1)
        s_lower = float(config.smu.s.edges[0])
        s_upper = float(config.smu.s.edges[-1])
        s_origin = float(config.smu.s.origin)
        s_inv_step = float(config.smu.s.inv_step)
        s_log_bins = config.smu.s.log_bins
        mu_lower = float(config.smu.mu.edges[0])
        mu_upper = float(config.smu.mu.edges[-1])
        mu_origin = float(config.smu.mu.origin)
        mu_inv_step = float(config.smu.mu.inv_step)

    outputs = (
        jax.ShapeDtypeStruct(rppi_shape, jnp.int32),
        jax.ShapeDtypeStruct(smu_shape, jnp.int32),
    )
    return jax.ffi.ffi_call("PairCountLeaf2LeafAniso", outputs)(
        ilist.ispl,
        ilist.iother,
        splT,
        xT,
        splQ,
        xQ,
        same_catalog=config.same_catalog,
        use_rppi=config.rppi is not None,
        use_smu=config.smu is not None,
        los_axis=np.int32(config.los_axis),
        boxsize=float(config.boxsize),
        rp_lower=rp_lower,
        rp_upper=rp_upper,
        rp_origin=rp_origin,
        rp_inv_step=rp_inv_step,
        rp_log_bins=rp_log_bins,
        pi_lower=pi_lower,
        pi_upper=pi_upper,
        pi_origin=pi_origin,
        pi_inv_step=pi_inv_step,
        s_lower=s_lower,
        s_upper=s_upper,
        s_origin=s_origin,
        s_inv_step=s_inv_step,
        s_log_bins=s_log_bins,
        mu_lower=mu_lower,
        mu_upper=mu_upper,
        mu_origin=mu_origin,
        mu_inv_step=mu_inv_step,
        block_size=np.int32(block_size),
    )


_paircount_leaf2leaf_aniso.jit = jax.jit(
    _paircount_leaf2leaf_aniso,
    static_argnames=("config", "block_size"),
)


def _prepare_paircount_inputs(
    pos1: jax.Array,
    pos2: jax.Array,
    *,
    num1: jax.Array,
    num2: jax.Array,
    same_catalog: bool,
    cfg: PairCountConfig,
):
    part1 = Pos(pos=pos1, num=num1)
    if same_catalog:
        part1z, th = zsort_and_tree.jit(part1, cfg_tree=cfg.tree)
        return part1z, part1z, th, 0, 0

    part2 = Pos(pos=pos2, num=num2)
    partzs, th = zsort_and_tree_multi_type.jit((part1, part2), cfg_tree=cfg.tree)
    return partzs[0], partzs[1], th, 0, 1


def _pair_counts(
    pos1: jax.Array,
    pos2: jax.Array,
    config: _PairCountRuntimeConfig,
    *,
    num1: jax.Array,
    num2: jax.Array,
    block_size: int = 128,
    cfg: PairCountConfig = PairCountConfig(),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    partz_q, partz_t, th, ptype_query, ptype_source = _prepare_paircount_inputs(
        pos1,
        pos2,
        num1=num1,
        num2=num2,
        same_catalog=config.same_catalog,
        cfg=cfg,
    )

    pimax = 0.0
    if config.rppi is not None:
        pimax = max(
            abs(float(config.rppi.pi.edges[0])),
            abs(float(config.rppi.pi.edges[-1])),
        )
    ilist = _paircount_dual_walk.jit(
        th,
        ptype_query=ptype_query,
        ptype_source=ptype_source,
        same_catalog=config.same_catalog,
        use_r=config.r is not None,
        rmax=0.0 if config.r is None else float(config.r.edges[-1]),
        use_rppi=config.rppi is not None,
        rpmax=0.0 if config.rppi is None else float(config.rppi.rp.edges[-1]),
        pimax=pimax,
        use_smu=config.smu is not None,
        smax=0.0 if config.smu is None else float(config.smu.s.edges[-1]),
        los_axis=config.los_axis,
        boxsize=float(config.boxsize),
        alloc_fac_ilist=cfg.alloc_fac_ilist,
    )

    splQ = th.splits_leaf_to_part(ptype=ptype_query)
    splT = th.splits_leaf_to_part(ptype=ptype_source)

    if config.r is None:
        dd_r = jnp.zeros((0,), dtype=jnp.int32)
    else:
        dd_r = _paircount_leaf2leaf_r.jit(
            ilist,
            splT,
            partz_t.pos,
            r=config.r,
            same_catalog=config.same_catalog,
            boxsize=float(config.boxsize),
            block_size=block_size,
            splQ=splQ,
            xQ=partz_q.pos,
        )

    if config.rppi is None and config.smu is None:
        dd_rppi = jnp.zeros((0, 0), dtype=jnp.int32)
        dd_smu = jnp.zeros((0, 0), dtype=jnp.int32)
    else:
        dd_rppi, dd_smu = _paircount_leaf2leaf_aniso.jit(
            ilist,
            splT,
            partz_t.pos,
            config=config,
            block_size=block_size,
            splQ=splQ,
            xQ=partz_q.pos,
        )

    return dd_r, dd_rppi, dd_smu


_pair_counts.jit = jax.jit(_pair_counts, static_argnames=("config", "block_size", "cfg"))


def _prepare_call_inputs(pos1, pos2, valid1, valid2, num1, num2):
    pos1_in = pos1
    same_catalog = pos2 is None or pos2 is pos1_in

    pos1 = jnp.asarray(pos1)
    if pos1.ndim != 2 or pos1.shape[-1] != 3:
        raise ValueError("pos1 must have shape (N, 3).")
    dtype = jnp.result_type(pos1.dtype, jnp.float32)
    pos1 = jnp.asarray(pos1, dtype=dtype)
    if same_catalog:
        pos2 = pos1
    else:
        pos2 = jnp.asarray(pos2, dtype=dtype)
        if pos2.ndim != 2 or pos2.shape[-1] != 3:
            raise ValueError("pos2 must have shape (N, 3).")

    pos1, num1 = _compact_points(pos1, valid1, num1)
    if same_catalog:
        pos2, num2 = pos1, num1
    else:
        pos2, num2 = _compact_points(pos2, valid2, num2)

    return pos1, pos2, num1, num2, same_catalog


def _run_pair_counts(
    pos1,
    *,
    boxsize: float,
    pos2=None,
    valid1=None,
    valid2=None,
    num1=None,
    num2=None,
    los_axis: int = 2,
    r: _BinSpec | None = None,
    rppi: _RppiSpec | None = None,
    smu: _SmuSpec | None = None,
    block_size: int = 128,
    cfg: PairCountConfig = PairCountConfig(),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if boxsize is None:
        raise ValueError("boxsize is required.")
    if los_axis not in (0, 1, 2):
        raise ValueError("los_axis must be 0, 1, or 2.")

    pos1, pos2, num1, num2, same_catalog = _prepare_call_inputs(pos1, pos2, valid1, valid2, num1, num2)

    return _pair_counts.jit(
        pos1,
        pos2,
        _PairCountRuntimeConfig(
            boxsize=float(boxsize),
            los_axis=los_axis,
            same_catalog=same_catalog,
            r=r,
            rppi=rppi,
            smu=smu,
        ),
        num1=num1,
        num2=num2,
        block_size=block_size,
        cfg=cfg,
    )


def DD(
    pos1,
    r,
    *,
    boxsize: float,
    pos2=None,
    valid1=None,
    valid2=None,
    num1=None,
    num2=None,
    block_size: int = 128,
    cfg: PairCountConfig = PairCountConfig(),
) -> jax.Array:
    """Raw periodic pair counts in radial bins.

    Bins are specified as ``(lower, upper, nbins, "lin"|"log")``.
    The final upper radial edge is included.
    """
    r = _format_bin_spec(r, allow_log=True, name="r", min_value=0.0)
    return _run_pair_counts(
        pos1,
        boxsize=boxsize,
        pos2=pos2,
        valid1=valid1,
        valid2=valid2,
        num1=num1,
        num2=num2,
        r=r,
        block_size=block_size,
        cfg=cfg,
    )[0]


def DDrppi(
    pos1,
    rp,
    pi,
    *,
    boxsize: float,
    pos2=None,
    valid1=None,
    valid2=None,
    num1=None,
    num2=None,
    los_axis: int = 2,
    block_size: int = 128,
    cfg: PairCountConfig = PairCountConfig(),
) -> jax.Array:
    """Raw periodic pair counts in projected-distance and signed-pi bins.

    Bins are specified as ``(lower, upper, nbins, "lin"|"log")`` for ``rp``
    and ``(lower, upper, nbins, "lin")`` for signed ``pi``.
    The final upper ``rp`` and ``pi`` edges are included.
    """
    rppi = _RppiSpec(
        rp=_format_bin_spec(rp, allow_log=True, name="rp", min_value=0.0),
        pi=_format_bin_spec(pi, allow_log=False, name="pi"),
    )
    return _run_pair_counts(
        pos1,
        boxsize=boxsize,
        pos2=pos2,
        valid1=valid1,
        valid2=valid2,
        num1=num1,
        num2=num2,
        los_axis=los_axis,
        rppi=rppi,
        block_size=block_size,
        cfg=cfg,
    )[1]


def DDsmu(
    pos1,
    s,
    mu,
    *,
    boxsize: float,
    pos2=None,
    valid1=None,
    valid2=None,
    num1=None,
    num2=None,
    los_axis: int = 2,
    block_size: int = 128,
    cfg: PairCountConfig = PairCountConfig(),
) -> jax.Array:
    """Raw periodic pair counts in separation and signed-mu bins.

    Bins are specified as ``(lower, upper, nbins, "lin"|"log")`` for ``s``
    and ``(lower, upper, nbins, "lin")`` for signed ``mu``.
    The final upper ``s`` edge is included; ``mu`` endpoints are excluded.
    """
    smu = _SmuSpec(
        s=_format_bin_spec(s, allow_log=True, name="s", min_value=0.0),
        mu=_format_bin_spec(mu, allow_log=False, name="mu", min_value=-1.0, max_value=1.0),
    )
    return _run_pair_counts(
        pos1,
        boxsize=boxsize,
        pos2=pos2,
        valid1=valid1,
        valid2=valid2,
        num1=num1,
        num2=num2,
        los_axis=los_axis,
        smu=smu,
        block_size=block_size,
        cfg=cfg,
    )[2]


DD.jit = jax.jit(DD, static_argnames=("r", "boxsize", "block_size", "cfg"))
DDrppi.jit = jax.jit(
    DDrppi,
    static_argnames=("rp", "pi", "boxsize", "los_axis", "block_size", "cfg"),
)
DDsmu.jit = jax.jit(
    DDsmu,
    static_argnames=("s", "mu", "boxsize", "los_axis", "block_size", "cfg"),
)
