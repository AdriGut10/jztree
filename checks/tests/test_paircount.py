import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jztree as jz
from jztree.data import InteractionList, Pos


def _edges(spec):
    lower, upper, nbins, spacing = spec
    if spacing == "log":
        return np.geomspace(lower, upper, nbins + 1)
    return np.linspace(lower, upper, nbins + 1)


def _wrap_dx(dx, boxsize):
    return dx - boxsize * np.rint(dx / boxsize)


def _bin(edges, x):
    if x < edges[0] or x > edges[-1]:
        return -1
    i = np.searchsorted(edges, x, side="right") - 1
    return min(i, len(edges) - 2)


def _brute_r_raw(pos1, spec, boxsize, pos2=None):
    edges = _edges(spec)
    pos1 = np.asarray(pos1)
    same = pos2 is None
    pos2 = pos1 if same else np.asarray(pos2)
    out = np.zeros(len(edges) - 1, dtype=np.int32)
    for x1 in pos1:
        for x2 in pos2:
            r = np.sqrt(np.sum(_wrap_dx(x2 - x1, boxsize) ** 2))
            ibin = _bin(edges, r)
            if ibin >= 0:
                out[ibin] += 1
    return out


def _brute_rppi_raw(pos1, rp_spec, pi_spec, boxsize, pos2=None, los_axis=2):
    rp_edges = _edges(rp_spec)
    pi_edges = _edges(pi_spec)
    pos1 = np.asarray(pos1)
    same = pos2 is None
    pos2 = pos1 if same else np.asarray(pos2)
    out = np.zeros((len(rp_edges) - 1, len(pi_edges) - 1), dtype=np.int32)
    for x1 in pos1:
        for x2 in pos2:
            dx = _wrap_dx(x2 - x1, boxsize)
            pi = dx[los_axis]
            rp = np.sqrt(np.sum(np.delete(dx, los_axis) ** 2))
            irp = _bin(rp_edges, rp)
            ipi = _bin(pi_edges, pi)
            if irp >= 0 and ipi >= 0:
                out[irp, ipi] += 1
    return out


def _brute_smu_raw(pos1, s_spec, mu_spec, boxsize, pos2=None, los_axis=2):
    s_edges = _edges(s_spec)
    mu_edges = _edges(mu_spec)
    pos1 = np.asarray(pos1)
    same = pos2 is None
    pos2 = pos1 if same else np.asarray(pos2)
    out = np.zeros((len(s_edges) - 1, len(mu_edges) - 1), dtype=np.int32)
    for x1 in pos1:
        for x2 in pos2:
            dx = _wrap_dx(x2 - x1, boxsize)
            s = np.sqrt(np.sum(dx**2))
            mu = dx[los_axis] / max(s, 1.0e-30)
            isbin = _bin(s_edges, s)
            imubin = -1 if mu <= mu_edges[0] or mu >= mu_edges[-1] else _bin(mu_edges, mu)
            if isbin >= 0 and imubin >= 0:
                out[isbin, imubin] += 1
    return out


def _pos(dtype=jnp.float32):
    return jnp.asarray(
        [
            [0.05, 0.05, 0.05],
            [0.15, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.05, 0.28, 0.05],
            [0.05, 0.05, 0.46],
            [0.64, 0.72, 0.81],
        ],
        dtype=dtype,
    )


def test_paircount_leaf2leaf_r_direct():
    pos = _pos()
    boxsize = 1.0
    r = (0.0, 0.8, 4, "lin")
    ilist = InteractionList(
        ispl=jnp.asarray([0, 1], dtype=jnp.int32),
        iother=jnp.asarray([0], dtype=jnp.int32),
    )
    spl = jnp.asarray([0, pos.shape[0]], dtype=jnp.int32)

    dd = jz.paircount._paircount_leaf2leaf_r.jit(
        ilist,
        spl,
        pos,
        r=jz.paircount._format_bin_spec(r, allow_log=True, name="r"),
        same_catalog=True,
        boxsize=boxsize,
        block_size=64,
    )

    assert np.array_equal(np.asarray(dd), _brute_r_raw(pos, r, boxsize))


def test_paircount_leaf2leaf_rppi_direct():
    pos = _pos()
    boxsize = 1.0
    rp = (0.0, 0.6, 3, "lin")
    pi = (-0.6, 0.6, 6, "lin")
    ilist = InteractionList(
        ispl=jnp.asarray([0, 1], dtype=jnp.int32),
        iother=jnp.asarray([0], dtype=jnp.int32),
    )
    spl = jnp.asarray([0, pos.shape[0]], dtype=jnp.int32)
    config = jz.paircount._PairCountRuntimeConfig(
        boxsize=boxsize,
        los_axis=2,
        same_catalog=True,
        rppi=jz.paircount._RppiSpec(
            rp=jz.paircount._format_bin_spec(rp, allow_log=True, name="rp"),
            pi=jz.paircount._format_bin_spec(pi, allow_log=False, name="pi"),
        ),
    )

    dd, _ = jz.paircount._paircount_leaf2leaf_aniso.jit(
        ilist,
        spl,
        pos,
        config=config,
        block_size=64,
    )

    assert np.array_equal(np.asarray(dd), _brute_rppi_raw(pos, rp, pi, boxsize))


def test_paircount_leaf2leaf_smu_direct():
    pos = _pos()
    boxsize = 1.0
    s = (0.0, 0.8, 4, "lin")
    mu = (-1.0, 1.0, 4, "lin")
    ilist = InteractionList(
        ispl=jnp.asarray([0, 1], dtype=jnp.int32),
        iother=jnp.asarray([0], dtype=jnp.int32),
    )
    spl = jnp.asarray([0, pos.shape[0]], dtype=jnp.int32)
    config = jz.paircount._PairCountRuntimeConfig(
        boxsize=boxsize,
        los_axis=2,
        same_catalog=True,
        smu=jz.paircount._SmuSpec(
            s=jz.paircount._format_bin_spec(s, allow_log=True, name="s"),
            mu=jz.paircount._format_bin_spec(mu, allow_log=False, name="mu"),
        ),
    )

    _, dd = jz.paircount._paircount_leaf2leaf_aniso.jit(
        ilist,
        spl,
        pos,
        config=config,
        block_size=64,
    )

    assert np.array_equal(np.asarray(dd), _brute_smu_raw(pos, s, mu, boxsize))


def test_paircount_dual_walk_diagnostic():
    pos = _pos()
    cfg = jz.config.PairCountConfig()
    partz, th = jz.tree.zsort_and_tree.jit(Pos(pos=pos, num=jnp.asarray(pos.shape[0], dtype=jnp.int32)), cfg_tree=cfg.tree)
    ilist = jz.paircount._paircount_dual_walk.jit(
        th,
        ptype_query=0,
        ptype_source=0,
        same_catalog=True,
        use_r=True,
        rmax=0.8,
        use_rppi=False,
        rpmax=0.0,
        pimax=0.0,
        use_smu=False,
        smax=0.0,
        los_axis=2,
        boxsize=1.0,
        alloc_fac_ilist=cfg.alloc_fac_ilist,
    )
    spl = th.splits_leaf_to_part(ptype=0)
    nleaf = np.asarray(th.num(0)).item()

    assert np.asarray(ilist.ispl[-1]).item() > 0
    assert np.asarray(spl[nleaf]).item() == pos.shape[0]


def test_paircount_raw_api_bruteforce():
    pos = _pos()
    boxsize = 1.0
    r = (0.0, 0.8, 4, "lin")
    rp = (0.0, 0.6, 3, "lin")
    pi = (-0.6, 0.6, 6, "lin")
    s = (0.0, 0.8, 4, "lin")
    mu = (-1.0, 1.0, 4, "lin")
    pos2 = jnp.asarray([[0.12, 0.05, 0.05], [0.06, 0.05, 0.49], [0.83, 0.91, 0.08]], dtype=jnp.float32)

    assert np.array_equal(np.asarray(jz.paircount.DD.jit(pos, r, boxsize=boxsize)), _brute_r_raw(pos, r, boxsize))
    assert np.array_equal(
        np.asarray(jz.paircount.DDrppi.jit(pos, rp, pi, boxsize=boxsize)),
        _brute_rppi_raw(pos, rp, pi, boxsize),
    )
    assert np.array_equal(
        np.asarray(jz.paircount.DDsmu.jit(pos, s, mu, boxsize=boxsize)),
        _brute_smu_raw(pos, s, mu, boxsize),
    )
    assert np.array_equal(
        np.asarray(jz.paircount.DD.jit(pos, r, boxsize=boxsize, pos2=pos2)),
        _brute_r_raw(pos, r, boxsize, pos2=pos2),
    )
    assert np.array_equal(
        np.asarray(jz.paircount.DDrppi.jit(pos, rp, pi, boxsize=boxsize, pos2=pos2)),
        _brute_rppi_raw(pos, rp, pi, boxsize, pos2=pos2),
    )
    assert np.array_equal(
        np.asarray(jz.paircount.DDsmu.jit(pos, s, mu, boxsize=boxsize, pos2=pos2)),
        _brute_smu_raw(pos, s, mu, boxsize, pos2=pos2),
    )


def test_paircount_log_bins_and_upper_edge():
    pos = jnp.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=jnp.float32)
    r = (0.1, 0.4, 2, "log")

    dd = jz.paircount.DD.jit(pos, r, boxsize=1.0)

    assert np.array_equal(np.asarray(dd), _brute_r_raw(pos, r, 1.0))


def test_paircount_los_axis_valid_num():
    pos = _pos()
    valid = jnp.asarray([True, True, False, True, True, True])
    pos_use = np.asarray(pos)[np.asarray(valid)]
    pos_use = pos_use[: np.sum(np.asarray(valid)[:5])]
    boxsize = 1.0
    r = (0.0, 0.9, 3, "lin")
    rp = (0.0, 0.9, 3, "lin")
    pi = (-0.5, 0.5, 4, "lin")

    dd_r = jz.paircount.DD.jit(pos, r, boxsize=boxsize, valid1=valid, num1=5)
    dd_rppi = jz.paircount.DDrppi.jit(
        pos,
        rp,
        pi,
        boxsize=boxsize,
        valid1=valid,
        num1=5,
        los_axis=0,
    )

    assert np.array_equal(np.asarray(dd_r), _brute_r_raw(pos_use, r, boxsize))
    assert np.array_equal(np.asarray(dd_rppi), _brute_rppi_raw(pos_use, rp, pi, boxsize, los_axis=0))


def test_paircount_double():
    with jax.enable_x64():
        pos = _pos(dtype=jnp.float64)
        r = (0.0, 0.9, 3, "lin")
        dd = jz.paircount.DD.jit(pos, r, boxsize=1.0)
        assert np.array_equal(np.asarray(dd), _brute_r_raw(pos, r, 1.0))


def test_paircount_invalid_bins():
    pos = _pos()
    with pytest.raises(ValueError):
        jz.paircount.DD(pos, (-0.1, 0.1, 1, "lin"), boxsize=1.0)
    with pytest.raises(ValueError):
        jz.paircount.DD(pos, (0.0, 0.1, 1, "log"), boxsize=1.0)
    with pytest.raises(ValueError):
        jz.paircount.DDrppi(pos, (-0.1, 0.1, 1, "lin"), (-0.2, 0.2, 2, "lin"), boxsize=1.0)
    with pytest.raises(ValueError):
        jz.paircount.DDrppi(pos, (0.0, 0.2, 1, "lin"), (-0.2, 0.2, 2, "log"), boxsize=1.0)
    with pytest.raises(ValueError):
        jz.paircount.DDsmu(pos, (0.0, 0.2, 1, "lin"), (-1.2, 1.0, 2, "lin"), boxsize=1.0)
