import importlib.util

import numpy as np
import pytest
import jax.numpy as jnp

import jztree as jz


corrfunc_required = pytest.mark.skipif(importlib.util.find_spec("Corrfunc") is None, reason="Corrfunc not installed")


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


def _brute_r(pos1, r, boxsize, pos2=None):
    edges = _edges(r)
    pos1 = np.asarray(pos1)
    pos2 = pos1 if pos2 is None else np.asarray(pos2)
    out = np.zeros(len(edges) - 1, dtype=np.int64)
    for x1 in pos1:
        for x2 in pos2:
            ibin = _bin(edges, np.sqrt(np.sum(_wrap_dx(x1 - x2, boxsize) ** 2)))
            if ibin >= 0:
                out[ibin] += 1
    return out


def _brute_rppi(pos1, rp, pi, boxsize, pos2=None, los_axis=2):
    rp_edges = _edges(rp)
    pi_edges = _edges(pi)
    pos1 = np.asarray(pos1)
    pos2 = pos1 if pos2 is None else np.asarray(pos2)
    out = np.zeros((len(rp_edges) - 1, len(pi_edges) - 1), dtype=np.int64)
    for x1 in pos1:
        for x2 in pos2:
            dx = _wrap_dx(x2 - x1, boxsize)
            irp = _bin(rp_edges, np.sqrt(np.sum(np.delete(dx, los_axis) ** 2)))
            ipi = _bin(pi_edges, dx[los_axis])
            if irp >= 0 and ipi >= 0:
                out[irp, ipi] += 1
    return out


def _brute_smu(pos1, s, mu, boxsize, pos2=None, los_axis=2):
    s_edges = _edges(s)
    mu_edges = _edges(mu)
    pos1 = np.asarray(pos1)
    pos2 = pos1 if pos2 is None else np.asarray(pos2)
    out = np.zeros((len(s_edges) - 1, len(mu_edges) - 1), dtype=np.int64)
    for x1 in pos1:
        for x2 in pos2:
            dx = _wrap_dx(x2 - x1, boxsize)
            sep = np.sqrt(np.sum(dx**2))
            m = dx[los_axis] / max(sep, 1.0e-30)
            isbin = _bin(s_edges, sep)
            imubin = -1 if m <= mu_edges[0] or m >= mu_edges[-1] else _bin(mu_edges, m)
            if isbin >= 0 and imubin >= 0:
                out[isbin, imubin] += 1
    return out


def _corrfunc_dd(pos, r, boxsize, pos2=None):
    from Corrfunc.theory import DD

    edges = _edges(r)
    pos = np.asarray(pos, dtype=np.float64)
    autocorr = pos2 is None
    if autocorr:
        res = DD(1, 1, edges, pos[:, 0], pos[:, 1], pos[:, 2], boxsize=boxsize, periodic=True)
    else:
        pos2 = np.asarray(pos2, dtype=np.float64)
        res = DD(
            0,
            1,
            edges,
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            X2=pos2[:, 0],
            Y2=pos2[:, 1],
            Z2=pos2[:, 2],
            boxsize=boxsize,
            periodic=True,
        )
    return np.asarray(res["npairs"], dtype=np.int64).copy()


def _corrfunc_drrppi(pos, rp, pi, boxsize, pos2=None):
    from Corrfunc.theory import DDrppi

    rp_edges = _edges(rp)
    pi_edges = _edges(pi)
    assert np.isclose(-pi_edges[0], pi_edges[-1])
    pos = np.asarray(pos, dtype=np.float64)
    autocorr = pos2 is None
    kwargs = {}
    if not autocorr:
        pos2 = np.asarray(pos2, dtype=np.float64)
        kwargs = {"X2": pos2[:, 0], "Y2": pos2[:, 1], "Z2": pos2[:, 2]}
    res = DDrppi(
        int(autocorr),
        1,
        rp_edges,
        pi_edges[-1],
        len(pi_edges) - 1,
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        boxsize=boxsize,
        periodic=True,
        **kwargs,
    )
    return np.asarray(res["npairs"], dtype=np.int64).reshape((len(rp_edges) - 1, len(pi_edges) - 1))


def _corrfunc_ddsmu(pos, s, mu, boxsize, pos2=None, gpu=False):
    from Corrfunc.theory import DDsmu

    s_edges = _edges(s)
    mu_edges = _edges(mu)
    assert np.isclose(-mu_edges[0], mu_edges[-1])
    pos = np.asarray(pos, dtype=np.float64)
    autocorr = pos2 is None
    kwargs = {}
    if not autocorr:
        pos2 = np.asarray(pos2, dtype=np.float64)
        kwargs = {"X2": pos2[:, 0], "Y2": pos2[:, 1], "Z2": pos2[:, 2]}
    res = DDsmu(
        int(autocorr),
        1,
        s_edges,
        mu_edges[-1],
        len(mu_edges) - 1,
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        boxsize=boxsize,
        periodic=True,
        gpu=gpu,
        **kwargs,
    )
    return np.asarray(res["npairs"], dtype=np.int64).reshape((len(s_edges) - 1, len(mu_edges) - 1))


@corrfunc_required
def test_paircount_corrfunc_dd():
    pos = jnp.asarray(np.random.default_rng(0).uniform(0.0, 1.0, size=(64, 3)), dtype=jnp.float32)
    r = (0.0, 0.45, 7, "lin")
    ref = _brute_r(pos, r, 1.0)
    cf = _corrfunc_dd(pos, r, 1.0)
    dd = jz.paircount.DD.jit(pos, r, boxsize=1.0)

    assert np.array_equal(cf, ref)
    assert np.array_equal(np.asarray(dd), cf)


@corrfunc_required
def test_paircount_corrfunc_drrppi():
    pos = jnp.asarray(np.random.default_rng(1).uniform(0.0, 1.0, size=(64, 3)), dtype=jnp.float32)
    rp = (0.0, 0.45, 5, "lin")
    pi = (-0.45, 0.45, 8, "lin")
    ref = _brute_rppi(pos, rp, pi, 1.0)
    cf = _corrfunc_drrppi(pos, rp, pi, 1.0)
    dd = jz.paircount.DDrppi.jit(pos, rp, pi, boxsize=1.0)

    assert np.array_equal(cf, ref)
    assert np.array_equal(np.asarray(dd), cf)


@corrfunc_required
@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
def test_paircount_corrfunc_ddsmu(gpu):
    pos = jnp.asarray(np.random.default_rng(2).uniform(0.0, 1.0, size=(64, 3)), dtype=jnp.float32)
    s = (0.0, 0.45, 5, "lin")
    mu = (-1.0, 1.0, 8, "lin")
    ref = _brute_smu(pos, s, mu, 1.0)
    cf = _corrfunc_ddsmu(pos, s, mu, 1.0, gpu=gpu)
    dd = jz.paircount.DDsmu.jit(pos, s, mu, boxsize=1.0)

    assert np.array_equal(cf, ref)
    assert np.array_equal(np.asarray(dd), cf)
