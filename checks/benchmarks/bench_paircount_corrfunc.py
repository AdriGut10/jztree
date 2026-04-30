import importlib.util
from time import perf_counter

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jztree as jz


corrfunc_required = pytest.mark.skipif(importlib.util.find_spec("Corrfunc") is None, reason="Corrfunc not installed")
CORRFUNC_THREADS = 16
MODES = ("DD", "DDrppi", "DDsmu")
DTYPES = ("float32", "float64")


def pytest_generate_tests(metafunc):
    if "mode_id" not in metafunc.fixturenames:
        return
    if metafunc.config.getoption("--single"):
        mode_ids = [1, 2, 3]
    elif metafunc.config.getoption("--double"):
        mode_ids = [4, 5, 6]
    else:
        mode_ids = [1, 2, 3, 4, 5, 6]
    metafunc.parametrize("mode_id", mode_ids, ids=[_case_id(mode_id) for mode_id in mode_ids])


def _case_id(mode_id):
    mode, dtype_name = _mode_dtype(mode_id)
    return f"{dtype_name}-{mode}"


def _mode_dtype(mode_id):
    mode_index = (mode_id - 1) % len(MODES)
    dtype_index = (mode_id - 1) // len(MODES)
    return MODES[mode_index], DTYPES[dtype_index]


def _edges(spec):
    lower, upper, nbins, spacing = spec
    if spacing == "log":
        return np.geomspace(lower, upper, nbins + 1)
    return np.linspace(lower, upper, nbins + 1)


def _print_count_diff(mode, backend, jz_counts, cf_counts):
    diff = np.abs(jz_counts - cf_counts)
    denom = max(int(np.max(cf_counts)), 1)
    print(f"{mode} Corrfunc {backend} max_abs_diff: {int(np.max(diff))}")
    print(f"{mode} Corrfunc {backend} max_rel_diff: {float(np.max(diff) / denom):.3e}")
    print(f"{mode} Corrfunc {backend} total_abs_diff: {int(np.sum(diff))}")


def _corrfunc_counts(mode, pos_np, bins, boxsize, nthreads, gpu_ddsmu):
    from Corrfunc.theory import DD as CorrfuncDD
    from Corrfunc.theory import DDrppi as CorrfuncDDrppi
    from Corrfunc.theory import DDsmu as CorrfuncDDsmu

    if mode == "DD":
        redges = _edges(bins["r"])
        res = CorrfuncDD(
            autocorr=1,
            nthreads=nthreads,
            binfile=redges,
            X1=pos_np[:, 0],
            Y1=pos_np[:, 1],
            Z1=pos_np[:, 2],
            boxsize=boxsize,
            periodic=True,
        )
        return np.asarray(res["npairs"], dtype=np.int64)

    if mode == "DDrppi":
        rp_edges = _edges(bins["rp"])
        pi_edges = _edges(bins["pi"])
        res = CorrfuncDDrppi(
            autocorr=1,
            nthreads=nthreads,
            binfile=rp_edges,
            pimax=max(abs(pi_edges[0]), abs(pi_edges[-1])),
            npibins=len(pi_edges) - 1,
            X1=pos_np[:, 0],
            Y1=pos_np[:, 1],
            Z1=pos_np[:, 2],
            boxsize=boxsize,
            periodic=True,
        )
        return np.asarray(res["npairs"], dtype=np.int64).reshape((len(rp_edges) - 1, len(pi_edges) - 1))

    s_edges = _edges(bins["s"])
    mu_edges = _edges(bins["mu"])
    res = CorrfuncDDsmu(
        autocorr=1,
        nthreads=nthreads,
        binfile=s_edges,
        mumax=max(abs(mu_edges[0]), abs(mu_edges[-1])),
        nmubins=len(mu_edges) - 1,
        X1=pos_np[:, 0],
        Y1=pos_np[:, 1],
        Z1=pos_np[:, 2],
        boxsize=boxsize,
        periodic=True,
        gpu=gpu_ddsmu,
    )
    return np.asarray(res["npairs"], dtype=np.int64).reshape((len(s_edges) - 1, len(mu_edges) - 1))


def _measure_corrfunc(mode, pos_np, bins, boxsize, nthreads, gpu_ddsmu):
    _corrfunc_counts(mode, pos_np, bins, boxsize, nthreads, gpu_ddsmu)
    t0 = perf_counter()
    counts = None
    for _ in range(5):
        counts = _corrfunc_counts(mode, pos_np, bins, boxsize, nthreads, gpu_ddsmu)
    return counts, (perf_counter() - t0) * 1e3 / 5


def _bench_paircount_corrfunc(jax_bench, npart, mode_id):
    mode, dtype_name = _mode_dtype(mode_id)
    boxsize = 1.0
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float32 if dtype_name == "float32" else jnp.float64
    np_dtype = np.float32 if dtype_name == "float32" else np.float64
    pos = jax.random.uniform(jax.random.key(0), (npart, 3), dtype=dtype)
    pos_np = np.asarray(pos, dtype=np_dtype)
    bins = {
        "r": (1.0e-4, 0.2, 20, "lin"),
        "rp": (1.0e-4, 0.2, 20, "lin"),
        "pi": (-0.4, 0.4, 80, "lin"),
        "s": (1.0e-4, 0.2, 20, "lin"),
        "mu": (-1.0, 1.0, 80, "lin"),
    }
    jb = jax_bench(jit_rounds=3, jit_warmup=1)

    if mode == "DD":
        jz_counts = jb.measure(
            fn_jit=jz.paircount.DD.jit,
            pos1=pos,
            r=bins["r"],
            boxsize=boxsize,
            tag=f"jztree_{mode}_{dtype_name}",
        )[1]
    elif mode == "DDrppi":
        jz_counts = jb.measure(
            fn_jit=jz.paircount.DDrppi.jit,
            pos1=pos,
            rp=bins["rp"],
            pi=bins["pi"],
            boxsize=boxsize,
            tag=f"jztree_{mode}_{dtype_name}",
        )[1]
    else:
        jz_counts = jb.measure(
            fn_jit=jz.paircount.DDsmu.jit,
            pos1=pos,
            s=bins["s"],
            mu=bins["mu"],
            boxsize=boxsize,
            tag=f"jztree_{mode}_{dtype_name}",
        )[1]
    jz_counts = np.asarray(jax.block_until_ready(jz_counts), dtype=np.int64)

    cf_counts, cf_ms = _measure_corrfunc(mode, pos_np, bins, boxsize, CORRFUNC_THREADS, False)
    _print_count_diff(mode, "CPU", jz_counts, cf_counts)
    print(f"corrfunc_{mode}_cpu_{dtype_name}: {cf_ms:.2f} ms")

    if mode == "DDsmu":
        cf_counts, cf_ms = _measure_corrfunc(mode, pos_np, bins, boxsize, CORRFUNC_THREADS, True)
        _print_count_diff(mode, "GPU", jz_counts, cf_counts)
        print(f"corrfunc_{mode}_gpu_{dtype_name}: {cf_ms:.2f} ms")


@corrfunc_required
def bench_paircount_corrfunc_quick(jax_bench, pytestconfig, mode_id):
    if not pytestconfig.getoption("--quick"):
        pytest.skip("quick-only smoke benchmark")
    _bench_paircount_corrfunc(jax_bench, int(1e5), mode_id)


@corrfunc_required
@pytest.mark.skip_in_quick
@pytest.mark.parametrize("npart", [int(5e4), int(1e5), int(3e5)])
def bench_paircount_corrfunc_scaling(jax_bench, npart, mode_id):
    _bench_paircount_corrfunc(jax_bench, npart, mode_id)
