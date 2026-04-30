from __future__ import annotations

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import jztree as jz


def _require_gpu():
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("This check requires a CUDA-visible JAX GPU backend.")
    print("Using GPU:", devices[0])


def _edges(spec):
    lower, upper, nbins, spacing = spec
    if spacing == "log":
        return np.geomspace(lower, upper, nbins + 1)
    return np.linspace(lower, upper, nbins + 1)


def _corrfunc_counts(mode, pos_np, bins, boxsize, nthreads, use_gpu_ddsmu):
    from Corrfunc.theory import DD as CorrfuncDD
    from Corrfunc.theory import DDrppi as CorrfuncDDrppi
    from Corrfunc.theory import DDsmu as CorrfuncDDsmu

    if mode == "DD":
        r_edges = _edges(bins["r"])
        res = CorrfuncDD(
            autocorr=1,
            nthreads=nthreads,
            binfile=r_edges,
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
        gpu=use_gpu_ddsmu,
    )
    return np.asarray(res["npairs"], dtype=np.int64).reshape((len(s_edges) - 1, len(mu_edges) - 1))


def _jztree_counts(mode, pos, bins, boxsize):
    if mode == "DD":
        return jz.paircount.DD.jit(pos, bins["r"], boxsize=boxsize)
    if mode == "DDrppi":
        return jz.paircount.DDrppi.jit(pos, bins["rp"], bins["pi"], boxsize=boxsize)
    return jz.paircount.DDsmu.jit(pos, bins["s"], bins["mu"], boxsize=boxsize)


def _measure_jztree(mode, pos, bins, boxsize, repeats, warmup):
    counts = None
    for _ in range(warmup):
        counts = np.asarray(jax.block_until_ready(_jztree_counts(mode, pos, bins, boxsize)), dtype=np.int64)

    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        counts = np.asarray(jax.block_until_ready(_jztree_counts(mode, pos, bins, boxsize)), dtype=np.int64)
        times.append((perf_counter() - t0) * 1.0e3)
    return counts, np.asarray(times)


def _measure_corrfunc(mode, pos_np, bins, boxsize, nthreads, use_gpu_ddsmu, repeats, warmup):
    counts = None
    for _ in range(warmup):
        counts = _corrfunc_counts(mode, pos_np, bins, boxsize, nthreads, use_gpu_ddsmu)

    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        counts = _corrfunc_counts(mode, pos_np, bins, boxsize, nthreads, use_gpu_ddsmu)
        times.append((perf_counter() - t0) * 1.0e3)
    return counts, np.asarray(times)


def _check_counts(mode, jz_counts, cf_counts, jztree_times, corrfunc_times, pos_np, max_rel_diff, backend):
    diff = np.abs(jz_counts - cf_counts)
    denom = max(int(np.max(cf_counts)), 1)
    rel_diff = float(np.max(diff) / denom)
    total_rel_diff = float(np.sum(diff) / max(int(np.sum(cf_counts)), 1))
    print(
        f"{mode}: jztree={np.mean(jztree_times):.2f} ms "
        f"(min {np.min(jztree_times):.2f}) corrfunc={np.mean(corrfunc_times):.2f} ms "
        f"(min {np.min(corrfunc_times):.2f}) "
        f"dtype={pos_np.dtype} "
        f"corrfunc_backend={backend} "
        f"max_abs_diff={int(np.max(diff))} max_rel_diff={rel_diff:.3e} "
        f"total_abs_diff={int(np.sum(diff))} total_rel_diff={total_rel_diff:.3e}"
    )
    if rel_diff > max_rel_diff:
        raise AssertionError(f"{mode} max_rel_diff={rel_diff:.3e} exceeds {max_rel_diff:.3e}")


def _check_mode(mode, pos, pos_np, bins, boxsize, nthreads, max_rel_diff, repeats, warmup, skip_corrfunc_gpu_ddsmu):
    jz_counts, jztree_times = _measure_jztree(mode, pos, bins, boxsize, repeats, warmup)

    cf_counts, corrfunc_times = _measure_corrfunc(mode, pos_np, bins, boxsize, nthreads, False, repeats, warmup)
    _check_counts(mode, jz_counts, cf_counts, jztree_times, corrfunc_times, pos_np, max_rel_diff, "CPU")

    if mode == "DDsmu" and not skip_corrfunc_gpu_ddsmu:
        cf_counts, corrfunc_times = _measure_corrfunc(mode, pos_np, bins, boxsize, nthreads, True, repeats, warmup)
        _check_counts(mode, jz_counts, cf_counts, jztree_times, corrfunc_times, pos_np, max_rel_diff, "GPU")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPU smoke check for JZTREE Corrfunc-style pair counts.")
    parser.add_argument("--npart", type=int, default=100_000)
    parser.add_argument("--boxsize", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corrfunc-threads", type=int, default=8)
    parser.add_argument("--skip-corrfunc-gpu-ddsmu", action="store_true", help="Skip the additional Corrfunc GPU DDsmu comparison.")
    parser.add_argument("--max-rel-diff", type=float, default=2.0e-5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup calls before measuring each backend.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    _require_gpu()

    boxsize = args.boxsize
    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    pos = jax.random.uniform(jax.random.key(args.seed), (args.npart, 3), minval=0.0, maxval=boxsize, dtype=dtype)
    pos_np = np.asarray(pos, dtype=np_dtype)
    bins = {
        "r": (1.0e-4 * boxsize, 0.2 * boxsize, 20, "lin"),
        "rp": (1.0e-4 * boxsize, 0.2 * boxsize, 20, "lin"),
        "pi": (-0.4 * boxsize, 0.4 * boxsize, 80, "lin"),
        "s": (1.0e-4 * boxsize, 0.2 * boxsize, 20, "lin"),
        "mu": (-1.0, 1.0, 80, "lin"),
    }

    for mode in ("DD", "DDrppi", "DDsmu"):
        _check_mode(
            mode,
            pos,
            pos_np,
            bins,
            boxsize,
            args.corrfunc_threads,
            args.max_rel_diff,
            args.repeats,
            args.warmup,
            args.skip_corrfunc_gpu_ddsmu,
        )


if __name__ == "__main__":
    main()
