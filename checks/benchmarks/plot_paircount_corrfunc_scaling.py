import argparse
import csv
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp

import jztree as jz


MODES = ("DD", "DDrppi", "DDsmu")
INT32_LIMIT = np.iinfo(np.int32).max


def _has_cuda():
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except RuntimeError:
        return False


def _parse_npart(value):
    text = value.replace("_", "").lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1_000)
    if text.endswith("m"):
        return int(float(text[:-1]) * 1_000_000)
    return int(float(text))


def _bins(args):
    return {
        "r": (args.rmin, args.rmax, args.nr, "lin"),
        "rp": (args.rmin, args.rmax, args.nr, "lin"),
        "pi": (-args.pimax, args.pimax, 2 * args.npi, "lin"),
        "s": (args.rmin, args.rmax, args.nr, "lin"),
        "mu": (-args.mumax, args.mumax, 2 * args.nmu, "lin"),
    }


def _edges(spec):
    lower, upper, nbins, spacing = spec
    if spacing == "log":
        return np.geomspace(lower, upper, nbins + 1)
    return np.linspace(lower, upper, nbins + 1)


def _jztree_counts(mode, pos, bins, boxsize):
    if mode == "DD":
        return jz.paircount.DD.jit(pos, bins["r"], boxsize=boxsize)
    if mode == "DDrppi":
        return jz.paircount.DDrppi.jit(pos, bins["rp"], bins["pi"], boxsize=boxsize)
    return jz.paircount.DDsmu.jit(pos, bins["s"], bins["mu"], boxsize=boxsize)


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


def _mode_bin_volumes(mode, bins, boxsize):
    box_volume = boxsize**3
    if mode == "DD":
        r = _edges(bins["r"])
        return 4.0 * math.pi / 3.0 * (r[1:] ** 3 - r[:-1] ** 3) / box_volume
    if mode == "DDrppi":
        rp = _edges(bins["rp"])
        pi = _edges(bins["pi"])
        return np.outer(math.pi * (rp[1:] ** 2 - rp[:-1] ** 2), pi[1:] - pi[:-1]).ravel() / box_volume
    s = _edges(bins["s"])
    mu = _edges(bins["mu"])
    return np.outer(2.0 * math.pi / 3.0 * (s[1:] ** 3 - s[:-1] ** 3), mu[1:] - mu[:-1]).ravel() / box_volume


def _estimated_max_bin_count(mode, npart, bins, boxsize):
    pair_count = npart * npart
    return pair_count * float(np.max(_mode_bin_volumes(mode, bins, boxsize)))


def _measure_jztree(mode, pos, bins, boxsize, repeats, warmup):
    t0 = perf_counter()
    counts = jax.block_until_ready(_jztree_counts(mode, pos, bins, boxsize))
    first_call_ms = (perf_counter() - t0) * 1e3

    for _ in range(warmup):
        counts = jax.block_until_ready(_jztree_counts(mode, pos, bins, boxsize))

    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        counts = jax.block_until_ready(_jztree_counts(mode, pos, bins, boxsize))
        times.append((perf_counter() - t0) * 1e3)

    return np.asarray(counts, dtype=np.int64), first_call_ms, np.asarray(times)


def _measure_corrfunc(mode, pos_np, bins, boxsize, nthreads, repeats, warmup, use_gpu_ddsmu):
    times = []
    counts = None
    for _ in range(warmup):
        counts = _corrfunc_counts(mode, pos_np, bins, boxsize, nthreads, use_gpu_ddsmu)
    for _ in range(repeats):
        t0 = perf_counter()
        counts = _corrfunc_counts(mode, pos_np, bins, boxsize, nthreads, use_gpu_ddsmu)
        times.append((perf_counter() - t0) * 1e3)
    return counts, np.asarray(times)


def _diff_summary(jztree_counts, corrfunc_counts):
    diff = np.abs(jztree_counts - corrfunc_counts)
    denom = max(int(np.max(corrfunc_counts)), 1)
    return {
        "max_abs_diff": int(np.max(diff)),
        "max_rel_diff": float(np.max(diff) / denom),
        "total_abs_diff": int(np.sum(diff)),
    }


def _write_csv(path, rows):
    fieldnames = [
        "npart",
        "mode",
        "jztree_mean_ms",
        "jztree_std_ms",
        "jztree_min_ms",
        "jztree_first_call_ms",
        "corrfunc_cpu_mean_ms",
        "corrfunc_cpu_std_ms",
        "corrfunc_cpu_min_ms",
        "corrfunc_cpu_speedup",
        "corrfunc_cpu_max_abs_diff",
        "corrfunc_cpu_max_rel_diff",
        "corrfunc_cpu_total_abs_diff",
        "corrfunc_gpu_mean_ms",
        "corrfunc_gpu_std_ms",
        "corrfunc_gpu_min_ms",
        "corrfunc_gpu_speedup",
        "corrfunc_gpu_max_abs_diff",
        "corrfunc_gpu_max_rel_diff",
        "corrfunc_gpu_total_abs_diff",
        "skipped_reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _plot(path, rows, modes):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; wrote CSV only")
        return False

    plotted_modes = [mode for mode in modes if any(row["mode"] == mode and not row.get("skipped_reason") for row in rows)]
    if not plotted_modes:
        return False

    fig, axes = plt.subplots(1, len(plotted_modes), figsize=(5.0 * len(plotted_modes), 4.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, mode in zip(axes, plotted_modes):
        subset = [row for row in rows if row["mode"] == mode and not row.get("skipped_reason")]
        subset = sorted(subset, key=lambda row: row["npart"])
        n = np.asarray([row["npart"] for row in subset])
        jztree_ms = np.asarray([row["jztree_mean_ms"] for row in subset])
        corrfunc_cpu_ms = np.asarray([row["corrfunc_cpu_mean_ms"] for row in subset])
        corrfunc_gpu_ms = np.asarray([row["corrfunc_gpu_mean_ms"] for row in subset])

        ax.plot(n, jztree_ms, marker="o", label="JZTREE GPU")
        if np.all(np.isfinite(corrfunc_cpu_ms)):
            ax.plot(n, corrfunc_cpu_ms, marker="s", label="Corrfunc CPU")
        if np.all(np.isfinite(corrfunc_gpu_ms)):
            ax.plot(n, corrfunc_gpu_ms, marker="^", label="Corrfunc GPU")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(mode)
        ax.set_xlabel("particles")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("time [ms]")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _print_row(row):
    if row.get("skipped_reason"):
        print(f'{row["mode"]} n={row["npart"]}: skipped ({row["skipped_reason"]})')
        return
    parts = [
        f'{row["mode"]} n={row["npart"]}:',
        f'JZTREE {row["jztree_mean_ms"]:.2f} ms',
        f'first call {row["jztree_first_call_ms"]:.2f} ms',
    ]
    if np.isfinite(row.get("corrfunc_cpu_mean_ms", float("nan"))):
        parts.append(
            f'Corrfunc CPU {row["corrfunc_cpu_mean_ms"]:.2f} ms '
            f'({row["corrfunc_cpu_speedup"]:.1f}x, diff {row["corrfunc_cpu_max_rel_diff"]:.3e})'
        )
    if np.isfinite(row.get("corrfunc_gpu_mean_ms", float("nan"))):
        parts.append(
            f'Corrfunc GPU {row["corrfunc_gpu_mean_ms"]:.2f} ms '
            f'({row["corrfunc_gpu_speedup"]:.1f}x, diff {row["corrfunc_gpu_max_rel_diff"]:.3e})'
        )
    print(", ".join(parts))


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot JZTREE GPU pair-count scaling against Corrfunc.")
    parser.add_argument("--sizes", nargs="+", type=_parse_npart, default=[50_000, 100_000, 300_000, 600_000])
    parser.add_argument("--stress-1e6", action="store_true", help="Append 1e6 particles to the requested sizes.")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--outdir", type=Path, default=Path("benchmarks/.results/paircount_corrfunc_scaling_plot"))
    parser.add_argument("--boxsize", type=float, default=1.0)
    parser.add_argument("--rmin", type=float, default=1.0e-4)
    parser.add_argument("--rmax", type=float, default=0.2)
    parser.add_argument("--pimax", type=float, default=0.4)
    parser.add_argument("--mumax", type=float, default=1.0)
    parser.add_argument("--nr", type=int, default=20)
    parser.add_argument("--npi", type=int, default=40)
    parser.add_argument("--nmu", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corrfunc-threads", type=int, default=8)
    parser.add_argument("--corrfunc-repeats", type=int, default=3)
    parser.add_argument("--corrfunc-warmup", type=int, default=1)
    parser.add_argument("--skip-corrfunc-gpu-ddsmu", action="store_true", help="Skip the additional Corrfunc GPU line for DDsmu.")
    parser.add_argument("--jztree-repeats", type=int, default=5)
    parser.add_argument("--jztree-warmup", type=int, default=2)
    parser.add_argument("--jztree-only", action="store_true")
    parser.add_argument("--allow-int32-risk", action="store_true", help="Run even when a uniform-bin estimate approaches int32 overflow.")
    return parser.parse_args()


def main():
    args = _parse_args()
    sizes = sorted(set(args.sizes + ([1_000_000] if args.stress_1e6 else [])))
    bins = _bins(args)

    if not _has_cuda():
        raise RuntimeError("JZTREE paircount benchmarks require a CUDA-visible JAX backend.")

    if not args.jztree_only:
        try:
            import Corrfunc  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Corrfunc is required unless --jztree-only is set.") from exc

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"device: {jax.devices()[0]}")
    for npart in sizes:
        pos = jax.random.uniform(jax.random.key(args.seed), (npart, 3), dtype=jnp.float32)
        pos_np = None if args.jztree_only else np.asarray(pos)

        for mode in args.modes:
            estimated = _estimated_max_bin_count(mode, npart, bins, args.boxsize)
            if estimated > 0.8 * INT32_LIMIT and not args.allow_int32_risk:
                row = {
                    "npart": npart,
                    "mode": mode,
                    "skipped_reason": f"estimated max bin count {estimated:.3e} is close to int32 limit",
                }
                rows.append(row)
                _print_row(row)
                continue

            jztree_counts, first_call_ms, jztree_times = _measure_jztree(
                mode, pos, bins, args.boxsize, args.jztree_repeats, args.jztree_warmup
            )
            row = {
                "npart": npart,
                "mode": mode,
                "jztree_mean_ms": float(np.mean(jztree_times)),
                "jztree_std_ms": float(np.std(jztree_times)),
                "jztree_min_ms": float(np.min(jztree_times)),
                "jztree_first_call_ms": float(first_call_ms),
                "corrfunc_cpu_mean_ms": float("nan"),
                "corrfunc_cpu_std_ms": float("nan"),
                "corrfunc_cpu_min_ms": float("nan"),
                "corrfunc_cpu_speedup": float("nan"),
                "corrfunc_cpu_max_abs_diff": "",
                "corrfunc_cpu_max_rel_diff": "",
                "corrfunc_cpu_total_abs_diff": "",
                "corrfunc_gpu_mean_ms": float("nan"),
                "corrfunc_gpu_std_ms": float("nan"),
                "corrfunc_gpu_min_ms": float("nan"),
                "corrfunc_gpu_speedup": float("nan"),
                "corrfunc_gpu_max_abs_diff": "",
                "corrfunc_gpu_max_rel_diff": "",
                "corrfunc_gpu_total_abs_diff": "",
            }

            if not args.jztree_only:
                corrfunc_cpu_counts, corrfunc_cpu_times = _measure_corrfunc(
                    mode, pos_np, bins, args.boxsize, args.corrfunc_threads, args.corrfunc_repeats, args.corrfunc_warmup, False
                )
                row["corrfunc_cpu_mean_ms"] = float(np.mean(corrfunc_cpu_times))
                row["corrfunc_cpu_std_ms"] = float(np.std(corrfunc_cpu_times))
                row["corrfunc_cpu_min_ms"] = float(np.min(corrfunc_cpu_times))
                row["corrfunc_cpu_speedup"] = float(row["corrfunc_cpu_mean_ms"] / row["jztree_mean_ms"])
                for key, value in _diff_summary(jztree_counts, corrfunc_cpu_counts).items():
                    row[f"corrfunc_cpu_{key}"] = value

                if mode == "DDsmu" and not args.skip_corrfunc_gpu_ddsmu:
                    corrfunc_gpu_counts, corrfunc_gpu_times = _measure_corrfunc(
                        mode,
                        pos_np,
                        bins,
                        args.boxsize,
                        args.corrfunc_threads,
                        args.corrfunc_repeats,
                        args.corrfunc_warmup,
                        True,
                    )
                    row["corrfunc_gpu_mean_ms"] = float(np.mean(corrfunc_gpu_times))
                    row["corrfunc_gpu_std_ms"] = float(np.std(corrfunc_gpu_times))
                    row["corrfunc_gpu_min_ms"] = float(np.min(corrfunc_gpu_times))
                    row["corrfunc_gpu_speedup"] = float(row["corrfunc_gpu_mean_ms"] / row["jztree_mean_ms"])
                    for key, value in _diff_summary(jztree_counts, corrfunc_gpu_counts).items():
                        row[f"corrfunc_gpu_{key}"] = value

            rows.append(row)
            _print_row(row)

    csv_path = args.outdir / "paircount_corrfunc_scaling.csv"
    png_path = args.outdir / "paircount_corrfunc_scaling.png"
    _write_csv(csv_path, rows)
    wrote_plot = _plot(png_path, rows, args.modes)
    print(f"wrote {csv_path}")
    if wrote_plot:
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
