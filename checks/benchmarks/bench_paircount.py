import pytest
import jax
import jax.numpy as jnp

import jztree as jz


def _bin_specs():
    r = (0.0, 0.2, 20, "lin")
    rp = (0.0, 0.2, 20, "lin")
    pi = (-0.4, 0.4, 80, "lin")
    s = (0.0, 0.2, 20, "lin")
    mu = (-1.0, 1.0, 80, "lin")
    return r, rp, pi, s, mu


@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [int(5e4), int(1e5), int(2e5), int(3e5)])
def bench_paircount_static_bins_DD(jax_bench, npart):
    boxsize = 1.0
    cfg = jz.config.PairCountConfig()
    jb = jax_bench(jit_rounds=3, jit_warmup=1)
    pos = jax.random.uniform(jax.random.key(0), (npart, 3), dtype=jnp.float32)
    r_spec = _bin_specs()[0]
    r = jz.paircount._format_bin_spec(r_spec, allow_log=True, name="r")

    partz, th = jb.measure(fn_jit=jz.tree.zsort_and_tree.jit, part=jz.data.Pos(pos=pos), cfg_tree=cfg.tree, tag="zsort_tree")[1]
    ilist = jb.measure(
        fn_jit=jz.paircount._paircount_dual_walk.jit,
        th=th,
        ptype_query=0,
        ptype_source=0,
        same_catalog=True,
        use_r=True,
        rmax=r.edges[-1],
        use_rppi=False,
        rpmax=0.0,
        pimax=0.0,
        use_smu=False,
        smax=0.0,
        los_axis=2,
        boxsize=boxsize,
        alloc_fac_ilist=cfg.alloc_fac_ilist,
        tag="treewalk_DD",
    )[1]
    spl = th.splits_leaf_to_part()
    jb.measure(
        fn_jit=jz.paircount._paircount_leaf2leaf_r.jit,
        ilist=ilist,
        splT=spl,
        xT=partz.pos,
        r=r,
        same_catalog=True,
        boxsize=boxsize,
        block_size=128,
        tag="leaf2leaf_DD",
    )
    jb.measure(fn_jit=jz.paircount.DD.jit, pos1=pos, r=r_spec, boxsize=boxsize, tag="total_DD")


@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [int(5e4), int(1e5), int(2e5), int(3e5)])
def bench_paircount_static_bins_DDrppi(jax_bench, npart):
    boxsize = 1.0
    cfg = jz.config.PairCountConfig()
    jb = jax_bench(jit_rounds=3, jit_warmup=1)
    pos = jax.random.uniform(jax.random.key(0), (npart, 3), dtype=jnp.float32)
    _, rp_spec, pi_spec, _, _ = _bin_specs()
    rppi = jz.paircount._RppiSpec(
        rp=jz.paircount._format_bin_spec(rp_spec, allow_log=True, name="rp", min_value=0.0),
        pi=jz.paircount._format_bin_spec(pi_spec, allow_log=False, name="pi"),
    )

    partz, th = jb.measure(fn_jit=jz.tree.zsort_and_tree.jit, part=jz.data.Pos(pos=pos), cfg_tree=cfg.tree, tag="zsort_tree")[1]
    ilist = jb.measure(
        fn_jit=jz.paircount._paircount_dual_walk.jit,
        th=th,
        ptype_query=0,
        ptype_source=0,
        same_catalog=True,
        use_r=False,
        rmax=0.0,
        use_rppi=True,
        rpmax=rppi.rp.edges[-1],
        pimax=max(abs(rppi.pi.edges[0]), abs(rppi.pi.edges[-1])),
        use_smu=False,
        smax=0.0,
        los_axis=2,
        boxsize=boxsize,
        alloc_fac_ilist=cfg.alloc_fac_ilist,
        tag="treewalk_DDrppi",
    )[1]
    spl = th.splits_leaf_to_part()
    jb.measure(
        fn_jit=jz.paircount._paircount_leaf2leaf_aniso.jit,
        ilist=ilist,
        splT=spl,
        xT=partz.pos,
        config=jz.paircount._PairCountRuntimeConfig(boxsize=boxsize, los_axis=2, same_catalog=True, rppi=rppi),
        block_size=128,
        tag="leaf2leaf_DDrppi",
    )
    jb.measure(
        fn_jit=jz.paircount.DDrppi.jit,
        pos1=pos,
        rp=rp_spec,
        pi=pi_spec,
        boxsize=boxsize,
        tag="total_DDrppi",
    )


@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [int(5e4), int(1e5), int(2e5), int(3e5)])
def bench_paircount_static_bins_DDsmu(jax_bench, npart):
    boxsize = 1.0
    cfg = jz.config.PairCountConfig()
    jb = jax_bench(jit_rounds=3, jit_warmup=1)
    pos = jax.random.uniform(jax.random.key(0), (npart, 3), dtype=jnp.float32)
    _, _, _, s_spec, mu_spec = _bin_specs()
    smu = jz.paircount._SmuSpec(
        s=jz.paircount._format_bin_spec(s_spec, allow_log=True, name="s", min_value=0.0),
        mu=jz.paircount._format_bin_spec(mu_spec, allow_log=False, name="mu", min_value=-1.0, max_value=1.0),
    )

    partz, th = jb.measure(fn_jit=jz.tree.zsort_and_tree.jit, part=jz.data.Pos(pos=pos), cfg_tree=cfg.tree, tag="zsort_tree")[1]
    ilist = jb.measure(
        fn_jit=jz.paircount._paircount_dual_walk.jit,
        th=th,
        ptype_query=0,
        ptype_source=0,
        same_catalog=True,
        use_r=False,
        rmax=0.0,
        use_rppi=False,
        rpmax=0.0,
        pimax=0.0,
        use_smu=True,
        smax=smu.s.edges[-1],
        los_axis=2,
        boxsize=boxsize,
        alloc_fac_ilist=cfg.alloc_fac_ilist,
        tag="treewalk_DDsmu",
    )[1]
    spl = th.splits_leaf_to_part()
    jb.measure(
        fn_jit=jz.paircount._paircount_leaf2leaf_aniso.jit,
        ilist=ilist,
        splT=spl,
        xT=partz.pos,
        config=jz.paircount._PairCountRuntimeConfig(boxsize=boxsize, los_axis=2, same_catalog=True, smu=smu),
        block_size=128,
        tag="leaf2leaf_DDsmu",
    )
    jb.measure(
        fn_jit=jz.paircount.DDsmu.jit,
        pos1=pos,
        s=s_spec,
        mu=mu_spec,
        boxsize=boxsize,
        tag="total_DDsmu",
    )


def bench_paircount_static_bins_setup(jax_bench):
    cfg = jz.config.PairCountConfig()
    jb = jax_bench(jit_rounds=2, jit_warmup=1)
    r, rp, pi, s, mu = _bin_specs()

    def total(pos):
        return (
            jz.paircount.DD.jit(pos, r, boxsize=1.0, cfg=cfg),
            jz.paircount.DDrppi.jit(pos, rp, pi, boxsize=1.0, cfg=cfg),
            jz.paircount.DDsmu.jit(pos, s, mu, boxsize=1.0, cfg=cfg),
        )

    total_jit = jax.jit(total)

    print("\nUniform:")
    with jz.stats.statistics() as st:
        pos = jax.random.uniform(jax.random.key(0), (int(2e5), 3), dtype=jnp.float32)
        jb.measure(fn_jit=total_jit, pos=pos, tag="uniform")
        st.print_suggestions(cfg)

    print("\nClustered:")
    with jz.stats.statistics() as st:
        pos = jax.random.normal(jax.random.key(1), (int(2e5), 3), dtype=jnp.float32) * 0.12 + 0.5
        pos = jnp.mod(pos, 1.0)
        jb.measure(fn_jit=total_jit, pos=pos, tag="clustered")
        st.print_suggestions(cfg)
