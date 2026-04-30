#ifndef PAIRCOUNT_H
#define PAIRCOUNT_H

#include <cub/cub.cuh>
#include <limits>

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

struct ConstInteractionList {
    const int32_t* spl;
    const int32_t* iother;
};

struct InteractionList {
    int32_t* spl;
    int32_t* iother;
};

template<int dim, typename tvec>
__device__ __forceinline__ tvec min_axis_dist(
    Vec<dim,tvec> x1,
    Vec<dim,tvec> x2,
    Vec<dim,tvec> width_half,
    int axis,
    tvec boxsize
) {
    tvec dcent = wrap_dx<tvec>(x1[axis] - x2[axis], boxsize);
    return max(abs(dcent) - width_half[axis], static_cast<tvec>(0));
}

template<int dim, typename tvec>
__device__ __forceinline__ tvec min_perp_dist2(
    Vec<dim,tvec> x1,
    Vec<dim,tvec> x2,
    Vec<dim,tvec> width_half,
    int axis,
    tvec boxsize
) {
    tvec r2 = static_cast<tvec>(0);
    #pragma unroll
    for(int i = 0; i < dim; i++) {
        if(i == axis) continue;
        tvec dcent = wrap_dx<tvec>(x1[i] - x2[i], boxsize);
        tvec dx = max(abs(dcent) - width_half[i], static_cast<tvec>(0));
        r2 += dx * dx;
    }
    return r2;
}

template<typename tvec>
__device__ __forceinline__ int bin_from_value(
    tvec value,
    tvec lower,
    tvec upper,
    tvec origin,
    tvec inv_step
) {
    tvec upper_safe = nextafter(upper, lower);
    tvec coord = min(value, upper_safe);
    return static_cast<int>(floor((coord - origin) * inv_step));
}

template<typename tvec>
__device__ __forceinline__ int bin_from_radius2(
    tvec r2,
    tvec lower,
    tvec upper,
    tvec origin,
    tvec inv_step,
    bool log_bins
) {
    if(log_bins) {
        tvec upper_safe = nextafter(upper, lower);
        tvec coord = static_cast<tvec>(0.5) * log(min(max(r2, lower * lower), upper_safe * upper_safe));
        return static_cast<int>(floor((coord - origin) * inv_step));
    }
    return bin_from_value<tvec>(sqrt(r2), lower, upper, origin, inv_step);
}

__device__ __forceinline__ void clear_hist(int* hist, int nbins) {
    for(int i = threadIdx.x; i < nbins; i += blockDim.x) {
        hist[i] = 0;
    }
    __syncthreads();
}

__device__ __forceinline__ void flush_hist(const int* hist, int* out, int nbins) {
    for(int i = threadIdx.x; i < nbins; i += blockDim.x) {
        atomicAdd(&out[i], hist[i]);
    }
}

template<int dim, typename tvec>
__device__ __forceinline__ void add_aniso_pair_to_hist(
    Vec<dim,tvec> dx,
    int32_t* hist_rppi,
    int32_t* hist_smu,
    bool use_rppi,
    bool use_smu,
    int los_axis,
    tvec rp_lower,
    tvec rp_upper,
    tvec rp_origin,
    tvec rp_inv_step,
    bool rp_log_bins,
    tvec pi_lower,
    tvec pi_upper,
    tvec pi_origin,
    tvec pi_inv_step,
    tvec s_lower,
    tvec s_upper,
    tvec s_origin,
    tvec s_inv_step,
    bool s_log_bins,
    tvec mu_lower,
    tvec mu_upper,
    tvec mu_origin,
    tvec mu_inv_step,
    int32_t nrp,
    int32_t npi,
    int32_t ns,
    int32_t nmu
) {
    tvec pi = dx[los_axis];
    tvec s2 = static_cast<tvec>(0);
    tvec rp2 = static_cast<tvec>(0);
    #pragma unroll
    for(int i = 0; i < dim; i++) {
        s2 += dx[i] * dx[i];
        if(i != los_axis) rp2 += dx[i] * dx[i];
    }

    if(use_rppi && rp2 >= rp_lower * rp_lower && rp2 <= rp_upper * rp_upper && pi >= pi_lower && pi <= pi_upper) {
        int irp = bin_from_radius2<tvec>(rp2, rp_lower, rp_upper, rp_origin, rp_inv_step, rp_log_bins);
        int ipi = bin_from_value<tvec>(pi, pi_lower, pi_upper, pi_origin, pi_inv_step);
        if(irp >= 0 && irp < nrp && ipi >= 0 && ipi < npi) {
            atomicAdd(&hist_rppi[irp * npi + ipi], 1);
        }
    }

    if(use_smu && s2 >= s_lower * s_lower && s2 <= s_upper * s_upper) {
        tvec s = sqrt(s2);
        tvec mu = pi / max(s, static_cast<tvec>(1.0e-30));
        if(mu > mu_lower && mu < mu_upper) {
            int is = bin_from_radius2<tvec>(s2, s_lower, s_upper, s_origin, s_inv_step, s_log_bins);
            int imu = bin_from_value<tvec>(mu, mu_lower, mu_upper, mu_origin, mu_inv_step);
            if(is >= 0 && is < ns && imu >= 0 && imu < nmu) {
                atomicAdd(&hist_smu[is * nmu + imu], 1);
            }
        }
    }
}

template<int pass, int dim, typename tvec>
__global__ void PairCountNode2NodeCountInsert(
    ConstInteractionList par_ilist,
    const int32_t* __restrict__ parent_spl,
    const Node<dim,tvec>* __restrict__ nodes,
    const int32_t* __restrict__ node_qcount,
    const int32_t* __restrict__ node_scount,
    int32_t* __restrict__ node_icount,
    InteractionList node_ilist,
    bool same_catalog,
    bool use_r,
    bool use_rppi,
    bool use_smu,
    int los_axis,
    tvec boxsize,
    tvec rmax2,
    tvec rpmax2,
    tvec pimax,
    tvec smax2,
    int32_t nmax
) {
    extern __shared__ unsigned char smem[];

    int parentQ = blockIdx.x;
    int inodeQ_start = parent_spl[parentQ];
    int inodeQ_end = parent_spl[parentQ + 1];

    for(int iqoff = inodeQ_start; iqoff < inodeQ_end; iqoff += blockDim.x) {
        int inodeQ = min(iqoff + threadIdx.x, inodeQ_end - 1);
        bool valid = iqoff + threadIdx.x < inodeQ_end;
        if(valid && node_qcount[inodeQ] <= 0) valid = false;

        NodeWithExt<dim,tvec> nodeQ = NodeLvlToHalfExt<dim,tvec>(nodes[inodeQ]);
        int32_t out_offset = pass == 1 ? node_ilist.spl[inodeQ] : 0;
        int32_t ncount = 0;

        PrefetchList<int32_t> pf_ilist(par_ilist.iother, par_ilist.spl[parentQ], par_ilist.spl[parentQ + 1]);

        while(!pf_ilist.finished()) {
            int parentT = pf_ilist.next();
            if(same_catalog && parentT < parentQ) continue;

            int inodeT_start = parent_spl[parentT];
            int inodeT_end = parent_spl[parentT + 1];

            NodeWithExt<dim,tvec>* nodeT = reinterpret_cast<NodeWithExt<dim,tvec>*>(smem);
            for(int itoff = inodeT_start; itoff < inodeT_end; itoff += blockDim.x) {
                int inodeT = itoff + threadIdx.x;
                if(inodeT < inodeT_end) {
                    nodeT[threadIdx.x] = NodeLvlToHalfExt<dim,tvec>(nodes[inodeT]);
                }
                __syncthreads();

                int ntile = min(inodeT_end - itoff, blockDim.x);
                for(int j = 0; j < ntile; j++) {
                    int inodeT_cur = itoff + j;
                    if(!valid) continue;
                    if(node_scount[inodeT_cur] <= 0) continue;
                    if(same_catalog && parentT == parentQ && inodeT_cur < inodeQ) continue;

                    Vec<dim,tvec> width_half = nodeQ.extent + nodeT[j].extent;
                    tvec r2min = mindist2<dim,tvec>(nodeT[j].center, nodeQ.center, width_half, boxsize);

                    bool keep = false;
                    if(use_r && r2min <= rmax2) keep = true;
                    if(use_smu && r2min <= smax2) keep = true;
                    if(use_rppi) {
                        tvec pi_min = min_axis_dist<dim,tvec>(nodeT[j].center, nodeQ.center, width_half, los_axis, boxsize);
                        tvec rp2_min = min_perp_dist2<dim,tvec>(nodeT[j].center, nodeQ.center, width_half, los_axis, boxsize);
                        keep = keep || ((rp2_min <= rpmax2) && (pi_min <= pimax));
                    }

                    if(keep) {
                        if(pass == 1) {
                            int32_t offset = out_offset + ncount;
                            if(offset < nmax) node_ilist.iother[offset] = inodeT_cur;
                        }
                        ncount += 1;
                    }
                }
                __syncthreads();
            }
        }

        if(pass == 0 && valid) node_icount[inodeQ] = ncount;
    }
}

template<int dim, typename tvec>
std::string PairCountNode2Node(
    cudaStream_t stream,
    const int32_t* parent_ilist_spl,
    const int32_t* parent_ilist_ioth,
    const int32_t* parent_spl,
    const Node<dim,tvec>* nodes,
    const int32_t* node_qcount,
    const int32_t* node_scount,
    int32_t* node_ilist_spl,
    int32_t* node_ilist_ioth,
    bool same_catalog,
    bool use_r,
    bool use_rppi,
    bool use_smu,
    int los_axis,
    double boxsize,
    double rmax,
    double rpmax,
    double pimax,
    double smax,
    int size_parent,
    int size_node,
    size_t size_node_ilist,
    int block_size
) {
    ConstInteractionList par_ilist = {parent_ilist_spl, parent_ilist_ioth};
    InteractionList node_ilist = {node_ilist_spl, node_ilist_ioth};

    if(size_node_ilist > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return std::string("PairCountNode2Node only supports int32 interaction-list allocations.");
    }

    cudaMemsetAsync(node_ilist.spl, 0, sizeof(int32_t) * (size_node + 1), stream);

    size_t smem_alloc_size = block_size * sizeof(NodeWithExt<dim,tvec>);
    PairCountNode2NodeCountInsert<0,dim,tvec><<<size_parent, block_size, smem_alloc_size, stream>>>(
        par_ilist,
        parent_spl,
        nodes,
        node_qcount,
        node_scount,
        node_ilist.spl + 1,
        node_ilist,
        same_catalog,
        use_r,
        use_rppi,
        use_smu,
        los_axis,
        static_cast<tvec>(boxsize),
        static_cast<tvec>(rmax * rmax),
        static_cast<tvec>(rpmax * rpmax),
        static_cast<tvec>(pimax),
        static_cast<tvec>(smax * smax),
        static_cast<int32_t>(size_node_ilist)
    );

    size_t tmp_bytes;
    cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes, node_ilist.spl + 1, node_ilist.spl + 1, size_node, stream);
    if(tmp_bytes > size_node_ilist * sizeof(int32_t)) {
        return std::string(
            "Scan allocation too small! Needed: " + std::to_string(tmp_bytes) + " bytes. " +
            "Have: " + std::to_string(size_node_ilist * sizeof(int32_t)) + " bytes."
        );
    }
    cub::DeviceScan::InclusiveSum(
        node_ilist.iother, tmp_bytes, node_ilist.spl + 1, node_ilist.spl + 1, size_node, stream
    );

    PairCountNode2NodeCountInsert<1,dim,tvec><<<size_parent, block_size, smem_alloc_size, stream>>>(
        par_ilist,
        parent_spl,
        nodes,
        node_qcount,
        node_scount,
        nullptr,
        node_ilist,
        same_catalog,
        use_r,
        use_rppi,
        use_smu,
        los_axis,
        static_cast<tvec>(boxsize),
        static_cast<tvec>(rmax * rmax),
        static_cast<tvec>(rpmax * rpmax),
        static_cast<tvec>(pimax),
        static_cast<tvec>(smax * smax),
        static_cast<int32_t>(size_node_ilist)
    );

    return std::string();
}

template<int dim, typename tvec>
__global__ void PairCountLeaf2LeafRKernel(
    ConstInteractionList ilist,
    const int32_t* __restrict__ splT,
    const Vec<dim,tvec>* __restrict__ xT,
    const int32_t* __restrict__ splQ,
    const Vec<dim,tvec>* __restrict__ xQ,
    int32_t* __restrict__ hist_out,
    bool same_catalog,
    tvec boxsize,
    tvec rlower,
    tvec rupper,
    tvec rorigin,
    tvec rinv_step,
    bool rlog_bins,
    int32_t nbins
) {
    int ileafQ = blockIdx.x;
    int iqstart = splQ[ileafQ];
    int iqend = splQ[ileafQ + 1];
    if(iqstart >= iqend) return;

    extern __shared__ unsigned char smem[];
    Vec<dim,tvec>* tileT = reinterpret_cast<Vec<dim,tvec>*>(smem);
    int32_t* hist = reinterpret_cast<int32_t*>(tileT + blockDim.x);

    clear_hist(hist, nbins);

    for(int qoff = iqstart; qoff < iqend; qoff += blockDim.x) {
        int ipartQ = min(qoff + threadIdx.x, iqend - 1);
        bool valid = qoff + threadIdx.x < iqend;
        Vec<dim,tvec> posQ = xQ[ipartQ];

        PrefetchList<int32_t> pf_ilist(ilist.iother, ilist.spl[ileafQ], ilist.spl[ileafQ + 1]);
        while(!pf_ilist.finished()) {
            int ileafT = pf_ilist.next();
            int itstart = splT[ileafT];
            int itend = splT[ileafT + 1];

            for(int toff = itstart; toff < itend; toff += blockDim.x) {
                int nload = min(itend - toff, blockDim.x);
                if(threadIdx.x < nload) tileT[threadIdx.x] = xT[toff + threadIdx.x];
                __syncthreads();

                if(valid) {
                    for(int j = 0; j < nload; j++) {
                        tvec r2 = distance_squared<dim,tvec>(posQ, tileT[j], boxsize);
                        if(r2 < rlower * rlower || r2 > rupper * rupper) continue;

                        int ibin = bin_from_radius2<tvec>(r2, rlower, rupper, rorigin, rinv_step, rlog_bins);
                        if(ibin >= 0 && ibin < nbins) {
                            int32_t inc = (same_catalog && ileafT != ileafQ) ? 2 : 1;
                            atomicAdd(&hist[ibin], inc);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    flush_hist(hist, hist_out, nbins);
}

template<int dim, typename tvec>
std::string PairCountLeaf2LeafR(
    cudaStream_t stream,
    const int32_t* ilist_spl,
    const int32_t* ilist_iother,
    const int32_t* splT,
    const Vec<dim,tvec>* xT,
    const int32_t* splQ,
    const Vec<dim,tvec>* xQ,
    int32_t* hist,
    bool same_catalog,
    double boxsize,
    double rlower,
    double rupper,
    double rorigin,
    double rinv_step,
    bool rlog_bins,
    size_t size_leaves_query,
    int32_t nbins,
    size_t block_size
) {
    ConstInteractionList ilist = {ilist_spl, ilist_iother};

    cudaMemsetAsync(hist, 0, sizeof(int32_t) * nbins, stream);

    size_t smem = block_size * sizeof(Vec<dim,tvec>) + nbins * sizeof(int32_t);
    PairCountLeaf2LeafRKernel<dim,tvec><<<size_leaves_query, block_size, smem, stream>>>(
        ilist,
        splT,
        xT,
        splQ,
        xQ,
        hist,
        same_catalog,
        static_cast<tvec>(boxsize),
        static_cast<tvec>(rlower),
        static_cast<tvec>(rupper),
        static_cast<tvec>(rorigin),
        static_cast<tvec>(rinv_step),
        rlog_bins,
        nbins
    );

    return std::string();
}

template<int dim, typename tvec>
__global__ void PairCountLeaf2LeafAnisoKernel(
    ConstInteractionList ilist,
    const int32_t* __restrict__ splT,
    const Vec<dim,tvec>* __restrict__ xT,
    const int32_t* __restrict__ splQ,
    const Vec<dim,tvec>* __restrict__ xQ,
    int32_t* __restrict__ hist_rppi_out,
    int32_t* __restrict__ hist_smu_out,
    bool same_catalog,
    bool use_rppi,
    bool use_smu,
    int los_axis,
    tvec boxsize,
    tvec rp_lower,
    tvec rp_upper,
    tvec rp_origin,
    tvec rp_inv_step,
    bool rp_log_bins,
    tvec pi_lower,
    tvec pi_upper,
    tvec pi_origin,
    tvec pi_inv_step,
    tvec s_lower,
    tvec s_upper,
    tvec s_origin,
    tvec s_inv_step,
    bool s_log_bins,
    tvec mu_lower,
    tvec mu_upper,
    tvec mu_origin,
    tvec mu_inv_step,
    int32_t nrp,
    int32_t npi,
    int32_t ns,
    int32_t nmu
) {
    int ileafQ = blockIdx.x;
    int iqstart = splQ[ileafQ];
    int iqend = splQ[ileafQ + 1];
    if(iqstart >= iqend) return;

    int32_t nbins_rppi = nrp * npi;
    int32_t nbins_smu = ns * nmu;

    extern __shared__ unsigned char smem[];
    Vec<dim,tvec>* tileT = reinterpret_cast<Vec<dim,tvec>*>(smem);
    int32_t* hist_rppi = reinterpret_cast<int32_t*>(tileT + blockDim.x);
    int32_t* hist_smu = hist_rppi + nbins_rppi;

    clear_hist(hist_rppi, nbins_rppi + nbins_smu);

    for(int qoff = iqstart; qoff < iqend; qoff += blockDim.x) {
        int ipartQ = min(qoff + threadIdx.x, iqend - 1);
        bool valid = qoff + threadIdx.x < iqend;
        Vec<dim,tvec> posQ = xQ[ipartQ];

        PrefetchList<int32_t> pf_ilist(ilist.iother, ilist.spl[ileafQ], ilist.spl[ileafQ + 1]);
        while(!pf_ilist.finished()) {
            int ileafT = pf_ilist.next();
            int itstart = splT[ileafT];
            int itend = splT[ileafT + 1];

            for(int toff = itstart; toff < itend; toff += blockDim.x) {
                int nload = min(itend - toff, blockDim.x);
                if(threadIdx.x < nload) tileT[threadIdx.x] = xT[toff + threadIdx.x];
                __syncthreads();

                if(valid) {
                    for(int j = 0; j < nload; j++) {
                        Vec<dim,tvec> dx;
                        #pragma unroll
                        for(int i = 0; i < dim; i++) dx[i] = wrap_dx<tvec>(tileT[j][i] - posQ[i], boxsize);

                        add_aniso_pair_to_hist<dim,tvec>(
                            dx,
                            hist_rppi,
                            hist_smu,
                            use_rppi,
                            use_smu,
                            los_axis,
                            rp_lower,
                            rp_upper,
                            rp_origin,
                            rp_inv_step,
                            rp_log_bins,
                            pi_lower,
                            pi_upper,
                            pi_origin,
                            pi_inv_step,
                            s_lower,
                            s_upper,
                            s_origin,
                            s_inv_step,
                            s_log_bins,
                            mu_lower,
                            mu_upper,
                            mu_origin,
                            mu_inv_step,
                            nrp,
                            npi,
                            ns,
                            nmu
                        );

                        if(same_catalog && ileafT != ileafQ) {
                            #pragma unroll
                            for(int i = 0; i < dim; i++) dx[i] = -dx[i];
                            add_aniso_pair_to_hist<dim,tvec>(
                                dx,
                                hist_rppi,
                                hist_smu,
                                use_rppi,
                                use_smu,
                                los_axis,
                                rp_lower,
                                rp_upper,
                                rp_origin,
                                rp_inv_step,
                                rp_log_bins,
                                pi_lower,
                                pi_upper,
                                pi_origin,
                                pi_inv_step,
                                s_lower,
                                s_upper,
                                s_origin,
                                s_inv_step,
                                s_log_bins,
                                mu_lower,
                                mu_upper,
                                mu_origin,
                                mu_inv_step,
                                nrp,
                                npi,
                                ns,
                                nmu
                            );
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    flush_hist(hist_rppi, hist_rppi_out, nbins_rppi);
    flush_hist(hist_smu, hist_smu_out, nbins_smu);
}

template<int dim, typename tvec>
std::string PairCountLeaf2LeafAniso(
    cudaStream_t stream,
    const int32_t* ilist_spl,
    const int32_t* ilist_iother,
    const int32_t* splT,
    const Vec<dim,tvec>* xT,
    const int32_t* splQ,
    const Vec<dim,tvec>* xQ,
    int32_t* hist_rppi,
    int32_t* hist_smu,
    bool same_catalog,
    bool use_rppi,
    bool use_smu,
    int los_axis,
    double boxsize,
    double rp_lower,
    double rp_upper,
    double rp_origin,
    double rp_inv_step,
    bool rp_log_bins,
    double pi_lower,
    double pi_upper,
    double pi_origin,
    double pi_inv_step,
    double s_lower,
    double s_upper,
    double s_origin,
    double s_inv_step,
    bool s_log_bins,
    double mu_lower,
    double mu_upper,
    double mu_origin,
    double mu_inv_step,
    size_t size_leaves_query,
    int32_t nrp,
    int32_t npi,
    int32_t ns,
    int32_t nmu,
    size_t block_size
) {
    ConstInteractionList ilist = {ilist_spl, ilist_iother};
    int32_t nbins_rppi = nrp * npi;
    int32_t nbins_smu = ns * nmu;

    cudaMemsetAsync(hist_rppi, 0, sizeof(int32_t) * nbins_rppi, stream);
    cudaMemsetAsync(hist_smu, 0, sizeof(int32_t) * nbins_smu, stream);

    size_t smem = block_size * sizeof(Vec<dim,tvec>) + (nbins_rppi + nbins_smu) * sizeof(int32_t);
    PairCountLeaf2LeafAnisoKernel<dim,tvec><<<size_leaves_query, block_size, smem, stream>>>(
        ilist,
        splT,
        xT,
        splQ,
        xQ,
        hist_rppi,
        hist_smu,
        same_catalog,
        use_rppi,
        use_smu,
        los_axis,
        static_cast<tvec>(boxsize),
        static_cast<tvec>(rp_lower),
        static_cast<tvec>(rp_upper),
        static_cast<tvec>(rp_origin),
        static_cast<tvec>(rp_inv_step),
        rp_log_bins,
        static_cast<tvec>(pi_lower),
        static_cast<tvec>(pi_upper),
        static_cast<tvec>(pi_origin),
        static_cast<tvec>(pi_inv_step),
        static_cast<tvec>(s_lower),
        static_cast<tvec>(s_upper),
        static_cast<tvec>(s_origin),
        static_cast<tvec>(s_inv_step),
        s_log_bins,
        static_cast<tvec>(mu_lower),
        static_cast<tvec>(mu_upper),
        static_cast<tvec>(mu_origin),
        static_cast<tvec>(mu_inv_step),
        nrp,
        npi,
        ns,
        nmu
    );

    return std::string();
}

#endif // PAIRCOUNT_H
