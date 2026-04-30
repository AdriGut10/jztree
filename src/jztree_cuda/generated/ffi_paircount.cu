// This file was automatically generated
// You can modify it, but I recommend automatically regenerating this code whenever you adapt
// one of the kernels. The FFI Bindings are very tedious in jax and they involve a lot of
// boilerplate code that is easy to mess up.

#include <map>
#include <tuple>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}
#include "../common/math.cuh"
#include "../paircount.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                           FFI call to CUDA kernel: PairCountNode2Node                          */
/* ---------------------------------------------------------------------------------------------- */

using PairCountNode2NodeDispatchFn = std::string (*) (
    cudaStream_t stream,
    const void* parent_ilist_spl,
    const void* parent_ilist_ioth,
    const void* parent_spl,
    const void* nodes,
    const void* node_qcount,
    const void* node_scount,
    void* node_ilist_spl,
    void* node_ilist_ioth,
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
);

template<int dim, typename tvec>
static std::string PairCountNode2NodeDispatchWrapper(
    cudaStream_t stream,
    const void* parent_ilist_spl,
    const void* parent_ilist_ioth,
    const void* parent_spl,
    const void* nodes,
    const void* node_qcount,
    const void* node_scount,
    void* node_ilist_spl,
    void* node_ilist_ioth,
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
    return PairCountNode2Node<dim, tvec>(
        stream,
        reinterpret_cast<const int32_t*>(parent_ilist_spl),
        reinterpret_cast<const int32_t*>(parent_ilist_ioth),
        reinterpret_cast<const int32_t*>(parent_spl),
        reinterpret_cast<const Node<dim,tvec>*>(nodes),
        reinterpret_cast<const int32_t*>(node_qcount),
        reinterpret_cast<const int32_t*>(node_scount),
        reinterpret_cast<int32_t*>(node_ilist_spl),
        reinterpret_cast<int32_t*>(node_ilist_ioth),
        same_catalog,
        use_r,
        use_rppi,
        use_smu,
        los_axis,
        boxsize,
        rmax,
        rpmax,
        pimax,
        smax,
        size_parent,
        size_node,
        size_node_ilist,
        block_size
    );
}

ffi::Error PairCountNode2NodeFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer parent_ilist_spl,
    ffi::AnyBuffer parent_ilist_ioth,
    ffi::AnyBuffer parent_spl,
    ffi::AnyBuffer nodes,
    ffi::AnyBuffer node_qcount,
    ffi::AnyBuffer node_scount,
    ffi::Result<ffi::AnyBuffer> node_ilist_spl,
    ffi::Result<ffi::AnyBuffer> node_ilist_ioth,
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
    int block_size,
    int dim
) {
    int size_parent = parent_spl.element_count() - 1;
    int size_node = node_qcount.element_count();
    size_t size_node_ilist = node_ilist_ioth->element_count();
    DT tvec = nodes.element_type();

    using TTuple = std::tuple<int, DT>;
    using TFunc = PairCountNode2NodeDispatchFn;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, &PairCountNode2NodeDispatchWrapper<2, float> },
        { {2, DT::F64}, &PairCountNode2NodeDispatchWrapper<2, double> },
        { {3, DT::F32}, &PairCountNode2NodeDispatchWrapper<3, float> },
        { {3, DT::F64}, &PairCountNode2NodeDispatchWrapper<3, double> }
    };

    const TTuple key = TTuple(dim, tvec);
    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"
            " in PairCountNode2NodeFFIHost -- Only supporting:\n"
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    PairCountNode2NodeDispatchFn instance = it->second;

    std::string result = instance(
        stream,
        parent_ilist_spl.untyped_data(),
        parent_ilist_ioth.untyped_data(),
        parent_spl.untyped_data(),
        nodes.untyped_data(),
        node_qcount.untyped_data(),
        node_scount.untyped_data(),
        node_ilist_spl->untyped_data(),
        node_ilist_ioth->untyped_data(),
        same_catalog,
        use_r,
        use_rppi,
        use_smu,
        los_axis,
        boxsize,
        rmax,
        rpmax,
        pimax,
        smax,
        size_parent,
        size_node,
        size_node_ilist,
        block_size
    );
    if (!result.empty()) {
        return ffi::Error::Internal(result);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PairCountNode2NodeFFI, PairCountNode2NodeFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // parent_ilist_spl
        .Arg<ffi::AnyBuffer>() // parent_ilist_ioth
        .Arg<ffi::AnyBuffer>() // parent_spl
        .Arg<ffi::AnyBuffer>() // nodes
        .Arg<ffi::AnyBuffer>() // node_qcount
        .Arg<ffi::AnyBuffer>() // node_scount
        .Ret<ffi::AnyBuffer>() // node_ilist_spl
        .Ret<ffi::AnyBuffer>() // node_ilist_ioth
        .Attr<bool>("same_catalog")
        .Attr<bool>("use_r")
        .Attr<bool>("use_rppi")
        .Attr<bool>("use_smu")
        .Attr<int>("los_axis")
        .Attr<double>("boxsize")
        .Attr<double>("rmax")
        .Attr<double>("rpmax")
        .Attr<double>("pimax")
        .Attr<double>("smax")
        .Attr<int>("block_size")
        .Attr<int>("dim"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                           FFI call to CUDA kernel: PairCountLeaf2LeafR                         */
/* ---------------------------------------------------------------------------------------------- */

using PairCountLeaf2LeafRDispatchFn = std::string (*) (
    cudaStream_t stream,
    const void* ilist_spl,
    const void* ilist_iother,
    const void* splT,
    const void* xT,
    const void* splQ,
    const void* xQ,
    void* hist,
    bool same_catalog,
    double boxsize,
    double rlower,
    double rupper,
    double rorigin,
    double rinv_step,
    bool rlog_bins,
    size_t size_leaves_query,
    int nbins,
    size_t block_size
);

template<int dim, typename tvec>
static std::string PairCountLeaf2LeafRDispatchWrapper(
    cudaStream_t stream,
    const void* ilist_spl,
    const void* ilist_iother,
    const void* splT,
    const void* xT,
    const void* splQ,
    const void* xQ,
    void* hist,
    bool same_catalog,
    double boxsize,
    double rlower,
    double rupper,
    double rorigin,
    double rinv_step,
    bool rlog_bins,
    size_t size_leaves_query,
    int nbins,
    size_t block_size
) {
    return PairCountLeaf2LeafR<dim, tvec>(
        stream,
        reinterpret_cast<const int32_t*>(ilist_spl),
        reinterpret_cast<const int32_t*>(ilist_iother),
        reinterpret_cast<const int32_t*>(splT),
        reinterpret_cast<const Vec<dim,tvec>*>(xT),
        reinterpret_cast<const int32_t*>(splQ),
        reinterpret_cast<const Vec<dim,tvec>*>(xQ),
        reinterpret_cast<int32_t*>(hist),
        same_catalog,
        boxsize,
        rlower,
        rupper,
        rorigin,
        rinv_step,
        rlog_bins,
        size_leaves_query,
        nbins,
        block_size
    );
}

ffi::Error PairCountLeaf2LeafRFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer ilist_spl,
    ffi::AnyBuffer ilist_iother,
    ffi::AnyBuffer splT,
    ffi::AnyBuffer xT,
    ffi::AnyBuffer splQ,
    ffi::AnyBuffer xQ,
    ffi::Result<ffi::AnyBuffer> hist,
    bool same_catalog,
    double boxsize,
    double rlower,
    double rupper,
    double rorigin,
    double rinv_step,
    bool rlog_bins,
    int block_size
) {
    size_t size_leaves_query = splQ.element_count() - 1;
    int nbins = hist->element_count();
    int dim = xT.dimensions()[1];
    DT tvec = xT.element_type();

    using TTuple = std::tuple<int, DT>;
    using TFunc = PairCountLeaf2LeafRDispatchFn;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, &PairCountLeaf2LeafRDispatchWrapper<2, float> },
        { {2, DT::F64}, &PairCountLeaf2LeafRDispatchWrapper<2, double> },
        { {3, DT::F32}, &PairCountLeaf2LeafRDispatchWrapper<3, float> },
        { {3, DT::F64}, &PairCountLeaf2LeafRDispatchWrapper<3, double> }
    };

    const TTuple key = TTuple(dim, tvec);
    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"
            " in PairCountLeaf2LeafRFFIHost -- Only supporting:\n"
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    PairCountLeaf2LeafRDispatchFn instance = it->second;

    std::string result = instance(
        stream,
        ilist_spl.untyped_data(),
        ilist_iother.untyped_data(),
        splT.untyped_data(),
        xT.untyped_data(),
        splQ.untyped_data(),
        xQ.untyped_data(),
        hist->untyped_data(),
        same_catalog,
        boxsize,
        rlower,
        rupper,
        rorigin,
        rinv_step,
        rlog_bins,
        size_leaves_query,
        nbins,
        block_size
    );
    if (!result.empty()) {
        return ffi::Error::Internal(result);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PairCountLeaf2LeafRFFI, PairCountLeaf2LeafRFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // ilist_spl
        .Arg<ffi::AnyBuffer>() // ilist_iother
        .Arg<ffi::AnyBuffer>() // splT
        .Arg<ffi::AnyBuffer>() // xT
        .Arg<ffi::AnyBuffer>() // splQ
        .Arg<ffi::AnyBuffer>() // xQ
        .Ret<ffi::AnyBuffer>() // hist
        .Attr<bool>("same_catalog")
        .Attr<double>("boxsize")
        .Attr<double>("rlower")
        .Attr<double>("rupper")
        .Attr<double>("rorigin")
        .Attr<double>("rinv_step")
        .Attr<bool>("rlog_bins")
        .Attr<int>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                        FFI call to CUDA kernel: PairCountLeaf2LeafAniso                        */
/* ---------------------------------------------------------------------------------------------- */

using PairCountLeaf2LeafAnisoDispatchFn = std::string (*) (
    cudaStream_t stream,
    const void* ilist_spl,
    const void* ilist_iother,
    const void* splT,
    const void* xT,
    const void* splQ,
    const void* xQ,
    void* hist_rppi,
    void* hist_smu,
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
    int nrp,
    int npi,
    int ns,
    int nmu,
    size_t block_size
);

template<int dim, typename tvec>
static std::string PairCountLeaf2LeafAnisoDispatchWrapper(
    cudaStream_t stream,
    const void* ilist_spl,
    const void* ilist_iother,
    const void* splT,
    const void* xT,
    const void* splQ,
    const void* xQ,
    void* hist_rppi,
    void* hist_smu,
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
    int nrp,
    int npi,
    int ns,
    int nmu,
    size_t block_size
) {
    return PairCountLeaf2LeafAniso<dim, tvec>(
        stream,
        reinterpret_cast<const int32_t*>(ilist_spl),
        reinterpret_cast<const int32_t*>(ilist_iother),
        reinterpret_cast<const int32_t*>(splT),
        reinterpret_cast<const Vec<dim,tvec>*>(xT),
        reinterpret_cast<const int32_t*>(splQ),
        reinterpret_cast<const Vec<dim,tvec>*>(xQ),
        reinterpret_cast<int32_t*>(hist_rppi),
        reinterpret_cast<int32_t*>(hist_smu),
        same_catalog,
        use_rppi,
        use_smu,
        los_axis,
        boxsize,
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
        size_leaves_query,
        nrp,
        npi,
        ns,
        nmu,
        block_size
    );
}

ffi::Error PairCountLeaf2LeafAnisoFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer ilist_spl,
    ffi::AnyBuffer ilist_iother,
    ffi::AnyBuffer splT,
    ffi::AnyBuffer xT,
    ffi::AnyBuffer splQ,
    ffi::AnyBuffer xQ,
    ffi::Result<ffi::AnyBuffer> hist_rppi,
    ffi::Result<ffi::AnyBuffer> hist_smu,
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
    int block_size
) {
    size_t size_leaves_query = splQ.element_count() - 1;
    int nrp = hist_rppi->dimensions()[0];
    int npi = hist_rppi->dimensions()[1];
    int ns = hist_smu->dimensions()[0];
    int nmu = hist_smu->dimensions()[1];
    int dim = xT.dimensions()[1];
    DT tvec = xT.element_type();

    using TTuple = std::tuple<int, DT>;
    using TFunc = PairCountLeaf2LeafAnisoDispatchFn;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2, DT::F32}, &PairCountLeaf2LeafAnisoDispatchWrapper<2, float> },
        { {2, DT::F64}, &PairCountLeaf2LeafAnisoDispatchWrapper<2, double> },
        { {3, DT::F32}, &PairCountLeaf2LeafAnisoDispatchWrapper<3, float> },
        { {3, DT::F64}, &PairCountLeaf2LeafAnisoDispatchWrapper<3, double> }
    };

    const TTuple key = TTuple(dim, tvec);
    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim, tvec)"
            " in PairCountLeaf2LeafAnisoFFIHost -- Only supporting:\n"
            "(2, float), (2, double), (3, float), (3, double)"
        );
    }
    PairCountLeaf2LeafAnisoDispatchFn instance = it->second;

    std::string result = instance(
        stream,
        ilist_spl.untyped_data(),
        ilist_iother.untyped_data(),
        splT.untyped_data(),
        xT.untyped_data(),
        splQ.untyped_data(),
        xQ.untyped_data(),
        hist_rppi->untyped_data(),
        hist_smu->untyped_data(),
        same_catalog,
        use_rppi,
        use_smu,
        los_axis,
        boxsize,
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
        size_leaves_query,
        nrp,
        npi,
        ns,
        nmu,
        block_size
    );
    if (!result.empty()) {
        return ffi::Error::Internal(result);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PairCountLeaf2LeafAnisoFFI, PairCountLeaf2LeafAnisoFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // ilist_spl
        .Arg<ffi::AnyBuffer>() // ilist_iother
        .Arg<ffi::AnyBuffer>() // splT
        .Arg<ffi::AnyBuffer>() // xT
        .Arg<ffi::AnyBuffer>() // splQ
        .Arg<ffi::AnyBuffer>() // xQ
        .Ret<ffi::AnyBuffer>() // hist_rppi
        .Ret<ffi::AnyBuffer>() // hist_smu
        .Attr<bool>("same_catalog")
        .Attr<bool>("use_rppi")
        .Attr<bool>("use_smu")
        .Attr<int>("los_axis")
        .Attr<double>("boxsize")
        .Attr<double>("rp_lower")
        .Attr<double>("rp_upper")
        .Attr<double>("rp_origin")
        .Attr<double>("rp_inv_step")
        .Attr<bool>("rp_log_bins")
        .Attr<double>("pi_lower")
        .Attr<double>("pi_upper")
        .Attr<double>("pi_origin")
        .Attr<double>("pi_inv_step")
        .Attr<double>("s_lower")
        .Attr<double>("s_upper")
        .Attr<double>("s_origin")
        .Attr<double>("s_inv_step")
        .Attr<bool>("s_log_bins")
        .Attr<double>("mu_lower")
        .Attr<double>("mu_upper")
        .Attr<double>("mu_origin")
        .Attr<double>("mu_inv_step")
        .Attr<int>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_paircount, m) {
    m.def("PairCountNode2Node", []() { return EncapsulateFfiCall(&PairCountNode2NodeFFI); });
    m.def("PairCountLeaf2LeafR", []() { return EncapsulateFfiCall(&PairCountLeaf2LeafRFFI); });
    m.def("PairCountLeaf2LeafAniso", []() { return EncapsulateFfiCall(&PairCountLeaf2LeafAnisoFFI); });
}
