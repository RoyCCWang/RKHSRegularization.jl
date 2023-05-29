# Custom composite types.

### 1-D input kernels for modeling a scalar-valued state function of
#   time.

abstract type ODEKernelType end


struct GaussianKernelODEType{T} <: ODEKernelType
    ϵ_sq::T
    s::T # amplitude scale.
end



### elemntary Kernels.

abstract type StationaryKernelType end

# first number is dimension, second is differentiability.
struct Spline12KernelType{T} <: StationaryKernelType
    a::T
end

struct Spline32KernelType{T} <: StationaryKernelType
    a::T
end

struct Spline34KernelType{T} <: StationaryKernelType
    a::T
end

struct RationalQuadraticKernelType{T} <: StationaryKernelType
    a::T
end

struct TunableRationalQuadraticKernelType{T} <: StationaryKernelType
    a::T
    w::T # variance.
end

struct ModulatedSqExpKernelType{T} <: StationaryKernelType
    ϵ_sq::T # decay.
    ν::T # frequency.
end

struct ModulatedMultivariateSqExpKernelType{T}
    ϵ_sq::T # decay.
    ν::Vector{T} # frequency.
end

struct AdaptiveModulatedSqExpKernelType{T} <: StationaryKernelType
    ϵ_sq::T # decay.
    ν::Vector{T} # frequency.
    ϕ::Function # warp map for Sq Exp.
    ψ::Function # warp map for cosine.
end

### Non-canonical kernels
struct KRWarpKernelType{KTv,KTx}
    θ_v::KTv
    θ_x::KTx
end

struct GaussianKernel1DType{T} <: StationaryKernelType
    ϵ_sq::T
end


struct DoubleProductKernelType{KT}
    θ_base::KT
    M::Int
end

abstract type BrownianBridgeKernelType end

struct BrownianBridge10{T} <: BrownianBridgeKernelType
    a::T
end

struct BrownianBridge20{T} <: BrownianBridgeKernelType
    a::T
end

struct BrownianBridge20CompactDomain{T} <: BrownianBridgeKernelType
    a::Vector{T} # lower bound of domain.
    b::Vector{T} # upper bound of domain.
end

# struct BrownianBridgeβ0 <: BrownianBridgeKernelType
#     β::Int
# end

struct BrownianBridge1ϵ{T} <: BrownianBridgeKernelType
    ϵ::T
end

struct BrownianBridge2ϵ{T} <: BrownianBridgeKernelType
    ϵ::T
end


struct BrownianBridgeSemiInfDomain{BT <: BrownianBridgeKernelType}
    θ_base::BT
end

struct BrownianBridgeCompactDomain{BT <: BrownianBridgeKernelType, T}
    θ_base::BT
    a::T # common lower bound for all input dimensions.
    b::T # common upper bound for all input dimensions.
end

### adaptive kernel.

struct AdaptiveKernelDPPType{KT}
    canonical_params::KT # Canonical kernel parameters.
    warpfunc::Function
end

struct AdaptiveKernelType{KT}
    canonical_params::KT # Canonical kernel parameters.
    warpfunc::Function
end

struct FastAdaptiveKernelType{KT, T, D}
    canonical_kernel::KT # Canonical kernel parameters.
    warpfuncs::Vector{Function}
    w_X::Array{T,D} # pre-computed warp map evaluations at training positions.
    s::Vector{T} # weights for each warp function.
end

struct AdaptiveKernelMultiWarpType{KT, T}
    canonical_params::KT # Canonical kernel parameters.
    warpfuncs::Vector{Function} # length M.

    # length M. Positive weights for the m-th warpfunc
    #   to the diagonal entries of the kernel matrix.
    a::Vector{T}
end

struct AdaptiveKernelMultiWarpDPPType{KT, T}
    canonical_params::KT # Canonical kernel parameters.
    warpfuncs::Vector{Function} # length M.

    # length M. Positive weights for the m-th warpfunc
    #   to the diagonal entries of the kernel matrix.
    a::Vector{T}
    self_gain::T
end

# sum of kernels. Weights should be positive.
"""
out = θ.base_gain * evalkernel(p, q, θ.base_kernel_parameters) + δ(p,q) * additive_function(p).
δ(p,q) is kronecker delta.

Type fields:
base_kernel_parameters::KT
additive_function::Function # takes input on 𝓧 to ℝ_{++}.
base_gain::T
zero_tol::T
"""
struct AdditiveVarianceKernelType{KT, T}
    base_kernel_parameters::KT
    additive_function::Function # takes input on 𝓧 to ℝ_{++}.
    base_gain::T
    zero_tol::T
end

struct AdaptiveKernelΨType{KT}
    canonical_params::KT # Canonical kernel parameters.
    ψ::Function
end

struct BrownianBridge20Adaptive
    β::Int # set to 2 for now.
    warpfunc::Function # must be between 0 and 1
end

struct SOSKernelType{KT}
    base_params::KT
end

struct ElementaryKRKernelType{KT,T}
    θ_a::AdaptiveKernelType{KT}
    θ_canonical::RationalQuadraticKernelType{T}
end

struct KRKernelType{KT,T}
    θ_EKR::Vector{ElementaryKRKernelType{KT,T}}
end

### for fitting conditional pdf, univaraite.
# p(A | B), B could be multivariate, A is univariate.
struct CPDF1DKernelType{T, KT}
    A::RationalQuadraticKernelType{T}
    B::AdaptiveKernelType{KT}
end

# struct CPDF1DKernelType{KT_B,KT_A}
#     B::KT_B
#     A::KT_A
# end

### Squared kernel, from multinomial theorem.

struct SquaredFunctionsKernelType{KT}
    base_params::KT
end

### approximation kernel.
# struct ApproxKernelType{Kernel_Type1, Kernel_Type2, T, X_Type}
#     c::Vector{Vector{T}} # solution of the problem.
#     X::Vector{Vector{X_Type}} # sampling locations.
#     θ1::Kernel_Type1 # outer kernel
#     θ2::Kernel_Type2 # inner kernel
#     σ²::T
# end



### warpmaps.

### For RKHS regularization problems.

# classic RKHS
struct RKHSProblemType{Kernel_Type,T,X_Type}
    c::Vector{T} # solution of the problem.
    X::Vector{X_Type} # sampling locations.
    θ::Kernel_Type
    σ²::T
end

struct AdaptiveRKHSProblemType{Kernel_Type,T,X_Type}
    c::Vector{T} # solution of the problem.
    X::Vector{X_Type} # sampling locations.
    θ::Kernel_Type
    σ²::Vector{T}
end
