
# # Additive noise models.
abstract type NoiseModel end

struct Variance{T <: AbstractFloat} <: NoiseModel
    #v::T
    v::Vector{T} # single-element.
end

function Variance(x::T)::Variance{T} where T <: AbstractFloat
    return Variance([x;])
end

struct DiagonalCovariance{T} <: NoiseModel
    v::Vector{T}
end

struct Covariance{T} <: NoiseModel
    v::Matrix{T}
end

function updatenoisemodel!(s::Variance{T}, v::T) where T <: AbstractFloat
    s.v[begin] = v
    return nothing
end

function updatenoisemodel!(s::DiagonalCovariance{T}, v::Vector{T}) where T <: AbstractFloat
    s.v[:] = v
    return nothing
end

function updatenoisemodel!(s::Covariance{T}, v::Array{T,D}) where {T <: AbstractFloat, D}
    s.v[:] = v # this is equivalent to s.v[:] = vec(v). Julia's multi-dim arrays use col-major ordering.
    return nothing
end

# # noise model traits.
abstract type NoiseModelTrait end


# # Inference option: {GP, RKHS} tensor product {adaptive kernel, non-adaptive kernel}
abstract type InferenceOption end
abstract type RKHSInference <: InferenceOption end
abstract type GPInference <: InferenceOption end

struct CholeskyGP{T} <: GPInference
    kq::Vector{T} # length(X)
    v::Vector{T} # length(X)
    L::LowerTriangular{T, Matrix{T}} # lower triangular cholesky factor of the training kernel matrix.
end

function CholeskyGP(::Type{T}, N::Integer)::CholeskyGP{T} where T
    return CholeskyGP(Vector{T}(undef, N), Vector{T}(undef, N), LowerTriangular(zeros(T,N,N)))
end

function allocatecontainer!(A::CholeskyGP{T}, N::Int)::CholeskyGP{T} where T # resizes A's buffer.
    resize!(A.kq, N)
    return CholeskyGP(A.kq, LowerTriangular(zeros(T,N,N)))
end


struct RKHS{T} <: RKHSInference
    kq::Vector{T} # length(X)
end

function RKHS(::Type{T}, N::Integer)::RKHS{T} where T
    return RKHS(Vector{T}(undef, N))
end

function allocatecontainer!(A::RKHS{T}, N::Int)::RKHS{T} where T # resizes A's buffer.
    resize!(A.kq, N)
    return A
end

# has pre-computed warp map evaluations at training positions.
struct AdaptiveCholeskyGP{T} <: GPInference
    kq::Vector{T} # length(X)
    v::Vector{T} # length(X)
    L::LowerTriangular{T, Matrix{T}} # lower triangular cholesky factor of the training kernel matrix.
    warp_X::Vector{T} # length(X)
end

function AdaptiveCholeskyGP(::Type{T}, N::Integer)::AdaptiveCholeskyGP{T} where T
    return AdaptiveCholeskyGP(Vector{T}(undef, N), Vector{T}(undef, N), LowerTriangular(zeros(T,N,N)), Vector{T}(undef, N))
end

function allocatecontainer!(A::AdaptiveCholeskyGP{T}, N::Int)::AdaptiveCholeskyGP{T} where T # resizes A's buffer.
    resize!(A.kq, N)
    resize!(A.warp_X, N)
    return AdaptiveCholeskyGP(
        A.kq,
        Vector{T}(undef, N),
        LowerTriangular(zeros(T,N,N)),
        A.warp_X,
    )
end

struct AdaptiveRKHS{T} <: RKHSInference
    kq::Vector{T} # length(X)
    warp_X::Vector{T} # length(X)
end

function AdaptiveRKHS(::Type{T}, N::Integer)::AdaptiveRKHS{T} where T
    return AdaptiveRKHS(Vector{T}(undef, N), Vector{T}(undef, N))
end

function allocatecontainer!(A::AdaptiveRKHS{T}, N::Int)::AdaptiveRKHS{T} where T # resizes A's buffer.
    resize!(A.kq, N)
    resize!(A.warp_X, N)
    return A
end

# ## evaluate kernels.

# ## types
abstract type PositiveDefiniteKernel end # see source files in \kernels\

# ## traits
abstract type InferenceTrait end

# for constructing InferenceOption
struct UseCholeskyGP <: InferenceTrait end
struct UseRKHS <: InferenceTrait end
struct UseAdaptiveCholeskyGP <: InferenceTrait end
struct UseAdaptiveRKHS <: InferenceTrait end

# # For RKHS regularization problems.
# contain quantities that are required for inference.

# classic RKHS
struct Problem{T <: AbstractFloat, KT <: PositiveDefiniteKernel, IT <: InferenceOption, ST <: NoiseModel}
    c::Vector{T} # solution of the problem.
    X::Vector{Vector{T}} # sampling locations.
    θ::KT
    noise_model::ST # σ²::ST

    inference::IT # inference-related intermediate/cache quantities.

    # this will contain kernel matrix plus noise model.
    U::Matrix{T} # to avoid allocation repeately when hyperparameter fitting. Also used for sequential/batch problem update.
end

function Problem(
    ::UseAdaptiveRKHS,
    X::Vector{Vector{T}},
    θ::KT,
    noise_model::ST,
    )::Problem{T, KT, AdaptiveRKHS{T}, ST} where {T, KT, ST <: NoiseModel}

    N = length(X)
    return Problem(
        Vector{T}(undef,N),
        X,
        θ,
        noise_model,
        AdaptiveRKHS(T, N),
        Matrix{T}(undef, N, N),
    )
end

function Problem(
    ::UseRKHS,
    X::Vector{Vector{T}},
    θ::KT,
    noise_model::ST,
    )::Problem{T, KT, RKHS{T}, ST} where {T, KT <: PositiveDefiniteKernel, ST <: NoiseModel}

    N = length(X)
    return Problem(
        Vector{T}(undef,N),
        X,
        θ,
        noise_model,
        RKHS(T, N),
        Matrix{T}(undef, N, N),
    )
end

function Problem(
    ::UseAdaptiveCholeskyGP,
    X::Vector{Vector{T}},
    θ::KT,
    noise_model::ST,
    )::Problem{T, KT, AdaptiveCholeskyGP{T}, ST} where {T, KT <: PositiveDefiniteKernel, ST <: NoiseModel}

    N = length(X)
    return Problem(
        Vector{T}(undef,N),
        X,
        θ,
        noise_model,
        AdaptiveCholeskyGP(T, N),
        Matrix{T}(undef, N, N),
    )
end

function Problem(
    ::UseCholeskyGP,
    X::Vector{Vector{T}},
    θ::KT,
    noise_model::ST,
    )::Problem{T, KT, CholeskyGP{T}, ST} where {T, KT <: PositiveDefiniteKernel, ST <: NoiseModel}

    N = length(X)
    return Problem(
        Vector{T}(undef,N),
        X,
        θ,
        noise_model,
        CholeskyGP(T, N),
        Matrix{T}(undef, N, N),
    )
end

function getnumsamples(η::Problem)::Int
    return length(η.X)
end

# compact RBF types

struct EdgeWeightContainer{KT}

    θs::Matrix{KT} # θs[src, dest] is the kernel for weight between src and dest nodes.
    indicators::BitMatrix # true if the edge is active. Avoids floating-point comparison between kernel evaluations.
end

function EdgeWeightContainer(::KT, N::Int)::EdgeWeightContainer{KT} where KT
    M = Matrix{KT}(undef, N, N)
    #M = fill!(KT(one(T)))
    return EdgeWeightContainer(M, falses(N,N))
end
