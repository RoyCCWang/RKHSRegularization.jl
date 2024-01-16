

### adaptive kernel.

# # To control the strength of the warp map.
# consider only one warp map.
# let A = norm(x-z)^2, B = (warp(x) - warp(z))^2.
# How should distance([x; warp(x)] - [x; warp(z)]) be?
abstract type WarpMapWeights end

struct EqualWeights <: WarpMapWeights end # dist = A + B. Corresponds to the Euclidean norm over [x; warp(x)] and [z; warp(z)].

function getparametercardinality(::EqualWeights)::Int
    return 0
end


# the weight is for the warp map. Use unity for the weight of the input.
# i.e.: dist = A + κ*B. Interpret κ as a regularization parameter.
struct SingleWeight{T <: AbstractFloat} <: WarpMapWeights
    #κ::T # non-negative.
    κ::Vector{T} # non-negative.
end

function SingleWeight(w::T) where T <: AbstractFloat
    SingleWeight([w;])
end

function getparametercardinality(::SingleWeight)::Int
    return 1
end

# Jensen's inequality: f( (1-t)*A + t*B ) <= (1-t)*f(A) + t*f(B), where f is a convex function.
# restated in probabilitie theory for a random var X: f( E[X] ) <= E[f(X)]. (misc info)
# We have f = norm(.)^2 in this setting.
# Upper and lower bounds dist to be in (min(A,B), max(A,B)).
struct ConvexWeights{T <: AbstractFloat} <: WarpMapWeights # dist = (1-κ)*A + κ*B.
    #κ::T # non-negative.
    #one_minus_κ::T # cache for speed.
    κ::Vector{T} # non-negative.
    one_minus_κ::Vector{T} # cache for speed.
end

function ConvexWeights(κ::T)::ConvexWeights{T} where T
    #@assert zero(T) <= κ <= one(T) # floating equality comparison, avoid. Use the following instead:
    t = one(T)-κ
    @assert !(κ < zero(T)) # assert non-negative-ness.
    @assert !(t < zero(T)) # assert non-negative-ness.

    return ConvexWeights([κ;], [t;])
end

function getparametercardinality(::ConvexWeights)::Int
    return 2
end


# # Warpmaps
abstract type WarpMap end

# ## Traits
abstract type WarpMapOption end
struct UseLoG <: WarpMapOption end

# ## most generic.
struct GenericFunction{FT} <: WarpMap
    warpfunc::FT
end

# ## Laplacian of Gaussian (LoG), normalized by standard deviation squared, which is 1/(2*a).
# for now, only use RKHS functions as the function for LoG. Use SquaredDistanceKernel kernels because they are differentiable wrt the input variable.
#struct LaplacianofGaussian{T, KT <: SquaredDistanceKernel, IT <: RKHSInference, ST <: NoiseModel} <: WarpMap
    #RKHS_problem::Problem{T,KT,IT,ST}
struct LaplacianofGaussian{PT} <: WarpMap
    container::PT # can be a generic function, or a Problem{}.
end

function constructwarpmap(
    ::UseLoG,
    θ::SqExpKernel{T},
    X::Vector{Vector{T}},
    y::Vector{T},
    noise_model::ST,
    )::LaplacianofGaussian{Problem{T, SqExpKernel{T}, RKHS{T}, ST}} where {T <: AbstractFloat, ST <: NoiseModel}

    η = Problem(UseRKHS(), X, θ, noise_model)
    fit!(η, y)

    return LaplacianofGaussian(η)
end

function evalwarpmap(
    x::Vector{T},
    warp::LaplacianofGaussian{Problem{T, SqExpKernel{T}, RKHS{T}, ST}},
    )::T where {T <: AbstractFloat, ST <: NoiseModel}

    @assert !isempty(x)

    # parse.
    η = warp.container
    a = η.θ.a[begin]

    # compute Laplacian.
    f_x = queryRKHS(x, η)
    nLoG = f_x*(2*a*sum( x[d]^2 for d in eachindex(x) ) - length(x))

    return nLoG
end

function evalwarpmap(x::Vector{T}, warp)::T where T <: AbstractFloat
    return warp(x)
end

# # the Container.
struct AdaptiveKernel{KT <: StationaryKernel, WMT <: WarpMap, WT <: WarpMapWeights} <: PositiveDefiniteKernel
    canonical::KT # Canonical kernel parameters.
    warp::WMT
    weights::WT
end

function getparametercardinality(θ::AdaptiveKernel)::Int
    return getparametercardinality(θ.canonical) + getparametercardinality(θ.weights)
end


#####

function getwarpsqdistance(::EqualWeights, r_sq::T, r_sq_warp::T)::T where T
    return r_sq + r_sq_warp
end

function getwarpsqdistance(A::SingleWeight{T}, r_sq::T, r_sq_warp::T)::T where T
    return r_sq + A.κ[begin]*r_sq_warp
end

function getwarpsqdistance(A::ConvexWeights{T}, r_sq::T, r_sq_warp::T)::T where T
    return A.one_minus_κ[begin]*r_sq + A.κ[begin]*r_sq_warp
end

# Adaptive kernel evaluation.
function evalkernel(
    p,
    q::Vector{T},
    θ::AdaptiveKernel,
    )::T where T

    # # the if-else might introduce divergence flow slow down.
    # if pointer(x1) == pointer(x2)
    #     return evalkernel(zero(T), θ)
    # end

    r_sq = devectorizel2normsq(p, q)
    r_sq_warp = (evalwarpmap(p, θ.warp) - evalwarpmap(q, θ.warp))^2
    τ_sq = getwarpsqdistance(θ.weights, r_sq, r_sq_warp)

    return evalkernel(τ_sq, θ.canonical)
end

# # for querying.

function evalkernel!(
    kq::Vector{T},
    p::Vector{T}, # length D
    X::Vector{Vector{T}}, # length N, length D.
    θ::AdaptiveKernel,
    warp_X::Vector{T}, # length N. cached warp map evaluations.
    ) where T
    
    # compute common quantity.
    warp_p = evalwarpmap(p, θ.warp)

    # kernel evaluation against each training position in X.
    resize!(kq, length(X))
    for n in eachindex(kq)

        r_sq = devectorizel2normsq(p, X[n])
        r_sq_warp = (warp_p - warp_X[n])^2

        # apply weights between the warpmap contribution and the input contribution to the resultant squared distance.
        τ_sq = getwarpsqdistance(θ.weights, r_sq, r_sq_warp)

        kq[n] = evalkernel(τ_sq, θ.canonical)
    end

    return nothing
end

function evalkernel(
    p::Vector{T}, # length D
    n::Integer,
    X::Vector{Vector{T}}, # length N, length D.
    θ::AdaptiveKernel,
    warp_X::Vector{T}, # length N. cached warp map evaluations.
    ) where T
    
    # compute common quantity.
    warp_p = evalwarpmap(p, θ.warp)

    r_sq = devectorizel2normsq(p, X[n])
    r_sq_warp = (warp_p - warp_X[n])^2

    # apply weights between the warpmap contribution and the input contribution to the resultant squared distance.
    τ_sq = getwarpsqdistance(θ.weights, r_sq, r_sq_warp)

    return evalkernel(τ_sq, θ.canonical)
end

# front end for adaptive kernel.
function evalkernel!(
    kq::Vector{T},
    p::Vector{T}, # length D
    X::Vector{Vector{T}}, # length N, length D.
    θ::AdaptiveKernel,
    inference::Union{AdaptiveCholeskyGP, AdaptiveRKHS},
    ) where T

    # faster, cached version.
    return evalkernel!(kq, p, X, θ, inference.warp_X)

    # # slower version.
    # for n in eachindex(X)
    #     kq[n] = evalkernel(p, X[n], θ)
    # end

    return nothing
end

# front end for non-adaptive kernel.
function evalkernel!(
    kq::Vector{T},
    p::Vector{T},
    X::Vector{Vector{T}},
    θ::PositiveDefiniteKernel,
    args...
    ) where T

    resize!(kq, length(X))

    for n in eachindex(kq)
        kq[n] = evalkernel(p, X[n], θ)
    end

    return nothing
end


# # Updates and constructions by parsing supplied hyperparams.
# used in hyperparameter optimization or inference.



function updatekernel!(
    θ::AdaptiveKernel,
    p::Vector{T},
    ) where T
    
    offset_ind = getparametercardinality(θ.canonical)

    # the kernel and weight must have non-zero number of parameters, or we'll have a out of bounds error.
    updatekernel!(θ.canonical, p[begin:begin+offset_ind-1])

    if offset_ind < length(p)
        
        # there are update values for the weights.

        updateweight!(θ.weights, p[begin+offset_ind:end])
    end

    return nothing
end


# # weights for warp map.

function updateweight!(w::ConvexWeights{T}, x::Vector{T}) where T

    w.κ[begin] = x[begin]
    w.one_minus_κ[begin] = one(T) - x[begin]
    
    return nothing
end

function parseweight(::ConvexWeights{T}, x::Vector{T})::ConvexWeights{T} where T
    return ConvexWeights(x[begin])
end

#
function updateweight!(w::SingleWeight{T}, x::Vector{T}) where T
    w.κ[begin] = x[begin]
    return nothing
end

function parseweight(::SingleWeight{T}, x::Vector{T})::SingleWeight{T} where T
    return SingleWeight(x[begin])
end

#
function updateweight!(::EqualWeights, args...)
    return nothing
end