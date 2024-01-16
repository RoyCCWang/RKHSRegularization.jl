

### elemntary Kernels.
# # All kernels are normalized to have maximum value of 1

abstract type StationaryKernel <: PositiveDefiniteKernel end

abstract type SquaredDistanceKernel <: StationaryKernel end
abstract type DistanceKernel <: StationaryKernel end

# no error checking on length of inputs.
function devectorizel2normsq(x, y::Vector{T})::T where T <: AbstractFloat
    
    out = zero(T)
    for i in eachindex(x)
        out += (x[i] - y[i])^2
    end
    return out
end

function evalkernel(x1, x2::Vector{T}, θ::SquaredDistanceKernel)::T where T <: Number

    # # the if-else might introduce divergence flow slow down.
    # if pointer(x1) == pointer(x2)
    #     return evalkernel(zero(T), θ)
    # end

    τ_sq = devectorizel2normsq(x1, x2) # same as norm(x1-x2)^2
    return evalkernel(τ_sq, θ)
end

# 1D version. For speed.
function evalkernel(x1, x2::T, θ::SquaredDistanceKernel)::T where T <: Number

    return evalkernel((x1-x2)^2,θ)
end


function devectorizel2norm(x, y::Vector{T})::T where T <: AbstractFloat
    
    out = zero(T)
    for i in eachindex(x)
        out += (x[i] - y[i])^2
    end
    return sqrt(out)
end

function evalkernel(x1, x2::Vector{T}, θ::DistanceKernel)::T where T <: Number

    τ = devectorizel2norm(x1, x2) # same as norm(x1-x2)
    return evalkernel(τ, θ)
end

# 1D version. For speed.
function evalkernel(x1, x2::T, θ::DistanceKernel)::T where T <: Number

    return evalkernel(abs(x1-x2),θ)
end


############# the different types of common kernels.


# # Compact spline kernels. See  Cor. 9.14, Wendland 2005.
# generalizes Spline12, 23, 34 kernels to arbitrary (D,q), where q is mean-square differentiability of the kernel.

abstract type DifferentiabilityTrait end
struct Order3 <: DifferentiabilityTrait end
struct Order2 <: DifferentiabilityTrait end
struct Order1 <: DifferentiabilityTrait end

#struct WendlandSplineKernel{T, DT <: DifferentiabilityTrait} <: DistanceKernel
struct WendlandSplineKernel{T <: AbstractFloat} <: DistanceKernel
    #a::T
    a::Vector{T}

    # buffer.
    l::Int
    c::Vector{Int}

    # dispatch
    #differentiability::DT
    m::Int # l + order, l = getsplineparameter(D, order)
    Z::Int # normalizing constant for the spline kernel to have a maximum of 1. It is the offset constant from Cor. 19.4 in the second factor.
end

# comment out if a::T
function WendlandSplineKernel(a::T, l::Int, c::Vector{Int}, m::Int, Z::Int)::WendlandSplineKernel{T} where T <: AbstractFloat
    return WendlandSplineKernel([a;], l, c, m, Z)
end


# Cor. 9.14, Wendland 2005.
function getsplineparameter(D::Int, q::Int)::Int
    return div(D,2)+q+1 # note that div(D,2) == floot(Int, D/2)
end

# Cor. 9.14, Wendland.
function WendlandSplineKernel(::Order3, a::T, D::Int)::WendlandSplineKernel{T} where T
    
    l = getsplineparameter(D, 3)
    c = Vector{Int}(undef, 3)
    c[1] = 15*l + 45
    c[2] = 6*l^2 + 36*l + 45
    c[3] = l^3 + 9*l^2 + 23*l + 15
    
    return WendlandSplineKernel(a, l, c, l+3, 15)
end

function WendlandSplineKernel(::Order2, a::T, D::Int)::WendlandSplineKernel{T} where T
    
    l = getsplineparameter(D, 2)
    c = Vector{Int}(undef, 2)
    c[1] = 3*l + 6
    c[2] = l^2 + 4*l + 3

    return WendlandSplineKernel(a, l, c, l+2, 3)
end

function WendlandSplineKernel(::Order1, a::T, D::Int)::WendlandSplineKernel{T} where T
    
    l = getsplineparameter(D, 1)
    c = Vector{Int}(undef, 1)
    c[1] = l + 1

    return WendlandSplineKernel(a, l, c, l+1, 1)
end

function evalkernel(τ::T, θ::WendlandSplineKernel{T}) where T

    c, m, Z = θ.c, θ.m, θ.Z
    
    r::T = τ*θ.a[begin]
    tmp = one(T)-r
    if sign(tmp) < zero(T)
        return zero(T)
    end
    
    factor1 = tmp^m
    factor2 = (sum( c[i]*r^i for i in eachindex(c) ) )/Z + 1 # use Horner's method if speed it an issue.
    return factor1*factor2
end


# # exponential kernel, aka Ornstein-Uhlenbeck covariance function.
struct ExponentialKernel{T <: AbstractFloat} <: DistanceKernel
    # k(t,z) = q/(2*λ) * exp(-λ*abs(t-z)).
    λ::Vector{T} # single-entry, above 0.
    q::Vector{T} # single_entry, above 0.
end

function ExponentialKernel(a::T, b::T)::ExponentialKernel{T} where {T <: AbstractFloat}
    return ExponentialKernel([a;], [b;])
end

function evalkernel(τ::T, θ::ExponentialKernel)::T where T <: Real
    
    q, λ = θ.q[1], θ.λ[1]
    
    return (q/(2*λ)) * exp(-λ*τ)
end

# Matern kernel.
# k(x,z) = b*(1 + a*norm(x-z))*exp(-a*norm(x-z))
struct Matern3Halfs{T <: AbstractFloat} <: DistanceKernel
    
    a::Vector{T} # single-entry, above 0.
    b::Vector{T} # single_entry, above 0.
end

function Matern3Halfs(a::T, b::T)::Matern3Halfs{T} where {T <: AbstractFloat}
    return Matern3Halfs([a;], [b;])
end

function evalkernel(τ::T, θ::Matern3Halfs)::T where T <: Real
    a, b = θ.a[1], θ.b[1]
    tmp = a*τ
    return b*(one(T)+tmp) * exp(-tmp)
end


# # Stationary kernels with squared distance as input.

# ## Eq. 4.19, pg 86, GPML 2006.
struct RationalQuadraticKernel{T <: AbstractFloat, ET <: Real} <: SquaredDistanceKernel
    #a::T # 1/(2*n*l^2) from GPML 2006.
    a::Vector{T}
    n::ET # exponent.
end

function RationalQuadraticKernel(a::T, n::ET)::RationalQuadraticKernel{T,ET} where {T <: AbstractFloat, ET <: Real}
    return RationalQuadraticKernel([a;], n)
end


function evalkernel(τ_sq::T, θ::RationalQuadraticKernel)::T where T <: Real
    return (one(T) + θ.a[begin] * τ_sq )^(θ.n)
end


# # algebraic sigmoid's univariate rational quadratic kernel.
# function evalkernel(τ::T, θ::RationalQuadraticKernel)::T where T <: Real
#     denominator = sqrt(θ.a[begin]+τ^2)^3
#     return sqrt(θ.a[begin])^3/denominator
# end

# ## Eq. 4.9, pg 83, GPML 2006.
struct SqExpKernel{T <: AbstractFloat} <: SquaredDistanceKernel
    #a::T # 1/(2*l^2) from GPML 2006.
    a::Vector{T}
end

function SqExpKernel(a::T) where T <: AbstractFloat
    return SqExpKernel([a;])
end

# This is the canonical kernel: 1D Gaussian.
function evalkernel(τ_sq::T, θ::SqExpKernel{T})::T where T <: Real

    return exp(-θ.a[begin]*τ_sq)
end


# # Updates and constructions by parsing supplied hyperparams.
# used in hyperparameter optimization or inference.

# for select kernels with scalar-valued hyperparameters.
function updatekernel!(
    θ::Union{SqExpKernel, WendlandSplineKernel, RationalQuadraticKernel},
    p::Union{Vector{T}, T}
    ) where T <: Real

    θ.a[begin] = p[begin]
    return nothing
end


function getparametercardinality(::SqExpKernel)::Int
    return 1
end

function getparametercardinality(::WendlandSplineKernel)::Int
    return 1
end

# do not allow optimization of the degree for now.
function getparametercardinality(::RationalQuadraticKernel)::Int
    return 1
end

