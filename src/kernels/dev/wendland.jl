## need to debug this script.

# # Compact spline kernels. See  Cor. 9.14, Wendland 2005.
# generalizes Spline12, 23, 34 kernels to arbitrary (D,q), where q is mean-square differentiability of the kernel.

abstract type DifferentiabilityTrait end
struct Order3 <: DifferentiabilityTrait end
struct Order2 <: DifferentiabilityTrait end
struct Order1 <: DifferentiabilityTrait end

#struct WendlandSpline{T, DT <: DifferentiabilityTrait} <: DistanceKernel
struct WendlandSpline{T <: AbstractFloat} <: DistanceKernel
    #a::T
    a::Memory{T} # single-entry.

    # buffer.
    l::Int
    c::Memory{T}

    # dispatch
    #differentiability::DT
    m::Int # l + order, l = getsplineparameter(D, order)
    Z::Int # normalizing constant for the spline kernel to have a maximum of 1. It is the offset constant from Cor. 19.4 in the second factor.
end

# comment out if a::T
function WendlandSpline(a::T, l::Int, c::Memory{T}, m::Int, Z::Int) where T <: AbstractFloat
    p = Memory{T}(undef, 1)
    p[begin] = a
    return WendlandSpline(p, l, c, m, Z)
end

# Cor. 9.14, Wendland 2005.
function getsplineparameter(D::Int, q::Int)
    return div(D,2)+q+1 # note that div(D,2) == floot(Int, D/2)
end

# Cor. 9.14, Wendland.
function WendlandSpline(::Order3, a::T, D::Int) where T <: AbstractFloat
    
    l = getsplineparameter(D, 3)
    c = Memory{T}(undef, 3)
    c[1] = 15*l + 45
    c[2] = 6*l^2 + 36*l + 45
    c[3] = l^3 + 9*l^2 + 23*l + 15
    
    return WendlandSpline(a, l, c, l+3, 15)
end

function WendlandSpline(::Order2, a::T, D::Int) where T <: AbstractFloat
    
    l = getsplineparameter(D, 2)
    c = Memory{T}(undef, 2)
    c[1] = 3*l + 6
    c[2] = l^2 + 4*l + 3

    return WendlandSpline(a, l, c, l+2, 3)
end

function WendlandSpline(::Order1, a::T, D::Int) where T <: AbstractFloat
    
    l = getsplineparameter(D, 1)
    c = Memory{T}(undef, 1)
    c[1] = l + 1

    return WendlandSpline(a, l, c, l+1, 1)
end

function evalkernel(τ::T, θ::WendlandSpline{T}) where T

    c, m, Z = θ.c, θ.m, θ.Z
    
    r = τ*θ.a[begin]
    tmp = one(T)-r
    @show tmp # TODO try a = 132, r = 0.1
    if sign(tmp) < zero(T)
        return zero(T)
    end
    
    factor1 = tmp^m
    factor2 = (sum( c[i]*r^i for i in eachindex(c) ) )/Z + one(T) # use Horner's method if speed it an issue.
    # factor2 = (evalpoly(r, [0;c]) )/Z + 1  # Horner's.
    return factor1*factor2
end