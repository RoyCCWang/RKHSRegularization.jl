# Kernel implementations.

function evalkernel(p, q::Vector{T},
                    θ::AdditiveVarianceKernelType{KT,T2})::T where {T,KT,T2}
    #
    out = θ.base_gain * evalkernel(p, q, θ.base_kernel_parameters)

    if norm(p-q) < θ.zero_tol
        return out + θ.additive_function(p)
    end

    return out
end

# SOS kernel evaluation.
function evalkernel(p, q::Vector{T}, θ::SOSKernelType{KT})::T where {T,KT}

    out = evalkernel(p, q, θ.base_params)
    return out^2
end

# KR kernel
function evalkernel(p, q::Vector{T}, θ::ElementaryKRKernelType{KT,T})::T where {T,KT}

    out_v = evalkernel(p[1:end-1], q[1:end-1], θ.θ_a)
    out_x = evalkernel(p[end], q[end], θ.θ_canonical)

    return out_v*out_x
end

# Adaptive kernel evaluation.
function evalkernel(p, q::Vector{T},
            θ::AdaptiveKernelType{KT} )::T where {T,KT}

    # if norm(p-q) < 1e-15 #zero_tol
    #     return one(T)
    # end

    warpfunc = θ.warpfunc
    p3 = warpfunc(p)
    q3 = warpfunc(q)
    r_new = p3-q3

    r = p-q
    #τ² = LinearAlgebra.dot(r,r) + LinearAlgebra.dot(r_new,r_new)#(p3-q3)*(p3-q3)
    #τ = sqrt(τ²)
    τ = sqrt(sum( r[d]^2 for d = 1:length(r) ) + r_new^2)

    #println("a: τ = ", τ)
    return evalkernel(τ, θ.canonical_params)
end

function evalkernel(p, q::Vector{T},
            θ::FastAdaptiveKernelType{KT,T,D} )::T where {T,KT,D}

    sum_r_warp_sq = sum( (θ.s[i]*(θ.warpfuncs[i](p) - θ.warpfuncs[i](q)))^2
                            for i = 1:length(θ.warpfuncs))

    # sb = one(T) - sum(θ.s)
    # @assert one(T) >= sb > zero(T)
    sb = one(T)

    sum_r_sq = sum( (sb*(p[d]-q[d]))^2 for d = 1:length(q) )

    τ = sqrt( sum_r_sq + sum_r_warp_sq )

    return evalkernel(τ, θ.canonical_kernel)
end

function evalkernel(p, q::Vector{T},
            θ::AdaptiveKernelDPPType{KT},
            zero_tol::T = eps(T)*2 )::T where {T,KT}

    if norm(p-q) < zero_tol
        return one(T) + (θ.warpfunc(p))^2
    end

    warpfunc = θ.warpfunc
    p3 = warpfunc(p)
    q3 = warpfunc(q)
    r_new = p3-q3

    r = p-q
    τ² = LinearAlgebra.dot(r,r) + LinearAlgebra.dot(r_new,r_new)#(p3-q3)*(p3-q3)
    τ = sqrt(τ²)

    return evalkernel(τ, θ.canonical_params)
end

# Adaptive kernel evaluation.
function evalkernel(p, q::Vector{T},
            θ::AdaptiveKernelMultiWarpType{KT},
            zero_tol::T = eps(T)*2 )::T where {T,KT}

    if norm(p-q) < zero_tol
        return one(T)
    end

    return evalkernel(θ.warpfuncs, θ.a, p, q, θ.canonical_params)
end

function evalkernel(p, q::Vector{T},
            θ::AdaptiveKernelMultiWarpDPPType{KT},
            zero_tol::T = eps(T)*2 )::T where {T,KT}

    if norm(p-q) < zero_tol
        #return one(T) + sum( θ.a[m]*(θ.warpfuncs[m](p))^2 for m = 1:length(θ.warpfuncs) )
        return one(T) + θ.self_gain*sum( θ.a[m]*abs(θ.warpfuncs[m](p)) for m = 1:length(θ.warpfuncs) )
    end

    return evalkernel(θ.warpfuncs, θ.a, p, q,
                            θ.canonical_params)
end

function evalkernel(warpfuncs::Vector{Function},
                    weights::Vector{T},
                    p, q::Vector{T},
                    θ_canonical::KT)::T where {T,KT}

    warp_terms = zero(T)
    for m = 1:length(warpfuncs)
        p3 = warpfuncs[m](p)
        q3 = warpfuncs[m](q)

        r_new = p3-q3
        warp_terms += weights[m]*LinearAlgebra.dot(r_new,r_new)
    end

    r = p-q
    τ² = LinearAlgebra.dot(r,r) + warp_terms
    τ = sqrt(τ²)

    return evalkernel(τ, θ_canonical)
end


# Adaptive kernel evaluation.
function evalkernel(p, q::Vector{T},
            θ::AdaptiveKernelΨType{KT})::T where {T,KT}

    if p == q
        return one(T)
    end

    r_new = θ.ψ(p,q)

    r = p-q
    τ² = LinearAlgebra.dot(r,r) + LinearAlgebra.dot(r_new,r_new)#(p3-q3)*(p3-q3)
    τ = sqrt(τ²)
#println("a: τ = ", τ)
    return evalkernel(τ, θ.canonical_params)
end

####### Brownian Bridge kernels. Use tensor product for multivariate input. Defined only on [0,1].

function evalkernel(p, q::T, θ::BrownianBridge10)::T where T<: Real
    return min(p,q)-p*q
end

# function evalkernel(x, z::T, θ::BrownianBridge20)::T where T<: Real
#     if x < z
#         return x*(1-z)*(x^2 +z^2 -2*z)/6
#     end
#
#     return (1-x)*z*(x^2 +z^2 -2*x)/6
# end

function evalkernel(x, z::T, θ::BrownianBridge1ϵ)::T where T<: Real
    ϵ = θ.ϵ
    denominator = ϵ*sinh(ϵ)
    numerator = sinh(ϵ*min(x,z))*sinh(ϵ*(1-max(x,z)))

    return numerator/denominator
end

function evalkernel(x, z::T, θ::BrownianBridge2ϵ)::T where T<: Real
    ϵ = θ.ϵ

    numerator = exp(-ϵ*(x+z))
    denominator = 4*ϵ^3*(exp(2*ϵ)-1)^2
    multiplier = numerator/denominator

    term1 = exp(2*ϵ)*(2*ϵ-ϵ*(x+z)-1)
    term2 = exp(4*ϵ)*(ϵ*(x+z)+1)
    term3 = exp(2*ϵ*(1+x+z))*(2*ϵ-ϵ*(x+z)+1)
    term4 = exp(2*ϵ*(x+z))*(ϵ*(x+z)-1)
    term5 = exp(2*ϵ*(2+min(x,z)))*(-ϵ*abs(x-z)-1)
    term6 = exp(2*ϵ*max(x,z))*(-ϵ*abs(x-z)+1)
    term7 = exp(2*ϵ*(1+min(x,z)))*(1-2*ϵ+ϵ*abs(x-z))
    term8 = exp(2*ϵ*(1+max(x,z)))*(1+2*ϵ-ϵ*abs(x-z))

    return multiplier*(term1+term2+term3+term4+term5+term6+term7+term8)
end

# tensor product kernels.
function evalkernel(p, q::Vector{T}, θ::BrownianBridgeKernelType)::T where T<: Real
    return prod( evalkernel(p[d],q[d],θ) for d = 1:length(q) )
end

function evalkernel(p, q::Vector{T}, θ::BrownianBridgeSemiInfDomain)::T where T<: Real
    return prod( evalkernel(p[d],q[d],θ) for d = 1:length(q) )
end

function evalkernel(p, q::Vector{T}, θ::BrownianBridgeCompactDomain)::T where T<: Real
    return prod( evalkernel(p[d],q[d],θ) for d = 1:length(q) )
end

# equation 4.2 from McCourt's "An Introduction to the Hilbert-Schmidt SVD using iterated Brownian bridge kernels".
# function evalkernel(x, z::T, θ::BrownianBridge20)::T where T<: Real
#     β::Int = 2
#
#     factor1 = (-1)^(β-1)*2^(2*β-1)/factorial(2*β)
#     factor2 = Bernoullipolynomial4(abs(x-z)/2) - Bernoullipolynomial4((x+z)/2)
#     return factor1*factor2
# end

# an un-numbered equation shortly after McCourt's equation 4.3.
function evalkernel(x, z::T, θ::BrownianBridge20)::T where T<: Real

    if z < x
        return -1/6*z*(1-x)*(x^2+z^2-2*x)
    end

    return -1/6*x*(1-z)*(x^2+z^2-2*z)
end


function evalkernel(x_in, z_in::T, θ::BrownianBridge20CompactDomain, d::Int = 1)::T where T<: Real

    # Map input, with ϵ as buffer from the end points.
    ϵ = 1e-6
    x_tmp = interval2itpindex(x_in, θ.a[d], θ.b[d], 10)
    z_tmp = interval2itpindex(z_in, θ.a[d], θ.b[d], 10)

    # assign a value of zero for inputs that are not within [1,10].
    if !(1 < x_tmp < 10)
        return zero(T)
    end

    if !(1 < z_tmp < 10)
        return zero(T)
    end

    # Map inputs to (0,1)
    x = (x_tmp -1)/(9 + ϵ)
    z = (z_tmp -1)/(9 + ϵ)

    # evaluate the Brownian Bridge 20 kernel.
    if z < x
        return -1/6*z*(1-x)*(x^2+z^2-2*x)
    end

    return -1/6*x*(1-z)*(x^2+z^2-2*z)
end

function evalkernel(x_in, z_in::T, θ::BrownianBridgeSemiInfDomain{KT})::T where {T<: Real, KT}

    # map [0,Inf) to [0,0.5).
    x = x_in/(2*(1+x_in))
    z = z_in/(2*(1+z_in))

    return evalkernel(x, z, θ.θ_base)
end

function evalkernel(x_in, z_in::T, θ::BrownianBridgeCompactDomain{KT,T})::T where {T<: Real, KT}

    # map [0,Inf) to [0,0.5).
    x = Utilities.convertcompactdomain(x_in, θ.a, θ.b, zero(T), one(T))
    z = Utilities.convertcompactdomain(z_in, θ.a, θ.b, zero(T), one(T))

    return evalkernel(x, z, θ.θ_base)
end

######## Conventional kernel function implementations
# All kernels are normalized to have maximum value of 1
# Support is r <= 1.
function evalkernel(x1, x2::Vector{T}, θ::StationaryKernelType)::T where T

    # norm(x1-x2)
    #r = x1-x2
    #τ² = LinearAlgebra.dot(r,r)
    #τ = sqrt(τ²)
    τ = norm(x1-x2)

#println("s: τ = ", τ)
    return evalkernel(τ,θ)
end

# 1D version. For speed.
function evalkernel(x1, x2::T, θ::StationaryKernelType)::T where T
    τ = abs(x1-x2)
    #τ = x1-x2

    return evalkernel(τ,θ)
end

# This is the canonical kernel: spline 34.
# outputs 1 at τ = 0, outputs 0 at τ >= 1.
function evalkernel(τ::T, θ::Spline34KernelType) where T

    r::T = τ*θ.a[1]

    tmp = one(T)-r
    if sign(tmp) < zero(T)
        return zero(T)
    end

    tmp = tmp^6
    out::T = ( (35.0*r^2 + 18.0*r + 3.0)*tmp )/3.0

    #@assert isfinite(out)
    return out
end

# This is the canonical kernel: spline 12.
function evalkernel(τ::T, θ::Spline12KernelType) where T

    r::T = τ*θ.a[1]

    tmp = one(T)-r
    if sign(tmp) < zero(T)
        return zero(T)
    end

    tmp = tmp^3
    out::T = (3.0*r + 1.0)*tmp

    #@assert isfinite(out)
    return out
end

# This is the canonical kernel: spline 32.
function evalkernel(τ::T, θ::Spline32KernelType) where T

    r::T = τ*θ.a[1]

    tmp = one(T)-r
    if sign(tmp) < zero(T)
        return zero(T)
    end

    tmp = tmp^4
    out::T = (4.0*r + 1.0)*tmp

    #@assert isfinite(out)
    return out
end

# This is the canonical kernel: 1D Gaussian.
function evalkernel(τ::T, θ::GaussianKernel1DType)::T where T <: Real

    # a is ϵ^2 in "Stable eval of Gaussian radial basis func interpolants", 2012.
    out::T = exp(-θ.ϵ_sq[1]*τ^2)

    #@assert isfinite(out)
    return out
end

# algebraic sigmoid's univariate rational quadratic kernel.
function evalkernel(τ::T, θ::RationalQuadraticKernelType)::T where T <: Real
    denominator = sqrt(θ.a[1]+τ^2)^3

    #return θ.a/denominator
    #return sqrt(θ.a)/denominator
    return sqrt(θ.a[1])^3/denominator
end

function evalkernel(τ::T, θ::TunableRationalQuadraticKernelType)::T where T <: Real
    denominator = sqrt(θ.a[1]+τ^2)^3

    #return θ.a/denominator
    #return sqrt(θ.a)/denominator
    return θ.w[1]*sqrt(θ.a[1])^3/denominator
end

function evalkernel(τ::T, θ::ModulatedSqExpKernelType)::T where T <: Real

    out::T = exp(-θ.ϵ_sq[1]*τ^2)*cos(θ.ν*τ)

    return out
end

function evalkernel(x, z::Vector{T}, θ::ModulatedSqExpKernelType)::T where T <: Real

    r = x-z
    τ = norm(r)

    out::T = exp(-θ.ϵ_sq[1]*τ^2)*cos(dot(r,θ.ν[1]))

    return out
end

function evalkernel(x, z::Vector{T}, θ::AdaptiveModulatedSqExpKernelType)::T where T <: Real
    ϕ = θ.ϕ
    ψ = θ.ψ

    r1 = [x; ϕ(x)] - [z; ϕ(z)]
    τ1 = norm(r1)

    r2 = [x; ψ(x)] - [z; ψ(z)]
    τ2 = norm(r2)

    #out::T = exp(-θ.ϵ_sq*τ1^2)*cos(dot(r2,θ.ν))
    out::T = exp(-θ.ϵ_sq[1]*τ1^2)*cos(r2[end]*θ.ν[end])

    return out
end

# Adaptive kernel for conditional univariate pdf evaluation:
# p(x[end] | x[1:end-1]).
function evalkernel(p, q::Vector{T},
                    θ::CPDF1DKernelType{T,KT})::T where {T,KT}

    d = length(p)

    # k(p,q)
    factor1 = evalkernel(p[1:d-1], q[1:d-1], θ.B)

    # h(p,q)
    factor2 = evalkernel(p[d], q[d], θ.A)

    return factor1*factor2
end

function evalkernel(x, z::Vector{T}, θ::KRWarpKernelType{KTv,KTx})::T where {T,KTv,KTx}

    out::T = evalkernel(x[1:end-1], z[1:end-1], θ.θ_v)
    return out*evalkernel(x[end], z[end], θ.θ_x)
end

# M is such that 1 <= M <= length(x)-1 == length(z)-1
function evalkernel(x, z::Vector{T}, θ::DoubleProductKernelType{KT})::T where {T,KT}
    M = θ.M
    return evalkernel(x[1:M], z[1:M], θ.θ_base)*evalkernel(x[M+1:end], z[M+1:end], θ.θ_base)
end
