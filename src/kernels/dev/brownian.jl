

abstract type BrownianBridgeKernel <: Kernel end

struct BrownianBridge10{T} <: BrownianBridgeKernel
    a::T
end

struct BrownianBridge20{T} <: BrownianBridgeKernel
    a::T
end

struct BrownianBridge20CompactDomain{T} <: BrownianBridgeKernel
    a::Vector{T} # lower bound of domain.
    b::Vector{T} # upper bound of domain.
end

# struct BrownianBridgeβ0 <: BrownianBridgeKernel
#     β::Int
# end

struct BrownianBridge1ϵ{T} <: BrownianBridgeKernel
    ϵ::T
end

struct BrownianBridge2ϵ{T} <: BrownianBridgeKernel
    ϵ::T
end


struct BrownianBridgeSemiInfDomain{BT <: BrownianBridgeKernel}
    θ_base::BT
end

struct BrownianBridgeCompactDomain{BT <: BrownianBridgeKernel, T}
    θ_base::BT
    a::T # common lower bound for all input dimensions.
    b::T # common upper bound for all input dimensions.
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
    ϵ = θ.ϵ[begin]
    denominator = ϵ*sinh(ϵ)
    numerator = sinh(ϵ*min(x,z))*sinh(ϵ*(1-max(x,z)))

    return numerator/denominator
end

function evalkernel(x, z::T, θ::BrownianBridge2ϵ)::T where T<: Real
    ϵ = θ.ϵ[begin]

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
function evalkernel(p, q::Vector{T}, θ::BrownianBridgeKernel)::T where T<: Real
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


function evalkernel(x_in, z_in::T, θ::BrownianBridge20CompactDomain; d::Int = 1)::T where T<: Real

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
    x = convertcompactdomain(x_in, θ.a, θ.b, zero(T), one(T))
    z = convertcompactdomain(z_in, θ.a, θ.b, zero(T), one(T))

    return evalkernel(x, z, θ.θ_base)
end