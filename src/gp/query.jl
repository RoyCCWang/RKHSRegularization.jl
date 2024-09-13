# inference.

struct RKHSState{T}
    kq::Memory{T}

    function RKHSState(::Type{T}, N::Integer) where T <: AbstractFloat
        return new{T}(Memory{T}(undef, N))
    end
end

struct RKHSModel{T, KT}
    c::Memory{T}
    X::Matrix{T}
    θ::KT
end

function RKHSModel(s::LikelihoodState)
    return RKHSModel(s.c, s.inputs, s.kernel)
end

function query!(s::RKHSState, xq::AbstractVector, ps::RKHSModel)
    evalkernel!(s.kq, xq, ps.X, ps.θ)
    return dot(s.kq, ps.c)
end

# front end for non-adaptive kernel.
function evalkernel!(kq::AbstractVector, xq::AbstractVector, X::Matrix, θ::Kernel, args...)

    for (n, x_n) in Iterators.zip(eachindex(kq), eachcol(X))
        kq[n] = evalkernel(xq, x_n, θ)
    end
    return nothing
end

function query!(f_xqs::AbstractArray, s::RKHSState, xqs::AbstractMatrix, ps::RKHSModel)

    for (m, xq) in Iterators.zip(eachindex(f_xqs), eachcol(xqs))
        f_xqs[m] = query!(s, xq, ps)
    end
    return nothing
end

struct GPState{T}
    kq::Memory{T}
    v::Memory{T}

    function GPState(::Type{T}, N::Integer) where T <: AbstractFloat
        return new{T}(Memory{T}(undef, N), Memory{T}(undef, N))
    end
end

struct GPModel{T, KT, LT}
    c::Memory{T}
    X::Matrix{T}
    θ::KT
    L::LT
end

function GPModel(s::LikelihoodState)
    L = cholesky(s.U) # allocates.
    return GPModel(s.c, s.inputs, s.kernel, L)
end

function query!(s::GPState, xq::AbstractVector, ps::GPModel)
    kq, v = s.kq, s.v
    c, X, θ, L = ps.c, ps.X, ps.θ, ps.L

    evalkernel!(kq, xq, X, θ)
    m_xq = dot(kq, c)

    ldiv!(v, L, kq)
    v_xq = evalkernel(xq, xq, θ) - dot(v,v) # no guard against negative variance.

    return m_xq, v_xq
end

function query!(m_xqs::AbstractArray, v_xqs::AbstractArray, s::GPState, xqs::AbstractMatrix, ps::GPModel)

    for (m, xq) in Iterators.zip(eachindex(m_xqs, v_xqs), eachcol(xqs))
        m_xqs[m], v_xqs[m] = query!(s, xq, ps)
    end
    return nothing
end
