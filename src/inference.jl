
# T is the output type.
# function queryRKHS(x::Vector{T}, η::Problem)::T where T
    
#     Xq = collect( x for _ = 1:1 )
#     return queryRKHS!(Xq, η)
# end

function queryRKHS(xq::Vector{T}, η::Problem)::T where T

    X, c, kq, θ = η.X, η.c, η.inference.kq, η.θ
    inference = η.inference

    # resize!(kq, length(X))
    # for i in eachindex(kq)
    #     kq[i] = evalkernel(xq, X[i], θ)
    # end
    evalkernel!(kq, xq, X, θ, inference)

    return dot(kq, c)
end


function batchqueryRKHS!(
    means::Vector{T}, # mutates
    xqs::Vector{Vector{T}},
    η::Problem,
    ) where T
    
    N = length(xqs)
    resize!(means, N)

    for n in eachindex(xqs)
        means[n] = queryRKHS(xqs[n], η)
    end

    return nothing
end

function batchqueryRKHS(xqs::Vector{Vector{T}}, η::Problem)::Vector{T} where T
    
    means = Vector{T}(undef, length(xqs))
    batchqueryRKHS!(means, xqs, η)
    
    return means
end

# # GP inference.

# inference.
function queryGP(xq::Vector{T}, η::Problem)::Tuple{T,T} where T

    # parse.
    X, c, θ, kq, L, v = η.X, η.c, η.θ, η.inference.kq, η.inference.L, η.inference.v
    inference = η.inference

    # mean.
    # for i in eachindex(X)
    #     kq[i] = evalkernel(xq, X[i], θ)
    # end
    evalkernel!(kq, xq, X, θ, inference)

    m_xq = dot(kq, c)

    # variance.
    #v[:] = L\kq
    ldiv!(v, L, kq)
    v_xq = evalkernel(xq, xq, θ) - dot(v,v) # no guard against negative variance.

    return m_xq, v_xq
end

# # Batch inference for both Problem and LTProblem.

function batchqueryGP!(
    means::Vector{T}, # mutates
    variances::Vector{T}, # mutates
    xqs::Vector{Vector{T}},
    η,
    ) where T
    
    N = length(xqs)
    resize!(means, N)
    resize!(variances, N)

    for n in eachindex(xqs)
        means[n], variances[n] = queryGP(xqs[n], η)
    end

    return nothing
end

function batchqueryGP(xqs::Vector{Vector{T}}, η)::Tuple{Vector{T},Vector{T}} where T
    
    means = Vector{T}(undef, length(xqs))
    variances = Vector{T}(undef, length(xqs))
    batchqueryGP!(means, variances, xqs, η)
    
    return means, variances
end