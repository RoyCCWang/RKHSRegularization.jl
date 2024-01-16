# Methods for fitting and querying under the RKHS regularization framework.

function constructkernelmatrix(X::Vector{Vector{T}}, θ::PositiveDefiniteKernel)::Matrix{T} where T

    K = Matrix{T}(undef,length(X),length(X))
    constructkernelmatrix!(K, X, θ)

    return K
end

function constructkernelmatrix!(
    K::Matrix{T},
    X::Union{Vector{Vector{T}},SubArray},
    θ::PositiveDefiniteKernel,
    ) where T

    M = length(X)
    @assert size(K) == (M,M)

    #fill!(K,Inf) # debug
    for j = 1:M
        for i = j:M
            K[i,j] = evalkernel(X[i], X[j], θ)
        end
    end

    for j = 2:M
        for i = 1:(j-1)
            K[i,j] = K[j,i]
        end
    end

    return nothing
end

# returns K_XZ. Entry i,j is k(x[i],z[j]).
function constructkernelmatrix(
    X::Vector{Vector{T}},
    Z::Vector{Vector{T}},
    θ::PositiveDefiniteKernel,
    )::Matrix{T} where T

    Nr = length(X)
    Nc = length(Z)

    K = Matrix{T}(undef, Nr, Nc)
    fill!(K,Inf) # debug
    for j = 1:Nc
        for i = 1:Nr
            K[i,j] = evalkernel(X[i], Z[j], θ)
        end
    end

    return K
end

# front end for all kernels.
function constructkernelmatrix!(K::Matrix{T}, η::Problem) where T
    constructkernelmatrix!(K, η.inference, η.X, η.θ)
    return nothing
end

# for non-adaptive kernels.
function constructkernelmatrix!(
    K::Matrix{T},
    ::Union{RKHS{T},CholeskyGP{T}},
    X::Union{Vector{Vector{T}},SubArray},
    θ::PositiveDefiniteKernel,
    ) where T

    constructkernelmatrix!(K, X, θ)
    return nothing
end

# adaptive kernel version.
function constructkernelmatrix!(
    K::Matrix{T},
    inference::Union{AdaptiveRKHS{T},AdaptiveCholeskyGP{T}}, # mutates.
    X::Union{Vector{Vector{T}},SubArray},
    θ::AdaptiveKernel,
    ) where T

    M = length(X)
    @assert size(K) == (M,M)

    warp = θ.warp
    warp_X = inference.warp_X

    # update warpmap evals.
    resize!(inference.warp_X, length(X))
    for n in eachindex(X)
        inference.warp_X[n] = evalwarpmap(X[n], warp)
        #@show evalwarpmap(X[n], warp)
    end

    #fill!(K,Inf) # debug
    for j = 1:M
        for i = j:M
            if i == j
                K[i,j] = one(T)
            else
                #K[i,j] = evalkernel(X[i], X[j], θ)
                K[i,j] = evalkernel(X[i], j, X, θ, warp_X)
            end
        end
    end

    for j = 2:M
        for i = 1:(j-1)
            K[i,j] = K[j,i]
        end
    end

    return nothing
end


#### Try separable RKHS via inverse. Assume N_r == D
# Assume η.a is same for all dimensions.

function fit!(η::Problem, y::Vector{T})::Nothing where T
    return fit!(η, y, η.θ)
end

# \theta overides η.θ.
function fit!(η::Problem, y::Vector{T}, θ::PositiveDefiniteKernel,)::Nothing where T
    return fit!(η.U, η, y, θ)
end

function fit!(
    U::Matrix{T}, # U is K with noise model applied, e.g. K + σ²I if using a variance model.
    η::Problem,
    y::Vector{T},
    θ::PositiveDefiniteKernel, # overides η.θ.
    )::Nothing where T

    @assert !isempty(η.X)
    @assert !isempty(y)

    M = length(η.X)
    @assert M == length(y)

    # update kernel matrix
    constructkernelmatrix!(U, η.inference, η.X, θ)

    # add observation model's noise.
    applynoisemodel!(U, η.noise_model)

    # solve.
    η.c[:] = U\y

    updateinferencecontainer!(η.inference, U)

    return nothing
end

function updateinferencecontainer!(C::Union{CholeskyGP{T},AdaptiveCholeskyGP{T}}, U::Matrix{T}) where T
    
    chol_U = cholesky(U)
    C.L[:] = chol_U.L

    return nothing
end

function updateinferencecontainer!(::Union{RKHS{T},AdaptiveRKHS{T}}, args...) where T
    return nothing
end

### apply additive normal distributed noise model.

function applynoisemodel!(U::Matrix{T}, x::Variance{T}) where T

    v = x.v[begin]
    for i in axes(U,1)
        U[i,i] += v
    end

    return nothing
end


function applynoisemodel!(U::Matrix{T}, x::DiagonalCovariance{T}) where T

    v = x.v
    #@show length(v), size(U)
    @assert length(v) == size(U,1)

    for i in axes(U,1)
        U[i,i] += v[i]
    end

    return nothing
end

function applynoisemodel!(U::Matrix{T}, x::Covariance{T}) where T

    v = x.v
    @assert size(v) == size(U)

    for i in eachindex(U)
        U[i] += v[i]
    end

    return nothing
end



