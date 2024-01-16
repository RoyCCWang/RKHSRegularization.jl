
# # Gaussian process problem containers.


# # update NoiseModel

struct VariableMapping{
    KT <: Union{Int, UnitRange{Int}, Nothing},
    NT <: Union{Int, Nothing}}

    kernel::KT
    σ²::NT
end

function updatekernel!(::PositiveDefiniteKernel, ::Nothing, args...)
    return nothing
end

function updatekernel!(θ::PositiveDefiniteKernel, var_range::UnitRange{Int}, p::Vector{T}) where T <: Real
    return updatekernel!(θ, p[var_range])
end

function updatekernel!(θ::PositiveDefiniteKernel, ind::Int, p::Vector)::Nothing
    return updatekernel!(θ, p[ind])
end

function updatenoisemodel!(::NoiseModel, ::Nothing, args...)
    return nothing
end

function updatenoisemodel!(s::Variance{T}, var_ind::Int, p::Vector{T})  where T <: Real
    return updatenoisemodel!(s, p[var_ind])
end

function updatenoisemodel!(s::Union{Covariance{T}, DiagonalCovariance{T}}, var_range::UnitRange{Int}, p::Vector{T})::Nothing  where T <: Real
    return updatenoisemodel!(s, p[var_range])
end


# # hyperparameter optimization cost function options
abstract type HyperparameterCostContainer end
struct MarginalLikelihood <: HyperparameterCostContainer end

# function getcostfunc(::MarginalLikelihood, η::Problem, y::Vector{T}) where T <: AbstractFloat

# direct, naive computation.
function lnnormalpdf(
    η::Problem,
    y::Vector{T}
    )::T where T

    L = η.inference.L
    
    out = log(2*π) + logdet(L)*2 + dot(y, η.c)
    out = -out/2
    return out
end

# marginal negative log likelihood. Eq. 5.9 from GPML, without constants.
function evalnll(
    p::Vector{T},
    var_mapping::VariableMapping,
    η::Problem,
    y::Vector{T},
    )::T where T <: AbstractFloat

    #η_new = duplicateproblem(η, θ_new)
    #fit!(η, y, θ_new)

    updatekernel!(η.θ, var_mapping.kernel, p)
    updatenoisemodel!(η.noise_model, var_mapping.σ², p)

    fit!(η, y)

    # dot(y, Ky\y) - logdet(Ky)
    L = η.inference.L
    logdet_term = 2*sum( log(L[i,i]) for i in axes(L,1) ) # might need guard against non-positive entries being in the diagonal of L when we compute it.
    
    #@show dot(y, η.c), logdet_term
    log_likelihood_eval = -dot(y, η.c) - logdet_term

    return -log_likelihood_eval
end
