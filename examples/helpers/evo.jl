
struct MetaheuristicsECAConfig{T <: AbstractFloat}
    f_calls_limit::Int
    initial_iterate::Matrix{T} # empty for no initial guess
end

function MetaheuristicsECAConfig(::Type{T}, N::Int)::MetaheuristicsECAConfig{T} where T <: AbstractFloat
    return MetaheuristicsECAConfig(N, Vector{T}(undef, 0))
end

function MetaheuristicsECAConfig(N::Int, x0::Vector{T})::MetaheuristicsECAConfig{T} where T <: AbstractFloat
    mat = Matrix{T}(undef, 1, length(x0))
    mat[1,:] = x0
    return MetaheuristicsECAConfig(N, mat)
end

function MetaheuristicsECAConfig(N::Int, x0s::Vector{Vector{T}})::MetaheuristicsECAConfig{T} where T <: AbstractFloat
    
    @assert !isempty(x0s)
    D = length(x0s[begin])
    
    mat = Matrix{T}(undef, length(x0s), D)
    for i in axes(mat,1)
        mat[i, :] = x0s[i]
    end

    return MetaheuristicsECAConfig(N, mat)
end

# result = runevofull(costfunc, lb, ub, f_calls_limit, x0)
function runevofull(costfunc, lb::Vector{T}, ub::Vector{T}, config) where T

    f_calls_limit, x0 = config.f_calls_limit, config.initial_iterate

    bounds = Metaheuristics.boxconstraints(lb = lb, ub = ub)
    algo  = Metaheuristics.ECA(
        N = 61,
        options = Metaheuristics.Options(
            f_calls_limit = f_calls_limit,
            seed = 1,
        ),
    )

    if !isempty(x0)
        Metaheuristics.set_user_solutions!(algo, x0, costfunc);
    end

    result = Metaheuristics.optimize(costfunc, bounds, algo)
    costfunc(result.best_sol.x) # force update of mgp with best solution.

    return result
end

# minimalist (and perhaps type-stable) return of best solution vector only, instead of custom results container.
function runevosimple(costfunc, lb::Vector{T}, ub::Vector{T}, config)::Vector{T} where T

    result = runevofull(costfunc, lb, ub, config)
    return result.best_sol.x
end