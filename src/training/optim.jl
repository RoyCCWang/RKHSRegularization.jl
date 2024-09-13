
# optimization of GP hyperparameters, using the marginal likelihood.
# allow external optimizers.

# struct OptimizerContainer{FT, CT}
#     runoptim::FT # function. Inputs: costfuc, lb, ub, config. Initial guesses go in config. Allow only box constraints on vars.
# end


# check if this is only for single hyperparameter kernels.
# the costfunc should be run once at the end of runoptim, to make sure the best solution is loaded into mgp.
function hyperoptim!(
    η::Problem,
    var_mapping::VariableMapping,
    runoptim,
    config,
    lb::Vector{T},
    ub::Vector{T},
    y::Vector{T},
    ) where T <: AbstractFloat
    
    @assert length(lb) == length(ub)
    
    # minimize the negative of likelihood
    costfunc = xx->evalnll!(η, xx, var_mapping, y)

    return runoptim(costfunc, lb, ub, config), costfunc
end
