module RKHSRegularization

using LinearAlgebra
#import Statistics
#using SparseArrays
#import NearestNeighbors as NB

# # RKHS
include("types.jl")


include("fit.jl")
include("gp/query.jl")
include("kernels/basic.jl")

# legacy


# include("fit_old.jl")


# include("inference.jl")

# include("utils.jl")

export evalnll!, initialize_params, LikelihoodState, FitOptions,
AllParameters, update_params!, get_flat, update_kernel!, update_noise!
GPState, GPModel, RKHSState, RKHSModel

end # module RKHSRegularization

