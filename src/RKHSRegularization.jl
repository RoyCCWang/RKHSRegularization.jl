module RKHSRegularization

using LinearAlgebra
#import Statistics
#using SparseArrays
#import NearestNeighbors as NB

# # RKHS
include("types.jl")

include("./kernels/basic.jl")
include("./kernels/adaptive.jl")
include("./kernels/brownian.jl")

include("fit.jl")
include("inference.jl")

include("utils.jl")

include("./gp/likelihood.jl")

# dimensionality reduction of training samples.


# # training
include("./training/optim.jl")


end # module RKHSRegularization

