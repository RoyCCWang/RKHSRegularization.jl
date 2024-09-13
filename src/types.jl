
# # Additive noise models.
abstract type NoiseModel end
# Interface requirements: update_noise!, get_num_params!, get_params

struct CommonVariance{T <: AbstractFloat} <: NoiseModel
    v::Memory{T} # single-element
    
    function CommonVariance(v::T) where T <: AbstractFloat
        a = Memory{T}(undef, 1)
        a[begin] = v
        return new{T}(a)
    end
end

function update_noise!(s::CommonVariance, v::Union{Real, AbstractVector})
    s.v[begin] = v[begin]
    return nothing
end

function get_num_params(::CommonVariance)
    return 1
end

struct DiagonalVariance{T <: AbstractFloat} <: NoiseModel
    v::Memory{T}
end

function update_noise!(s::DiagonalVariance, v::AbstractVector)
    copy!(s.v, v)
    return nothing
end

function get_num_params(s::DiagonalVariance)
    return length(s)
end

function get_params(s::Union{DiagonalVariance, CommonVariance})
    return s.v
end

# # Inference states
# This means these are buffers that mutate during inference.

struct InferenceState{T <: AbstractFloat, LT}
    kq::Memory{T} # length(X)
    v::Memory{T} # length(X)
    L::LT # lower triangular cholesky factor of the training kernel matrix.

    function InferenceState(::Type{T}, N::Integer, L::LT) where {T, LT}
        return new{T,LT}(Memory{T}(undef, N), Memory{T}(undef, N), L)
    end
end

struct DEInferenceState{T <: AbstractFloat, LT}
    kq::Memory{T} # length(X)
    v::Memory{T} # length(X)
    warp_X::Memory{T}
    L::LT # lower triangular cholesky factor of the training kernel matrix.

    function DEInferenceState(::Type{T}, N::Integer, L::LT) where {T, LT}
        return new{T,LT}(Memory{T}(undef, N), Memory{T}(undef, N), Memory{T}(undef, N), LT)
    end
end

# # Training states
# Train a combo of noise, kernel.

# For GP network: Train coefficients, positions, noise.

abstract type Kernel end
abstract type NonDEKernel <: Kernel end
abstract type DEKernel <: Kernel end


# # Updates


abstract type KernelfitOption end
struct VariableKernel <: KernelfitOption end
struct ConstantKernel <: KernelfitOption end

abstract type NoisefitOption end
struct VariableNoise <: NoisefitOption end
struct ConstantNoise <: NoisefitOption end

abstract type OutputsFitOption end
struct VariableOutputs <: OutputsFitOption end
struct ConstantOutputs <: OutputsFitOption end

abstract type InputsFitOption end
struct VariableInputs <: InputsFitOption end
struct ConstantInputs <: InputsFitOption end

function parse_kernel!(::ConstantKernel, args...)
    return nothing
end

function parse_noise!(::ConstantNoise, args...)
    return nothing
end

function parse_outputs!(::ConstantOutputs, args...)
    return nothing
end

function parse_inputs!(::ConstantInputs, args...)
    return nothing
end

function parse_kernel!(::VariableKernel, θ::Kernel, p::AbstractVector)
    return update_kernel!(θ, p)
end

function parse_noise!(::VariableNoise, s::NoiseModel, v::AbstractVector)
    return update_noise!(s, v)
end

function parse_outputs!(::VariableOutputs, y::AbstractVector, y_in::AbstractVector)
    return copy!(y, y_in)
end

function parse_inputs!(::VariableInputs, X::AbstractMatrix, x::AbstractVector)
    return copyto!(X, x)
end



# ######### legacy

# # # For RKHS regularization problems.
# # contain quantities that are required for inference.

# # classic RKHS

# struct GPModel{T <: AbstractFloat, KT <: Kernel, ST <: NoiseModel}
#     c::Memory{T} # solution of the problem.
#     X::Matrix{T}
#     θ::KT
#     noise_model::ST # σ²::ST

#     # this will contain kernel matrix plus noise model.
#     U::Matrix{T} # to avoid allocation repeately when hyperparameter fitting. Also used for sequential/batch problem update.

#     function GPModel(::Type{T}, D::Integer, N::Integer, θ::KT, noise::ST) where {T, KT, ST}
#         returnnew{T,KT,ST}(Memory{T}(undef, N), zeros(T, D, N), θ, noise, zeros(T, N, N))
#     end
# end

# function Problem(
#     ::UseAdaptiveRKHS,
#     X::Vector{Vector{T}},
#     θ::KT,
#     noise_model::ST,
#     )::Problem{T, KT, AdaptiveRKHS{T}, ST} where {T, KT, ST <: NoiseModel}

#     N = length(X)
#     return Problem(
#         Vector{T}(undef,N),
#         X,
#         θ,
#         noise_model,
#         AdaptiveRKHS(T, N),
#         Matrix{T}(undef, N, N),
#     )
# end

# function Problem(
#     ::UseRKHS,
#     X::Vector{Vector{T}},
#     θ::KT,
#     noise_model::ST,
#     )::Problem{T, KT, RKHS{T}, ST} where {T, KT <: Kernel, ST <: NoiseModel}

#     N = length(X)
#     return Problem(
#         Vector{T}(undef,N),
#         X,
#         θ,
#         noise_model,
#         RKHS(T, N),
#         Matrix{T}(undef, N, N),
#     )
# end

# function Problem(
#     ::UseAdaptiveCholeskyGP,
#     X::Vector{Vector{T}},
#     θ::KT,
#     noise_model::ST,
#     )::Problem{T, KT, AdaptiveCholeskyGP{T}, ST} where {T, KT <: Kernel, ST <: NoiseModel}

#     N = length(X)
#     return Problem(
#         Vector{T}(undef,N),
#         X,
#         θ,
#         noise_model,
#         AdaptiveCholeskyGP(T, N),
#         Matrix{T}(undef, N, N),
#     )
# end

# function Problem(
#     ::UseCholeskyGP,
#     X::Vector{Vector{T}},
#     θ::KT,
#     noise_model::ST,
#     )::Problem{T, KT, CholeskyGP{T}, ST} where {T, KT <: Kernel, ST <: NoiseModel}

#     N = length(X)
#     return Problem(
#         Vector{T}(undef,N),
#         X,
#         θ,
#         noise_model,
#         CholeskyGP(T, N),
#         Matrix{T}(undef, N, N),
#     )
# end

# function getnumsamples(η::Problem)::Int
#     return length(η.X)
# end

# # compact RBF types

# struct EdgeWeightContainer{KT}

#     θs::Matrix{KT} # θs[src, dest] is the kernel for weight between src and dest nodes.
#     indicators::BitMatrix # true if the edge is active. Avoids floating-point comparison between kernel evaluations.
# end

# function EdgeWeightContainer(::KT, N::Int)::EdgeWeightContainer{KT} where KT
#     M = Matrix{KT}(undef, N, N)
#     #M = fill!(KT(one(T)))
#     return EdgeWeightContainer(M, falses(N,N))
# end
