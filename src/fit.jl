
# all parameters to a RKHS regularization problem.
struct AllParameters{T <: AbstractFloat, ST <: SubArray, RT <: AbstractArray}
    contents::Memory{T}
    kernel::ST
    noise::ST
    outputs::ST
    inputs::RT # reshaped subarray

    function AllParameters(::Type{T}, D::Integer, N::Integer, N_kernel::Integer, N_noise::Integer) where T <: AbstractFloat
        #
        N_vars = D*N + N + N_noise + N_kernel
        contents = Memory{T}(undef, N_vars)

        st_ind = 1
        fin_ind = st_ind + N_kernel -1
        v_kernel = view(contents, st_ind:fin_ind)

        st_ind = fin_ind + 1
        fin_ind = st_ind + N_noise -1
        v_noise = view(contents, st_ind:fin_ind)

        st_ind = fin_ind + 1
        fin_ind = st_ind + N -1
        v_out = view(contents, st_ind:fin_ind)

        st_ind = fin_ind + 1
        fin_ind = st_ind + D*N -1
        v_in = reshape(view(contents, st_ind:fin_ind), D, N)

        fin_ind == N_vars || error("Length mismatch. Please report this issue.")
        return new{T, typeof(v_out), typeof(v_in)}(contents, v_kernel, v_noise, v_out, v_in)
    end
end

function get_flat(p::AllParameters)
    return p.contents
end

function update_params!(p::AllParameters, x::AbstractVector)
    v = view(p.contents, 1:length(x))
    copy!(v, x)
    return nothing
end

function initialize_params(X::Matrix{T}, y::AbstractVector{T}, θ::Kernel, noise::NoiseModel) where T <: AbstractFloat
    N_kernel = get_num_params(θ)
    N_noise = get_num_params(noise)
    D, N = size(X)
    length(y) == N || error("Length mismatch.")

    p = AllParameters(T, D, N, N_kernel, N_noise)
    copy!(p.kernel, get_params(θ))
    copy!(p.noise, get_params(noise))
    copy!(p.outputs, y)
    copy!(p.inputs, X)

    return p
end

# # when we just want the parameters for the mean query equation.
# struct QueryParameter{T <: AbstractFloat, ST <: SubArray}

#     contents::Memory{T}
#     kernel::ST
#     coeffs::ST
# end

struct FitOptions{KT <: KernelfitOption, NT <: NoisefitOption, OT <: OutputsFitOption, IT <: InputsFitOption}
    kernel::KT
    noise::NT
    outputs::OT
    inputs::IT
end

function FitOptions(kernel::KernelfitOption)
    return FitOptions(kernel, ConstantNoise(), ConstantOutputs(), ConstantInputs())
end

function FitOptions(kernel::KernelfitOption, noise::NoisefitOption)
    return FitOptions(kernel, noise, ConstantOutputs(), ConstantInputs())
end

function get_flat(p::AllParameters, op::FitOptions)
    
    N_vars = 0
    if typeof(op.kernel) <: VariableKernel
        N_vars += length(p.kernel)
    end

    if typeof(op.noise) <: VariableNoise
        N_vars += length(p.noise)
    end

    if typeof(op.outputs) <: VariableOutputs
        N_vars += length(p.outputs)
    end

    if typeof(op.inputs) <: VariableInputs
        N_vars += length(p.inputs)
    end

    return view(p.contents, 1:N_vars)
end

struct LikelihoodState{T <: AbstractFloat, PT <: AllParameters, KT <: Kernel, NT <: NoiseModel}
    p::PT

    U::Matrix{T}
    c::Memory{T}

    kernel::KT
    noise::NT
    outputs::Memory{T}
    inputs::Matrix{T}

    function LikelihoodState(::Type{T}, D::Integer, N::Integer, θ::KT, σ²::NT) where {T <: AbstractFloat, KT <: Kernel, NT <: NoiseModel}

        p = AllParameters(T, D, N, get_num_params(θ), get_num_params(σ²))
        outputs = collect(p.outputs)
        inputs = collect(p.inputs)

        return new{T, typeof(p), KT, NT}(
            p,
            zeros(T, N, N),
            Memory{T}(undef, N),
            θ,
            σ²,
            outputs,
            inputs,
        )
    end
    
    function LikelihoodState(X::Matrix{T}, y::AbstractVector{T}, θ::KT, σ²::NT) where {T <: AbstractFloat, KT <: Kernel, NT <: NoiseModel}

        D, N = size(X)
        length(y) == N || error("Size mismatch.")

        p = initialize_params(X, y, θ, σ²)
        outputs = collect(p.outputs)
        inputs = collect(p.inputs)

        return new{T, typeof(p), KT, NT}(
            p,
            zeros(T, N, N),
            Memory{T}(undef, N),
            θ,
            σ²,
            outputs,
            inputs,
        )
    end
end

# does not mutate
function eval_nll!(
    s::LikelihoodState{T}, # mutates
    x::AbstractVector{T},
    op::FitOptions,
    )::T where T <: AbstractFloat

    # load parameters
    p = s.p
    update_params!(p, x)

    parse_kernel!(op.kernel, s.kernel, p.kernel)
    parse_noise!(op.noise, s.noise, p.noise)
    parse_outputs!(op.outputs, s.outputs, p.outputs)
    parse_inputs!(op.inputs, s.inputs, p.inputs)
    
    # kernel matrix
    U, noise, kernel = s.U, s.noise, s.kernel
    X, y = s.inputs, s.outputs
    update_K!(U, X, kernel)
    applynoisemodel!(U, noise)
    #@show norm(U)

    # fit GP.
    c = s.c
    L = cholesky(U) # allocates.
    copy!(c, L\y) # allocates.

    # dot(y, Ky\y) - logdet(Ky)
    #logdet_term = 2*sum( log(L[i,i]) for i in axes(L,1) ) # might need guard against non-positive entries being in the diagonal of L when we compute it.
    logdet_term = logdet(L) # cannot explicitly index into a Cholesky factor as of Julia v1.11.

    #@show dot(y, η.c), logdet_term
    log_likelihood = -dot(y, c) - logdet_term

    return -log_likelihood
end

function update_K!(K::Matrix, X::Matrix, θ::Kernel)

    M = size(X,2)
    size(K,1) == size(K,2) == M || error("Size mismatch.")

    # for j = 1:M
    #     for i = j:M

    ## buggy code.
    # fill!(K, Inf) # debug.
    # for (j, x_j) in Iterators.zip(1:M, eachcol(X))
    #     for (i, x_i) in Iterators.zip(j:M, eachcol(X))
    #         K[i,j] = evalkernel(x_i, x_j, θ)
    #     end
    # end
    # for j = 2:M
    #     for i = 1:(j-1)
    #         K[i,j] = K[j,i]
    #     end
    # end

    # twice slower, but accurate.
    fill!(K, Inf) # debug.
    for (j, x_j) in Iterators.zip(axes(K,2), eachcol(X))
        for (i, x_i) in Iterators.zip(axes(K,1), eachcol(X))
            K[i,j] = evalkernel(x_i, x_j, θ)
        end
    end


    return nothing
end

### apply additive normal distributed noise model.

function applynoisemodel!(U::Matrix{T}, x::CommonVariance{T}) where T

    v = x.v[begin]
    for i in axes(U,1)
        U[i,i] += v
    end

    return nothing
end

function applynoisemodel!(U::Matrix{T}, x::DiagonalVariance{T}) where T

    v = x.v
    #@show length(v), size(U)
    @assert length(v) == size(U,1)

    for i in axes(U,1)
        U[i,i] += v[i]
    end

    return nothing
end