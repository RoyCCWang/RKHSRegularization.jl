# coordinates.
# x ∈ prod( [a[d], b[d]] for d = 1:D ).
# same sampling interval for each dimension.
# M[d] is the number of samples in dimension d.
# out ∈ prod( 1:M[d] for d = 1:D )
function convert2itpindex(  x::Vector{T},
    a::Vector{T},
    b::Vector{T},
    M::Vector{Int})::Vector{T} where T <: Real

    D = length(x)
    @assert D == length(a) == length(b)

    out = Vector{T}(undef,D)
    for d = 1:D
    offset = b[d]-a[d]
    u = (x[d]-a[d])/offset # ∈ [0,1]
    i = u*(M[d]-1) # ∈ [0,M-1]
    out[d] = i + 1
    end

    return out
end
# x =[-0.01; 2]
# a = [-0.1; -1 ]
# b = [0.0; 3]
# M = [10; 5]
# p = convert2itpindex([-0.01; 3], a, b, M) # should be [9;5]

function isonehot(x::Vector{T}; atol = 1e-5)::Bool where T

    if !isapprox(sum(x), one(T); atol = atol)
        #println("one)")
        return false
    end

    for n in eachindex(x)

        is_zero = isapprox(x[n], zero(T); atol = atol)
        is_one = isapprox(x[n], one(T); atol = atol)

        if !(is_zero || is_one)
            #println("$n")
            return false
        end
    end

    return true
end

function array2matrix(X::Vector{Vector{T}})::Matrix{T} where T

    N = length(X)
    D = length(X[1])

    out = Matrix{T}(undef,D,N)
    for n = 1:N
        out[:,n] = X[n]
    end

    return out
end



# function getflattenmapping(subset_inds0::Vector{Vector{Int}})
    
#     out_ranges = Vector{UnitRange{Int}}(undef, length(subset_inds0))
#     st_ind = 0
#     fin_ind = 0
#     for m in eachindex(subset_inds0)
#         st_ind = fin_ind + 1
#         fin_ind = st_ind + length(subset_inds0[m]) -1

#         out_ranges[m] = st_ind:fin_ind
#     end

#     return out_ranges
# end


function solvesystem!(
    out::Vector{T},
    Ls::Vector{LowerTriangular{T, Matrix{T}}},
    Us::Vector{UpperTriangular{T, Matrix{T}}},
    v::Vector{T},
    mapping::Vector{UnitRange{Int}},
    ) where T <: AbstractFloat

    @assert length(Ls) == length(Us)

    #out_nested = collect( view(out, mapping[m]) for m in eachindex(mapping) )

    for m in eachindex(Ls)
        
        # this might be faster for smaller lengths of out and v.
        # out_m = view(out, mapping[m])
        # ldiv!(out_m, Ls[m], v[mapping[m]])
        # ldiv!(out_m, Us[m], out_m)

        # less floating-point allocation.
        out_m = view(out, mapping[m])
        v_m = view(v, mapping[m])
        ldiv!(out_m, Ls[m], v_m)
        ldiv!(out_m, Us[m], out_m)
    end

    return nothing
end

function solvesystem!(
    out::Vector{Vector{T}},
    Ls::Vector{LowerTriangular{T, Matrix{T}}},
    Us::Vector{UpperTriangular{T, Matrix{T}}},
    v_set::Vector{Vector{T}},
    ) where T <: AbstractFloat

    @assert length(Ls) == length(Us) == length(v_set) == length(out)

    for m in eachindex(Ls)

        # less floating-point allocation.
        out_m = out[m]

        ldiv!(out_m, Ls[m], v_set[m])
        ldiv!(out_m, Us[m], out_m)
    end

    return nothing
end


