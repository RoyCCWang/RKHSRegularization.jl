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
