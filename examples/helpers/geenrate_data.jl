

abstract type TestFuncSelection end
struct Ellipse2D <: TestFuncSelection end

function maketestfunc(::Ellipse2D, offset::Vector{T}) where T

    A = convert(Matrix{T}, [1.0 0.4; 0.4 1.0] .* 0.1)
    f = xx->sinc((dot((xx-offset),A*(xx-offset))/3.2)^2)*(norm(xx)/4)^3

    return f
end

function makeellipsefunc(
    offset::Vector{T};
    ellipse_matrix::Matrix{T} = Matrix{T}(undef, 0,0),
    ) where T

    D = length(offset)
    A = ellipse_matrix
    if isempty(A)
        A = randn(D,D)
        A = A'*A
    end
    f = xx->sinc((dot((xx-offset),A*(xx-offset))/3.2)^2)*(norm(xx)/4)^3
end

function generategridpoints(
    ::Type{T},
    ::Val{D},
    x_ranges::Vector{LinRange{T, Int}},
    N::Int,
    f,
    )::Tuple{Array{Vector{T}, D} , Array{T,D}} where {T,D}
   
    X_nD = ranges2collection(reverse(x_ranges), Val(D)) # x1 is horizontal.
    for i  in eachindex(X_nD)
        X_nD[i] = reverse(X_nD[i])
    end
    f_X_nD = f.(X_nD)

    return X_nD, f_X_nD
end

function generatepoints(N::Int, lbs::Vector{T}, ubs::Vector{T})::Vector{Vector{T}} where T
    
    @assert length(ubs) == length(lbs)
    D = length(ubs)

    return collect( 
        collect(
            convertcompactdomain(rand(T), zero(T), one(T), lbs[d], ubs[d])
            for d = 1:D
        )
        for n = 1:N
    )
end
