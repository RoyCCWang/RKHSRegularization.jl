
# 2D regression.

import Images

fig_num = 1
PLT.close("all")

Random.seed!(25)

T = Float64

#PLT.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

img = Images.load("./data/k05_helmet_30.png")
gray_img = Images.Gray.(img)
y_nD = convert(Array{Float64}, gray_img)
N1_y, N2_y = size(y_nD)

##### regression
#θ = RK.Spline34Kernel(0.04)
θ = RK.WendlandSplineKernel(RK.Order2(), 0.04, 3)

# user inputs.
#σ² = 1e-3
σ² = 1e-6
#σ² = 0.0

zoom_factor = 3
xq1_range = LinRange(1, N1_y, N1_y*zoom_factor)
xq2_range = LinRange(1, N2_y, N2_y*zoom_factor)

x1_range = LinRange(1, N1_y, N1_y)
x2_range = LinRange(1, N2_y, N2_y)


#DelimitedFiles.writedlm( "query.txt",  xq_range, '\n')

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
X_nD = collect( [x1_range[i1]; x2_range[i2]] for i1 in eachindex(x1_range), i2 in eachindex(x2_range) )
X = vec(X_nD)
y = vec(y_nD)

# check posdef.
K = RK.constructkernelmatrix(X, θ)
println("rank(K) = ", rank(K))

println("isposdef = ", isposdef(K))

# fit RKHS.
η = RK.Problem(RK.UseRKHS(), X, θ, RK.Variance(σ²))
RK.fit!(η, y)


# query.

Xq_nD = collect( [xq1_range[i1]; xq2_range[i2]] for i1 in eachindex(xq1_range), i2 in eachindex(xq2_range) )

yq = Vector{T}(undef, length(Xq_nD))
RK.batchqueryRKHS!(yq, vec(Xq_nD), η)
Yq_nD = reshape(yq, size(Xq_nD))

# Visualize regression result.
PLT.figure(fig_num)
fig_num += 1
PLT.imshow(y_nD)
PLT.title("input image")
PLT.legend()

PLT.figure(fig_num)
fig_num += 1
PLT.imshow(Yq_nD)
PLT.title("output image")
PLT.legend()

##### end of regression


# discrepancy over the original training inputs X. Should be zero if σ² is 0.
yq2 = xx->RK.queryRKHS(xx, η)
@show norm(yq2.(vec(X_nD)) - y )
