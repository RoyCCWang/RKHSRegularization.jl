

PLT.close("all")

Random.seed!(25)

fig_num = 1

T = Float64

##### regression

# not mean-square differentiable.
θ = RK.BrownianBridge10(1.0)

# # not mean-square differentiable.
# λ = 0.01
# θ = RK.ExponentialKernel(λ, 2*λ)

#θ = RK.SqExpKernel(10.0)
#θ = RK.WendlandSplineKernel(RK.Order2(), 0.4, 3) # valid posdef kernel for input dimensions up to 3.
#θ = RK.Matern3Halfs(1.0, 1.0)


σ² = 1e-5
#σ² = 0.0
N = 15

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
x_range = LinRange(1e-5, 1.0-1e-5, N)

X = collect( [x_range[n]] for n = 1:N )
coordwrapper = xx->RK.convert2itpindex(xx, X[1], X[end], [length(X)])

f = xx->sinc(4*xx)*xx^3
y = f.(x_range)

# check posdef.
K = RK.constructkernelmatrix(X, θ)
println("rank(K) = ", rank(K))

println("isposdef = ", isposdef(K))



# fit RKHS.
η = RK.Problem(
    RK.UseRKHS(),
    X,
    θ,
    RK.Variance(σ²),
)
RK.fit!(η, y)



# query.
Nq = 100
xq_range = LinRange(0.0, 1.0, Nq)
xq = collect( [xq_range[n]] for n = 1:Nq )

f_xq = f.(xq_range) # generating function.

yq = Vector{T}(undef, Nq)
RK.batchqueryRKHS!(yq, xq, η)

X_display = collect( X[n][begin] for n in eachindex(X) )
xq_display = collect( xq[n][begin] for n in eachindex(xq) )

# Visualize regression result.
PLT.figure(fig_num)
fig_num += 1

PLT.plot(X_display, y, ".", label = "observed")
PLT.plot(xq_display, yq, label = "query - RKHS")

PLT.plot(xq_display, f_xq, label = "true")

title_string = "1-D RKHS demo"
PLT.title(title_string)
PLT.legend()
##### end of regression

# discrepancy over the original training inputs X. Should be zero if σ² is 0.
yq2 = xx->RK.queryRKHS(xx, η)
@show norm(yq2.(X) - y ), σ²

#@assert 1==2

###################### spline kernel.


spline_dim = 3
θ = RK.WendlandSplineKernel(RK.Order2(), 0.04, spline_dim) # valid posdef kernel for input dimensions up to 3.

# user inputs.
#σ² = 1e-1
σ² = 1e-4
#σ² = 0.0
N = 15

x_range = LinRange(-24.0, 45.0, N)
Nq = 1000
#xq_range = LinRange(-30.0, 50.0, Nq)
Nq = 1000
xq_range = LinRange(x_range[1], x_range[end], Nq)

#DelimitedFiles.writedlm( "query.txt",  xq_range, '\n')

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
X = collect( [x_range[n]] for n = 1:N )

f = xx->sinc(4*xx)*xx^3
y = f.(x_range)

# check posdef.
K = RK.constructkernelmatrix(X, θ)
println("rank(K) = ", rank(K))
println("isposdef = ", isposdef(K))

# fit RKHS.
η = RK.Problem(RK.UseRKHS(), X, θ, RK.Variance(σ²))
RK.fit!(η, y)

# query.
xq = collect( [xq_range[n]] for n = 1:Nq )

f_xq = f.(xq_range) # generating function.

yq = RK.batchqueryRKHS(xq,η)

X_display = collect( X[n][begin] for n in eachindex(X) )
xq_display = collect( xq[n][begin] for n in eachindex(xq) )


# Visualize regression result.
PLT.figure(fig_num)
fig_num += 1

PLT.plot(X_display, y, ".", label = "observed")
PLT.plot(xq_display, yq, label = "GP mean function")

title_string = "1D GP regression, spline kernel"
PLT.title(title_string)
PLT.legend()
##### end of regression


# discrepancy over the original training inputs X. Should be zero if σ² is 0.
yq2 = xx->RK.queryRKHS(xx, η)
@show norm(yq2.(X) - y ), σ²

nothing