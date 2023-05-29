
# 1D regression.

fig_num = 1
PyPlot.close("all")

Random.seed!(25)


#PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

##### regression
θ = RKReg.Spline34KernelType(0.04)

# user inputs.
σ² = 1e-3
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
K = RKReg.constructkernelmatrix(X, θ)
println("rank(K) = ", rank(K))

println("isposdef = ", isposdef(K))



# fit RKHS.
η = RKReg.RKHSProblemType( zeros(Float64,length(X)),
                     X,
                     θ,
                     σ²)
RKReg.fitRKHS!(η, y)



# query.
xq = collect( [xq_range[n]] for n = 1:Nq )

f_xq = f.(xq_range) # generating function.

yq = Vector{Float64}(undef, Nq)
RKReg.query!(yq,xq,η)

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(X, y, ".", label = "observed")
PyPlot.plot(xq, yq, label = "GP mean function")

title_string = "1D GP regression"
PyPlot.title(title_string)
PyPlot.legend()
##### end of regression
