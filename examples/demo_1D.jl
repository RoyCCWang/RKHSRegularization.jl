
Random.seed!(25)
rng_main = Random.Xoshiro(0)

PLT.close("all")
fig_num = 1

T = Float64
D = 1

# # Setup RKHS problem.
# Kernel.
θ = RK.SqExp(T(10))
#θ = RK.WendlandSpline(RK.Order2(), T(0.4), 3) # valid posdef kernel for input dimensions up to 3.
#θ = RK.Matern3Halfs(1.0, 1.0)

# observation noise variance.
σ² = RK.CommonVariance(T(1e-5))
#σ² = zero(T)

# create oracle
N = 15
x_range = LinRange(-one(T), one(T), N)
X = reshape(collect(x_range), D, N)

oracle_func = xx->sinc(4*xx)*xx^3
y = oracle_func.(x_range)

# # Fit kernel hyperparameter
options = RK.FitOptions(RK.VariableKernel())
state = RK.LikelihoodState(X, y, θ, σ²)

struct CostCallable{ST, OT}
    state::ST
    options::OT
end

function (f::CostCallable)(x)
    return RK.eval_nll!(f.state, x, f.options)
end

p0 = RK.initialize_params(X, y, θ, σ²)
x0 = collect(RK.get_flat(p0, options))

f = CostCallable(state, options)

Random.seed!(25)
ret = Optim.optimize(
    f,
    x0,
    Optim.ParticleSwarm(;
        lower = [T(1e-3);],
        upper = [T(100);],
        n_particles = 60,
    ),
)

x_star = Optim.minimizer(ret)
println("Fitted kernel hyperparameter is: ", x_star)
println()

# ## update θ with x_star,

# either:
p = RK.initialize_params(X, y, θ, σ²)
RK.update_params!(p, x_star)
RK.update_kernel!(θ, x_star)

# or:
f(x_star)



# query.
Nq = 1000
xqs_range = LinRange(T(-2), T(2), Nq)
xqs = reshape(collect(xqs_range), D, Nq)


f_xqs = oracle_func.(xqs_range) # generating function.

# If we use the GP model, we can get predictive mean and variance.
mq = Memory{T}(undef, Nq)
vq = Memory{T}(undef, Nq)

gp_state = RK.GPState(T, N)
gp_model = RK.GPModel(state)

RK.query!(mq, vq, gp_state, xqs, gp_model)

# If we use the RKHS model, we can only get the predictive mean. This computes faster than the GP model.
q_xqs = Memory{T}(undef, Nq)

rkhs_state = RK.RKHSState(T, N)
rkhs_model = RK.RKHSModel(state)

RK.query!(q_xqs, rkhs_state, xqs, rkhs_model)

@show norm(mq - q_xqs) # should be no difference.

# Visualize regression result.
PLT.figure(fig_num)
fig_num += 1

PLT.plot(vec(X), y, ".", label = "data")
PLT.plot(vec(xqs), mq, label = "mean")

PLT.plot(vec(xqs), f_xqs, label = "oracle")

title_string = "1-D demo"
PLT.title(title_string)
PLT.legend()
##### end of regression

# # discrepancy over the original training inputs X. Should be zero if σ² is 0.
# yq2 = xx->RK.queryRKHS(xx, η)
# @show norm(yq2.(X) - y ), σ²

nothing