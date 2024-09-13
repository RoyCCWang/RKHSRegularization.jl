
import Images

fig_num = 1
PLT.close("all")

Random.seed!(25)

T = Float64
D = 2

# # Setup data
img = Images.load(joinpath("data", "k05_helmet_30.png"))
gray_img = Images.Gray.(img)
y_nD = convert(Array{Float64}, gray_img)
N1_y, N2_y = size(y_nD)

# query and training positions.
zoom_factor = 3
xq1_range = LinRange(1, N1_y, N1_y*zoom_factor)
xq2_range = LinRange(1, N2_y, N2_y*zoom_factor)

x1_range = LinRange(1, N1_y, N1_y)
x2_range = LinRange(1, N2_y, N2_y)

# # Setup Model
θ = RK.SqExp(T(5))
σ² = RK.CommonVariance(T(1e-5))

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
X_nD = collect( [x1_range[i1]; x2_range[i2]] for i1 in eachindex(x1_range), i2 in eachindex(x2_range) )
X = reshape(collect(Iterators.flatten(vec(X_nD))), D, length(X_nD))
y = vec(y_nD)

Xq_nD = collect( [xq1_range[i1]; xq2_range[i2]] for i1 in eachindex(xq1_range), i2 in eachindex(xq2_range) )
Xq = reshape(collect(Iterators.flatten(vec(Xq_nD))), D, length(Xq_nD))

# # Fit kernel hyperparameter
options = RK.FitOptions(RK.VariableKernel(), RK.VariableNoise())
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
        lower = [T(1e-3); T(1e-9)],
        upper = [T(100); T(100)],
        n_particles = 3,
    ),
)

x_star = Optim.minimizer(ret)
println("Fitted kernel hyperparameter is: ", x_star)
println()


# # Query.
xqs = Xq

# If we use the GP model, we can get predictive mean and variance.
mq = zeros(T, size(Xq_nD))
vq = zeros(T, size(Xq_nD))

N = length(X_nD)
gp_state = RK.GPState(T, N)
gp_model = RK.GPModel(state)

RK.query!(mq, vq, gp_state, xqs, gp_model)

# If we use the RKHS model, we can only get the predictive mean. This computes faster than the GP model.
q_xqs = zeros(T, size(Xq_nD))

rkhs_state = RK.RKHSState(T, N)
rkhs_model = RK.RKHSModel(state)

RK.query!(q_xqs, rkhs_state, xqs, rkhs_model)

@show norm(mq - q_xqs) # should be no difference.

PLT.figure(fig_num)
fig_num += 1
PLT.imshow(y_nD)
PLT.title("data")

PLT.figure(fig_num)
fig_num += 1
PLT.imshow(mq)
PLT.title("predictive mean")

nothing