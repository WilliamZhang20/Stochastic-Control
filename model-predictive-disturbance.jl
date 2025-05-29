# Model Predictive Control for 1D System Disturbance Prediction
using LinearAlgebra
using SparseArrays
using OSQP
using Plots

# System matrices (1D system)
A = reshape([1.0], 1, 1)   # 1x1 matrix
B = reshape([1.0], 1, 1)
Bd = reshape([0.5], 1, 1)
nx, nu = size(B)

# MPC parameters
Q = I(nx) * 1.0
R = I(nu) * 0.1
N = 10        # Prediction horizon
Tf = 30       # Total time steps

# Constraints
umin, umax = -1.0, 1.0
xmin, xmax = -10.0, 10.0  # (unused here, but could be added)

# Initial state
x0 = [0.0]

# Known disturbances for all time steps + horizon
d_all = [0.2 * sin(0.2 * k) for k in 0:(Tf + N)]

# Preallocate state and control trajectories
X = zeros(nx, Tf + 1)
U = zeros(nu, Tf)
X[:, 1] .= x0

# Build prediction matrices function
function build_prediction_matrices(A, B, Bd, N)
    nx, nu = size(B)
    A_bar = zeros(nx * N, nx)
    B_bar = zeros(nx * N, nu * N)
    Bd_bar = zeros(nx * N, N)

    for i in 1:N
        A_power = A^i
        A_bar[(nx*(i-1)+1):(nx*i), :] .= A_power
        for j in 1:i
            AB = A^(i-j) * B
            ADBd = A^(i-j) * Bd
            B_bar[(nx*(i-1)+1):(nx*i), (nu*(j-1)+1):(nu*j)] .= AB
            Bd_bar[(nx*(i-1)+1):(nx*i), j] .= ADBd
        end
    end

    return A_bar, B_bar, Bd_bar
end

# Cost matrices for horizon
Q_bar = kron(I(N), Q)
R_bar = kron(I(N), R)

# Identity for input constraints (box constraints)
A_constr = sparse(I, N, N)

# Input constraints bounds
u_lower = fill(umin, N)
u_upper = fill(umax, N)

# MPC loop
for t in 1:Tf
    x_t = X[:, t]
    d_horizon = d_all[t+1 : t+N]

    # Build prediction matrices for this horizon
    A_bar, B_bar, Bd_bar = build_prediction_matrices(A, B, Bd, N)

    # Predicted disturbance vector
    d_vec = vcat(d_horizon...)

    # Predicted reference trajectory part (state propagation without control)
    x_ref = A_bar * x_t .+ Bd_bar * d_vec

    # Quadratic cost terms
    f = 2.0 * B_bar' * Q_bar * x_ref
    P = B_bar' * Q_bar * B_bar + R_bar
    P = (P + P') / 2  # Make sure P is symmetric

    # Setup and solve QP with OSQP
    model = OSQP.Model()
    OSQP.setup!(model; P = sparse(P), q = f, A = A_constr, l = u_lower, u = u_upper, verbose = false)
    results = OSQP.solve!(model)
    u_opt = results.x
    println("Applying control input: ", u_opt)

    # Apply first control input only
    u_apply = u_opt[1]
    U[:, t] .= u_apply

    # System forward step
    x_next = A * x_t + B * u_apply + Bd * d_all[t+1]
    X[:, t+1] = x_next
end

# Plot results
time = 0:(Tf-1)
plot(time, U[1, :], label="Control input u[k]", xlabel="Time step", ylabel="u", legend=:topright)
plot!(time, d_all[1:Tf], label="Disturbance d[k]", linestyle=:dash)

plot(0:Tf, X[1, :], label="State x[k]", xlabel="Time step", ylabel="x")

savefig("mpc_disturbance_prediction.png")
