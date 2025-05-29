# Linear Quadratic Regulator
# On simple 2x2 double integrator

using LinearAlgebra
using Plots
using Distributions

# System Dynamics
A = [0.0 1.0;
     0.0 0.0]
B = [0.0; 1.0]
B = reshape(B, 2, 1)

# Cost Matrices
Q = [10.0 0.0;
     0.0 1.0]
R = [1.0;;]

# Infinite horizon case
function solve_care(A, B, Q, R)
    n = size(A, 1)
    H = [A -B * inv(R) * B'; -Q -A']  # build the Hamiltonian matrix
    eig_decomp = eigen(H)
    eigvals = eig_decomp.values
    eigvecs = eig_decomp.vectors

    # Select stable eigenvalues (real part < 0)
    idxs = findall(x -> real(x) < 0.0, eigvals)
    V = eigvecs[:, idxs]

    # Partition V into upper and lower blocks
    V1 = V[1:n, :]
    V2 = V[n+1:end, :]

    # Solve for P (X in your case)
    X = real(V2 * inv(V1))
    return X
end

X = solve_care(A, B, Q, R)
K = inv(R) * B' * X

println("LQR Gain K: ", K)

# Simulation parameters
dt = 0.01
T = 5.0
N = Int(T / dt)
x = zeros(2, N)
x[:,1] = [1.0, 0.0]  # Initial state

# Noise: Gaussian process noise w ~ N(0, W)
W = 0.01 * I(2)  # small noise covariance
noise_dist = MvNormal([0.0, 0.0], W)

# Simulate LQR control
for i in 1:N-1
    u = -K * x[:,i]
    w = rand(noise_dist)
    dx = A * x[:,i] + B * u + w
    x[:,i+1] = x[:,i] + dt * dx
end

# Plot results
time = range(0, step=dt, length=N)
p = plot(time, x[1,:], label="Position", xlabel="Time (s)", ylabel="State", title="LQR Control")
plot!(p, time, x[2,:], label="Velocity")
savefig("lqr_double_integrator.png")
