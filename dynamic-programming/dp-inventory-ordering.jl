#=
Dynamic Programming, offline for inventory ordering
=#

#=
Credits: https://stanford.edu/class/ee365/lectures/coding_dp.pdf
Repeating MATLAB-inspired pseudocode in Julia
=#
# An inventory has stock value and a stochastic (but cateogircally distributed) demand
# We use DP to compute ordering policy

using LinearAlgebra
using Distributions
using Random
using Plots

# Constants
C = 6          # Capacity
T = 50         # Horizon
x₀ = 6         # Initial state
s = 0.1       # Holding cost coefficient
o = 1.0        # Fixed cost for ordering
X = 0:C        # Possible inventory levels
U = 0:C        # Possible controls
W = [0, 1, 2]  # Demand realizations
wdist = [0.7, 0.2, 0.1]  # Demand distribution

n = length(X)  # # of states
m = length(U)  # # of controls
p = length(W)  # # of disturbances

# Transition dynamics: f[i, j, k] = next_state_index
f = zeros(Int, n, m, p)
for i in 1:n, j in 1:m, k in 1:p
    x = X[i]
    u = U[j]
    w = W[k]
    next_x = clamp(x - w + u, 0, C)
    f[i, j, k] = next_x + 1  # +1 because Julia arrays are 1-indexed
end

# Stage costs: g[i, j] = cost at x_i using u_j
g = zeros(n, m)
for i in 1:n, j in 1:m
    x = X[i]
    u = U[j]
    if 2 - x <= u <= C
        g[i, j] = s * x + (u > 0 ? o : 0)
    else
        g[i, j] = Inf  # infeasible
    end
end

# Final cost is zero
g_final = zeros(n)

function value_iteration(f, g, g_final, wdist, T)
    n, m, p = size(f)
    V = zeros(n, T+1)
    V[:, end] .= g_final
    μ = zeros(Int, n, T)  # optimal control per state per time

    for t in T:-1:1
        for i in 1:n
            costs = fill(Inf, m)
            for j in 1:m
                stage = g[i, j]
                expected = sum(wdist[k] * V[f[i, j, k], t+1] for k in 1:p)
                costs[j] = stage + expected
            end
            V[i, t], μ[i, t] = findmin(costs)
        end
    end
    return μ, V
end

function cloop(μ, f, wdist, x₀, T)
    x_traj = zeros(Int, T+1)
    u_traj = zeros(Int, T)
    cost = 0.0
    x_traj[1] = x₀ + 1  # 1-based indexing

    for t in 1:T
        i = x_traj[t]
        u = μ[i, t]
        u_traj[t] = u
        w = rand(Categorical(wdist))  # draw w ∈ {1,2,3}
        x_next = f[i, u+1, w]  # u+1 for 1-based control
        cost += g[i, u+1]
        x_traj[t+1] = x_next
    end

    return x_traj .- 1, u_traj, cost  # convert back to 0-based state
end

function ftop(f, wdist)
    n, m, p = size(f)
    P = zeros(n, n, m)  # P[i,j,u] = Pr(x_{t+1}=j | x_t=i, u)

    for i in 1:n, u in 1:m
        for k in 1:p
            j = f[i, u, k]
            P[i, j, u] += wdist[k]
        end
    end
    return P
end

function plot_inventory(x_traj, u_traj)
    T = length(u_traj)
    time = 0:T

    p1 = plot(time, x_traj, label="Inventory Level", xlabel="Time", ylabel="xₜ", title="Inventory Level Over Time", lw=2)
    p2 = plot(1:T, u_traj, label="Order Quantity", xlabel="Time", ylabel="uₜ", title="Order Quantity Over Time", lw=2)

    plot(p1, p2, layout=(2, 1), size=(800, 500))
end

μ, V = value_iteration(f, g, g_final, wdist, T)
x_traj, u_traj, total_cost = cloop(μ, f, wdist, x₀, T)
println("Total cost: ", total_cost)
plot_inventory(x_traj, u_traj)
savefig("dp_inventory_policy.png")