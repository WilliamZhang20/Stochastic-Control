#=
Dynamic Programming for Optimal Control
=#

# Concept: https://stanford.edu/class/ee365/lectures/dp.pdf
# https://stanford.edu/class/ee365/lectures/coding_dp.pdf

using LinearAlgebra, Plots

# State space: x ∈ {-5, -4, ..., 5}
states = collect(-5:5)
n_states = length(states)

goal = 5   # Target state
actions = [-1, 0, 1]  # Control inputs

# Discount factor
gamma = 0.9

# 1. Compute Value Function using Dynamic Programming
function compute_value_function()
    V = fill(Inf, n_states)
    V[end] = 0  # Goal state has zero cost
    
    threshold = 1e-3
    converged = false
    while !converged
        V_new = copy(V)
        for i in 1:n_states-1 
            # Apply bellman equation
            costs = [1 + gamma * V[clamp(i + u, 1, n_states)] for u in actions]
            V_new[i] = minimum(costs)
        end
        converged = maximum(abs.(V_new - V)) < threshold
        V = V_new
    end
    return V
end

# 2. Compute Optimal Policy
function optimal_policy(V)
    policy = Dict()
    for i in 1:n_states-1  # Ignore goal
        best_action = argmin([1 + gamma * V[clamp(i + u, 1, n_states)] for u in actions])
        policy[states[i]] = actions[best_action]
    end
    return policy
end

# 3. Closed-Loop Feedback Control Simulation
function feedback_control_loop(policy, start_state)
    trajectory = [start_state]
    while trajectory[end] != goal
        u = policy[trajectory[end]]
        new_state = clamp(trajectory[end] + u, -5, 5)
        push!(trajectory, new_state)
    end
    return trajectory
end

# Compute Value Function and Policy
V = compute_value_function()
policy = optimal_policy(V)

# Simulate the Control Loop
start_state = -5
trajectory = feedback_control_loop(policy, start_state)

# Plot Results
plot(states, V, label="Value Function", xlabel="State", ylabel="Cost-to-Go", lw=2)
scatter!(trajectory, [V[findfirst(==(s), states)] for s in trajectory], markershape=:circle, label="Trajectory")
savefig("dp_feedback_control.png")
