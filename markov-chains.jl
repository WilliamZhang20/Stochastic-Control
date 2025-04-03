using Random, Plots, Distributions

function gen_random_markov_matrix(n::Int)
    P = rand(n, n)
    P ./= sum(P, dims=2) # vectorized normalization
    return P
end

function gen_upper_triangular_markov_matrix(n::Int)
    P = rand(n, n)
    for i in 1:n
        for j in 1:i-1
            P[i, j] = 0  # Set lower triangular part to zero
        end
    end
    P ./= sum(P, dims=2) .+ eps()  # Normalize rows to sum to 1, prevent division by 0
    return P
end

# Run markov chain state transitions
function sim_markov_chain(P::Matrix{Float64}, steps::Int, start_state::Int)
    n = size(P, 1) # state vector size
    states = zeros(Int, steps)
    states[1] = start_state

    for t in 2:steps
        states[t] = rand(Categorical(P[states[t-1], :])) # sample next state
    end
    return states
end

# Plot state transitions
function plot_markov_chain(states::Vector{Int})
    plot(states, xlabel="Time Step", ylabel="State", marker=:circle, label="State Transitions")
    savefig("markov_chain_plot.png")
end

# Defining a Markov Chain
n_states = 5
steps = 50
start_state = 1
P = gen_random_markov_matrix(n_states)

states = sim_markov_chain(P, steps, start_state)

plot_markov_chain(states)