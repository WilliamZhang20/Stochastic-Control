# Stochastic Shortest Path Optimization
# Applied to a trading example

using Random
using Plots

const PRICE_LEVELS = 0:4
const ACTIONS = [:hold, :sel]
const TERMINAL = :TERMINAL
const HORIZON = 10

function transition(price::Int, action::Symbol) 
    if action == :sell
        return TERMINAL, price # go to terminal
    else action == :hold
        # Symmetric Gaussian process
        delta = rand() < 0.5 ? -1 : 1
        new_price = clamp(price + delta, minimum(PRICE_LEVELS), maximum(PRICE_LEVELS))
        return new_price, 0.0
    end
end

function ssp_value_iteration() 
    V = Dict{Any, Vector{Float64}}()
    Pi = Dict{Any, Vector{Symbol}}()

    for s in PRICE_LEVELS
        V[s] = zeros(HORIZON + 1)
        Pi[s] = fill(:hold, HORIZON + 1)
    end

    V[TERMINAL] = zeros(HORIZON + 1)

    # Backwards iteration
    for t = HORIZON:-1:1
        for s in PRICE_LEVELS
            # Evaluate various actions
            values = Dict{Symbol, Float64}()
            # Action: Sell
            values[:sell] = s + V[TERMINAL][t+1]
            # Action: Hold (expected value over price up/down)
            up = clamp(s+1, minimum(PRICE_LEVELS), maximum(PRICE_LEVELS))
            down = clamp(s-1, minimum(PRICE_LEVELS), maximum(PRICE_LEVELS))
            values[:hold] = 0.5 * V[up][t+1] + 0.5 * V[down][t+1]

            best_a = argmax(values)
            V[s][t] = values[best_a]
            Pi[s][t] = best_a
        end
    end
    return V, Pi
end

V, Pi = ssp_value_iteration()

println("Optimal Policy (pi):")
for t in 1:HORIZON
    println("Time $t:")
    for s in PRICE_LEVELS
        print("  Price $s → $(Pi[s][t])\n")
    end
end

# Simulate policy from initial state
function simulate_policy(Pi, s_o::Int)
    s = s_o
    trajectory = Tuple{Int, Any}[(0, s)]
    for t in 1:HORIZON
        a = Pi[s][t]
        sPrime, r = transition(s, a)
        push!(trajectory, (t, sPrime))
        if sPrime == TERMINAL
            println("Sold at time $t for price $s")
            break
        end
        s = sPrime
    end
    return trajectory
end

# Simulate and plot
Random.seed!(42)
trajectory = simulate_policy(Pi, 2)

times = [p[1] for p in trajectory if p[2] isa Int]
prices = [p[2] for p in trajectory if p[2] isa Int]

filtered_times = []
filtered_prices = []
for (t, p) in trajectory
    if p isa Int
        push!(filtered_times, t)
        push!(filtered_prices, p)
    end
end

plot(times, prices, label="Price trajectory", marker=:circle,
           xlabel="Time", ylabel="Price", ylim=(-0.5, 4.5),
           title="Trading Policy Execution")
savefig("stochastic_sp_trader.png")