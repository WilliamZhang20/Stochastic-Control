using Random, Plots, Statistics

function diffusion_walk(grid_size::Int, steps::Int, n_walkers::Int, start_pos::Tuple{Int,Int}, obstacle_mask::BitMatrix)
    visits = zeros(Float64, grid_size, grid_size)

    for _ in 1:n_walkers
        pos = start_pos
        for _ in 1:steps
            visits[pos...] += 1.0

            # Possible directions
            neighbors = [
                (pos[1]-1, pos[2]),
                (pos[1]+1, pos[2]),
                (pos[1], pos[2]-1),
                (pos[1], pos[2]+1),
            ]

            # Filter valid moves
            valid = [(i,j) for (i,j) in neighbors if 
                1 ≤ i ≤ grid_size && 
                1 ≤ j ≤ grid_size && 
                !obstacle_mask[i,j]]

            # Stay in place if blocked
            pos = isempty(valid) ? pos : rand(valid)
        end
    end

    return visits ./ sum(visits)
end

function plot_smooth_surface(probabilities::Matrix{Float64})
    # Apply smoothing visually
    smooth_vals = sqrt.(probabilities)  # or use log.(probabilities .+ eps())
    surface(smooth_vals, xlabel="X", ylabel="Y", zlabel="√Visit Probability", c=:viridis)
end

# Parameters
grid_size = 30
steps = 200  # Increased step count
n_walkers = 50_000  # More walkers = more smoothness
start_pos = (15, 15)

# Obstacle far from start
obstacle_mask = falses(grid_size, grid_size)
obstacle_mask[5:7, 5:7] .= true  # Move obstacle away

# Run and plot
visit_probs = diffusion_walk(grid_size, steps, n_walkers, start_pos, obstacle_mask)
plot_smooth_surface(visit_probs)
savefig("diffusion_smooth_surface.png")
