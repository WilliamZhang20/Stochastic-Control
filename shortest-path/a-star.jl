# Coding the A-Star Algorithm
# ... on a grid!

using Random, Plots, DataStructures

function manhattan_distance(p1, p2)
    return sum(abs.(p1 .- p2)) # vectorized to all dims
end

function generate_grid(rows, cols, obstacle_prob=0.2)
    grid = rand(rows, cols) .> obstacle_prob # broadcast probability
    return grid
end

# A* Algorithm for Path between two vertices
# https://stanford.edu/class/ee365/lectures/astar.pdf
function a_star(grid, start, goal)
    rows, cols = size(grid)
    open_set = PriorityQueue{Tuple{Int, Int}, Int}()
    push!(open_set, start => 0)

    # The Closed Set => wrapped by the open_set
    dist = Dict{Tuple{Int, Int}, Int}(start => 0) 
    closed = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    directions = [(0,1), (1,0), (0,-1), (-1,0)]

    while !isempty(open_set)
        current = dequeue!(open_set)

        if current == goal
            println("Found Destination")
            # terminate & trace path
            path = []
            while current in keys(closed)
                pushfirst!(path, current)
                current = closed[current]
            end
            pushfirst!(path, start)
            plot_grid(grid, closed, path, open_set, start, goal)
            return path
        end

        for (dx, dy) in directions
            neighbor = (current[1] + dx, current[2] + dy)
            if 1 <= neighbor[1] <= rows && 1 <= neighbor[2] <= cols && grid[neighbor...]
                new_cost = dist[current] + 1
                if get(dist, neighbor, Inf) > new_cost
                    closed[neighbor] = current
                    dist[neighbor] = new_cost
                    push!(open_set, neighbor => new_cost + manhattan_distance(neighbor, goal))
                end
            end
        end
    end
    return nothing
end

# Plot grid + discovered path
function plot_grid(grid, came_from, path, open_set, start, goal)
    rows, cols = size(grid)
    img = ones(rows, cols) # White background
    
    # Obstacles
    for r in 1:rows, c in 1:cols
        if !grid[r, c]
            img[r, c] = 0 # Black for obstacles
        end
    end
    
    # Visited nodes (came_from)
    for (node, _) in came_from
        img[node...] = 0.2 # Blue
    end
    
    # Priority queue nodes
    for node_pair in open_set # Corrected line
        node = node_pair.first # Extract the node coordinates
        img[node...] = 0.8 # Yellow
    end
    
    # Path
    for node in path
        img[node...] = 0.67 # Red
    end
    
    img[start...] = 0.5 # Start (gray)
    img[goal...] = 0.7 # Goal (lighter gray)
    
    heatmap(img, c=:viridis, axis=false, legend=false)
    savefig("a_star_path.png")
end

# World Definition
rows, cols = 30, 30

start, goal = (3, 5), (25, 17)

grid = generate_grid(rows, cols)
grid[start...] = true
grid[goal...] = true

println("Determining path via A* Algorithm")
path = a_star(grid, start, goal)
println("All done")