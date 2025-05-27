function bellman_ford(num_vertices::Int, edges::Vector{Tuple{Int, Int, Float64}}, source::Int)
    # Init huge distances
    dist = fill(Inf, num_vertices)
    dist[source] = 0.0

    # Repeat relaxation of edges
    for i in 1:(num_vertices-1)
        for (u, v, w) in edges
            # Minimizing dist[v] from nodes before it
            # We want argmin_(j)(dist[j] + w_ij) - j is a node connected to i (node of interest)
            # Requiring the iteration through all 'subproblems' => DP
            if dist[u] + w < dist[v] 
                dist[v] = dist[u] + w
            end
        end
    end

    for (u, v, w) in edges
        if dist[u] + w < dist[v]
            error("Graph contains a negative-weight cycle")
        end
    end

    return dist

end

num_vertices = 5

# Edges represented as (source, target, weight)
edges = [
    (1, 2, 6.0),
    (1, 3, 7.0),
    (2, 3, 8.0),
    (2, 4, 5.0),
    (2, 5, -4.0),
    (3, 4, -3.0),
    (3, 5, 9.0),
    (4, 2, -2.0),
    (5, 1, 2.0),
    (5, 4, 7.0)
]

# Source vertex
source = 1

# Run Bellman-Ford
distances = bellman_ford(num_vertices, edges, source)

println("Shortest distances from node $source:")
for i in 1:num_vertices
    println("To node $i: $(distances[i])")
end