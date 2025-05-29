#-----
# Discrete-Time LQR
#-----
using LinearAlgebra
using Plots

function finite_horizon_lqr(A, B, Q, R, Qf, N)
    P = Vector{Matrix{Float64}}(undef, N+1)  # Riccati matrices
    K = Vector{Matrix{Float64}}(undef, N)    # Feedback gains

    P[N+1] = Qf  # Terminal cost

    for t = N:-1:1
        Pt = P[t+1]
        BtP = B' * Pt
        gain_vec = (R + BtP * B) \ (BtP * A)   # This is a (2, 1) vector (column)
        K[t] = gain_vec'  # transpose to (1, 2) row vector
        P[t] = Q + A' * Pt * A - A' * Pt * B * K[t]
    end

    return K, P
end

function dare(A, B, Q, R; tol=1e-8, maxiter=1000)
    """Solves Discrete-Time Algebraic Riccati Equation"""
    """Using method of iteration rather than recursion"""
    """KEY: this is INFINITE HORIZON EXAMPLE"""
    P = copy(Q)
    for i in 1:maxiter
        K = (R + B' * P * B) \ (B' * P * A)
        P_new = Q + A' * P * A - A' * P * B * K
        if norm(P_new - P) < tol
            break
        end
        P = P_new
    end
    return P
end

function dlqr(A, B, Q, R)
    """Returns the optimal gain matrix K"""
    P = dare(A, B, Q, R)
    K = (R + B' * P * B) \ (B' * P * A)
    return K, P
end

# ----------------------------
# Simulate closed-loop system
# ----------------------------

function simulate_infinite_horizon_lqr(A, B, Q, R)
    K, P = dlqr(A, B, Q, R)
    println("Optimal K = $K")
    println("Dimensions of K: ", size(K))

    x = [5.0; 0.0]
    trajectory = []

    for t in 1:30
        u = -K * x
        x = A * x + B * u
        push!(trajectory, x)
    end

    println("Final state: ", x)
    xs = hcat(trajectory...)
    plot(xs', xlabel="Time step", ylabel="State", label=["Position" "Velocity"], lw=2)
    savefig("Discrete-Time-Inf-Horiz.png")
end


"""
Exploring finite horizon alternative
"""

function simulate_finite_horizon_lqr(A, B, Q, R)
    Qf = Diagonal([10.0, 10.0])
    N = 30
    Kseq, _ = finite_horizon_lqr(A, B, Q, R, Qf, N)
    println("Optimal K = $Kseq")
    println("Dimensions of K: ", size(Kseq))

    x = [5.0; 0.0]
    trajectory = [x]
    controls = []

    for t in 1:N
        u = -Kseq[t] * x
        x = A * x + B * u
        push!(trajectory, x)
        push!(controls, u)
    end

    xs = hcat(trajectory...)
    us = hcat(controls...)
    plot(xs', label=["position" "velocity"], xlabel="Time", ylabel="State", lw=2)
    plot!(us', label="control", xlabel="Time", ylabel="u", lw=2)
    savefig("Discrete-Time-Finite-Horiz.png")
end

function main()
    A = [1.0 1.0; 0.0 1.0]
    B = [0.0; 1.0]
    Q = Diagonal([10.0, 1.0])
    R = 1.0

    simulate_infinite_horizon_lqr(A, B, Q, R)
    # simulate_finite_horizon_lqr(A, B, Q, R)
end

main()