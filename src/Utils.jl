export haar_random_unitary,
       circ_haar_random_unitary,
       wasserstein_distance,
       KL_divergence,
       skewness

"""
    haar_random_unitary(n::Int) -> Matrix{ComplexF64}

Generates a Haar-random unitary matrix of size `n x n`.

# Arguments
- `n::Int`: The size of the unitary matrix.

# Returns
- `Matrix{ComplexF64}`: The Haar-random unitary matrix.
"""
function haar_random_unitary(n::Int)
    U = randn(ComplexF64, n, n) + im*randn(ComplexF64, n, n)
    Z = qr(U)
    D = diagm(0 => diag(Z.R)./abs.(diag(Z.R)))
    return Z.Q * D
end

"""
    circ_haar_random_unitary(n::Int) -> ChainBlock

Generates a quantum circuit with a Haar-random unitary matrix of size `2^n x 2^n`.

# Arguments
- `n::Int`: The number of qubits.

# Returns
- `ChainBlock`: The quantum circuit with the Haar-random unitary matrix.
"""
function circ_haar_random_unitary(n::Int)
    U = haar_random_unitary(2^n)
    circ = chain(n)
    push!(circ, matblock(U))
    return circ
end

"""
    wasserstein_distance(p::Vector{Float64}, q::Vector{Float64}) -> Float64

Computes the Wasserstein distance between two probability distributions.

# Arguments
- `p::Vector{Float64}`: The first probability distribution.
- `q::Vector{Float64}`: The second probability distribution.

# Returns
- `Float64`: The Wasserstein distance.

# Throws
- An error if the input vectors are not valid probability distributions.
"""
function wasserstein_distance(p::Vector{Float64}, q::Vector{Float64})
    if abs(sum(p) - 1.0) > 1e-6 || abs(sum(q) - 1.0) > 1e-6
        error("Input vectors must be valid probability distributions.")
    end
    cdf_p = cumsum(p)
    cdf_q = cumsum(q)
    distance = sum(abs.(cdf_p .- cdf_q))
    return distance
end

"""
    KL_divergence(p::Vector{Float64}, q::Vector{Float64}) -> Float64

Computes the Kullback-Leibler (KL) divergence between two probability distributions.

# Arguments
- `p::Vector{Float64}`: The first probability distribution.
- `q::Vector{Float64}`: The second probability distribution.

# Returns
- `Float64`: The KL divergence.

# Throws
- An error if the input vectors are not valid probability distributions or if `p` and `q` have different lengths.
- An error if `p` is non-zero where `q` is zero.
"""
function KL_divergence(p::Vector{Float64}, q::Vector{Float64})
    if abs(sum(p) - 1.0) > 1e-6 || abs(sum(q) - 1.0) > 1e-6
        error("Input vectors must be valid probability distributions.")
    end
    length(p) == length(q) || error("Input vectors must have the same length.")
    for i in eachindex(p)
        if q[i] == 0 && p[i] != 0
            error("p must be 0 whenever q is 0 for the KL divergence to be well-defined.")
        end
    end
    return sum(p .* log.(p ./ q))
end

"""
    skewness(vec::Vector{Float64}) -> Float64

Computes the skewness of a given vector.

# Arguments
- `vec::Vector{Float64}`: The input vector.

# Returns
- `Float64`: The skewness of the vector.
"""
function skewness(vec::Vector{Float64})
    n = length(vec)
    mean_v = mean(vec)
    std_v = std(vec)
    skewness = (n / ((n-1)*(n-2))) * sum(((x - mean_v) / std_v)^3 for x in vec)
    return skewness
end