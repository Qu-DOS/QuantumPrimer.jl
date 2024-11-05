export haar_random_unitary,
       wasserstein_distance,
       KL_divergence,
       skewness,
       density_matrix_from_vector,
       register_from_vector,
       pauli_decomposition

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

"""
    density_matrix_from_vector(states::Vector{Vector{Int}}; coeffs=nothing::Union{Nothing, Vector{Float64}})

Create a density matrix from a vector of integer vectors. Each integer vector is converted to an `ArrayReg` using `bit_literal`.

# Arguments
- `states::Vector{Vector{Int}}`: A vector of integer vectors to be included in the density matrix.
- `coeffs::Union{Nothing, Vector{Float64}}`: Optional coefficients for the states. If provided, the coefficients are used in the density matrix calculation.

# Returns
- A density matrix created from the given integer vectors and coefficients.
"""
function density_matrix_from_vector(states::Vector{Vector{Int}}; coeffs=nothing::Union{Nothing, Vector{Float64}})
    register = []
    for ele in states
        push!(register, ArrayReg(bit_literal(ele...)))
    end
    if isnothing(coeffs)
        return density_matrix(sum(register) |> normalize!)
    else
        return density_matrix(sum(register.*coeffs) |> normalize!)
    end
end

"""
    register_from_vector(states::Vector{Vector{Int}}; coeffs=nothing::Union{Nothing, Vector{Float64}})

Create a quantum register from a vector of integer vectors. Each integer vector is converted to an `ArrayReg` using `bit_literal`.

# Arguments
- `states::Vector{Vector{Int}}`: A vector of integer vectors to be included in the register.
- `coeffs::Union{Nothing, Vector{Float64}}`: Optional coefficients for the states. If provided, the coefficients are normalized.

# Returns
- A quantum register created from the given integer vectors and coefficients.
"""
function register_from_vector(states::Vector{Vector{Int}}; coeffs=nothing::Union{Nothing, Vector{Float64}})
    register = []
    for ele in states
        push!(register, ArrayReg(bit_literal(ele...)))
    end
    if isnothing(coeffs)
        return sum(register./sqrt(length(states)))
    else
        coeffs /= sqrt(sum(coeffs.^2))
        return sum(register.*coeffs)
    end
end

"""
    pauli_decomposition(matrix::Union{ChainBlock, Matrix{ComplexF64}, Matrix{T}} where T<:Real)

Decompose a matrix into a dictionary of Pauli strings.

# Arguments
- `matrix::Union{ChainBlock, Matrix{ComplexF64}, Matrix{T}} where T<:Real`: The input matrix to be decomposed. The matrix must be square with dimensions 2^n x 2^n.

# Returns
- A dictionary where the keys are vectors of integers representing Pauli strings, and the values are the corresponding coefficients.
"""
function pauli_decomposition(matrix::Union{ChainBlock, Matrix{ComplexF64}, Matrix{T}} where T<:Real)
    n = trunc(Int, log2(size(matrix, 1)))
    if n != log2(size(matrix, 1))
        throw(ArgumentError("Matrix must be square and have dimensions 2^n x 2^n"))
    end
    Paulis = [0.5*I2, 0.5*X, 0.5*Y, 0.5*Z]
    repeated_vector = repeat([0, 1, 2, 3], outer=n)
    res = Dict{Vector{Int}, Float64}()
    matrix = Matrix(matrix)
    C = similar(matrix)
    for i in multiset_permutations(repeated_vector, n)
        P = [Paulis[i[j]+1] for j=1:n]
        mul!(C, Matrix(kron(P...)), matrix)
        coefficient = tr(C)
        coefficient !=0 ? res[i] = real(coefficient) : nothing
    end
    return res
end