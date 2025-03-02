export haar_random_unitary,
       wasserstein_distance,
       KL_divergence,
       density_matrix_from_vector,
       register_from_vector,
       pauli_decomposition,
       pauli_decomposition_kronecker,
       save_circuit_to_txt,
       normalize_vector,
       prepare_dos_encoded_state

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

"""
    pauli_decomposition_kronecker(matrix::Union{ChainBlock, Matrix{ComplexF64}, Matrix{T}} where T<:Real)

Decompose a matrix into a dictionary of Pauli strings using the package Kronecker.

# Arguments
- `matrix::Union{ChainBlock, Matrix{ComplexF64}, Matrix{T}} where T<:Real`: The input matrix to be decomposed. The matrix must be square with dimensions 2^n x 2^n.

# Returns
- A dictionary where the keys are vectors of integers representing Pauli strings, and the values are the corresponding coefficients.
"""
function pauli_decomposition_kronecker(matrix::Union{ChainBlock, Matrix{ComplexF64}, Matrix{T}} where T<:Real)
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
        P = Kronecker.kronecker([Matrix(Paulis[i[j]+1]) for j=1:n]...)
        mul!(C, P, matrix)
        coefficient = tr(C)
        if coefficient != 0
            res[i] = real(coefficient)
        end
    end
    return res
end

"""
    save_circuit_to_txt(circuit::AbstractBlock; filename::String="output.txt")

Save a quantum circuit to a text file as a console output.

# Arguments
- `circuit::AbstractBlock`: The quantum circuit to be saved.
- `filename::String="yao_circuit.txt"`: The name of the file to save the circuit to. Defaults to "yao_circuit.txt".

# Returns
- Nothing. The function writes the circuit to the specified file.
"""
function save_circuit_to_txt(circuit::AbstractBlock; filename::String="yao_circuit.txt")
    open(filename, "w") do io
        redirect_stdout(io) do
            println(circuit)
        end
    end
end

"""
    normalize_vector(vector::Vector{T} where T <: Number) -> Vector{T}

Normalize a vector by dividing each element by the vector's Euclidean norm.

# Arguments
- `vector::Vector{T} where T <: Number`: The input vector to be normalized.

# Returns
- `Vector{T}`: The normalized vector.

# Example
```julia
v = [3.0, 4.0]
normalize_vector(v)  # returns [0.6, 0.8]
"""
function normalize_vector(vector::Vector{T} where T <: Number)
    return vector / sqrt(sum(abs2, vector))
end

"""
    prepare_dos_encoded_state(unitary::Union{AbstractMatrix, AbstractBlock}, n_times::Int; nshots::Int=1000, extended_output::Bool=false, focussed_output::Bool=true)

Prepare a state for the DOS algorithm by encoding a unitary matrix into a quantum state.

# Arguments
- `unitary::Union{AbstractMatrix, AbstractBlock}`: The unitary matrix to be encoded.
- `n_times::Int`: The number of times to apply the unitary matrix.
- `nshots::Int=1000`: The number of shots to use for the measurement.
- `extended_output::Bool=false`: If `true`, the function returns the state and the measurement counts. If `false`, the function returns only the state.
- `focussed_output::Bool=true`: If `true`, the function returns the state with the ancilla qubits removed. If `false`, the function returns the full state.

# Returns
- If `extended_output` is `true`, the function returns a tuple `(dos_state, counts_register_after_qpe)`. If `extended_output` is `false`, the function returns only `dos_state`.
"""
function prepare_dos_encoded_state(unitary::Union{AbstractMatrix, AbstractBlock}, n_times::Int; nshots::Int=1000, extended_output::Bool=false, focussed_output::Bool=true)
     if typeof(unitary) <: AbstractBlock
        unitary = Matrix(unitary)
    end
    n_qubits = Int(log2(size(unitary, 1)))
    if n_qubits != log2(size(unitary, 1))
        throw(ArgumentError("The size of the unitary matrix must be a power of 2."))
    end
    all_n = n_times + 2n_qubits # include the ancilla qubits for purification
    initial_state = zero_state(all_n)
    initial_state |> circ_append_ancillas(circ_purified_maximally_mixed(n_qubits), n_times; pos_ancillas=:top) |> circ_append_ancillas(circ_qpe(n_times, unitary; reverse_qubits=false), n_qubits) |> chain(all_n, put(1:n_times => circ_reverse_order(n_times))) # reverse the readout if using Yao convention
    meas_register_after_qpe = measure(initial_state, 1:n_times; nshots=nshots)
    counts_register_after_qpe = [count(x->x==i, meas_register_after_qpe) for i in 0:2^n_times-1] / nshots
    dos_state = focussed_output ? focus!(initial_state, 1:n_times) : initial_state
    if extended_output
        return dos_state, counts_register_after_qpe
    else
        return dos_state
    end
end