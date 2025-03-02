export circ_gate_single,
       circ_gate_n,
       circ_gate_sum,
       circ_gate_where,
       circ_block_single,
       circ_block_n,
       circ_block_sum,
       circ_block_where,
       circ_X,
       circ_Y,
       circ_Z,
       circ_X_where,
       circ_Y_where,
       circ_Z_where,
       circ_Xn,
       circ_Yn,
       circ_Zn,
       circ_Xsum,
       circ_Ysum,
       circ_Zsum,
       circ_Hn,
       circ_CNOT_2regs,
       circ_Ry_simple_conv,
       circ_Ry_conv,
       circ_SU4_conv,
       circ_Rx_layer,
       circ_Ry_layer,
       circ_Rz_layer,
       circ_CNOT_layer,
       circ_CZ_layer,
       circ_HEA,
       circ_HEA_real,
       circ_phase_flip,
       circ_hypergraph_state,
       circ_swap_decomposed,
       circ_swap_all,
       circ_swap_test,
       circ_destructive_swap_test,
       circ_obs_times_swap,
       circ_LCU,
       circ_haar_random_unitary,
       circ_purified_maximally_mixed,
       circ_phase_encoding,
       circ_qft,
       circ_qpe,
       circ_append_ancillas,
       circ_reverse_order

"""
    circ_gate_single(n::Int, i::Int, gate::ConstantGate) -> ChainBlock

Creates a quantum circuit with a single gate applied to the `i`-th qubit.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the qubit to which the gate is applied.
- `gate::ConstantGate`: The gate to be applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_gate_single(n::Int, i::Int, gate::ConstantGate) = chain(n, put(i=>gate))

"""
    circ_gate_n(n::Int, gate::ConstantGate) -> ChainBlock

Creates a quantum circuit with the same gate applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `gate::ConstantGate`: The gate to be applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_gate_n(n::Int, gate::ConstantGate) = chain(n, put(i=>gate) for i=1:n)

"""
    circ_gate_sum(n::Int, gate::ConstantGate) -> ChainBlock

Creates a quantum circuit with the sum of the same gate applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `gate::ConstantGate`: The gate to be applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_gate_sum(n::Int, gate::ConstantGate) = sum(chain(n, put(i=>gate) for i=1:n))

"""
    circ_gate_where(n::Int, gate::ConstantGate, ones_where::Vector{Int}) -> ChainBlock

Creates a quantum circuit with the gate applied to specific qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `gate::ConstantGate`: The gate to be applied.
- `ones_where::Vector{Int}`: The indices of the qubits to which the gate is applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_gate_where(n::Int, gate::ConstantGate, ones_where::Vector{Int}) = chain(n, put(i=>gate) for i in ones_where)

"""
    circ_block_single(n::Int, i::Int, block::ChainBlock{2}) -> ChainBlock

Creates a quantum circuit with a single block applied to the `i`-th qubit.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the qubit to which the block is applied.
- `block::ChainBlock{2}`: The block to be applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_block_single(n::Int, i::Int, block::ChainBlock{2}) = chain(n, put(i=>block))

"""
    circ_block_n(n::Int, block::ChainBlock{2}) -> ChainBlock

Creates a quantum circuit with the same block applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `block::ChainBlock{2}`: The block to be applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_block_n(n::Int, block::ChainBlock{2}) = chain(n, put(i=>block) for i=1:n)

"""
    circ_block_sum(n::Int, block::ChainBlock{2}) -> ChainBlock

Creates a quantum circuit with the sum of the same block applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `block::ChainBlock{2}`: The block to be applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_block_sum(n::Int, block::ChainBlock{2}) = sum(chain(n, put(i=>block) for i=1:n))

"""
    circ_block_where(n::Int, block::ChainBlock{2}, ones_where::Vector{Int}) -> ChainBlock

Creates a quantum circuit with the block applied to specific qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `block::ChainBlock{2}`: The block to be applied.
- `ones_where::Vector{Int}`: The indices of the qubits to which the block is applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_block_where(n::Int, block::ChainBlock{2}, ones_where::Vector{Int}) = chain(n, put(i=>block) for i in ones_where)

"""
    circ_X(n::Int, i::Int) -> ChainBlock

Creates a quantum circuit with an X gate applied to the `i`-th qubit.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the qubit to which the X gate is applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_X(n::Int, i::Int) = circ_gate_single(n, i, X)

"""
    circ_X(n::Int) -> ChainBlock

Creates a quantum circuit with an X gate applied to the first qubit.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_X(n::Int) = circ_gate_single(n, 1, X)

"""
    circ_X_where(n::Int, ones_where::Vector{Int}) -> ChainBlock

Creates a quantum circuit with X gates applied to specific qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `ones_where::Vector{Int}`: The indices of the qubits to which the X gates are applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_X_where(n::Int, ones_where::Vector{Int}) = circ_gate_where(n, X, ones_where)

"""
    circ_Y(n::Int, i::Int) -> ChainBlock

Creates a quantum circuit with a Y gate applied to the `i`-th qubit.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the qubit to which the Y gate is applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Y(n::Int, i::Int) = circ_gate_single(n, i, Y)

"""
    circ_Y(n::Int) -> ChainBlock

Creates a quantum circuit with a Y gate applied to the first qubit.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Y(n::Int) = circ_gate_single(n, 1, Y)

"""
    circ_Y_where(n::Int, ones_where::Vector{Int}) -> ChainBlock

Creates a quantum circuit with Y gates applied to specific qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `ones_where::Vector{Int}`: The indices of the qubits to which the Y gates are applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Y_where(n::Int, ones_where::Vector{Int}) = circ_gate_where(n, Y, ones_where)

"""
    circ_Z(n::Int, i::Int) -> ChainBlock

Creates a quantum circuit with a Z gate applied to the `i`-th qubit.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the qubit to which the Z gate is applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Z(n::Int, i::Int) = circ_gate_single(n, i, Z)

"""
    circ_Z(n::Int) -> ChainBlock

Creates a quantum circuit with a Z gate applied to the first qubit.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Z(n::Int) = circ_gate_single(n, 1, Z)

"""
    circ_Z_where(n::Int, ones_where::Vector{Int}) -> ChainBlock

Creates a quantum circuit with Z gates applied to specific qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `ones_where::Vector{Int}`: The indices of the qubits to which the Z gates are applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Z_where(n::Int, ones_where::Vector{Int}) = circ_gate_where(n, Z, ones_where)

"""
    circ_Xn(n::Int) -> ChainBlock

Creates a quantum circuit with X gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Xn(n::Int) = chain(n, put(i=>X) for i=1:n)

"""
    circ_Yn(n::Int) -> ChainBlock

Creates a quantum circuit with Y gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Yn(n::Int) = chain(n, put(i=>Y) for i=1:n)

"""
    circ_Zn(n::Int) -> ChainBlock

Creates a quantum circuit with Z gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Zn(n::Int) = chain(n, put(i=>Z) for i=1:n)

"""
    circ_Xsum(n::Int) -> ChainBlock

Creates a quantum circuit with the sum of X gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Xsum(n::Int) = sum(chain(n, put(i=>X) for i=1:n))

"""
    circ_Ysum(n::Int) -> ChainBlock

Creates a quantum circuit with the sum of Y gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Ysum(n::Int) = sum(chain(n, put(i=>Y) for i=1:n))

"""
    circ_Zsum(n::Int) -> ChainBlock

Creates a quantum circuit with the sum of Z gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Zsum(n::Int) = sum(chain(n, put(i=>Z) for i=1:n))

"""
    circ_Hn(n::Int) -> ChainBlock

Creates a quantum circuit with Hadamard gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Hn(n::Int) = chain(n, put(i=>H) for i=1:n)

"""
    circ_CNOT_2regs(n::Int) -> ChainBlock

Creates a quantum circuit with CNOT gates applied between two registers.

# Arguments
- `n::Int`: The total number of qubits in each register.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_CNOT_2regs(n::Int) = chain(n, control(i, i+n÷2 => X) for i=1:n÷2)

"""
    circ_Ry_simple_conv(n::Int, i::Int, j::Int) -> ChainBlock

Creates a simple quantum circuit with Ry gates and a controlled-X gate.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the first qubit.
- `j::Int`: The index of the second qubit.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Ry_simple_conv(n::Int, i::Int, j::Int) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X))

"""
    circ_Ry_conv(n::Int, i::Int, j::Int) -> ChainBlock

Creates a quantum circuit with multiple Ry gates and controlled-X gates.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the first qubit.
- `j::Int`: The index of the second qubit.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
circ_Ry_conv(n::Int, i::Int, j::Int) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)),
                            put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)), put(j=>Ry(0)))

"""
    circ_SU4_conv(n::Int, i::Int, j::Int) -> ChainBlock

Creates a quantum circuit with SU(4) gates and controlled-X gates.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the first qubit.
- `j::Int`: The index of the second qubit.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_SU4_conv(n::Int, i::Int, j::Int)
    squ() = chain(Rz(0), Ry(0), Rz(0))
    chain(n, put(i=>squ()), put(j=>squ()), control(i,j=>X), put(i=>Ry(0)), put(j=>Rz(0)),
            control(j,i=>X), put(i=>Ry(0)), control(i,j=>X), put(i=>squ()), put(j=>squ()))
end

"""
    circ_Rx_layer(n::Int) -> ChainBlock

Creates a quantum circuit layer with Rx gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit layer.
"""
circ_Rx_layer(n::Int) = chain(n, put(i=>Rx(0)) for i=1:n)

"""
    circ_Ry_layer(n::Int) -> ChainBlock

Creates a quantum circuit layer with Ry gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit layer.
"""
circ_Ry_layer(n::Int) = chain(n, put(i=>Ry(0)) for i=1:n)

"""
    circ_Rz_layer(n::Int) -> ChainBlock

Creates a quantum circuit layer with Rz gates applied to all qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit layer.
"""
circ_Rz_layer(n::Int) = chain(n, put(i=>Rz(0)) for i=1:n)

"""
    circ_CNOT_layer(n::Int) -> ChainBlock

Creates a quantum circuit layer with CNOT gates applied between adjacent qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit layer.
"""
circ_CNOT_layer(n::Int) = chain(n, control(i, mod(i, n) + 1 => X) for i=1:n)

"""
    circ_CZ_layer(n::Int) -> ChainBlock

Creates a quantum circuit layer with CZ gates applied between adjacent qubits.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit layer.
"""
circ_CZ_layer(n::Int) = chain(n, control(i, mod(i, n) + 1 => Z) for i=1:n)

"""
    circ_HEA(n::Int) -> ChainBlock

Creates a Hardware Efficient Ansatz (HEA) quantum circuit.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_HEA(n::Int)
    circ = chain(n)
    push!(circ, circ_Rx_layer(n))
    push!(circ, circ_Ry_layer(n))
    push!(circ, circ_CNOT_layer(n))
    return circ
end

"""
    circ_HEA_real(n::Int) -> ChainBlock

Creates a Hardware Efficient Ansatz (HEA) quantum circuit. This version is for real-valued ground states.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_HEA_real(n::Int)
    circ = chain(n)
    push!(circ, circ_Ry_layer(n))
    push!(circ, circ_CNOT_layer(n))
    return circ
end

"""
    circ_phase_flip(n::Int, x::Int) -> ChainBlock

Creates a quantum circuit that performs a phase flip on the state `x`.

# Arguments
- `n::Int`: The total number of qubits.
- `x::Int`: The state to be phase-flipped.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_phase_flip(n::Int, x::Int)
    circ = chain(n)
    x2 = digits(2^n - 1 - x, base=2, pad=n)
    map(i -> x2[i] == 1 ? push!(circ, chain(n, put(i => X))) : nothing, 1:n)
    push!(circ, chain(n, control(1:n-1, n => Z)))
    map(i -> x2[i] == 1 ? push!(circ, chain(n, put(i => X))) : nothing, 1:n)
    return circ
end

"""
    circ_hypergraph_state(vec::Vector{Int}) -> ChainBlock

Creates a quantum circuit that prepares a hypergraph state.

# Arguments
- `vec::Vector{Int}`: The vector representing the hypergraph state.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_hypergraph_state(vec::Vector{Int})
    n = trunc(Int, log2(length(vec)))
    circ = chain(n)
    push!(circ, chain(n, put(i => H) for i in 1:n))
    map(i -> vec[i+1] == -1 ? push!(circ, circ_phase_flip(n, i)) : nothing, 0:2^n-1)
    return circ
end

"""
    circ_hypergraph_state(n::Int, edges::Vector{Vector{Int}}) -> ChainBlock

Creates a quantum circuit that prepares a hypergraph state from a list of edges.

# Arguments
- `n::Int`: The number of qubits in the circuit.
- `edges::Vector{Vector{Int}}`: A vector of vectors, where each inner vector represents an (hyper-)edge in the hypergraph.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_hypergraph_state(n::Int, edges::Vector{Vector{Int}})
    circ = chain(n)
    push!(circ, chain(n, put(i => H) for i in 1:n))
    for edge in edges
        push!(circ, chain(n, control(edge[1:end-1], edge[end] => Z)))
    end
    return circ
end

"""
    circ_swap_decomposed(n::Int, i::Int, j::Int) -> ChainBlock

Creates a quantum circuit that performs a decomposed SWAP operation between qubits `i` and `j`.

# Arguments
- `n::Int`: The total number of qubits.
- `i::Int`: The index of the first qubit.
- `j::Int`: The index of the second qubit.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_swap_decomposed(n::Int, i::Int, j::Int)
    return sum(chain(n, put(i=>ele/sqrt(2)), put(j=>ele/sqrt(2))) for ele in [I2, X, Y, Z])
end

"""
    circ_swap_all(n::Int; decompose::Bool=false) -> ChainBlock

Creates a quantum circuit that performs SWAP operations between pairs of qubits.

# Arguments
- `n::Int`: The total number of qubits.
- `decompose::Bool`: Whether to use the decomposed SWAP operation.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_swap_all(n::Int; decompose::Bool=false)
    isodd(n) ? error("n must be even") : nothing
    if decompose
        return chain(n, put((i, n÷2+i) => circ_swap_decomposed(n, i, n÷2+1)) for i in 1:n÷2)
    else
        return chain(n, put((i, n÷2+i) => SWAP) for i in 1:n÷2)
    end
end

"""
    circ_swap_test(n::Int) -> ChainBlock

Creates a quantum circuit that performs a SWAP test.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_swap_test(n::Int)
    circ = chain(1+2n)
    push!(circ, put(1=>H))
    for i in 1:n
        push!(circ, control(1, (1+i, 1+n+i)=>SWAP))
    end
    push!(circ, put(1=>H))
    return circ
end

"""
    circ_destructive_swap_test(n::Int) -> ChainBlock

Creates a quantum circuit that performs a destructive SWAP test.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_destructive_swap_test(n::Int)
    circ = chain(2n)
    for i in 1:n
        push!(circ, control(i, n+i=>X))
        push!(circ, put(i=>H))
    end
    return circ
end

"""
    circ_obs_times_swap(n::Int, obs::Union{ChainBlock, Add}) -> ChainBlock

Creates a quantum circuit that performs an observable times SWAP operation.

# Arguments
- `n::Int`: The total number of qubits.
- `obs::Union{ChainBlock, Add}`: The observable to be applied.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_obs_times_swap(n::Int, obs::Union{ChainBlock, Add})
    paulis = [I2, X, Y, Z]
    n_paulis = length(paulis)
    cartesian_product = Base.Iterators.product(ntuple(i -> 1:n_paulis, 2n)...) # uses decomposition: SWAP = (I2⊗I2 + X⊗X + Y⊗Y + Z⊗Z) / 2
    circ = sum(chain(chain(2n, put(i => paulis[ele[i]]) for i in eachindex(ele)), obs) for ele in cartesian_product)
    return circ
end

"""
    circ_LCU(n::Int, U_vec::Vector{ChainBlock}; initial_layer::Bool=true, final_layer::Bool=true) -> ChainBlock

Creates a quantum circuit that performs a Linear Combination of Unitaries (LCU) operation.

# Arguments
- `n::Int`: The total number of qubits.
- `U_vec::Vector{ChainBlock}`: The vector of unitary operations.
- `initial_layer::Bool`: Whether to include the initial layer of Hadamard gates.
- `final_layer::Bool`: Whether to include the final layer of Hadamard gates.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_LCU(n::Int, U_vec::Vector{ChainBlock{T}} where T <: Any; initial_layer::Bool=true, final_layer::Bool=true)
    n_ancillas = Int(ceil(log(2, length(U_vec))))
    K = length(U_vec)
    N = n + n_ancillas
    circ = chain(N)
    initial_layer ? push!(circ, chain(N, put(i => H) for i = 1:n_ancillas)) : nothing
    for k = 0:K-1
        push!(circ, chain(N, put(1:n_ancillas => circ_X_where(n_ancillas, findall(j -> j == 0, digits(k, base=2, pad=n_ancillas))))))
        push!(circ, control(1:n_ancillas, n_ancillas+1:N => U_vec[k+1])) # NB: U_vec is 1-indexed
        push!(circ, chain(N, put(1:n_ancillas => circ_X_where(n_ancillas, findall(j -> j == 0, digits(k, base=2, pad=n_ancillas))))))
    end
    final_layer ? push!(circ, chain(N, put(i => H) for i = 1:n_ancillas)) : nothing
    return circ
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
    circ_purified_maximally_mixed(n::Int) -> ChainBlock

Creates a quantum circuit that prepares a purified maximally mixed state.

# Arguments
- `n::Int`: The number of qubits in each register.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_purified_maximally_mixed(n::Int)
    reg1 = 1:n
    reg2 = n+1:2n
    circ = chain(2n,
        put(reg1 => circ_Hn(n)),
        put(vcat(reg1, reg2) => circ_CNOT_2regs(2n)))
    return circ
end

"""
    circ_phase_encoding(n::Int, unitary_block::AbstractBlock; reverse_qubits::Bool=false) -> ChainBlock

Creates a quantum circuit for phase encoding using a unitary block.

# Arguments
- `n::Int`: The number of qubits for the time register.
- `unitary_block::AbstractBlock`: The unitary block to be applied.
- `reverse_qubits::Bool`: Whether to reverse the order of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_phase_encoding(n::Int, unitary_block::AbstractBlock; reverse_qubits::Bool=false)
    reg_times = 1:n
    n_eigvec = nqubits(unitary_block)
    reg_eigvec = n+1:n+n_eigvec
    if !reverse_qubits
        circ_control = chain(n+n_eigvec, control(i+1, reg_eigvec => unitary_block^(2^i)) for i=0:n-1)
    else
        circ_control = chain(n+n_eigvec, control(n-i, reg_eigvec => unitary_block^(2^i)) for i=0:n-1)
    end
    circ = chain(n+n_eigvec,
                 subroutine(circ_Hn(n), reg_times),
                 circ_control)
    return circ
end

"""
    circ_phase_encoding(n::Int, unitary_matrix::AbstractMatrix; reverse_qubits::Bool=false) -> ChainBlock

Creates a quantum circuit for phase encoding using a unitary matrix.

# Arguments
- `n::Int`: The number of qubits for the time register.
- `unitary_matrix::AbstractMatrix`: The unitary matrix to be applied.
- `reverse_qubits::Bool`: Whether to reverse the order of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_phase_encoding(n::Int, unitary_matrix::AbstractMatrix; reverse_qubits::Bool=false)
    n_eigvec = Int(log2(size(unitary_matrix, 1)))
    if n_eigvec != log2(size(unitary_matrix, 1))
        throw(ArgumentError("The size of the unitary matrix must be a power of 2."))
    end
    reg_times = 1:n
    reg_eigvec = n+1:n+n_eigvec
    if !reverse_qubits
        circ_control = chain(n+n_eigvec, control(i+1, reg_eigvec => matblock(unitary_matrix^(2^i))) for i=0:n-1)
    else
        circ_control = chain(n+n_eigvec, control(n-i, reg_eigvec => matblock(unitary_matrix^(2^i))) for i=0:n-1)
    end
    circ = chain(n+n_eigvec,
                 subroutine(circ_Hn(n), reg_times),
                 circ_control)
    return circ
end

"""
    circ_qft(n::Int; reverse_qubits::Bool=false) -> ChainBlock

Creates a quantum circuit for the Quantum Fourier Transform (QFT).

# Arguments
- `n::Int`: The number of qubits.
- `reverse_qubits::Bool`: Whether to reverse the order of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_qft(n::Int; reverse_qubits::Bool=false)
    circ = chain(n)
    if reverse_qubits
        push!(circ, circ_reverse_order(n))
    end
    for j in 1:n
        push!(circ, put(j=>H))
        for k in (j+1):n
            angle = 2π / 2^(k-j + 1)
            push!(circ, control(k, j=>shift(angle)))
        end
    end
    if reverse_qubits
        push!(circ, circ_reverse_order(n))
    end
    return circ
end

"""
    circ_qpe(n::Int, unitary_matrix::Union{AbstractMatrix, AbstractBlock}; reverse_qubits::Bool=false) -> ChainBlock

Creates a quantum circuit for Quantum Phase Estimation (QPE).

# Arguments
- `n::Int`: The number of qubits for the time register.
- `unitary_matrix::Union{AbstractMatrix, AbstractBlock}`: The unitary matrix to be applied.
- `reverse_qubits::Bool`: Whether to reverse the order of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit.
"""
function circ_qpe(n::Int, unitary_matrix::Union{AbstractMatrix, AbstractBlock}; reverse_qubits::Bool=false)
    if typeof(unitary_matrix) <: AbstractBlock
        unitary_matrix = Matrix(unitary_matrix)
    end
    n_eigvec = Int(log2(size(unitary_matrix, 1)))
    if n_eigvec != log2(size(unitary_matrix, 1))
        throw(ArgumentError("The size of the unitary matrix must be a power of 2."))
    end
    reg_times = 1:n
    reg_eigvec = n+1:n+n_eigvec
    circ = chain(n+n_eigvec,
                 circ_phase_encoding(n, unitary_matrix; reverse_qubits=reverse_qubits),
                 subroutine(circ_qft(n; reverse_qubits=reverse_qubits)', reg_times))
    return circ
end

"""
    circ_append_ancillas(circ::AbstractBlock, n_ancillas::Int; pos_ancillas::Symbol=:bottom) -> ChainBlock

Appends ancilla qubits to a quantum circuit.

# Arguments
- `circ::AbstractBlock`: The quantum circuit.
- `n_ancillas::Int`: The number of ancilla qubits.
- `pos_ancillas::Symbol`: The position of the ancilla qubits (`:bottom` or `:top`).

# Returns
- `ChainBlock`: The resulting quantum circuit with ancilla qubits.
"""
function circ_append_ancillas(circ::AbstractBlock, n_ancillas::Int; pos_ancillas::Symbol=:bottom)
    n_qubits = nqubits(circ)
    if pos_ancillas == :bottom
        reg_main = 1:n_qubits
        reg_ancilla = n_qubits+1:n_qubits+n_ancillas
        reg_all = vcat(reg_main, reg_ancilla)
    elseif pos_ancillas == :top
        reg_main = n_ancillas+1:n_ancillas+n_qubits
        reg_ancilla = 1:n_ancillas
        reg_all = vcat(reg_ancilla, reg_main)
    end
    circ = chain(n_qubits+n_ancillas,
        put(reg_main => circ))
    return circ
end

"""
    circ_reverse_order(n::Int) -> ChainBlock

Creates a quantum circuit that reverses the order of qubits using SWAP gates.

# Arguments
- `n::Int`: The total number of qubits.

# Returns
- `ChainBlock`: The resulting quantum circuit that reverses the order of the qubits.
"""
function circ_reverse_order(n::Int)
    return chain(n, put((i, n+1-i) => SWAP) for i in 1:n÷2)
end
