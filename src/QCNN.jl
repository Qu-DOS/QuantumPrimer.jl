# Exports
export build_QCNN

"""
    build_QCNN(n::Int; ansatz=circ_Ry_conv::Function) -> ChainBlock

Builds a Quantum Convolutional Neural Network (QCNN) circuit.

# Arguments
- `n::Int`: The number of qubits.
- `ansatz::Function`: The ansatz function to use for the convolutional layers, default is `circ_Ry_conv`.

# Returns
- `ChainBlock`: The constructed QCNN circuit.
"""
function build_QCNN(n::Int; ansatz=circ_Ry_conv::Function)
    circ = chain(n)
    n_q = n
    while n_q > 1
        if n_q % 2 == 0
            map(i -> push!(circ, chain(n, ansatz(n, i, mod(i, n_q) + 1))), 1:2:n_q-1)
            map(i -> push!(circ, chain(n, ansatz(n, i, mod(i, n_q) + 1))), 2:2:n_q)
        else 
            map(i -> push!(circ, chain(n, ansatz(n, i, mod(i, n_q) + 1))), 1:2:n_q)
            map(i -> push!(circ, chain(n, ansatz(n, i, mod(i, n_q) + 1))), 2:2:n_q-1)
        end
        n_q = Int(ceil(n_q / 2))
    end
    return circ
end