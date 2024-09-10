# Exports
export build_QNN

"""
    build_QNN(n::Int, depth::Int; ansatz=circ_HEA::Function) -> ChainBlock

Builds a Quantum Neural Network (QNN) circuit.

# Arguments
- `n::Int`: The number of qubits.
- `depth::Int`: The depth of the QNN, i.e., the number of layers.
- `ansatz::Function`: The ansatz function to use for each layer, default is `circ_HEA`.

# Returns
- `ChainBlock`: The constructed QNN circuit.
"""
function build_QNN(n::Int, depth::Int; ansatz=circ_HEA::Function)
    circ = chain(n)
    for _ = 1:depth
        push!(circ, ansatz(n))
    end
    return circ
end