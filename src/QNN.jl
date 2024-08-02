function build_QNN(n::Int, depth::Int; ansatz=circ_HEA::Function)
    return ansatz(n, depth)
end