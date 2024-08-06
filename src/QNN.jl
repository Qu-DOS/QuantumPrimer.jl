# Exports
export build_QNN

function build_QNN(n::Int, depth::Int; ansatz=circ_HEA::Function)
    circ = chain(n)
    for _ = 1:depth
        push!(circ, ansatz(n))
    end
    return circ
end