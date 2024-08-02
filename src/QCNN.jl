function build_QCNN(n::Int; ansatz=conv_Ry::Function)
    circ = chain(n)
    n_q = n
    while n_q>1
        if n_q%2 == 0
            map(i->push!(circ, chain(n, ansatz(n, i, mod(i, n_q)+1))), 1:2:n_q-1)
            map(i->push!(circ, chain(n, ansatz(n, i, mod(i, n_q)+1))), 2:2:n_q)
        else 
            map(i->push!(circ, chain(n, ansatz(n, i, mod(i, n_q)+1))), 1:2:n_q)
            map(i->push!(circ, chain(n, ansatz(n, i, mod(i, n_q)+1))), 2:2:n_q-1)
        end
        n_q = Int(ceil(n_q/2))
    end
    return circ
end