function circ_phase_flip(n::Int, x::Int)
    circ = chain(n)
    x2 = digits(2^n - 1 - x, base=2, pad=n)
    map(i -> x2[i] == 1 ? push!(circ, chain(n, put(i => X))) : nothing, 1:n)
    push!(circ, chain(n, control(1:n-1, n => Z)))
    map(i -> x2[i] == 1 ? push!(circ, chain(n, put(i => X))) : nothing, 1:n)
    return circ
end

function circ_hypergraph_state(vec::Vector{Int})
    n = trunc(Int, log2(length(vec)))
    circ = chain(n)
    push!(circ, chain(n, put(i => H) for i in 1:n))
    map(i -> vec[i+1] == -1 ? push!(circ, circ_phase_flip(n, i)) : nothing, 0:2^n-1)
    return circ
end