circ_Z(n::Int) = chain(n, put(1=>Z))

circ_Z(n::Int, i::Int) = chain(n, put(i=>Z))

circ_Zn(n::Int) = chain(n, put(i=>Z) for i = 1:n)

circ_Zsum(n::Int) = sum(chain(n, put(i=>Z) for i=1:n))

conv_Ry(n::Int, i::Int, j::Int) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X))

conv_Ry2(n::Int, i::Int, j::Int) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)),
                            put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)), put(j=>Ry(0)))

function conv_SU4(n::Int, i::Int, j::Int)
    squ() = chain(Rz(0), Ry(0), Rz(0))
    chain(n, put(i=>squ()), put(j=>squ()), control(i,j=>X), put(i=>Ry(0)), put(j=>Rz(0)),
            control(j,i=>X), put(i=>Ry(0)), control(i,j=>X), put(i=>squ()), put(j=>squ()))
end

layer_Rx(n::Int) = chain(n, put(i => Rx(0)) for i = 1:n)

layer_Ry(n::Int) = chain(n, put(i => Ry(0)) for i = 1:n)

layer_CNOT(n::Int) = chain(n, control(i, mod(i, n) + 1 => X) for i = 1:n)

function circ_HEA(n::Int, depth::Int)
    circ = chain(n)
    for _ = 1:depth
        push!(circ, layer_Rx(n))
        push!(circ, layer_Ry(n))
        push!(circ, layer_CNOT(n))
    end
    return circ
end

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