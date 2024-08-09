# Exports
export circ_gate_single,
       circ_gate_n,
       circ_gate_sum,
       circ_block_single,
       circ_block_n,
       circ_block_sum,
       circ_X,
       circ_Y,
       circ_Z,
       circ_Xn,
       circ_Yn,
       circ_Zn,
       circ_Xsum,
       circ_Ysum,
       circ_Zsum,
       circ_Ry_conv,
       circ_Ry2_conv,
       circ_SU4_conv,
       circ_Rx_layer,
       circ_Ry_layer,
       circ_Rz_layer,
       circ_CNOT_layer,
       circ_HEA,
       circ_phase_flip,
       circ_hypergraph_state,
       circ_swap_all,
       circ_swap_test,
       circ_destructive_swap_test

circ_gate_single(n::Int, i::Int, gate::ConstantGate) = chain(n, put(i=>gate))
circ_gate_n(n::Int, gate::ConstantGate) = chain(n, put(i=>gate) for i=1:n)
circ_gate_sum(n::Int, gate::ConstantGate) = sum(chain(n, put(i=>gate) for i=1:n))

circ_block_single(n::Int, i::Int, block::ChainBlock{2}) = chain(n, put(i=>block))
circ_block_n(n::Int, block::ChainBlock{2}) = chain(n, put(i=>block) for i=1:n)
circ_block_sum(n::Int, block::ChainBlock{2}) = sum(chain(n, put(i=>block) for i=1:n))

circ_X(n::Int, i::Int) = circ_gate_single(n, i, X)
circ_X(n::Int) = circ_gate_single(n, 1, X)

circ_Y(n::Int, i::Int) = circ_gate_single(n, i, Y)
circ_Y(n::Int) = circ_gate_single(n, 1, Y)

circ_Z(n::Int, i::Int) = circ_gate_single(n, i, Z)
circ_Z(n::Int) = circ_gate_single(n, 1, Z)

circ_Xn(n::Int) = chain(n, put(i=>X) for i=1:n)
circ_Yn(n::Int) = chain(n, put(i=>Y) for i=1:n)
circ_Zn(n::Int) = chain(n, put(i=>Z) for i=1:n)

circ_Xsum(n::Int) = sum(chain(n, put(i=>X) for i=1:n))
circ_Ysum(n::Int) = sum(chain(n, put(i=>Y) for i=1:n))
circ_Zsum(n::Int) = sum(chain(n, put(i=>Z) for i=1:n))

circ_Ry_conv(n::Int, i::Int, j::Int) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X))

circ_Ry2_conv(n::Int, i::Int, j::Int) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)),
                            put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)), put(j=>Ry(0)))

function circ_SU4_conv(n::Int, i::Int, j::Int)
    squ() = chain(Rz(0), Ry(0), Rz(0))
    chain(n, put(i=>squ()), put(j=>squ()), control(i,j=>X), put(i=>Ry(0)), put(j=>Rz(0)),
            control(j,i=>X), put(i=>Ry(0)), control(i,j=>X), put(i=>squ()), put(j=>squ()))
end

circ_Rx_layer(n::Int) = chain(n, put(i=>Rx(0)) for i=1:n)

circ_Ry_layer(n::Int) = chain(n, put(i=>Ry(0)) for i=1:n)

circ_Rz_layer(n::Int) = chain(n, put(i=>Rz(0)) for i=1:n)

circ_CNOT_layer(n::Int) = chain(n, control(i, mod(i, n) + 1 => X) for i=1:n)

function circ_HEA(n::Int)
    circ = chain(n)
    push!(circ, circ_Rx_layer(n))
    push!(circ, circ_Ry_layer(n))
    push!(circ, circ_CNOT_layer(n))
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

function circ_swap_all(n::Int)
    circ = chain(2n)
    for i in 1:n
        push!(circ, swap(i, n+i))
    end
    return circ
end

function circ_swap_test(n::Int)
    circ = chain(1+2n)
    push!(circ, put(1=>H))
    for i in 1:n
        push!(circ, control(1, (1+i, 1+n+i)=>SWAP))
    end
    push!(circ, put(1=>H))
    return circ
end

function circ_destructive_swap_test(n::Int)
    circ = chain(2n)
    for i in 1:n
        push!(circ, control(i, n+i=>X))
        push!(circ, put(i=>H))
    end
    return circ
end