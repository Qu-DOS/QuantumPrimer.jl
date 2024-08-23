# Exports
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
       circ_Ry_simple_conv,
       circ_Ry_conv,
       circ_SU4_conv,
       circ_Rx_layer,
       circ_Ry_layer,
       circ_Rz_layer,
       circ_CNOT_layer,
       circ_HEA,
       circ_phase_flip,
       circ_hypergraph_state,
       circ_swap_decomposed,
       circ_swap_all,
       circ_swap_test,
       circ_destructive_swap_test,
       circ_obs_times_swap,
       circ_LCU

circ_gate_single(n::Int, i::Int, gate::ConstantGate) = chain(n, put(i=>gate))
circ_gate_n(n::Int, gate::ConstantGate) = chain(n, put(i=>gate) for i=1:n)
circ_gate_sum(n::Int, gate::ConstantGate) = sum(chain(n, put(i=>gate) for i=1:n))
circ_gate_where(n::Int, gate::ConstantGate, ones_where::Vector{Int}) = chain(n, put(i=>gate) for i in ones_where)

circ_block_single(n::Int, i::Int, block::ChainBlock{2}) = chain(n, put(i=>block))
circ_block_n(n::Int, block::ChainBlock{2}) = chain(n, put(i=>block) for i=1:n)
circ_block_sum(n::Int, block::ChainBlock{2}) = sum(chain(n, put(i=>block) for i=1:n))
circ_block_where(n::Int, block::ChainBlock{2}, ones_where::Vector{Int}) = chain(n, put(i=>block) for i in ones_where)

circ_X(n::Int, i::Int) = circ_gate_single(n, i, X)
circ_X(n::Int) = circ_gate_single(n, 1, X)
circ_X_where(n::Int, ones_where::Vector{Int}) = circ_gate_where(n, X, ones_where)

circ_Y(n::Int, i::Int) = circ_gate_single(n, i, Y)
circ_Y(n::Int) = circ_gate_single(n, 1, Y)
circ_Y_where(n::Int, ones_where::Vector{Int}) = circ_gate_where(n, Y, ones_where)

circ_Z(n::Int, i::Int) = circ_gate_single(n, i, Z)
circ_Z(n::Int) = circ_gate_single(n, 1, Z)
circ_Z_where(n::Int, ones_where::Vector{Int}) = circ_gate_where(n, Z, ones_where)

circ_Xn(n::Int) = chain(n, put(i=>X) for i=1:n)
circ_Yn(n::Int) = chain(n, put(i=>Y) for i=1:n)
circ_Zn(n::Int) = chain(n, put(i=>Z) for i=1:n)

circ_Xsum(n::Int) = sum(chain(n, put(i=>X) for i=1:n))
circ_Ysum(n::Int) = sum(chain(n, put(i=>Y) for i=1:n))
circ_Zsum(n::Int) = sum(chain(n, put(i=>Z) for i=1:n))

circ_Ry_simple_conv(n::Int, i::Int, j::Int) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X))

circ_Ry_conv(n::Int, i::Int, j::Int) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)),
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

function circ_swap_decomposed(n::Int, i::Int, j::Int)
    return sum(chain(n, put(i=>ele/sqrt(2)), put(j=>ele/sqrt(2))) for ele in [I2, X, Y, Z])
end
 
function circ_swap_all(n::Int; decompose::Bool=false)
    isodd(n) ? error("n must be even") : nothing
    if decompose
        return chain(n, put((i, n÷2+i) => circ_swap_decomposed(n, i, n÷2+1)) for i in 1:n÷2)
    else
        return chain(n, put((i, n÷2+i) => SWAP) for i in 1:n÷2)
    end
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

function circ_obs_times_swap(n::Int, obs::Union{ChainBlock, Add})
    paulis = [I2, X, Y, Z]
    n_paulis = length(paulis)
    cartesian_product = Base.Iterators.product(ntuple(i -> 1:n_paulis, 2n)...) # uses decomposition: SWAP = (I2⊗I2 + X⊗X + Y⊗Y + Z⊗Z) / 2
    circ = sum(chain(chain(2n, put(i => paulis[ele[i]]) for i in eachindex(ele)), obs) for ele in cartesian_product)
    return circ
end

function circ_LCU(n::Int, U_vec::Vector{ChainBlock}; initial_layer::Bool=true, final_layer::Bool=true)
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