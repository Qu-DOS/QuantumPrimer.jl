function circ_swap_test(n::Int)
    circ = chain(1+2n)
    push!(circ, put(1=>H))
    for i in 1:n
        push!(circ, control(1, (1+i, 1+n+i)=>SWAP))
    end
    push!(circ, put(1=>H))
    return circ
end

function swap_test(state1::ArrayReg, state2::ArrayReg; nshots=1000)
    n = nqubits(state1)
    circ = circ_swap_test(n)
    measurements = measure(join(state2, state1, zero_state(1)) |> circ, 1; nshots=nshots)
    P0 = count(i->i==0, measurements) / nshots
    P1 = count(i->i==1, measurements) / nshots
    res = P0 - P1
    res = res > 0 ? res : 0
    return res
end

function circ_destructive_swap_test(n::Int)
    circ = chain(2n)
    for i in 1:n
        push!(circ, control(i, n+i=>X))
        push!(circ, put(i=>H))
    end
    return circ
end

function destructive_swap_test(state1::ArrayReg, state2::ArrayReg; nshots=1000)
    n = nqubits(state1)
    circ = circ_destructive_swap_test(n)
    measurements = measure(join(state2, state1) |> circ, 1:2n; nshots=nshots)
    P_fail = 0
    for i in 1:nshots
        binary_a = join(string(x) for x in measurements[i][1:n])
        int_a = parse(UInt, binary_a; base=2)
        binary_b = join(string(x) for x in measurements[i][n+1:2n])
        int_b = parse(UInt, binary_b; base=2)
        P_fail += count_ones(int_a & int_b) % 2 # Test failing if bitwise AND has odd parity
    end
    P_fail /= nshots
    res = 1 - 2*P_fail
    res = res > 0 ? res : 0
    return res
end