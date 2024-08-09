# Export
export correlation,
       projected_quantum_kernel,
       swap_test,
       destructive_swap_test,
       entanglement_difference,
       overlap

function correlation(state1::ArrayReg, state2::ArrayReg, obs_A::ChainBlock, obs_B::ChainBlock)
    n = nqubits(state1)
    joined_state = join(state2, state1)
    A = sandwich(joined_state, obs_A * circ_swap_all(n), joined_state)
    B = sandwich(joined_state, obs_B * circ_swap_all(n), joined_state)
    AB = sandwich(joined_state, obs_A * obs_B + obs_B * obs_A, joined_state) / 2 # symmetrized
    return real(AB - A*B)
end

function projected_quantum_kernel(state1::ArrayReg, state2::ArrayReg; gamma=1.::Float64) # S110 in huang2021power
    n = nqubits(state1)
    pauli_basis = [X, Y, Z]
    summ = 0
    for pauli_op in pauli_basis
        for i in 1:n
            circ = circ_gate_single(n, i, pauli_op)
            exp_value1 = expect(circ, copy(state1))
            exp_value2 = expect(circ, copy(state2))
            summ += (exp_value1 - exp_value2)^2
        end
    end
    return exp(-gamma*summ)
end

function swap_test(state1::ArrayReg, state2::ArrayReg; nshots=1000::Int)
    n = nqubits(state1)
    circ = circ_swap_test(n)
    measurements = measure(join(state2, state1, zero_state(1)) |> circ, 1; nshots=nshots)
    P0 = count(i->i==0, measurements) / nshots
    P1 = count(i->i==1, measurements) / nshots
    res = P0 - P1
    res = res > 0 ? res : 0
    return res
end

function destructive_swap_test(state1::ArrayReg, state2::ArrayReg; nshots=1000::Int)
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

function overlap(state1::ArrayReg, state2::ArrayReg)
    return abs2(dot(state1.state, state2.state))
end

function entanglement_difference(state1::ArrayReg, state2::ArrayReg)
    n = nqubits(state1)
    entropy1 = 0
    entropy2 = 0
    for i in 1:n
        entropy1 += von_neumann_entropy(state1, i)
        entropy2 += von_neumann_entropy(state2, i)
    end
    entropy1 /= n
    entropy2 /= n
    return abs(entropy1 - entropy2)
end