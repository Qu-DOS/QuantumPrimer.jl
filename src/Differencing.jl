# Export
export covariance,
       covariance_normalized,
       projected_quantum_kernel,
       swap_test,
       destructive_swap_test,
       entanglement_difference,
       overlap

function covariance(state1::ArrayReg, state2::ArrayReg, obs_A::Union{ChainBlock, Add}, obs_B::Union{ChainBlock, Add})
    n = nqubits(state1)
    joined_state = join(state2, state1)
    A = sandwich(joined_state, circ_swap_all(2n) * obs_A, joined_state)
    B = sandwich(joined_state, circ_swap_all(2n) * obs_B, joined_state)
    AB = sandwich(joined_state, circ_swap_all(2n) * obs_A * obs_B, joined_state)
    BA = sandwich(joined_state, circ_swap_all(2n) * obs_B * obs_A, joined_state)
    AB_sym = (AB + BA) / 2
    return AB_sym - A*B
end

function covariance_normalized(state1::ArrayReg, state2::ArrayReg, obs_A::Union{ChainBlock, Add}, obs_B::Union{ChainBlock, Add})
    n = nqubits(state1)
    A = sandwich(state1, obs_A, state2)
    B = sandwich(state1, obs_B, state2)
    AB = sandwich(state1, obs_A * obs_B, state2)
    BA = sandwich(state1, obs_B * obs_A, state2)
    AB_sym = (AB + BA) / 2
    return AB_sym - A*B
end

function covariance_XZ(state::ArrayReg)
    n = nqubits(state)
    cost1 = real(expect(circ_X(n), copy(state))) * real(expect(circ_Z(n, (n%2)+1), copy(state)))
    cost2 = real(expect(circ_X(n)*circ_Z(n, (n%2)+1), copy(state)))
    cost = abs(cost2 - cost1)
    return cost
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