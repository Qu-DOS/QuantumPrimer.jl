# Export
export covariance,
       covariance_siamese,
       covariance_siamese_commuting_obs,
       projected_quantum_kernel,
       swap_test,
       destructive_swap_test,
       entanglement_difference

function covariance(output::Symbol, state::Union{ArrayReg, Pair}, obs_A::Union{ChainBlock, Add}, obs_B::Union{ChainBlock, Add})
    A = expect(obs_A, state)
    B = expect(obs_B, state)
    AB = expect(obs_A * obs_B, state)
    BA = expect(obs_B * obs_A, state)
    AB_sym = (AB + BA) / 2
    if output == :loss
        return abs(AB_sym - A*B)
    elseif output == :grad
        _, dA = expect'(obs_A, state)
        _, dB = expect'(obs_B, state)
        _, dAB = expect'(obs_A * obs_B, state)
        _, dBA = expect'(obs_B * obs_A, state)
        dAB_sym = (dAB + dBA) / 2
        return sign(AB_sym - A * B) * (dAB_sym - (A * dB + B * dA))
    end
end

function covariance_siamese_commuting_obs(output::Symbol, state1::Union{ArrayReg, Pair}, state2::Union{ArrayReg, Pair}, obs_A::Union{ChainBlock, Add}, obs_B::Union{ChainBlock, Add}; model=nothing::Union{AbstractModel, Nothing})
    # If the observables commute, the covariance can be computed as usual - obs * SWAP is Hermitian and can perform expectation value
    n = 0
    try
        n = nactive(state1)
    catch
        n = nactive(state1[1])
    end
    if output == :loss
        joined_state = join(state2, state1)
        A = expect(circ_swap_all(2n) * obs_A, joined_state) # NB: depending on obs_A, SWAP and obs_A may not commute, thus circ_swap_all(2n) * obs_A is not Hermitian
        B = expect(circ_swap_all(2n) * obs_B, joined_state)
        AB = expect(circ_swap_all(2n) * obs_A * obs_B, joined_state)
        BA = expect(circ_swap_all(2n) * obs_B * obs_A, joined_state)
        AB_sym = (AB + BA) / 2
        return AB_sym - A*B
    elseif output == :grad
        circ_full = chain(2n, put(1:n => state1[2]), put(n+1:2n => state2[2]))
        joined_state = join(state2[1], state1[1])
        A = expect(circ_swap_all(2n) * obs_A, copy(joined_state) |> circ_full) # NB: depending on obs_A, SWAP and obs_A may not commute, thus circ_swap_all(2n) * obs_A is not Hermitian
        B = expect(circ_swap_all(2n) * obs_B, copy(joined_state) |> circ_full)
        _, dA = expect'(circ_swap_all(2n) * obs_A, copy(joined_state) => circ_full)
        _, dB = expect'(circ_swap_all(2n) * obs_B, copy(joined_state) => circ_full)
        _, dAB = expect'(circ_swap_all(2n) * obs_A * obs_B, copy(joined_state) => circ_full)
        _, dBA = expect'(circ_swap_all(2n) * obs_B * obs_A, copy(joined_state) => circ_full)
        dAB_sym = (dAB + dBA) / 2
        return dAB_sym - (A * dB + B * dA)
    end
end

function projected_quantum_kernel(state1::ArrayReg, state2::ArrayReg; gamma=1.::Float64) # S110 in huang2021power
    n = nactive(state1)
    pauli_basis = [X, Y, Z]
    summ = 0
    for pauli_op in pauli_basis
        for i in 1:n
            circ = circ_gate_single(n, i, pauli_op)
            exp_value1 = expect(circ, state1)
            exp_value2 = expect(circ, state2)
            summ += (exp_value1 - exp_value2)^2
        end
    end
    return exp(-gamma*summ)
end

function swap_test(state1::ArrayReg, state2::ArrayReg; nshots=1000::Int)
    n = nactive(state1)
    circ = circ_swap_test(n)
    measurements = measure(join(state2, state1, zero_state(1)) |> circ, 1; nshots=nshots)
    P0 = count(i->i==0, measurements) / nshots
    P1 = count(i->i==1, measurements) / nshots
    res = P0 - P1
    res = res > 0 ? res : 0
    return res
end

function destructive_swap_test(state1::ArrayReg, state2::ArrayReg; nshots=1000::Int)
    n = nactive(state1)
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

function entanglement_difference(state1::ArrayReg, state2::ArrayReg)
    n = nactive(state1)
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