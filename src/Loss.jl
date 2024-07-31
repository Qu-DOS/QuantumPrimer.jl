sigmoid(x) = 1/(1+exp(-x))

function eval_loss(states::Tuple{T, T} where T<:ArrayReg, p::GenericParams)
    state1, state2 = states
    circ = p.circ
    dispatch!(circ, p.params)
    state1_transformed = copy(state1) |> circ
    state2_transformed = copy(state2) |> circ
    loss = destructive_swap_test(state1_transformed, state2_transformed)
    return loss
end

function eval_loss(state, p::AbstractParams)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    return real(expect(cost, copy(state)|>circ)) #copy so that actual state isn't affected
end

function eval_full_loss(d::Union{Data, DataSiamese}, p::AbstractParams, sig)
    total_loss = 0
    for i in eachindex(d.s)
        loss = eval_loss(d.s[i], p)
        if sig == true
            total_loss += (2*sigmoid(10*loss)-1-d.l[i])^2
        else
            total_loss += (loss-d.l[i])^2
        end
    end
    total_loss *= 1/length(d.s)
    return total_loss
end