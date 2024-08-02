sigmoid(x) = 1/(1+exp(-x))

circ_z(n::Int) = chain(n, put(1=>Z))

function eval_loss(states::NTuple{2, ArrayReg}, p::AbstractParams; cost=overlap::Function)
    state1, state2 = states
    circ = p.circ
    dispatch!(circ, expand_params(p))
    return cost(copy(state1) |> circ, copy(state2) |> circ)
end

function eval_loss(state::ArrayReg, p::AbstractParams; cost=circ_z::Function)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    return real(expect(cost(p.n), copy(state) |> circ)) #copy so that actual state isn't affected
end

function eval_full_loss(d::AbstractData, p::AbstractParams, sig::Bool)
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