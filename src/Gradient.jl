function eval_grad(state::ArrayReg, p::AbstractParams)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    cost = chain(p.n, put(1=>Z)) # Z gate on first qubit
    _, grads = expect'(cost, copy(state)=>circ)
    return grads
end

function eval_grad(state::Tuple{T, T} where T <: ArrayReg, p::AbstractParams)
    state1, state2 = state
    n = nqubits(state1)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    cost = destructive_swap_test(state1, state2)
    cost = circ_swap_test(n)
    _, grads = expect'(cost, (copy(state1), copy(state2)) => circ)
    return grads
end

function eval_full_grad(d::Data, p::AbstractParams, sig)
    all_grads = []
    for i in eachindex(d.s)
        grad = eval_grad(d.s[i], p)
        loss = eval_loss(d.s[i], p)
        if sig == true
            push!(all_grads, (2*sigmoid(10*loss)-1-d.l[i])*-2*sigmoid(10*loss)^2*-10*grad*exp(-10*loss))
        else
            push!(all_grads, grad*(loss-d.l[i]))
        end
    end
    total_grads = 2/length(d.s)*sum(all_grads)
    return reduce_params(p, total_grads)
end