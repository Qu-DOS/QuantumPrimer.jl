function eval_grad(state::ArrayReg, p::AbstractParams)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    cost = chain(p.n, put(1=>Z)) # Z gate on first qubit
    _, grads = expect'(cost, copy(state)=>circ)
    return grads
end

# uses parameter-shift rule (or finite difference) to evaluate the gradient 
function eval_grad(state::NTuple{2, T} where T <: ArrayReg, p::AbstractParams; cost_func=entanglement_difference::Function, epsilon=π/2)
    state1, state2 = state
    circ = p.circ
    p_expanded = expand_params(p)
    p_plus = deepcopy(p_expanded)
    p_minus = deepcopy(p_expanded)
    grads = similar(p_expanded)
    for i in eachindex(p_expanded)
        p_plus[i] += epsilon
        p_minus[i] -= epsilon
        dispatch!(circ, p_plus)
        cost_plus = cost_func(copy(state1) |> circ, copy(state2) |> circ)
        dispatch!(circ, p_minus)
        cost_minus = cost_func(copy(state1) |> circ, copy(state2) |> circ)
        grads[i] = epsilon == π/2 ? (cost_plus-cost_minus)/2 : (cost_plus-cost_minus)/(2*epsilon)
        p_plus[i] = p_expanded[i]
        p_minus[i] = p_expanded[i]
    end
    return grads
end

function eval_full_grad(d::AbstractData, p::AbstractParams, sig::Bool)
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