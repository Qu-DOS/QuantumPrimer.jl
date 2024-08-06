# Exports
export eval_grad,
       eval_full_grad

function eval_grad(state::ArrayReg, model::AbstractModel)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    _, grads = expect'(model.cost(model.n), copy(state)=>circ)
    return grads
end

# uses parameter-shift rule (or finite difference if epsilon!=π/2) to evaluate the gradient 
function eval_grad(state::NTuple{2, ArrayReg}, model::AbstractModel; epsilon=(π/2)::Float64)
    state1, state2 = state
    circ = model.circ
    p_expanded = expand_params(model)
    p_plus = deepcopy(p_expanded)
    p_minus = deepcopy(p_expanded)
    grads = similar(p_expanded)
    for i in eachindex(p_expanded)
        p_plus[i] += epsilon
        p_minus[i] -= epsilon
        dispatch!(circ, p_plus)
        cost_plus = model.cost(copy(state1) |> circ, copy(state2) |> circ)
        dispatch!(circ, p_minus)
        cost_minus = model.cost(copy(state1) |> circ, copy(state2) |> circ)
        grads[i] = epsilon == π/2 ? (cost_plus-cost_minus)/2 : (cost_plus-cost_minus)/(2*epsilon)
        p_plus[i] = p_expanded[i]
        p_minus[i] = p_expanded[i]
    end
    return grads
end

function eval_full_grad(data::AbstractData, model::AbstractModel; sig=true::Bool)
    all_grads = []
    for i in eachindex(data.states)
        grad = eval_grad(data.states[i], model)
        loss = eval_loss(data.states[i], model)
        if sig == true
            push!(all_grads, (2*sigmoid(10*loss)-1-data.labels[i])*-2*sigmoid(10*loss)^2*-10*grad*exp(-10*loss))
        else
            push!(all_grads, grad*(loss-data.labels[i]))
        end
    end
    total_grads = 2/length(data.states)*sum(all_grads)
    return reduce_params(model, total_grads)
end