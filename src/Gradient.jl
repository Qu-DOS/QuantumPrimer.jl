# Exports
export regularize_grads,
       eval_grad,
       eval_full_grad

function regularize_grads(grads::Vector{Float64}, model::AbstractModel; lambda=1.::Float64, regularization=:nothing::Symbol)
    if regularization == :l1
        l1 = sign.(expand_params(model))
        grads += lambda * l1
    elseif regularization == :l2
        l2 = 2*expand_params(model)
        grads += lambda * l2
    end
    return grads
end

function eval_grad(state::ArrayReg, model::AbstractModel; lambda=1.::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    _, grads = expect'(model.cost(model.n), copy(state)=>circ)
    grads = convert.(Float64, grads)
    grads = regularize_grads(grads, model; lambda=lambda, regularization=regularization)
    return grads
end

# uses parameter-shift rule (or finite difference if epsilon!=π/2) to evaluate the gradient 
function eval_grad(state::NTuple{2, ArrayReg}, model::AbstractModel; epsilon=(π/2)::Float64, lambda=1.::Float64, regularization=:nothing::Symbol)
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
    grads = regularize_grads(grads, model; lambda=lambda, regularization=regularization)
    return grads
end

# uses parameter-shift rule (or finite difference if epsilon!=π/2) to evaluate the gradient 
function eval_grad(state::ArrayReg, model1::AbstractModel, model2::AbstractModel; epsilon=(π/2)::Float64, lambda=1.::Float64, regularization=:nothing::Symbol)
    circ1 = model1.circ
    circ2 = model2.circ
    circ_full = chain(model1.n*2, put!(1:model1.n => circ1), put!(model1.n+1:model1.n*2 => circ2))
    p_expanded1 = expand_params(model1)
    p_expanded2 = expand_params(model2)
    p_expanded_full = vcat(p_expanded1, p_expanded2)
    n_parameters_full = length(p_expanded_full)
    p_plus = deepcopy(p_expanded_full)
    p_minus = deepcopy(p_expanded_full)
    grads = similar(p_expanded_full)
    for i in eachindex(p_expanded_full)
        p_plus[i] += epsilon
        p_minus[i] -= epsilon
        dispatch!(circ1, p_plus[1:n_parameters_full÷2])
        dispatch!(circ2, p_plus[n_parameters_full÷2+1:end])
        cost_plus = model.cost(copy(state) |> circ_full)
        dispatch!(circ1, p_minus[1:n_parameters_full÷2])
        dispatch!(circ2, p_minus[n_parameters_full÷2+1:end])
        cost_minus = model.cost(copy(state) |> circ_full)
        grads[i] = epsilon == π/2 ? (cost_plus-cost_minus)/2 : (cost_plus-cost_minus)/(2*epsilon)
        p_plus[i] = p_expanded_full[i]
        p_minus[i] = p_expanded_full[i]
    end
    return grads
end

function eval_full_grad(data::AbstractData, model::AbstractModel; lambda=1.::Float64, regularization=:nothing::Symbol)
    all_grads = Vector{Vector{Float64}}()
    for i in eachindex(data.states)
        grad = eval_grad(data.states[i], model; lambda=lambda, regularization=regularization)
        loss = eval_loss(data.states[i], model; lambda=lambda, regularization=regularization)
        activation_derivative = ForwardDiff.derivative(model.activation, loss)
        push!(all_grads, grad * 2 * (model.activation(loss) - data.labels[i]) * activation_derivative)
    end
    total_grads = sum(all_grads) / length(data.states)
    return reduce_params(model, total_grads)
end