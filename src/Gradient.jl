# Exports
export parameter_shift_rule,
       regularize_grads,
       eval_grad,
       eval_full_grad

function parameter_shift_rule(obs::Union{ChainBlock, Add}, state::ArrayReg, model::AbstractModel; epsilon=(π/2)::Float64)
    circ = model.circ
    p_expanded = expand_params(model)
    p_plus = deepcopy(p_expanded)
    p_minus = deepcopy(p_expanded)
    grads = similar(p_expanded, ComplexF64) # change to ComplexF64 because of sandwich function
    for i in eachindex(p_expanded)
        p_plus[i] += epsilon
        p_minus[i] -= epsilon
        dispatch!(circ, p_plus)
        cost_plus = expect(obs, copy(state) |> circ)
        dispatch!(circ, p_minus)
        cost_minus = expect(obs, copy(state) |> circ)
        grads[i] = epsilon == π/2 ? (cost_plus-cost_minus)/2 : (cost_plus-cost_minus)/(2*epsilon)
        p_plus[i] = p_expanded[i]
        p_minus[i] = p_expanded[i]
    end
    return grads
end

function parameter_shift_rule(obs::Union{ChainBlock, Add}, state1::ArrayReg, state2::ArrayReg, model::AbstractModel; epsilon=(π/2)::Float64)
    circ = model.circ
    p_expanded = expand_params(model)
    p_plus = deepcopy(p_expanded)
    p_minus = deepcopy(p_expanded)
    grads = similar(p_expanded, ComplexF64) # change to ComplexF64 because of sandwich function
    for i in eachindex(p_expanded)
        p_plus[i] += epsilon
        p_minus[i] -= epsilon
        dispatch!(circ, p_plus)
        cost_plus = sandwich(copy(state1) |> circ, obs, copy(state2) |> circ)
        dispatch!(circ, p_minus)
        cost_minus = sandwich(copy(state1) |> circ, obs, copy(state2) |> circ)
        grads[i] = epsilon == π/2 ? (cost_plus-cost_minus)/2 : (cost_plus-cost_minus)/(2*epsilon)
        p_plus[i] = p_expanded[i]
        p_minus[i] = p_expanded[i]
    end
    return grads
end

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

function eval_grad(state::ArrayReg, model::AbstractModel, cost::CircuitCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    _, grads = expect'(cost.cost(model.n), copy(state) => circ)
    grads = convert.(Float64, grads)
    grads = regularize_grads(grads, model; lambda=lambda, regularization=regularization)
    return grads
end

function eval_grad(states::NTuple{2, ArrayReg}, model::AbstractModel, cost::GeneralCost; epsilon=(π/2)::Float64, lambda=1.::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    circ = model.circ
    p_expanded = expand_params(model)
    dispatch!(circ, p_expanded)
    grads = cost.cost(:grad, copy(state1) => circ, copy(state2) => circ; model=model)
    grads = convert.(Float64, grads)
    grads = grads[1:length(p_expanded)]
    # grads1 = grads[1:length(p_expanded)]
    # grads2 = grads[length(p_expanded)+1:end]
    # grads = (grads1 + grads2) / 2
    # println(grads1)
    # println(grads2)
    # println(grads)
    # println("Length of model params: ", length(model.params))
    # println("Length of grads: ", length(grads))
    # println("Length of p_expanded: ", length(p_expanded))
    grads = regularize_grads(grads, model; lambda=lambda, regularization=regularization)
    return grads
end

function eval_grad(state::ArrayReg, models::NTuple{2, AbstractModel}, cost::GeneralCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    model1, model2 = models
    n = model1.n
    circ_full = chain(2n, put(1:n => model1.circ), put(n+1:2n => model2.circ))
    p_expanded1 = expand_params(model1)
    p_expanded2 = expand_params(model2)
    p_expanded_full = vcat(p_expanded1, p_expanded2)
    dispatch!(circ_full, p_expanded_full)
    grads = cost.cost(:grad, copy(state) => circ_full)
    grads = convert.(Float64, grads)
    grads = regularize_grads(grads, model1; lambda=lambda, regularization=regularization)
    grads = regularize_grads(grads, model2; lambda=lambda, regularization=regularization)
    return grads
end

function eval_full_grad(data::AbstractData, model::Union{AbstractModel, NTuple{2, AbstractModel}}, cost::AbstractCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    all_grads = Vector{Vector{Float64}}()
    for i in eachindex(data.states)
        grad = eval_grad(data.states[i], model, cost; lambda=lambda, regularization=regularization)
        loss = eval_loss(data.states[i], model, cost; lambda=lambda, regularization=regularization)
        activation_derivative = ForwardDiff.derivative(cost.activation, loss)
        push!(all_grads, 2 * (cost.activation(loss) - data.labels[i]) * activation_derivative * grad)
    end
    total_grads = sum(all_grads) / length(data.states)
    if typeof(model) <: Tuple
        n1 = length(expand_params(model[1]))
        return vcat(reduce_params(model[1], total_grads[1:n1]), reduce_params(model[2], total_grads[n1+1:end]))
    else
        return reduce_params(model, total_grads)
    end
end