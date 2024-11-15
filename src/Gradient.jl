# Exports
export parameter_shift_rule,
       regularize_grads,
       eval_grad,
       eval_full_grad

"""
    parameter_shift_rule(obs::Union{ChainBlock, Add}, state::ArrayReg, model::AbstractModel; epsilon=(π/2)::Float64) -> Vector{ComplexF64}

Computes the gradient of the expectation value of an observable using the parameter shift rule.

# Arguments
- `obs::Union{ChainBlock, Add}`: The observable.
- `state::ArrayReg`: The quantum state.
- `model::AbstractModel`: The quantum model.
- `epsilon::Float64`: The shift parameter, default is `π/2`.

# Returns
- `Vector{ComplexF64}`: The computed gradients.
"""
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

"""
    parameter_shift_rule(obs::Union{ChainBlock, Add}, state::ArrayReg, model::AbstractModel; epsilon=(π/2)::Float64) -> Vector{ComplexF64}

Computes the gradient of the expectation value of an observable using the parameter shift rule.

# Arguments
- `obs::Union{ChainBlock, Add}`: The observable.
- `state::ArrayReg`: The quantum state.
- `model::AbstractModel`: The quantum model.
- `epsilon::Float64`: The shift parameter, default is `π/2`.

# Returns
- `Vector{ComplexF64}`: The computed gradients.
"""
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

"""
    regularize_grads(grads::Vector{Float64}, model::AbstractModel; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Vector{Float64}

Regularizes the gradients using L1 or L2 regularization.

# Arguments
- `grads::Vector{Float64}`: The gradients to be regularized.
- `model::AbstractModel`: The quantum model.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Vector{Float64}`: The regularized gradients.
"""
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

"""
    regularize_grads(grads::Vector{Float64}, models::NTuple{<:Any, AbstractModel}; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Vector{Float64}

Regularizes the gradients for a tuple of models using L1 or L2 regularization.

# Arguments
- `grads::Vector{Float64}`: The gradients to be regularized.
- `models::NTuple{<:Any, AbstractModel}`: The tuple of quantum models.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Vector{Float64}`: The regularized gradients.
"""
function regularize_grads(grads::Vector{Float64}, models::NTuple{<:Any, AbstractModel}; lambda=1.::Float64, regularization=:nothing::Symbol)
    n = models[1].n
    p_expanded_full = repeat(vcat([expand_params(model) for model in models]...), (n÷2)*2)
    if regularization == :l1
        l1 = sign.(p_expanded_full)
        grads += lambda * l1
    elseif regularization == :l2
        l2 = 2*p_expanded_full
        grads += lambda * l2
    end
    return grads
end

"""
    eval_grad(state::ArrayReg, model::AbstractModel, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Vector{Float64}

Evaluates the gradient of the cost function for a given quantum state and model.

# Arguments
- `state::ArrayReg`: The quantum state.
- `model::AbstractModel`: The quantum model.
- `cost::CircuitCost`: The cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Vector{Float64}`: The evaluated gradients.
"""
function eval_grad(state::ArrayReg, model::AbstractModel, cost::CircuitCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    _, grads = expect'(cost.cost(model.n), copy(state) => circ)
    grads = convert.(Float64, grads)
    grads = regularize_grads(grads, model; lambda=lambda, regularization=regularization)
    return grads
end

"""
    eval_grad(states::NTuple{2, ArrayReg}, models::NTuple{2, AbstractModel}, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Vector{Float64}

Evaluates the gradient of the cost function for a pair of quantum states and models.

# Arguments
- `states::NTuple{2, ArrayReg}`: The tuple of quantum states.
- `models::NTuple{2, AbstractModel}`: The tuple of quantum models.
- `cost::CircuitCost`: The cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Vector{Float64}`: The evaluated gradients.
"""
function eval_grad(states::NTuple{2, ArrayReg}, models::NTuple{2, AbstractModel}, cost::CircuitCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    model1, model2 = models
    n = model1.n
    circ_full = chain(2n, put(1:n => model1.circ + model2.circ), put(n+1:2n => model1.circ + model2.circ))
    p_expanded1 = expand_params(model1)
    p_expanded2 = expand_params(model2)
    p_expanded_full = vcat(p_expanded1, p_expanded2, p_expanded1, p_expanded2)
    dispatch!(circ_full, p_expanded_full)
    _, grads = expect'(cost.cost(2n), join(state2, state1) => circ_full)
    # grads = cost.cost(:grad, join(state2, state1) => circ_full)
    grads = convert.(Float64, grads)
    grads = regularize_grads(grads, models; lambda=lambda, regularization=regularization)
    # grads = regularize_grads(grads, model1; lambda=lambda, regularization=regularization)
    # grads = regularize_grads(grads, model2; lambda=lambda, regularization=regularization)
    avg_grads = (grads[1:length(p_expanded_full)÷2] + grads[length(p_expanded_full)÷2+1:end]) # can be /2 or not, depending if seen as chain rule or average of same parameters
    return avg_grads
end

"""
    eval_grad(states::NTuple{2, ArrayReg}, models::NTuple{<:Any, AbstractModel}, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Vector{Float64}

Evaluates the gradient of the cost function for a pair of quantum states and models.

# Arguments
- `states::NTuple{2, ArrayReg}`: The tuple of quantum states.
- `models::NTuple{<:Any, AbstractModel}`: The tuple of quantum models.
- `cost::CircuitCost`: The cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Vector{Float64}`: The evaluated gradients.
"""
function eval_grad(states::NTuple{2, ArrayReg}, models::NTuple{<:Any, AbstractModel}, cost::CircuitCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    n = models[1].n
    n_models = length(models)
    LCU = sum([model.circ for model in models])
    circ_full = chain(2n, put(1:n => LCU^(n÷2)), put(n+1:2n => LCU^(n÷2)))
    p_expanded_full = repeat(vcat([expand_params(model) for model in models]...), (n÷2)*2) # repeat the expanded parameters of the single unitaries the number of exponentiations (n_models) and for the registers (2)
    dispatch!(circ_full, p_expanded_full)
    _, grads = expect'(cost.cost(2n), join(state2, state1) => circ_full)
    grads = convert.(Float64, grads)
    grads = regularize_grads(grads, models; lambda=lambda, regularization=regularization)
    m = length(p_expanded_full)÷n_models
    avg_grads = sum([grads[1+(i-1)*m:i*m] for i in 1:n_models])
    return avg_grads
end

"""
    eval_grad(states::NTuple{2, ArrayReg}, model::AbstractModel, cost::GeneralCost; epsilon=(π/2)::Float64, lambda=1.0::Float64, regularization=:nothing::Symbol) -> Vector{Float64}

Evaluates the gradient of the general cost function for a pair of quantum states and a model.

# Arguments
- `states::NTuple{2, ArrayReg}`: The tuple of quantum states.
- `model::AbstractModel`: The quantum model.
- `cost::GeneralCost`: The general cost function.
- `epsilon::Float64`: The shift parameter, default is `π/2`.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Vector{Float64}`: The evaluated gradients.
"""
function eval_grad(states::NTuple{2, ArrayReg}, model::AbstractModel, cost::GeneralCost; epsilon=(π/2)::Float64, lambda=1.::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    circ = model.circ
    p_expanded = expand_params(model)
    dispatch!(circ, p_expanded)
    grads = cost.cost(:grad, copy(state1) => circ, copy(state2) => circ; model=model)
    grads = convert.(Float64, grads)
    grads = grads[1:length(p_expanded)]
    grads = regularize_grads(grads, model; lambda=lambda, regularization=regularization)
    return grads
end

"""
    eval_grad(state::ArrayReg, models::NTuple{2, AbstractModel}, cost::GeneralCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Vector{Float64}

Evaluates the gradient of the general cost function for a quantum state and a tuple of models.

# Arguments
- `state::ArrayReg`: The quantum state.
- `models::NTuple{2, AbstractModel}`: The tuple of quantum models.
- `cost::GeneralCost`: The general cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Vector{Float64}`: The evaluated gradients.
"""
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

"""
    eval_full_grad(data::AbstractData, model::Union{AbstractModel, NTuple{<:Any, AbstractModel}}, cost::AbstractCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Vector{Float64}

Evaluates the full gradient of the cost function over a dataset for a given model or tuple of models.

# Arguments
- `data::AbstractData`: The dataset containing quantum states and corresponding labels.
- `model::Union{AbstractModel, NTuple{<:Any, AbstractModel}}`: The quantum model or tuple of models.
- `cost::AbstractCost`: The cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Vector{Float64}`: The evaluated full gradients.
"""
function eval_full_grad(data::AbstractData, model::Union{AbstractModel, NTuple{<:Any, AbstractModel}}, cost::AbstractCost; lambda=1.::Float64, regularization=:nothing::Symbol)
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
        return vcat([reduce_params(model[i], total_grads[1+(i-1)*n1:i*n1]) for i in eachindex(model)]...)
    else
        return reduce_params(model, total_grads)
    end
end