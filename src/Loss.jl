# Exports
export regularize_loss,
       eval_loss,
       eval_full_loss

"""
    regularize_loss(loss::Float64, model::AbstractModel; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Regularizes the loss using L1 or L2 regularization.

# Arguments
- `loss::Float64`: The initial loss value.
- `model::AbstractModel`: The quantum model.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The regularized loss.
"""
function regularize_loss(loss::Float64, model::AbstractModel; lambda=1.0::Float64, regularization=:nothing::Symbol)
    if regularization == :l1
        l1 = sum(abs(param) for param in expand_params(model))
        loss += lambda * l1
    elseif regularization == :l2
        l2 = sum(param^2 for param in expand_params(model))
        loss += lambda * l2
    end
    return loss
end

"""
    eval_loss(state::ArrayReg, model::AbstractModel, cost::GeneralCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the loss for a given quantum state and model using a general cost function.

# Arguments
- `state::ArrayReg`: The quantum state.
- `model::AbstractModel`: The quantum model.
- `cost::GeneralCost`: The general cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated loss.
"""
function eval_loss(state::ArrayReg, model::AbstractModel, cost::GeneralCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    loss = cost.cost(copy(state) |> circ)
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

"""
    eval_loss(state::ArrayReg, model::AbstractModel, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the loss for a given quantum state and model using a circuit cost function.

# Arguments
- `state::ArrayReg`: The quantum state.
- `model::AbstractModel`: The quantum model.
- `cost::CircuitCost`: The circuit cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated loss.
"""
function eval_loss(state::ArrayReg, model::AbstractModel, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    loss = expect(cost.cost(model.n), copy(state) |> circ)
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

"""
    eval_loss(states::NTuple{2, ArrayReg}, models::NTuple{2, AbstractModel}, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the loss for a pair of quantum states and models using a circuit cost function.

# Arguments
- `states::NTuple{2, ArrayReg}`: The tuple of quantum states.
- `models::NTuple{2, AbstractModel}`: The tuple of quantum models.
- `cost::CircuitCost`: The circuit cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated loss.
"""
function eval_loss(states::NTuple{2, ArrayReg}, models::NTuple{2, AbstractModel}, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    model1, model2 = models
    n = model1.n
    circ_full = chain(2n, put(1:n => model1.circ + model2.circ), put(n+1:2n => model1.circ + model2.circ))
    dispatch!(model1.circ, expand_params(model1))
    dispatch!(model2.circ, expand_params(model2))
    loss = expect(cost.cost(2n), join(state2, state1) |> circ_full)
    loss = regularize_loss(loss, model1; lambda=lambda, regularization=regularization)
    loss = regularize_loss(loss, model2; lambda=lambda, regularization=regularization)
    return loss
end

"""
    eval_loss(states::NTuple{2, ArrayReg}, models::NTuple{<:Any, AbstractModel}, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the loss for a pair of quantum states and models using a circuit cost function.

# Arguments
- `states::NTuple{2, ArrayReg}`: The tuple of quantum states.
- `models::NTuple{<:Any, AbstractModel}`: The tuple of quantum models.
- `cost::CircuitCost`: The circuit cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated loss.
"""
function eval_loss(states::NTuple{2, ArrayReg}, models::NTuple{<:Any, AbstractModel}, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    n = models[1].n
    n_models = length(models)
    LCU = sum([model.circ for model in models])
    circ_full = chain(2n, put(1:n => LCU^n_models), put(n+1:2n => LCU^n_models))
    for model in models
        dispatch!(model.circ, expand_params(model))
    end
    loss = expect(cost.cost(2n), join(state2, state1) |> circ_full |> normalize) # NORMALIZATION IS TO THE FULL CIRCUIT. IT SHOULD BE TO THE SINGLE STATES AFTER CIRC_FULL
    for model in models
        loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    end
    return loss
end

"""
    eval_loss(states::NTuple{2, ArrayReg}, model::AbstractModel, cost::GeneralCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the loss for a pair of quantum states and a model using a general cost function.

# Arguments
- `states::NTuple{2, ArrayReg}`: The tuple of quantum states.
- `model::AbstractModel`: The quantum model.
- `cost::GeneralCost`: The general cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated loss.
"""
function eval_loss(states::NTuple{2, ArrayReg}, model::AbstractModel, cost::GeneralCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    circ = model.circ
    dispatch!(circ, expand_params(model))
    loss = cost.cost(:loss, copy(state1) |> circ, copy(state2) |> circ)
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

"""
    eval_loss(states::NTuple{2, ArrayReg}, model::AbstractModel, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the loss for a pair of quantum states and a model using a general cost function.

# Arguments
- `states::NTuple{2, ArrayReg}`: The tuple of quantum states.
- `model::AbstractModel`: The quantum model.
- `cost::CircuitCost`: The circuit cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated loss.
"""
function eval_loss(states::NTuple{2, ArrayReg}, model::AbstractModel, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    circ = model.circ
    n = model.n
    dispatch!(circ, expand_params(model))
    circ_full = chain(2n, put(1:n => circ), put(n+1:2n => circ))
    loss = expect(cost.cost(2n)*circ_swap_all(2n), join(state2, state1) |> circ_full)
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

"""
    eval_loss(state::ArrayReg, models::NTuple{2, AbstractModel}, cost::GeneralCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the loss for a quantum state and a tuple of models using a general cost function.

# Arguments
- `state::ArrayReg`: The quantum state.
- `models::NTuple{2, AbstractModel}`: The tuple of quantum models.
- `cost::GeneralCost`: The general cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated loss.
"""
function eval_loss(state::ArrayReg, models::NTuple{2, AbstractModel}, cost::GeneralCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    model1, model2 = models
    n = model1.n
    circ_full = chain(2n, put(1:n => model1.circ), put(n+1:2n => model2.circ))
    dispatch!(model1.circ, expand_params(model1))
    dispatch!(model2.circ, expand_params(model2))
    loss = cost.cost(:loss, copy(state) |> circ_full)
    loss = regularize_loss(loss, model1; lambda=lambda, regularization=regularization)
    loss = regularize_loss(loss, model2; lambda=lambda, regularization=regularization)
    return loss
end

"""
    eval_loss(state::ArrayReg, models::NTuple{2, AbstractModel}, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the loss for a quantum state and a tuple of models using a circuit cost function.

# Arguments
- `state::ArrayReg`: The quantum state.
- `models::NTuple{2, AbstractModel}`: The tuple of quantum models.
- `cost::CircuitCost`: The circuit cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated loss.
"""
function eval_loss(state::ArrayReg, models::NTuple{2, AbstractModel}, cost::CircuitCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    model1, model2 = models
    n = model1.n
    circ_full = chain(2n, put(1:n => model1.circ), put(n+1:2n => model2.circ))
    dispatch!(model1.circ, expand_params(model1))
    dispatch!(model2.circ, expand_params(model2))
    loss = expect(cost.cost(n), copy(state) |> circ_full)
    loss = regularize_loss(loss, model1; lambda=lambda, regularization=regularization)
    loss = regularize_loss(loss, model2; lambda=lambda, regularization=regularization)
    return loss
end

"""
    eval_full_loss(data::AbstractData, model::Union{AbstractModel, NTuple{<:Any, AbstractModel}}, cost::AbstractCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Float64

Evaluates the full loss over a dataset for a given model or tuple of models.

# Arguments
- `data::AbstractData`: The dataset containing quantum states and corresponding labels.
- `model::Union{AbstractModel, NTuple{<:Any, AbstractModel}}`: The quantum model or tuple of models.
- `cost::AbstractCost`: The cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Float64`: The evaluated full loss.
"""
function eval_full_loss(data::AbstractData, model::Union{AbstractModel, NTuple{<:Any, AbstractModel}}, cost::AbstractCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
    total_loss = 0
    for i in eachindex(data.states)
        loss = eval_loss(data.states[i], model, cost; lambda=lambda, regularization=regularization)
        total_loss += (cost.activation(loss)-data.labels[i])^2
    end
    total_loss *= 1/length(data.states)
    return total_loss
end