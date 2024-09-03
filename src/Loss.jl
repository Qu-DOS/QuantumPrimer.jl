# Exports
export regularize_loss,
       eval_loss,
       eval_full_loss

function regularize_loss(loss::Float64, model::AbstractModel; lambda=1.::Float64, regularization=:nothing::Symbol)
    if regularization == :l1
        l1 = sum(abs(param) for param in expand_params(model))
        loss += lambda * l1
    elseif regularization == :l2
        l2 = sum(param^2 for param in expand_params(model))
        loss += lambda * l2
    end
    return loss
end

function eval_loss(state::ArrayReg, model::AbstractModel, cost::GeneralCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    loss = cost.cost(copy(state) |> circ)
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

function eval_loss(state::ArrayReg, model::AbstractModel, cost::CircuitCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    loss = expect(cost.cost(model.n), copy(state) |> circ)
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

function eval_loss(states::NTuple{2, ArrayReg}, models::NTuple{2, AbstractModel}, cost::CircuitCost; lambda=1.::Float64, regularization=:nothing::Symbol)
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

function eval_loss(states::NTuple{2, ArrayReg}, model::AbstractModel, cost::GeneralCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    circ = model.circ
    dispatch!(circ, expand_params(model))
    loss = cost.cost(:loss, copy(state1) |> circ, copy(state2) |> circ)
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

function eval_loss(state::ArrayReg, models::NTuple{2, AbstractModel}, cost::GeneralCost; lambda=1.::Float64, regularization=:nothing::Symbol)
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

function eval_loss(state::ArrayReg, models::NTuple{2, AbstractModel}, cost::CircuitCost; lambda=1.::Float64, regularization=:nothing::Symbol)
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

function eval_full_loss(data::AbstractData, model::Union{AbstractModel, NTuple{2, AbstractModel}}, cost::AbstractCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    total_loss = 0
    for i in eachindex(data.states)
        loss = eval_loss(data.states[i], model, cost; lambda=lambda, regularization=regularization)
        total_loss += (cost.activation(loss)-data.labels[i])^2
    end
    total_loss *= 1/length(data.states)
    return total_loss
end