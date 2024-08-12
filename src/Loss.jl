# Exports
export regularize_loss,
       eval_loss,
       eval_full_loss

function regularize_loss(loss::Float64, model::AbstractModel; lambda=1.::Float64, regularization=:nothing::Symbol)
    if regularization == :l1
        l1 = sum(abs(param) for param in model.params)
        loss += lambda * l1
    elseif regularization == :l2
        l2 = sum(param^2 for param in model.params)
        loss += lambda * l2
    end
    return loss
end

function eval_loss(states::NTuple{2, ArrayReg}, model::AbstractModel; lambda=1.::Float64, regularization=:nothing::Symbol)
    state1, state2 = states
    circ = model.circ
    dispatch!(circ, expand_params(model))
    cost_function = try
        model.cost(model.n)
    catch
        model.cost
    end
    if typeof(cost_function) <: ChainBlock
        loss = real(sandwich(copy(state1) |> circ, cost_function, copy(state2) |> circ)) # For QSCNN costs. A SWAP gate between state1 and state2 is implied when considering the state |state1 state2>.
    else
        loss = cost_function(copy(state1) |> circ, copy(state2) |> circ)
    end
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

function eval_loss(state::ArrayReg, model::AbstractModel; lambda=1.::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    loss = expect(model.cost(model.n), copy(state) |> circ) # copy so that actual state isn't affected
    loss = regularize_loss(loss, model; lambda=lambda, regularization=regularization)
    return loss
end

function eval_loss(state::ArrayReg, models::NTuple{2, AbstractModel}; lambda=1.::Float64, regularization=:nothing::Symbol)
    model1, model2 = models
    circ1 = model1.circ
    circ2 = model2.circ
    circ_full = chain(model1.n*2, put(1:model1.n => circ1), put(model1.n+1:model1.n*2 => circ2))
    dispatch!(circ1, expand_params(model1))
    dispatch!(circ2, expand_params(model2))
    cost_function = try
        model1.cost(model1.n)
    catch
        model1.cost
    end
    if typeof(cost_function) <: ChainBlock
        loss = expect(model1.cost(model1.n), copy(state) |> circ) # copy so that actual state isn't affected
    else
        loss = cost_function(copy(state) |> circ_full)
    end
    return loss
end

function eval_full_loss(data::AbstractData, model::Union{AbstractModel, NTuple{2, AbstractModel}}; lambda=1.::Float64, regularization=:nothing::Symbol)
    total_loss = 0
    for i in eachindex(data.states)
        loss = eval_loss(data.states[i], model; lambda=lambda, regularization=regularization)
        if typeof(model) <: NTuple{2, AbstractModel}
            model1, _ = model
            total_loss += (model1.activation(loss)-data.labels[i])^2
        else
            total_loss += (model.activation(loss)-data.labels[i])^2
        end
    end
    total_loss *= 1/length(data.states)
    return total_loss
end