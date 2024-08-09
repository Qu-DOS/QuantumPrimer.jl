# Exports
export eval_loss,
       eval_full_loss

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
    if regularization == :l1
        l1 = sum(abs(param) for param in model.params)
        loss += lambda * l1
    elseif regularization == :l2
        l2 = sum(param^2 for param in model.params)
        loss += lambda * l2
    end
    return loss
end

function eval_loss(state::ArrayReg, model::AbstractModel; lambda=1.::Float64, regularization=:nothing::Symbol)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    loss = expect(model.cost(model.n), copy(state) |> circ) # copy so that actual state isn't affected
    if regularization == :l1
        l1 = sum(abs(param) for param in model.params)
        loss += lambda * l1
    elseif regularization == :l2
        l2 = sum(param^2 for param in model.params)
        loss += lambda * l2
    end
    return loss
end

function eval_full_loss(data::AbstractData, model::AbstractModel; sig=true::Bool, lambda=1.::Float64, regularization=:nothing::Symbol)
    total_loss = 0
    for i in eachindex(data.states)
        loss = eval_loss(data.states[i], model; lambda=lambda, regularization=regularization)
        if sig == true
            total_loss += (2*sigmoid(10*loss)-1-data.labels[i])^2
        else
            total_loss += (loss-data.labels[i])^2
        end
    end
    total_loss *= 1/length(data.states)
    return total_loss
end