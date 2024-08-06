# Exports
export eval_loss,
       eval_full_loss

function eval_loss(states::NTuple{2, ArrayReg}, model::AbstractModel)
    state1, state2 = states
    circ = model.circ
    dispatch!(circ, expand_params(model))
    return model.cost(copy(state1) |> circ, copy(state2) |> circ)
end

function eval_loss(state::ArrayReg, model::AbstractModel)
    circ = model.circ
    dispatch!(circ, expand_params(model))
    return real(expect(model.cost(model.n), copy(state) |> circ)) #copy so that actual state isn't affected
end

function eval_full_loss(data::AbstractData, model::AbstractModel; sig=true::Bool)
    total_loss = 0
    for i in eachindex(data.states)
        loss = eval_loss(data.states[i], model)
        if sig == true
            total_loss += (2*sigmoid(10*loss)-1-data.labels[i])^2
        else
            total_loss += (loss-data.labels[i])^2
        end
    end
    total_loss *= 1/length(data.states)
    return total_loss
end