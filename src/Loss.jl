#sigmoid function
sigmoid(x) = 1/(1+exp(-x))

#evaluates model applied to state
function eval_loss(state, p::GenericParams)
    circ = p.circ
    dispatch!(circ, p.params)
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    return real(expect(cost, copy(state)|>circ)) #copy so that actual state isn't affected
end

#evaluates model applied to state
function eval_loss(state, p::InvariantParams)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    return real(expect(cost, copy(state)|>circ)) #copy so that actual state isn't affected
end

#evaluate full mse - with or without sigmoid (without seems to work better overall)
#assuming a binary classification - one class labelled 1, other class labelled -1
function eval_full_loss(states, labels, p::Params, sig)
    total_loss = 0
    for i in eachindex(states)
        loss = eval_loss(states[i], p)
        if sig == true
            total_loss += (2*sigmoid(10*loss)-1-labels[i])^2
        else
            total_loss += (loss-labels[i])^2
        end
    end
    total_loss *= 1/length(states)
    return total_loss
end