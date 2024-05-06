"""
    sigmoid(x)

Compute the sigmoid function.

## Arguments
- `x`: Input value.

## Returns
The sigmoid of the input value.

"""
sigmoid(x) = 1/(1+exp(-x))

"""
    eval_loss(state, p::GenericParams)

Evaluate the loss of a model (with no invariance) applied to a state.

## Arguments
- `state`: Input state.
- `p::GenericParams`: Model parameters.

## Returns
The loss value.

"""
function eval_loss(state, p::GenericParams)
    circ = p.circ
    dispatch!(circ, p.params)
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    return real(expect(cost, copy(state)|>circ)) #copy so that actual state isn't affected
end

"""
    eval_loss(state, p::InvariantParams)

Evaluate the loss of a translational invariant model applied to a state.

## Arguments
- `state`: Input state.
- `p::InvariantParams`: Model parameters.

## Returns
The loss value.

"""
function eval_loss(state, p::InvariantParams)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    return real(expect(cost, copy(state)|>circ)) #copy so that actual state isn't affected
end

"""
    eval_full_loss(d::Data, p::Params, sig)

Evaluate the full mean squared error (MSE) loss of a model on input data.

## Arguments
- `d::Data`: Input data and labels.
- `p::Params`: Model parameters.
- `sig`: Boolean indicating whether to use sigmoid activation. 

## Returns
The total mean squared error loss.

"""
function eval_full_loss(d::Data, p::Params, sig)
    total_loss = 0
    for i in eachindex(d.s)
        loss = eval_loss(d.s[i], p)
        if sig == true
            total_loss += (2*sigmoid(10*loss)-1-d.l[i])^2
        else
            total_loss += (loss-d.l[i])^2
        end
    end
    total_loss *= 1/length(d.s)
    return total_loss
end