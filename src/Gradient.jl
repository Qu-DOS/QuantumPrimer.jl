"""
    eval_grad(state, p::GenericParams)

Compute the gradient of the cost function for a given state.

## Arguments
- `state`: Input state.
- `p::GenericParams`: Model parameters.

## Returns
The gradient of the cost function.

"""
function eval_grad(state, p::GenericParams)
    circ = p.circ
    dispatch!(circ, p.params)
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    _, grads = expect'(cost, copy(state)=>circ)
    return grads
end

"""
    eval_grad(state, p::InvariantParams)

Compute the gradient of the cost function for a given state with parameters with translational invariance.

## Arguments
- `state`: Input state.
- `p::InvariantParams`: Model parameters.

## Returns
The gradient of the cost function.

"""
function eval_grad(state, p::InvariantParams)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    _, grads = expect'(cost, copy(state)=>circ)
    return grads
end

"""
    eval_full_grad(d::Data, p::GenericParams, sig)

Compute the full gradient of the cost function for input data.

## Arguments
- `d::Data`: Input data and labels.
- `p::GenericParams`: Model parameters.
- `sig`: Boolean indicating whether to use sigmoid activation.

## Returns
The total gradient of the cost function.

"""
#finds full gradient of cost function - with or without sigmoid
function eval_full_grad(d::Data, p::GenericParams, sig)
    all_grads = []
    for i in eachindex(d.s)
        grad = eval_grad(d.s[i], p)
        loss = eval_loss(d.s[i], p)
        if sig == true
            push!(all_grads, (2*sigmoid(10*loss)-1-d.l[i])*-2*sigmoid(10*loss)^2*-10*grad*exp(-10*loss))
        else
            push!(all_grads, grad*(loss-d.l[i]))
        end
    end
    total_grads = 2/length(d.s)*sum(all_grads)
    return total_grads
end

"""
    eval_full_grad(d::Data, p::InvariantParams, sig)

Compute the full gradient of the cost function for input data with parameters with translational invariance.

## Arguments
- `d::Data`: Input data and labels.
- `p::InvariantParams`: Model parameters.
- `sig`: Boolean indicating whether to use sigmoid activation.

## Returns
The total gradient of the cost function.

"""
function eval_full_grad(d::Data, p::InvariantParams, sig)
    all_grads = []
    for i in eachindex(d.s)
        grad = eval_grad(d.s[i], p)
        loss = eval_loss(d.s[i], p)
        if sig == true
            push!(all_grads, (2*sigmoid(10*loss)-1-d.l[i])*-2*sigmoid(10*loss)^2*-10*grad*exp(-10*loss))
        else
            push!(all_grads, grad*(loss-d.l[i]))
        end
    end
    total_grads = 2/length(d.s)*sum(all_grads)
    return reduce_params(p.n, total_grads,p)
end