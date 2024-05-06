#find grads for one state
function eval_grad(state, p::GenericParams)
    circ = p.circ
    dispatch!(circ, p.params)
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    _, grads = expect'(cost, copy(state)=>circ)
    return grads
end

#find grads for one state
function eval_grad(state, p::InvariantParams)
    circ = p.circ
    dispatch!(circ, expand_params(p))
    cost = sum(chain(p.n, put(1=>Z))) #Z gate on first qubit
    _, grads = expect'(cost, copy(state)=>circ)
    return grads
end

#finds full gradient of cost function - with or without sigmoid
function eval_full_grad(states, labels, p::GenericParams, sig)
    all_grads = []
    for i in eachindex(states)
        grad = eval_grad(states[i], p)
        loss = eval_loss(states[i], p)
        if sig == true
            push!(all_grads, (2*sigmoid(10*loss)-1-labels[i])*-2*sigmoid(10*loss)^2*-10*grad*exp(-10*loss))
        else
            push!(all_grads, grad*(loss-labels[i]))
        end
    end
    total_grads = 2/length(states)*sum(all_grads)
    return total_grads
end

#finds full gradient of cost function - with or without sigmoid
function eval_full_grad(states, labels, p::InvariantParams, sig)
    all_grads = []
    for i in eachindex(states)
        grad = eval_grad(states[i], p)
        loss = eval_loss(states[i], p)
        if sig == true
            push!(all_grads, (2*sigmoid(10*loss)-1-labels[i])*-2*sigmoid(10*loss)^2*-10*grad*exp(-10*loss))
        else
            push!(all_grads, grad*(loss-labels[i]))
        end
    end
    total_grads = 2/length(states)*sum(all_grads)
    return reduce_params(p.n, total_grads)
end