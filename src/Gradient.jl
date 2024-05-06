#find grads for one state
function eval_grad(n,state,params; invariant=false)
    circ = build_QCNN(n)
    if invariant
        dispatch!(circ, expand_params(params))
    else
        dispatch!(circ,params)
    end
    cost=sum(chain(n,put(1=>Z))) #Z gate on first qubit
    _, grads = expect'(cost,copy(state)=>circ)
    return grads
end

#finds full gradient of cost function - with or without sigmoid
function eval_full_grad(n,states,labels,params,sig; invariant=false)
    all_grads = []
    for i in eachindex(states)
        grad = eval_grad(n,states[i],params; invariant=invariant)
        loss = eval_loss(n,states[i],params; invariant=invariant)
        if sig == true
            push!(all_grads, (2*sigmoid(10*loss)-1-labels[i])*-2*sigmoid(10*loss)^2*-10*grad*exp(-10*loss))
        else
            push!(all_grads, grad*(loss-labels[i]))
        end
    end
    total_grads = 2/length(states)*sum(all_grads)
    if invariant
        res = reduce_params(n, total_grads)
    else
        res = total_grads
    end
    return res
end