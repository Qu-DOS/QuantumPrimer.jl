#sigmoid function
sigmoid(x) = 1/(1+exp(-x))

#evaluates model applied to state
function eval_loss(n,state,params; invariant=false)
    circ = build_QCNN(n)
    if invariant
        dispatch!(circ, expand_params(params))
    else
        dispatch!(circ, params)
    end
    cost=sum(chain(n,put(1=>Z))) #Z gate on first qubit
    return real(expect(cost,copy(state)|>circ)) #copy so that actual state isn't affected
end

#evaluate full mse - with or without sigmoid (without seems to work better overall)
#assuming a binary classification - one class labelled 1, other class labelled -1
function eval_full_loss(n,states,labels,params,sig; invariant=false)
    total_loss = 0
    for i in eachindex(states)
        loss = eval_loss(n,states[i],params; invariant=invariant)
        if sig == true
            total_loss+=(2*sigmoid(10*loss)-1-labels[i])^2
        else
            total_loss+=(loss-labels[i])^2
        end
    end
    total_loss*=1/length(states)
    return total_loss
end