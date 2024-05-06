#evaluate classification accuracy of model - with or without sigmoid
function test_model(n,states,labels,params,sig; invariant=false)
    suc = 0
    preds = []
    suc_inds = []
    for i in eachindex(states)
        loss = eval_loss(n,states[i],params; invariant=invariant)
        model_pred = 0
        true_pred = labels[i] #(assuming binary labels ±1)
        fx=0
        if sig == true
            fx = 2*sigmoid(10*loss)-1
        else
            fx = loss
        end
        if fx>0 #binary classification determined by sign of cost function
            model_pred=1
        else
            model_pred=-1
        end
        push!(preds,fx)
        if model_pred ≈ true_pred
            suc+=1
            push!(suc_inds,i)
        end
    end
    suc_rate = suc/length(states)
    return preds, suc_rate, suc_inds
end

#trains model - with or without sigmoid, outputs final predictions + train/test accuracy. 
#Adam usually works fine, usually set learning_rate = 0.1 or reduce to 0.01 if optimisation is not smooth
function train_test_model(n,states,labels,test_states,test_labels,params,iters,sig,lr; invariant=false, output=false)
    opt = Optimisers.setup(Optimisers.ADAM(lr), params)
    loss_track=zeros(0)
    train_track=zeros(0)
    test_track=zeros(0)
    l1=eval_full_loss(n,states,labels,params,sig; invariant=invariant)
    tr_preds,tr_acc,inds1 = test_model(n,states,labels,params,sig; invariant=invariant)
    te_preds,te_acc,inds2 = test_model(n,test_states,test_labels,params,sig; invariant=invariant)
    println("Initial: loss = $(l1),tr_acc = $(tr_acc), te_acc = $(te_acc)")
    append!(loss_track,l1)
    append!(train_track,tr_acc)
    append!(test_track,te_acc)
    intervals = collect(0:10:iters)
    for i in 1:iters
        Optimisers.update!(opt, params, eval_full_grad(n,states,labels,params,sig; invariant=invariant))
        l1=eval_full_loss(n,states,labels,params,sig; invariant=invariant)
        tr_preds,tr_acc,inds1 = test_model(n,states,labels,params,sig; invariant=invariant)
        te_preds,te_acc,inds2 = test_model(n,test_states,test_labels,params,sig; invariant=invariant)
        append!(loss_track,l1)
        append!(train_track,tr_acc)
        append!(test_track,te_acc)
        if output
            if i in intervals
                println("Iteration $(i): loss = $(l1), tr_acc = $(tr_acc), te_acc = $(te_acc)")
            end
        end
        if l1 < 1e-12
            break
        end
    end
    l1=loss_track[end]
    tr_acc = train_track[end]
    te_acc= test_track[end]
    println("Final: loss = $(l1),tr_acc = $(tr_acc), te_acc = $(te_acc)")
    return params,loss_track,train_track,test_track,tr_preds,te_preds
end
