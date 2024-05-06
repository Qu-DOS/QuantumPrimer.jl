#evaluate classification accuracy of model - with or without sigmoid
function test_model(states, labels, p::Params, sig)
    suc = 0
    preds = []
    suc_inds = []
    for i in eachindex(states)
        loss = eval_loss(states[i], p)
        model_pred = 0
        true_pred = labels[i] #(assuming binary labels ±1)
        fx = 0
        if sig == true
            fx = 2*sigmoid(10*loss)-1
        else
            fx = loss
        end
        if fx>0 #binary classification determined by sign of cost function
            model_pred = 1
        else
            model_pred = -1
        end
        push!(preds, fx)
        if model_pred ≈ true_pred
            suc += 1
            push!(suc_inds, i)
        end
    end
    suc_rate = suc/length(states)
    return preds, suc_rate, suc_inds
end

#trains model - with or without sigmoid, outputs final predictions + train/test accuracy. 
#Adam usually works fine, usually set learning_rate = 0.1 or reduce to 0.01 if optimisation is not smooth
function train_test_model(states, labels, test_states, test_labels, p::Params, iters, sig, lr; output=false)
    opt = Optimisers.setup(Optimisers.ADAM(lr), p.params)
    loss_track = zeros(0)
    tr_track = zeros(0)
    te_track = zeros(0)
    l1 = eval_full_loss(states, labels, p, sig)
    tr_preds,tr_acc,inds1 = test_model(states, labels, p, sig)
    te_preds,te_acc,inds2 = test_model(test_states, test_labels, p, sig)
    println("Initial: loss = $(l1),tr_acc = $(tr_acc), te_acc = $(te_acc)")
    append!(loss_track, l1)
    append!(tr_track, tr_acc)
    append!(te_track, te_acc)
    intervals = collect(0:10:iters)
    for i in 1:iters
        Optimisers.update!(opt, p.params, eval_full_grad(states, labels, p, sig))
        l1 = eval_full_loss(states, labels, p, sig)
        tr_preds, tr_acc, inds1 = test_model(states, labels, p, sig)
        te_preds, te_acc, inds2 = test_model(test_states, test_labels, p, sig)
        append!(loss_track, l1)
        append!(tr_track, tr_acc)
        append!(te_track, te_acc)
        if output
            i in intervals ? println("Iteration $(i): loss = $(l1), tr_acc = $(tr_acc), te_acc = $(te_acc)") : nothing
        end
        l1 < 1e-12 ? break : nothing
    end
    l1 = loss_track[end]
    tr_acc = tr_track[end]
    te_acc = te_track[end]
    println("Final: loss = $(l1),tr_acc = $(tr_acc), te_acc = $(te_acc)")
    return p.params, loss_track, tr_track, te_track, tr_preds, te_preds
end
