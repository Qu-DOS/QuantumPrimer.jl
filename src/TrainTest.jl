"""
    test_model(d::Data, p::Params, sig)

Evaluate classification accuracy of the model.

## Arguments
- `d::Data`: Input data and labels.
- `p::Params`: Model parameters.
- `sig`: Boolean indicating whether to use sigmoid activation.

## Returns
- `preds`: Predicted values.
- `suc_rate`: Success rate.
- `suc_inds`: Indices of successful predictions.

"""
function test_model(d::Data, p::Params, sig)
    preds = []
    suc_inds = []
    for i in eachindex(d.s)
        loss = eval_loss(d.s[i], p)
        model_pred = 0
        fx = 0
        sig==true ? fx=2*sigmoid(10*loss)-1 : fx=loss
        fx>0 ? model_pred=1 : model_pred=-1 # binary classification determined by sign of cost function
        push!(preds, fx)
        model_pred ≈ d.l[i] ? push!(suc_inds, i) : nothing
    end
    suc_rate = length(suc_inds)/length(d.s)
    return preds, suc_rate, suc_inds
end

"""
    train_test_model(d1::Data, d2::Data, p::Params, iters, sig, lr; output=false)

Train and test the model, outputting final predictions and train/test accuracy.

## Arguments
- `d1::Data`: Training data and labels.
- `d2::Data`: Testing data and labels.
- `p::Params`: Initial model parameters.
- `iters`: Number of training iterations.
- `sig`: Boolean indicating whether to use sigmoid activation.
- `lr`: Learning rate for optimization.
- `output`: Boolean indicating whether to output intermediate training information.

## Returns
- `p.params`: Final model parameters.
- `loss_track`: Track of loss values during training.
- `tr_track`: Track of training accuracy during training.
- `te_track`: Track of testing accuracy during training.
- `tr_preds`: Predictions on training data.
- `te_preds`: Predictions on testing data.

"""
function train_test_model(d1::Data, d2::Data, p::Params, iters, sig, lr; output=false)
    opt = Optimisers.setup(Optimisers.ADAM(lr), p.params)
    loss_track = zeros(0)
    tr_track = zeros(0)
    te_track = zeros(0)
    l1 = eval_full_loss(d1, p, sig)
    tr_preds,tr_acc,inds1 = test_model(d1, p, sig)
    te_preds,te_acc,inds2 = test_model(d2, p, sig)
    println("Initial: loss = $(l1),tr_acc = $(tr_acc), te_acc = $(te_acc)")
    append!(loss_track, l1)
    append!(tr_track, tr_acc)
    append!(te_track, te_acc)
    intervals = collect(0:10:iters)
    for i in 1:iters
        Optimisers.update!(opt, p.params, eval_full_grad(d1, p, sig))
        l1 = eval_full_loss(d1, p, sig)
        tr_preds, tr_acc, inds1 = test_model(d1, p, sig)
        te_preds, te_acc, inds2 = test_model(d2, p, sig)
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
