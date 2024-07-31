function test_model(d::Data, p::AbstractParams, sig)
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

function train_test_model(d1::Data, d2::Data, p::AbstractParams, iters, optim, sig; output=false)
    opt = Optimisers.setup(optim, p.params)
    loss_track = Float64[]
    tr_track = Float64[]
    te_track = Float64[]
    initial_loss = eval_full_loss(d1, p, sig)
    tr_preds, tr_acc, _ = test_model(d1, p, sig)
    te_preds, te_acc, _ = test_model(d2, p, sig)
    println("Initial: loss = $(initial_loss), tr_acc = $(tr_acc), te_acc = $(te_acc)")
    append!(loss_track, initial_loss)
    append!(tr_track, tr_acc)
    append!(te_track, te_acc)
    intervals = collect(0:10:iters)
    for i in 1:iters
        Optimisers.update!(opt, p.params, eval_full_grad(d1, p, sig))
        current_loss = eval_full_loss(d1, p, sig)
        tr_preds, tr_acc, _ = test_model(d1, p, sig)
        te_preds, te_acc, _ = test_model(d2, p, sig)
        append!(loss_track, current_loss)
        append!(tr_track, tr_acc)
        append!(te_track, te_acc)
        if output && (i in intervals)
            println("Iteration $(i): loss = $(current_loss), tr_acc = $(tr_acc), te_acc = $(te_acc)")
        end
        if current_loss < 1e-12
            break
        end
    end
    println("Final: loss = $(loss_track[end]), tr_acc = $(tr_track[end]), te_acc = $(te_track[end])")
    return p.params, loss_track, tr_track, te_track, tr_preds, te_preds
end