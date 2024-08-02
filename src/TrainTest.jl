function test_model(data::AbstractData, model::AbstractModel; sig=true::Bool)
    preds = []
    suc_inds = []
    for i in eachindex(data.states)
        loss = eval_loss(data.states[i], model)
        model_pred = 0
        fx = 0
        sig==true ? fx=2*sigmoid(10*loss)-1 : fx=loss
        fx>0 ? model_pred=1 : model_pred=-1 # binary classification determined by sign of cost function
        push!(preds, fx)
        model_pred ≈ data.labels[i] ? push!(suc_inds, i) : nothing
    end
    suc_rate = length(suc_inds)/length(data.states)
    return preds, suc_rate, suc_inds
end

function train_test_model(data1::AbstractData, data2::AbstractData, model::AbstractModel, iters::Int, optim; sig=true::Bool, verbose=false)
    opt = Optimisers.setup(optim, model.params)
    loss_track = Float64[]
    tr_track = Float64[]
    te_track = Float64[]
    initial_loss = eval_full_loss(data1, model; sig=sig)
    tr_preds, tr_acc, _ = test_model(data1, model; sig=sig)
    te_preds, te_acc, _ = test_model(data2, model; sig=sig)
    println("Initial: loss = $(initial_loss), tr_acc = $(tr_acc), te_acc = $(te_acc)")
    append!(loss_track, initial_loss)
    append!(tr_track, tr_acc)
    append!(te_track, te_acc)
    intervals = collect(0:10:iters)
    for i in 1:iters
        Optimisers.update!(opt, model.params, eval_full_grad(data1, model; sig=sig))
        current_loss = eval_full_loss(data1, model; sig=sig)
        tr_preds, tr_acc, _ = test_model(data1, model; sig=sig)
        te_preds, te_acc, _ = test_model(data2, model; sig=sig)
        append!(loss_track, current_loss)
        append!(tr_track, tr_acc)
        append!(te_track, te_acc)
        if verbose && (i in intervals)
            println("Iteration $(i): loss = $(current_loss), tr_acc = $(tr_acc), te_acc = $(te_acc)")
        end
        if current_loss < 1e-12
            break
        end
    end
    println("Final: loss = $(loss_track[end]), tr_acc = $(tr_track[end]), te_acc = $(te_track[end])")
    return model.params, loss_track, tr_track, te_track, tr_preds, te_preds
end