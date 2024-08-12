# Exports
export test_model,
       train_test_model
       
function test_model(data::AbstractData, model::Union{AbstractModel, NTuple{2, AbstractModel}}; lambda=1.::Float64, regularization=:nothing::Symbol)
    preds = zeros(Float64, length(data.states))
    suc_inds = Vector{Int}()
    for i in eachindex(data.states)
        loss = eval_loss(data.states[i], model; lambda=lambda, regularization=regularization)
        model_pred = 0
        if typeof(model) <: AbstractModel
            fx = model.activation(loss)
        elseif typeof(model) <: NTuple{2, AbstractModel}
            fx = model[1].activation(loss)
        end
        fx>0 ? model_pred=1 : model_pred=-1 # binary classification determined by sign of cost function
        preds[i] = fx
        model_pred ≈ data.labels[i] ? push!(suc_inds, i) : nothing
    end
    suc_rate = length(suc_inds)/length(data.states)
    return preds, suc_rate, suc_inds
end

function train_test_model(
        data1::AbstractData,
        data2::AbstractData,
        model::Union{AbstractModel, NTuple{2, AbstractModel}},
        iters::Int,
        optim::AbstractRule;
        lambda=0.2::Float64,
        regularization=:nothing::Symbol,
        verbose=false::Bool)
    if typeof(model) <: AbstractModel
        opt = Optimisers.setup(optim, model.params)
        loss_track = Float64[]
        tr_track = Float64[]
        te_track = Float64[]
        initial_loss = eval_full_loss(data1, model; lambda=lambda, regularization=regularization)
        tr_preds, tr_acc, _ = test_model(data1, model; lambda=lambda, regularization=regularization)
        te_preds, te_acc, _ = test_model(data2, model; lambda=lambda, regularization=regularization)
        println("Initial: loss = $(initial_loss), tr_acc = $(tr_acc), te_acc = $(te_acc)")
        append!(loss_track, initial_loss)
        append!(tr_track, tr_acc)
        append!(te_track, te_acc)
        intervals = collect(0:10:iters)
        for i in 1:iters
            Optimisers.update!(opt, model.params, eval_full_grad(data1, model; lambda=lambda, regularization=regularization))
            current_loss = eval_full_loss(data1, model; lambda=lambda, regularization=regularization)
            tr_preds, tr_acc, _ = test_model(data1, model; lambda=lambda, regularization=regularization)
            te_preds, te_acc, _ = test_model(data2, model; lambda=lambda, regularization=regularization)
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
    elseif typeof(model) <: NTuple{2, AbstractModel}
        model1, model2 = model
        opt1 = Optimisers.setup(optim, model1.params)
        opt2 = Optimisers.setup(optim, model2.params)
        loss_track = Float64[]
        tr_track = Float64[]
        te_track = Float64[]
        initial_loss = eval_full_loss(data1, model; lambda=lambda, regularization=regularization)
        tr_preds, tr_acc, _ = test_model(data1, model; lambda=lambda, regularization=regularization)
        te_preds, te_acc, _ = test_model(data2, model; lambda=lambda, regularization=regularization)
        println("Initial: loss = $(initial_loss), tr_acc = $(tr_acc), te_acc = $(te_acc)")
        append!(loss_track, initial_loss)
        append!(tr_track, tr_acc)
        append!(te_track, te_acc)
        intervals = collect(0:10:iters)
        for i in 1:iters
            Optimisers.update!(opt1, model1.params, eval_full_grad(data1, model, 1; lambda=lambda, regularization=regularization))
            Optimisers.update!(opt2, model2.params, eval_full_grad(data2, model, 2; lambda=lambda, regularization=regularization))
            current_loss = eval_full_loss(data1, model; lambda=lambda, regularization=regularization)
            tr_preds, tr_acc, _ = test_model(data1, model; lambda=lambda, regularization=regularization)
            te_preds, te_acc, _ = test_model(data2, model; lambda=lambda, regularization=regularization)
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
    end
    println("Final: loss = $(loss_track[end]), tr_acc = $(tr_track[end]), te_acc = $(te_track[end])")
    return model.params, loss_track, tr_track, te_track, tr_preds, te_preds
end