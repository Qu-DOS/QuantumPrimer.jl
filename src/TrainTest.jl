# Exports
export test_model,
       train_test_model
       
function test_model(data::AbstractData, model::Union{AbstractModel, NTuple{2, AbstractModel}}, cost::AbstractCost; lambda=1.::Float64, regularization=:nothing::Symbol)
    model_preds = zeros(Float64, length(data.states))
    model_labels = zeros(Float64, length(data.states))
    suc_inds = Vector{Int}()
    for i in eachindex(data.states)
        loss = eval_loss(data.states[i], model, cost; lambda=lambda, regularization=regularization)
        model_preds[i] = cost.activation(loss)
        model_labels[i] = Int(sign(model_preds[i])) # binary classification determined by sign of cost function
        model_labels[i] ≈ data.labels[i] ? push!(suc_inds, i) : nothing
    end
    suc_rate = length(suc_inds) / length(data.states)
    return model_preds, suc_rate, suc_inds
end

function train_test_model(data1::AbstractData,
                          data2::AbstractData,
                          model::AbstractModel,
                          cost::AbstractCost,
                          iters::Int,
                          optim::AbstractRule;
                          lambda=0.2::Float64,
                          regularization=:nothing::Symbol,
                          verbose=false::Bool)
    opt = Optimisers.setup(optim, model.params)
    loss_track = Float64[]
    tr_track = Float64[]
    te_track = Float64[]
    initial_loss = eval_full_loss(data1, model, cost; lambda=lambda, regularization=regularization)
    tr_preds, tr_acc, _ = test_model(data1, model, cost; lambda=lambda, regularization=regularization)
    te_preds, te_acc, _ = test_model(data2, model, cost; lambda=lambda, regularization=regularization)
    println("Initial: loss = $(initial_loss), tr_acc = $(tr_acc), te_acc = $(te_acc)")
    append!(loss_track, initial_loss)
    append!(tr_track, tr_acc)
    append!(te_track, te_acc)
    intervals = collect(0:iters÷8:iters)
    for i in 1:iters
        Optimisers.update!(opt, model.params, eval_full_grad(data1, model, cost; lambda=lambda, regularization=regularization))
        current_loss = eval_full_loss(data1, model, cost; lambda=lambda, regularization=regularization)
        tr_preds, tr_acc, _ = test_model(data1, model, cost; lambda=lambda, regularization=regularization)
        te_preds, te_acc, _ = test_model(data2, model, cost; lambda=lambda, regularization=regularization)
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

function train_test_model(data1::AbstractData,
                          data2::AbstractData,
                          models::NTuple{2, AbstractModel},
                          cost::AbstractCost,
                          iters::Int,
                          optim::AbstractRule;
                          lambda=0.2::Float64,
                          regularization=:nothing::Symbol,
                          verbose=false::Bool)
    model1, model2 = models
    all_params = vcat(model1.params, model2.params)
    opt = Optimisers.setup(optim, all_params)
    loss_track = Float64[]
    tr_track = Float64[]
    te_track = Float64[]
    initial_loss = eval_full_loss(data1, models, cost; lambda=lambda, regularization=regularization)
    tr_preds, tr_acc, _ = test_model(data1, models, cost; lambda=lambda, regularization=regularization)
    te_preds, te_acc, _ = test_model(data2, models, cost; lambda=lambda, regularization=regularization)
    println("Initial: loss = $(initial_loss), tr_acc = $(tr_acc), te_acc = $(te_acc)")
    append!(loss_track, initial_loss)
    append!(tr_track, tr_acc)
    append!(te_track, te_acc)
    intervals = collect(0:iters÷8:iters)
    for i in 1:iters
        Optimisers.update!(opt, all_params, eval_full_grad(data1, models, cost; lambda=lambda, regularization=regularization))
        model1.params = all_params[1:length(model1.params)]
        model2.params = all_params[length(model1.params)+1:end]
        current_loss = eval_full_loss(data1, models, cost; lambda=lambda, regularization=regularization)
        tr_preds, tr_acc, _ = test_model(data1, models, cost; lambda=lambda, regularization=regularization)
        te_preds, te_acc, _ = test_model(data2, models, cost; lambda=lambda, regularization=regularization)
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
    return model1.params, model2.params, loss_track, tr_track, te_track, tr_preds, te_preds
end