export test_model,
       train_test_model
       
"""
    test_model(data::AbstractData, model::Union{AbstractModel, NTuple{2, AbstractModel}}, cost::AbstractCost; lambda=1.0::Float64, regularization=:nothing::Symbol) -> Tuple{Vector{Float64}, Float64, Vector{Int}}

Tests a quantum model on a given dataset.

# Arguments
- `data::AbstractData`: The dataset containing quantum states and corresponding labels.
- `model::Union{AbstractModel, NTuple{2, AbstractModel}}`: The quantum model or tuple of models.
- `cost::AbstractCost`: The cost function.
- `lambda::Float64`: The regularization parameter, default is 1.0.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.

# Returns
- `Tuple{Vector{Float64}, Float64, Vector{Int}}`: A tuple containing the model predictions, success rate, and indices of successful predictions.
"""
function test_model(data::AbstractData, model::Union{AbstractModel, NTuple{<:Any, AbstractModel}}, cost::AbstractCost; lambda=1.0::Float64, regularization=:nothing::Symbol)
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

"""
    train_test_model(data1::AbstractData,
                     data2::AbstractData,
                     model::AbstractModel,
                     cost::AbstractCost,
                     iters::Int,
                     optim::AbstractRule;
                     lambda=0.2::Float64,
                     regularization=:nothing::Symbol,
                     verbose=false::Bool) -> Tuple{AbstractVector{T} where T<:Real, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Trains and tests a quantum model on given datasets.

# Arguments
- `data1::AbstractData`: The training dataset.
- `data2::AbstractData`: The testing dataset.
- `model::AbstractModel`: The quantum model.
- `cost::AbstractCost`: The cost function.
- `iters::Int`: The number of training iterations.
- `optim::AbstractRule`: The optimization rule.
- `lambda::Float64`: The regularization parameter, default is 0.2.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.
- `verbose::Bool`: Whether to print progress information, default is false.

# Returns
- `Tuple{AbstractVector{T} where T<:Real, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}`: A tuple containing the model parameters, loss track, training accuracy track, testing accuracy track, training predictions, and testing predictions.
"""
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

"""
    train_test_model(data1::AbstractData,
                     data2::AbstractData,
                     models::NTuple{<:Any, AbstractModel},
                     cost::AbstractCost,
                     iters::Int,
                     optim::AbstractRule;
                     lambda=0.2::Float64,
                     regularization=:nothing::Symbol,
                     verbose=false::Bool) -> Tuple{NTuple{<:Any, AbstractModel}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Trains and tests a tuple of quantum models on given datasets.

# Arguments
- `data1::AbstractData`: The training dataset.
- `data2::AbstractData`: The testing dataset.
- `models::NTuple{<:Any, AbstractModel}`: The tuple of quantum models.
- `cost::AbstractCost`: The cost function.
- `iters::Int`: The number of training iterations.
- `optim::AbstractRule`: The optimization rule.
- `lambda::Float64`: The regularization parameter, default is 0.2.
- `regularization::Symbol`: The type of regularization (`:l1` or `:l2`), default is `:nothing`.
- `verbose::Bool`: Whether to print progress information, default is false.

# Returns
- `Tuple{NTuple{<:Any, AbstractModel}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}`: A tuple containing all models, loss track, training accuracy track, testing accuracy track, training predictions, and testing predictions.
"""
function train_test_model(data1::AbstractData,
                          data2::AbstractData,
                          models::NTuple{<:Any, AbstractModel},
                          cost::AbstractCost,
                          iters::Int,
                          optim::AbstractRule;
                          lambda=0.2::Float64,
                          regularization=:nothing::Symbol,
                          verbose=false::Bool)
    all_params = vcat([model.params for model in models]...)
    n_params = length(models[1].params)
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
        for i in eachindex(models)
            models[i].params = all_params[(i-1)*n_params+1:i*n_params]
        end
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
    return models, loss_track, tr_track, te_track, tr_preds, te_preds
end