# Exports
export AbstractModel,
       GeneralModel,
       InvariantModel,
       initialize_params,
       expand_params,
       reduce_params

abstract type AbstractModel end

Base.@kwdef mutable struct GeneralModel{NN<:Int, CC<:ChainBlock, AA<:Function, FF<:Function, VV<:Function, TT<:AbstractVector{T} where T<:Real} <: AbstractModel
    n::NN
    n_layers::NN = ceil(Int, log2(n))
    circ::CC
    ansatz::AA
    cost::FF
    activation::VV = identity
    params::TT = Float64[]
end

Base.@kwdef mutable struct InvariantModel{NN<:Int, CC<:ChainBlock, AA<:Function, FF<:Function, VV<:Function, TT<:AbstractVector{T} where T<:Real} <: AbstractModel
    n::NN
    n_layers::NN = ceil(Int, log2(n))
    circ::CC
    ansatz::AA
    cost::FF
    activation::VV = identity
    params::TT = Float64[]
end

function initialize_params(model::GeneralModel)
    !ispow2(model.n) ? error("The register dimension has to be a power of 2") : nothing
    n_params = nparameters(model.circ)
    model.params = 2pi * rand(n_params)
end

function initialize_params(model::InvariantModel)
    !ispow2(model.n) ? error("The register dimension has to be a power of 2") : nothing
    n_params = trunc(Int, nparameters(model.ansatz(model.n, 1, 2)) * model.n_layers)
    model.params = 2pi * rand(n_params)
end

expand_params(model::GeneralModel) = model.params

function expand_params(model::InvariantModel)
    active_qubits = map(i->ceil(Int, model.n/2^i), 0:model.n_layers-1)
    n_params = nparameters(model.ansatz(model.n, 1, 2))
    return vcat([repeat(model.params[(i-1)*n_params+1:i*n_params], active_qubits[i]) for i in 1:model.n_layers]...)
end

reduce_params(model::GeneralModel, vect) = vect

function reduce_params(model::InvariantModel, vect)
    active_qubits = map(i->ceil(Int, model.n/2^i), 0:model.n_layers-1)
    n_ansatz = nparameters(model.ansatz(model.n, 1, 2))
    res = []
    for i in 1:model.n_layers
        active_params = vect[sum(active_qubits[1:i-1])*n_ansatz+1 : sum(active_qubits[1:i])*n_ansatz]
        for j in 1:n_ansatz
            push!(res, sum(active_params[j:n_ansatz:j+n_ansatz*(active_qubits[i]-1)]))
        end
    end
    return res
end