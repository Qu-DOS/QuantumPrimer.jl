# Exports
export AbstractModel,
       GeneralModel,
       InvariantModel,
       initialize_params!,
       expand_params,
       reduce_params

"""
    AbstractModel

An abstract type for representing quantum models.
"""
abstract type AbstractModel end

"""
    GeneralModel{NN<:Int, CC<:ChainBlock, AA<:Function, TT<:AbstractVector{T} where T<:Real}

A mutable struct representing a general quantum model.

# Fields
- `n::NN`: The number of qubits.
- `n_layers::NN`: The number of layers in the model, default is `ceil(Int, log2(n))`.
- `circ::CC`: The quantum circuit.
- `ansatz::AA`: The ansatz function.
- `params::TT`: The parameters of the model, default is an empty vector.

# Constructor
- `GeneralModel(n::NN, circ::CC, ansatz::AA; n_layers::NN=ceil(Int, log2(n)), params::TT=Float64[])`: Creates a `GeneralModel` instance.
"""
Base.@kwdef mutable struct GeneralModel{NN<:Int, CC<:ChainBlock, AA<:Function, TT<:AbstractVector{T} where T<:Real} <: AbstractModel
    n::NN
    n_layers::NN = ceil(Int, log2(n))
    circ::CC
    ansatz::AA
    params::TT = Float64[]
end

"""
    InvariantModel{NN<:Int, CC<:ChainBlock, AA<:Function, TT<:AbstractVector{T} where T<:Real}

A mutable struct representing an invariant quantum model.

# Fields
- `n::NN`: The number of qubits.
- `n_layers::NN`: The number of layers in the model, default is `ceil(Int, log2(n))`.
- `circ::CC`: The quantum circuit.
- `ansatz::AA`: The ansatz function.
- `params::TT`: The parameters of the model, default is an empty vector.

# Constructor
- `InvariantModel(n::NN, circ::CC, ansatz::AA; n_layers::NN=ceil(Int, log2(n)), params::TT=Float64[])`: Creates an `InvariantModel` instance.
"""
Base.@kwdef mutable struct InvariantModel{NN<:Int, CC<:ChainBlock, AA<:Function, TT<:AbstractVector{T} where T<:Real} <: AbstractModel
    n::NN
    n_layers::NN = ceil(Int, log2(n))
    circ::CC
    ansatz::AA
    params::TT = Float64[]
end

"""
    initialize_params!(model::GeneralModel)

Initializes the parameters of a `GeneralModel` instance.

# Arguments
- `model::GeneralModel`: The general quantum model.

# Throws
- An error if the number of qubits is not a power of 2.
"""
function initialize_params!(model::GeneralModel)
    !ispow2(model.n) ? error("The register dimension has to be a power of 2") : nothing
    n_params = nparameters(model.circ)
    model.params = 2pi * rand(n_params)
end

"""
    initialize_params!(model::InvariantModel)

Initializes the parameters of an `InvariantModel` instance.

# Arguments
- `model::InvariantModel`: The invariant quantum model.

# Throws
- An error if the number of qubits is not a power of 2.
"""
function initialize_params!(model::InvariantModel)
    !ispow2(model.n) ? error("The register dimension has to be a power of 2") : nothing
    n_params = trunc(Int, nparameters(model.ansatz(model.n, 1, 2)) * model.n_layers)
    model.params = 2pi * rand(n_params)
end

"""
    expand_params(model::GeneralModel) -> AbstractVector{T} where T<:Real

Expands the parameters of a `GeneralModel` instance.

# Arguments
- `model::GeneralModel`: The general quantum model.

# Returns
- `AbstractVector{T} where T<:Real`: The expanded parameters.
"""
expand_params(model::GeneralModel) = model.params

"""
    expand_params(model::InvariantModel) -> AbstractVector{T} where T<:Real

Expands the parameters of an `InvariantModel` instance.

# Arguments
- `model::InvariantModel`: The invariant quantum model.

# Returns
- `AbstractVector{T} where T<:Real`: The expanded parameters.
"""
function expand_params(model::InvariantModel)
    active_qubits = map(i->ceil(Int, model.n/2^i), 0:model.n_layers-1)
    n_params = nparameters(model.ansatz(model.n, 1, 2))
    return vcat([repeat(model.params[(i-1)*n_params+1:i*n_params], active_qubits[i]) for i in 1:model.n_layers]...)
end

"""
    reduce_params(model::GeneralModel, vect::AbstractVector{T} where T<:Real) -> AbstractVector{T} where T<:Real

Reduces the parameters of a `GeneralModel` instance.

# Arguments
- `model::GeneralModel`: The general quantum model.
- `vect::AbstractVector{T} where T<:Real`: The vector of parameters to be reduced.

# Returns
- `AbstractVector{T} where T<:Real`: The reduced parameters.
"""
reduce_params(model::GeneralModel, vect) = vect

"""
    reduce_params(model::InvariantModel, vect::AbstractVector{T} where T<:Real) -> AbstractVector{T} where T<:Real

Reduces the parameters of an `InvariantModel` instance.

# Arguments
- `model::InvariantModel`: The invariant quantum model.
- `vect::AbstractVector{T} where T<:Real`: The vector of parameters to be reduced.

# Returns
- `AbstractVector{T} where T<:Real`: The reduced parameters.
"""
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