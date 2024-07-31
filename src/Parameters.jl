abstract type AbstractParams end

Base.@kwdef mutable struct GenericParams{NN<:Integer, CC<:ChainBlock, TT<:AbstractVector{T} where T<:Real} <: AbstractParams
    n::NN
    circ::CC
    params::TT = Float64[]
end

Base.@kwdef mutable struct InvariantParams{NN<:Integer, CC<:ChainBlock, AA<:Function, TT<:AbstractVector{T} where T<:Real} <: AbstractParams
    n::NN
    circ::CC
    ansatz::AA
    params::TT = Float64[]
end

function initialize_params(p::GenericParams)
    !ispow2(p.n) ? error("The register dimension has to be a power of 2") : nothing
    n_params = nparameters(p.circ)
    p.params = 2pi * rand(n_params)
end

function initialize_params(p::InvariantParams)
    !ispow2(p.n) ? error("The register dimension has to be a power of 2") : nothing
    n_layers = ceil(Int,log2(p.n))
    n_params = trunc(Int, nparameters(p.ansatz(p.n, 1, 2)) * n_layers)
    p.params = 2pi * rand(n_params)
end

expand_params(p::GenericParams) = p.params

function expand_params(p::InvariantParams)
    n_layers = ceil(Int, log2(p.n))
    active_qubits = map(i->ceil(Int, p.n/2^i), 0:n_layers-1)
    n_params = nparameters(p.ansatz(p.n, 1, 2))
    return vcat([repeat(p.params[(i-1)*n_params+1:i*n_params], active_qubits[i]) for i in 1:n_layers]...)
end

reduce_params(p::GenericParams, vect) = vect

function reduce_params(p::InvariantParams, vect)
    n_layers = ceil(Int, log2(p.n))
    active_qubits = map(i->ceil(Int, p.n/2^i), 0:n_layers-1)
    n_ansatz = nparameters(p.ansatz(p.n, 1, 2))
    res = []
    for i in 1:n_layers
        active_params = vect[sum(active_qubits[1:i-1])*n_ansatz+1 : sum(active_qubits[1:i])*n_ansatz]
        for j in 1:n_ansatz
            push!(res, sum(active_params[j:n_ansatz:j+n_ansatz*(active_qubits[i]-1)]))
        end
    end
    return res
end