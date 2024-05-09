"""
    abstract type Params end

Abstract type representing parameters for the QCNN.

"""
abstract type Params end

"""
    Base.@kwdef mutable struct GenericParams{NN<:Integer, CC<:ChainBlock, TT<:AbstractVector{T} where T<:Real} <:Params

Defines parameters for a quantum circuit without translational invariance.

## Fields
- `n::NN`: Dimension of the quantum register (must be a power of 2).
- `circ::CC`: Circuit structure of the QCNN.
- `params::TT = Float64[]`: Vector of parameters, initialized to an empty vector of Float64 values.

"""
Base.@kwdef mutable struct GenericParams{NN<:Integer, CC<:ChainBlock, TT<:AbstractVector{T} where T<:Real} <:Params
    n::NN
    circ::CC
    params::TT = Float64[]
end

"""
    Base.@kwdef mutable struct InvariantParams{NN<:Integer, CC<:ChainBlock, TT<:AbstractVector{T} where T<:Real} <:Params

Defines parameters for a quantum circuit with translational invariance.

## Fields
- `n::NN`: Dimension of the quantum register (must be a power of 2).
- `circ::CC`: Circuit structure of the QCNN.
- `params::TT = Float64[]`: Vector of parameters, initialized to an empty vector of Float64 values.

"""
Base.@kwdef mutable struct InvariantParams{NN<:Integer, CC<:ChainBlock, AA<:Function, TT<:AbstractVector{T} where T<:Real} <:Params
    n::NN
    circ::CC
    ansatz::AA
    params::TT = Float64[]
end

"""
    initialize_params(p::GenericParams)

Initialize parameters for a quantum circuit without translational invariance.

## Arguments
- `p::GenericParams`: Parameters object containing circuit information.

## Errors
- Throws an error if the register dimension is not a power of 2.

"""
#initial params (no translational invariance). Take in a seed for reproducibility
function initialize_params(p::GenericParams)
    !ispow2(p.n) ? error("The register dimension has to be a power of 2") : nothing
    n_params = nparameters(p.circ)
    p.params = 2pi * rand(n_params)
end

"""
    initialize_params(p::InvariantParams)

Initialize parameters for a quantum circuit with translational invariance.

## Arguments
- `p::InvariantParams`: Parameters object containing circuit information.

## Errors
- Throws an error if the register dimension is not a power of 2.

"""
function initialize_params(p::InvariantParams)
    !ispow2(p.n) ? error("The register dimension has to be a power of 2") : nothing
    #n_params = trunc(Int, 2*log2(p.n)) # extended vector dim is trunc(Int, 2*sum([2^i for i in 1:log2(n)]))
    n_layers=ceil(Int,log2(p.n))
    n_params = trunc(Int,nparameters(p.ansatz(p.n,1,2))*n_layers)
    p.params = 2pi * rand(n_params)
end

"""
    expand_params(p::InvariantParams)

Expand unique translational invariant parameters to a full vector.

## Arguments
- `p::InvariantParams`: Parameters object containing circuit information.

## Returns
A vector representing expanded parameters.

"""
function expand_params(p::InvariantParams)
    n_layers = ceil(Int, log2(p.n))
    active_qs = map(i->ceil(Int, p.n/2^i), 0:n_layers-1)
    return vcat([repeat(p.params[(i-1)*nparameters(p.ansatz(p.n, 1, 2))+1:i*nparameters(p.ansatz(p.n, 1, 2))], active_qs[i]) for i in 1:n_layers]...)
end

"""
    reduce_params(n, params)

Reduce a vector of parameters to the unique set of parameters in the translational invariant QCNN. Performs the 
sum of the corresponding parameters in the circuit. Used in eval_full_grad.

## Arguments
- `n`: Dimension of the quantum register.
- `params`: Vector of "extended" parameters.

## Returns
A vector of unique parameters.

"""
function reduce_params(p::InvariantParams, vect)
    n_layers = ceil(Int, log2(p.n))
    active_qs = map(i->ceil(Int, p.n/2^i), 0:n_layers-1)
    n_ansatz = nparameters(p.ansatz(p.n, 1, 2))
    res = []
    for i in 1:n_layers
        active_pms = vect[sum(active_qs[1:i-1])*n_ansatz+1:sum(active_qs[1:i])*n_ansatz]
        for j in 1:n_ansatz
            push!(res, sum(active_pms[j:n_ansatz:j+n_ansatz*(active_qs[i]-1)]))
        end
    end
    return res
end