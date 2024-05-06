"""
    Base.@kwdef mutable struct GenericParams{NN<:Integer, CC<:ChainBlock, TT<:AbstractVector{T} where T<:Real} <:Params

Defines parameters for a quantum circuit without translational invariance.

## Fields
- `n::NN`: Dimension of the quantum register (must be a power of 2).
- `circ::CC`: Circuit structure of the QCNN.
- `params::TT = Float64[]`: Vector of parameters, initialized to an empty vector of Float64 values.

"""
struct Data{S <: Vector{SS} where SS <: ArrayReg, T <: Vector{TT} where TT <: Real}
    # train
    s1::S
    l1::T
    # test
    s2::S
    l2::T
end