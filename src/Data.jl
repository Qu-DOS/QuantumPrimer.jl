export AbstractData,
       Data,
       DataSiamese

"""
    AbstractData

An abstract type for representing data in quantum machine learning.
"""
abstract type AbstractData end

"""
    Data{S <: Vector{SS} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real}

A struct representing a dataset with quantum states and corresponding labels.

# Fields
- `states::S`: A vector of quantum states.
- `labels::L`: A vector of labels corresponding to the quantum states.

# Constructor
- `Data(states::S, labels::L)`: Creates a `Data` instance.
"""
struct Data{S <: Vector{SS} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real} <: AbstractData
    states::S
    labels::L
end

"""
    DataSiamese{S <: Vector{NTuple{2, SS}} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real}

A struct representing a dataset for Siamese networks with pairs of quantum states and corresponding labels.

# Fields
- `states::S`: A vector of tuples, each containing a pair of quantum states.
- `labels::L`: A vector of labels corresponding to the pairs of quantum states.

# Constructor
- `DataSiamese(states::S, labels::L)`: Creates a `DataSiamese` instance.
"""
struct DataSiamese{S <: Vector{NTuple{2, SS}} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real} <: AbstractData
    states::S
    labels::L
end