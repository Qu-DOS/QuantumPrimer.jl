export AbstractData,
       Data,
       DataSiamese,
       split_single_train_test,
       split_train_test,
       label_states,
       shuffle_data

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

"""
    split_single_train_test(states::Union{Vector{T}, Vector{NTuple{2, T}}}, n_samples::Int, test_ratio::Real) where T <: ArrayReg

Split a dataset into training and test sets.

# Arguments
- `states::Union{Vector{T}, Vector{NTuple{2, T}}}`: A vector of quantum states or pairs of quantum states.
- `n_samples::Int`: The number of samples to split.
- `test_ratio::Real`: The ratio of samples to use for the test set.

# Returns
- `train_states::Union{Vector{T}, Vector{NTuple{2, T}}}`: The training set.
- `test_states::Union{Vector{T}, Vector{NTuple{2, T}}}`: The test set.
"""
function split_single_train_test(states::Union{Vector{T}, Vector{NTuple{2, T}}}, n_samples::Int, test_ratio::Real) where T <: ArrayReg
    n_max = length(states)
    if n_samples > n_max
        throw(ArgumentError("The number of samples requested is too high. The maximum number of samples is $n_max"))
    end
    n_test = ceil(Int, n_samples * test_ratio)
    n_train = n_samples - n_test
    train_states = sample(states, n_train, replace=false)
    test_states = sample(states, n_test, replace=false)
    return train_states, test_states
end

"""
    split_train_test(states1::Union{Vector{T}, Vector{NTuple{2, T}}}, states2::Union{Vector{T}, Vector{NTuple{2, T}}}, n_samples::Int, test_ratio::Real) where T <: ArrayReg

Split two datasets into training and test sets.

# Arguments
- `states1::Union{Vector{T}, Vector{NTuple{2, T}}}`: A vector of quantum states or pairs of quantum states.
- `states2::Union{Vector{T}, Vector{NTuple{2, T}}}`: A vector of quantum states or pairs of quantum states.
- `n_samples::Int`: The number of samples to split.
- `test_ratio::Real`: The ratio of samples to use for the test set.

# Returns
- `train_states::Union{Vector{T}, Vector{NTuple{2, T}}}`: The training set.
- `test_states::Union{Vector{T}, Vector{NTuple{2, T}}}`: The test set.
"""
function split_train_test(states1::Union{Vector{T}, Vector{NTuple{2, T}}}, states2::Union{Vector{T}, Vector{NTuple{2, T}}}, n_samples::Int, test_ratio::Real) where T <: ArrayReg
    train_states1, test_states1 = split_single_train_test(states1, n_samples, test_ratio)
    train_states2, test_states2 = split_single_train_test(states2, n_samples, test_ratio)
    train_states = vcat(train_states1, train_states2)
    test_states = vcat(test_states1, test_states2)
    return train_states, test_states
end

"""
    label_states(states_to_label::Union{Vector{T}, Vector{NTuple{2, T}}}, states1::Union{Vector{T}, Vector{NTuple{2, T}}}, states2::Union{Vector{T}, Vector{NTuple{2, T}}}) where T <: ArrayReg

Label a set of quantum states based on two sets of quantum states.

# Arguments
- `states_to_label::Union{Vector{T}, Vector{NTuple{2, T}}}`: A vector of quantum states or pairs of quantum states to label.
- `states1::Union{Vector{T}, Vector{NTuple{2, T}}}`: A vector of quantum states or pairs of quantum states.
- `states2::Union{Vector{T}, Vector{NTuple{2, T}}}`: A vector of quantum states or pairs of quantum states.

# Returns
- `labels::Vector{Int}`: A vector of labels corresponding to the quantum states.
"""
function label_states(states_to_label::Union{Vector{T}, Vector{NTuple{2, T}}}, states1::Union{Vector{T}, Vector{NTuple{2, T}}}, states2::Union{Vector{T}, Vector{NTuple{2, T}}}) where T <: ArrayReg
    labels = Vector{Int}()
    for state in states_to_label
        if state in states1
            push!(labels, 1)
        elseif state in states2
            push!(labels, -1)
        end
    end
    return labels
end

"""
    shuffle_data(v1::Union{Vector{T}, Vector{NTuple{2, T}}}, v2::Vector{Int}) where T <: ArrayReg

Shuffle two vectors of quantum states and corresponding labels.

# Arguments
- `v1::Union{Vector{T}, Vector{NTuple{2, T}}}`: A vector of quantum states or pairs of quantum states.
- `v2::Vector{Int}`: A vector of labels corresponding to the quantum states.

# Returns
- `v1::Union{Vector{T}, Vector{NTuple{2, T}}}`: The shuffled vector of quantum states.
- `v2::Vector{Int}`: The shuffled vector of labels.
"""
function shuffle_data(v1::Union{Vector{T}, Vector{NTuple{2, T}}}, v2::Vector{Int}) where T <: ArrayReg
    if length(v1) != length(v2)
        throw(ArgumentError("The vectors in input must have equal dimension."))
    end
    permutation = randperm(length(v1))
    v1 = v1[permutation]
    v2 = v2[permutation]
    return v1, v2
end