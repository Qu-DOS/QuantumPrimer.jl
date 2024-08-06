# Exports
export AbstractData,
       Data,
       DataSiamese

abstract type AbstractData end

struct Data{S <: Vector{SS} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real} <: AbstractData
    states::S
    labels::L
end

struct DataSiamese{S <: Vector{NTuple{2, SS}} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real} <: AbstractData
    states::S
    labels::L
end