abstract type AbstractData end

struct Data{S <: Vector{SS} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real} <: AbstractData
    s::S
    l::L
end

struct DataSiamese{S <: Vector{Tuple{SS}} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real} <: AbstractData
    s::S
    l::L
end