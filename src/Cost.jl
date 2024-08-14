export AbstractCost,
       GeneralCost,
       CircuitCost

abstract type AbstractCost end

struct GeneralCost{FF<:Function, AA<:Function} <: AbstractCost
    cost::FF
    activation::AA
    function GeneralCost(cost::FF, activation::AA=identity) where {FF<:Function, AA<:Function}
        new{FF, AA}(cost, activation)
    end
end

struct CircuitCost{FF<:Function, AA<:Function} <: AbstractCost
    cost::FF
    activation::AA
    function CircuitCost(cost::FF, activation::AA=identity) where {FF<:Function, AA<:Function}
        new{FF, AA}(cost, activation)
    end
end