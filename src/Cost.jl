export AbstractCost,
       GeneralCost,
       CircuitCost

"""
    AbstractCost

An abstract type for representing cost functions in quantum circuits.
"""
abstract type AbstractCost end

"""
    GeneralCost{FF<:Function, AA<:Function}

A struct representing a general cost function with an optional activation function.

# Fields
- `cost::FF`: The cost function.
- `activation::AA`: The activation function, default is `identity`.

# Constructor
- `GeneralCost(cost::FF, activation::AA=identity)`: Creates a `GeneralCost` instance.
"""
struct GeneralCost{FF<:Function, AA<:Function} <: AbstractCost
    cost::FF
    activation::AA
    function GeneralCost(cost::FF, activation::AA=identity) where {FF<:Function, AA<:Function}
        new{FF, AA}(cost, activation)
    end
end

"""
    CircuitCost{FF<:Function, AA<:Function}

A struct representing a circuit-specific cost function with an optional activation function.

# Fields
- `cost::FF`: The cost function.
- `activation::AA`: The activation function, default is `identity`.

# Constructor
- `CircuitCost(cost::FF, activation::AA=identity)`: Creates a `CircuitCost` instance.
"""
struct CircuitCost{FF<:Function, AA<:Function} <: AbstractCost
    cost::FF
    activation::AA
    function CircuitCost(cost::FF, activation::AA=identity) where {FF<:Function, AA<:Function}
        new{FF, AA}(cost, activation)
    end
end