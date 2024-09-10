export sigmoid,
       hyperbolic_tangent

"""
    sigmoid(x::Real, a::Real, b::Real) -> Real

Computes the sigmoid function with parameters `a` and `b`.

# Arguments
- `x::Real`: The input value.
- `a::Real`: The slope parameter.
- `b::Real`: The shift parameter.

# Returns
- `Real`: The computed sigmoid value.
"""
sigmoid(x::Real, a::Real, b::Real) = 2 / (1 + exp(-a * (x - b))) - 1

"""
    sigmoid(x::Real) -> Real

Computes the sigmoid function with default parameters `a = 1` and `b = 0`.

# Arguments
- `x::Real`: The input value.

# Returns
- `Real`: The computed sigmoid value.
"""
sigmoid(x::Real) = sigmoid(x, 1, 0)

"""
    hyperbolic_tangent(x::Real, a::Real, b::Real) -> Real

Computes the hyperbolic tangent function with parameters `a` and `b`.

# Arguments
- `x::Real`: The input value.
- `a::Real`: The slope parameter.
- `b::Real`: The shift parameter.

# Returns
- `Real`: The computed hyperbolic tangent value.
"""
hyperbolic_tangent(x::Real, a::Real, b::Real) = tanh(a * (abs(x) - b))