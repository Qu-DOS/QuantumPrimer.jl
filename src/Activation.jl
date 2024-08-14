# Exports
export sigmoid,
       hyperbolic_tangent

sigmoid(x::Real) = 2 / (1 + exp(-x)) - 1

hyperbolic_tangent(x::Real, a::Real, b::Real) = tanh(a * (abs(x) - b))