# Exports
export sigmoid,
       hyperbolic_tangent

sigmoid(x::Real, a::Real, b::Real) = 2 / (1 + exp(-a * (x - b))) - 1

sigmoid(x::Real) = sigmoid(x, 1, 0)

hyperbolic_tangent(x::Real, a::Real, b::Real) = tanh(a * (abs(x) - b))