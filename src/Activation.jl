# Exports
export sigmoid,
       hyper_tan

sigmoid(x::Real) = 2 / (1 + exp(-x)) - 1

hyper_tan(x::Real, a::Real, b::Real) = tanh(a * (abs(x) - b))