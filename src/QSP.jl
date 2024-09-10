# Exports
export W,
       S,
       Usp,
       eval_Usp,
       loss,
       pcp,
       block_encode2,
       QSVT_square

"""
    W(a::Real) -> ChainBlock

Creates a quantum circuit block for the W gate.

# Arguments
- `a::Real`: The parameter for the W gate.

# Returns
- `ChainBlock`: The quantum circuit block for the W gate.
"""
W(a::Real) = chain(X, Ry(-2asin(a)))

"""
    S(phi::Real) -> Rz

Creates a quantum gate for the S gate.

# Arguments
- `phi::Real`: The parameter for the S gate.

# Returns
- `Rz`: The quantum gate for the S gate.
"""
S(phi::Real) = Rz(-2(phi)) 

"""
    Usp(phis::Vector{Real}, a::Real) -> ChainBlock

Creates a quantum circuit for the Usp unitary.

# Arguments
- `phis::Vector{Real}`: The vector of phase parameters.
- `a::Real`: The parameter for the W gate.

# Returns
- `ChainBlock`: The quantum circuit for the Usp unitary.
"""
function Usp(phis::Vector{Real}, a::Real)
    d = length(phis)
    circ = chain(1)
    for k = 1:d-1
        push!(circ, chain(1, S(phis[k]), W(a)))
    end
    push!(circ, chain(1, S(phis[d])))
    return circ
end

"""
    eval_Usp(x::Real, phis::Vector{Real}) -> Real

Evaluates the Usp unitary for a given input and phase parameters.

# Arguments
- `x::Real`: The input parameter.
- `phis::Vector{Real}`: The vector of phase parameters.

# Returns
- `Real`: The evaluated value of the Usp unitary.
"""
eval_Usp(x::Real, phis::Vector{Real}) = real(expect(Usp(phis, x), zero_state(1))) # Transform P(a)=<0|Usp|0> (or <+|Usp|+> if basis is changed) 

"""
    loss(target::Function, xs::Vector{Real}, phis::Vector{Real}) -> Real

Computes the loss function for a given target function, inputs, and phase parameters.

# Arguments
- `target::Function`: The target function.
- `xs::Vector{Real}`: The vector of input parameters.
- `phis::Vector{Real}`: The vector of phase parameters.

# Returns
- `Real`: The computed loss.
"""
loss(target::Function, xs::Vector{Real}, phis::Vector{Real}) = sum(map(i -> (eval_Usp(xs[i], phis) - target(xs[i]))^2, 1:length(xs)))

"""
    pcp(n::Int, phi::Real) -> ChainBlock

Creates a projected controlled phase gate.

# Arguments
- `n::Int`: The number of qubits.
- `phi::Real`: The phase parameter.

# Returns
- `ChainBlock`: The quantum circuit block for the projected controlled phase gate.
"""
pcp(n::Int, phi::Real) = chain(n+1, put(n+1 => Rz(ComplexF64(-2phi)))) # projected controlled phase gate

"""
    block_encode2(n::Int, A::AbstractMatrix) -> ChainBlock

Creates a block encoding for a given matrix.

# Arguments
- `n::Int`: The number of qubits.
- `A::AbstractMatrix`: The matrix to be block encoded.

# Returns
- `ChainBlock`: The quantum circuit block for the block encoding.
"""
block_encode2(n::Int, A::AbstractMatrix) = chain(n+1, put(n+1 => X), control(n+1, 1:n => matblock(ComplexF64.(A))), put(n+1 => X)) # example block encoding method

"""
    QSVT_square(n::Int, d::Int, phis::Vector{Real}, A::AbstractMatrix) -> ChainBlock

Creates a quantum singular value transformation (QSVT) circuit for a given matrix.

# Arguments
- `n::Int`: The number of qubits.
- `d::Int`: The degree of the polynomial.
- `phis::Vector{Real}`: The vector of phase parameters.
- `A::AbstractMatrix`: The matrix to be transformed.

# Returns
- `ChainBlock`: The quantum circuit for the QSVT.
"""
function QSVT_square(n::Int, d::Int, phis::Vector{Real}, A::AbstractMatrix)
    circ = chain(n+1)
    if d % 2 == 0 # even d
        for i = div(d, 2):-1:1
            push!(circ, chain(n+1, pcp(n, phis[2*i+1]), matblock(block_encode2(n, A))', pcp(n, phis[2*i]), matblock(block_encode2(n, A))))
        end
        push!(circ, pcp(n, phis[1]))
    else # odd d
        for i = div(d+1, 2):-1:2
            push!(circ, chain(pcp(n, phis[2*i]), matblock(block_encode2(n, A)), pcp(n, phis[2*i-1]), matblock(block_encode2(n, A))'))
        end
        push!(circ, chain(pcp(n, phis[2]), matblock(block_encode2(n, A)), pcp(n, phis[1])))
    end
    return circ
end