"""
    W(a)

Build the signal processing operator.

## Arguments
- `a`: Angle parameter.

## Returns
Signal rotation operator.

"""
W(a) = chain(X,Ry(-2asin(a)))


"""
    S(phi)

Build the signal processing operator.

## Arguments
- `phi`: Phase angle.

## Returns
Rotation gate over Rz.

"""
S(phi) = Rz(-2(phi)) 

"""
    Usp(phis, a)

Build the signal processing operator.

## Arguments
- `phis`: Array of phase angles.
- `a`: Angle parameter.

## Returns
The signal processing operator circuit.

"""
function Usp(phis, a)
    d=length(phis)
    circ=chain(1)
    for k=1:d-1
        push!(circ, chain(1,S(phis[k]),W(a)))
    end
    push!(circ,chain(1,S(phis[d])))
end

"""
    eval_Usp(x, phis)

Evaluate the signal processing operator.

## Arguments
- `x`: Input value.
- `phis`: Array of phase angles.

## Returns
The result of applying the signal processing operator.

"""
eval_Usp(x,phis) = real(expect(Usp(phis,x),zero_state(1))) #Transform P(a)=<0|Usp|0> (or <+|Usp|+> if basis is changed) 


"""
    loss(target, xs, phis)

Compute the loss function for a given target, inputs, and phase angles.

## Arguments
- `target`: Target value.
- `xs`: Array of input values.
- `phis`: Array of phase angles.

## Returns
The loss value.

"""
loss(target,xs,phis) = sum(map(i->(eval_Usp(xs[i],phis)-target(xs[i]))^2,1:length(xs)))


#QSVT functions. Here used 1st qubit as ancilla due to Yao convention (this puts matrix into top left corner). 
#Therefore, matrix transform applied on qubits 2:n-1

"""
    pcp(n, phi)

Generate a projected controlled phase gate.

## Arguments
- `n`: Number of qubits.
- `phi`: Phase angle.

## Returns
The projected controlled phase gate circuit.

"""
pcp(n,phi) = chain(n+1,put(n+1=>Rz(ComplexF64(-2phi)))) #projected controlled phase gate

"""
    block_encode2(n, A)

Encode a block using a method.

## Arguments
- `n`: Number of qubits.
- `A`: Matrix to encode.

## Returns
The encoded block circuit.

"""
block_encode2(n,A)=chain(n+1,put(n+1=>X),control(n+1,1:n=>matblock(ComplexF64.(A))),put(n+1=>X)) # example block encoding method

"""
    QSVT_square(n, d, phis, A)

Build the QSVT (Quantum Singular Value Transform) sequence.

## Arguments
- `n`: Number of qubits.
- `d`: Dimensionality parameter.
- `phis`: Array of phase angles.
- `A`: Matrix to transform.

## Returns
The QSVT sequence circuit.

"""
function QSVT_square(n,d,phis,A)
    circ=chain(n+1)
    if d%2 == 0 #even d
        for i=div(d,2):-1:1
            push!(circ,chain(n+1,pcp(n,phis[2*i+1]),matblock(block_encode2(n,A))',pcp(n,phis[2*i]),matblock(block_encode2(n,A))))
        end
        push!(circ,pcp(n,phis[1]))
    else #odd d
        for i=div(d+1,2):-1:2
            push!(circ,chain(pcp(n,phis[2*i]),matblock(block_encode2(n,A)),pcp(n,phis[2*i-1]),matblock(block_encode2(n,A))'))
        end
        push!(circ,chain(pcp(n,phis[2]),matblock(block_encode2(n,A)),pcp(n,phis[1])))
    end
    return circ
end