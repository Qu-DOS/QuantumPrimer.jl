"""
    conv_Ry(n, i, j)

Ansatz for the convolutional layer. Use Ry gate to keep ansatz real (adapted from arxiv:2108.00661v2).

## Arguments
- `n`: Dimension of the quantum register.
- `i`: Index of the first qubit.
- `j`: Index of the second qubit.

## Returns
A quantum circuit implementing convolutional layers with Ry gates.

"""
conv_Ry(n, i, j) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X))

"""
    build_QCNN(n)

Build a quantum circuit for a Quantum Convolutional Neural Network (QCNN).

## Arguments
- `n`: Dimension of the quantum register.

## Returns
A quantum circuit representing the QCNN.

"""
function build_QCNN(n)
    circ = chain(n)
    n_q = n
    while n_q>1
        if n_q%2 == 0
            map(i->push!(circ, chain(n, conv_Ry(n, i, mod(i, n_q)+1))), 1:2:n_q-1)
            map(i->push!(circ, chain(n, conv_Ry(n, i, mod(i, n_q)+1))), 2:2:n_q)
        else 
            map(i->push!(circ, chain(n, conv_Ry(n, i, mod(i, n_q)+1))), 1:2:n_q)
            map(i->push!(circ, chain(n, conv_Ry(n, i, mod(i, n_q)+1))), 2:2:n_q-1)
        end
        n_q = Int(ceil(n_q/2))
    end
    return circ
end