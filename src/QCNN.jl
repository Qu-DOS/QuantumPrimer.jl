"""
    conv_Ry(n, i, j)

Ansatz for the convolutional layer. Use Ry gate to keep ansatz real (adapted from arxiv:2108.00661v2).

## Arguments
- `n`: Dimension of the quantum register.
- `i`: Index of the first qubit.
- `j`: Index of the second qubit.

## Returns
A quantum circuit implementing a simple convolutional layer with Ry gates.

"""
conv_Ry(n, i, j) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X))

"""
    conv_Ry2(n, i, j)

Ansatz for the convolutional layer. Use Ry gate to keep ansatz real (adapted from arxiv:2108.00661v2).

## Arguments
- `n`: Dimension of the quantum register.
- `i`: Index of the first qubit.
- `j`: Index of the second qubit.

## Returns
A quantum circuit implementing a convolutional layer with Ry gates.

"""
conv_Ry2(n, i, j) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)),
                            put(j=>Ry(0)), control(i,j=>X), put(i=>Ry(0)), put(j=>Ry(0)))

"""
    conv_SU4(n, i, j)

Ansatz for the convolutional layer, applying an arbitrary SU(4) gate (adapted from arxiv:2108.00661v2).

## Arguments
- `n`: Dimension of the quantum register.
- `i`: Index of the first qubit.
- `j`: Index of the second qubit.

## Returns
A quantum circuit implementing a convolutional layer.

"""
function conv_SU4(n, i, j)
    squ() = chain(Rz(0), Ry(0), Rz(0))
    chain(n, put(i=>squ()), put(j=>squ()), control(i,j=>X), put(i=>Ry(0)), put(j=>Rz(0)),
            control(j,i=>X), put(i=>Ry(0)), control(i,j=>X), put(i=>squ()), put(j=>squ()))
end

"""
    build_QCNN(n; ansatz=conv_Ry)

Build a quantum circuit for a Quantum Convolutional Neural Network (QCNN) with a given ansatz.

## Arguments
- `n`: Dimension of the quantum register.
- `ansatz`: Convolutional ansatz

## Returns
A quantum circuit representing the QCNN. Plot with YaoPlots.plot() to visualize.

"""
function build_QCNN(n; ansatz=conv_Ry)
    circ = chain(n)
    n_q = n
    while n_q>1
        if n_q%2 == 0
            map(i->push!(circ, chain(n, ansatz(n, i, mod(i, n_q)+1))), 1:2:n_q-1)
            map(i->push!(circ, chain(n, ansatz(n, i, mod(i, n_q)+1))), 2:2:n_q)
        else 
            map(i->push!(circ, chain(n, ansatz(n, i, mod(i, n_q)+1))), 1:2:n_q)
            map(i->push!(circ, chain(n, ansatz(n, i, mod(i, n_q)+1))), 2:2:n_q-1)
        end
        n_q = Int(ceil(n_q/2))
    end
    return circ
end