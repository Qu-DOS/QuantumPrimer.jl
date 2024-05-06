#Ansatz for convolutional layer (not parameterized yet) - used Rys to keep ansatz real - adapted from [arXiv:2108.00661v2]
#Can change this ansatz and code should still work
conv_Ry(n, i, j) = chain(n, put(i=>Ry(0)), put(j=>Ry(0)), control(i,j=>X))

#Should work fine for all n, but definitely = 2^a (a integer), e.g. n=8
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