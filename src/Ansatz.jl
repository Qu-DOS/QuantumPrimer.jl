#Ansatz for convolutional layer (not parameterized yet) - used Rys to keep ansatz real - adapted from [arXiv:2108.00661v2]
#Can change this ansatz and code should still work
conv_Ry(n,i,j) = chain(n,put(i=>Ry(0)),put(j=>Ry(0)),control(i,j=>X))