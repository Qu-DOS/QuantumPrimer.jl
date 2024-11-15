module QuantumPrimer

using BitBasis
using Combinatorics
using ForwardDiff
using Graphs
using Kronecker
using LinearAlgebra
using Optimisers
using Random
using Statistics
using Yao

include("Circuit.jl");
include("Data.jl");
include("Activation.jl");
include("QCNN.jl");
include("QNN.jl");
include("Model.jl");
include("Cost.jl");
include("Loss.jl");
include("Gradient.jl");
include("Differencing.jl");
include("TrainTest.jl");

include("Graph.jl")
include("Utils.jl");

include("QSP.jl");

include("VQE.jl");

end