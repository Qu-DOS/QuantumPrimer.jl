module qcomp

using Yao
using Random
using Optimisers
using LinearAlgebra
using Statistics
using Combinatorics

include("Ansatz.jl")
include("Gradient.jl")
include("Loss.jl")
include("Parameters.jl")
include("QCNN.jl");
include("TrainTest.jl")

# export functions QCNN
export build_QCNN, initialize_params, test_model, train_test_model

end