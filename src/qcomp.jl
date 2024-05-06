module qcomp

using Yao
using Random
using Optimisers
using LinearAlgebra
using Statistics
using Combinatorics

include("QCNN.jl");
include("Parameters.jl")
include("Loss.jl")
include("Gradient.jl")
include("TrainTest.jl")

# export functions QCNN
export Params, GenericParams, InvariantParams, initialize_params, expand_params, reduce_params
export build_QCNN, test_model, train_test_model

end