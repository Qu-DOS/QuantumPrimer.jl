module QComp

using Yao
using Random
using Optimisers
using LinearAlgebra
using Statistics
using Combinatorics

include("Data.jl");
include("QCNN.jl");
include("Parameters.jl");
include("Loss.jl");
include("Gradient.jl");
include("TrainTest.jl");

include("QSP.jl");

# exports QCNN
export Data, Params, GenericParams, InvariantParams, initialize_params, expand_params, reduce_params
export conv_Ry, conv_Ry2, conv_SU4, build_QCNN, test_model, train_test_model

# exports QSP
export loss, eval_Usp, block_encode2, QSVT_square

end