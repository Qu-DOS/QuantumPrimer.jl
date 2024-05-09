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
export eval_grad, eval_full_grad, sigmoid, eval_loss, eval_full_loss
export conv_Ry, build_QCNN, test_model, train_test_model

# exports QSP
export W, S, Usp, eval_Usp, loss, pcp, block_encode2, QSVT_square

end