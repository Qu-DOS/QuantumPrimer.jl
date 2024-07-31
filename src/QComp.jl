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
include("DifferencingLayer.jl");

include("QSP.jl");

# exports QCNN
export Data, Params, GenericParams, InvariantParams, initialize_params, expand_params, reduce_params
export eval_grad, eval_full_grad, sigmoid, eval_loss, eval_full_loss
export conv_Ry, conv_Ry2, conv_SU4, build_QCNN, test_model, train_test_model
export circ_swap_test, swap_test, circ_destructive_swap_test, destructive_swap_test

# exports QSP
export W, S, Usp, eval_Usp, loss, pcp, block_encode2, QSVT_square

end