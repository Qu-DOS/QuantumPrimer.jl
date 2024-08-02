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
include("BitBasis.jl")
include("HypergraphStates.jl");

include("QSP.jl");

# exports QCNN
export AbstractData, Data, DataSiamese, AbstractParams, GenericParams, InvariantParams, initialize_params, expand_params, reduce_params
export eval_grad, eval_full_grad, sigmoid, eval_loss, eval_full_loss
export conv_Ry, conv_Ry2, conv_SU4, build_QCNN, test_model, train_test_model
export circ_swap_test, swap_test, circ_destructive_swap_test, destructive_swap_test, entanglement_difference, overlap, circ_z
export circ_phase_flip, circ_hypergraph_state

# exports QSP
export W, S, Usp, eval_Usp, loss, pcp, block_encode2, QSVT_square

end