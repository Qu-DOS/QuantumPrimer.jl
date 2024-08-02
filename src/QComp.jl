module QComp

using Yao
using Random
using Optimisers
using LinearAlgebra
using Statistics
using Combinatorics

include("Circuit.jl");
include("Data.jl");
include("Activation.jl")
include("QCNN.jl");
include("QNN.jl");
include("Model.jl");
include("Loss.jl");
include("Gradient.jl");
include("Differencing.jl");
include("TrainTest.jl");

include("QSP.jl");

# Exports
export AbstractData, Data, DataSiamese, AbstractModel, GeneralModel, InvariantModel, initialize_params, expand_params, reduce_params
export eval_grad, eval_full_grad, sigmoid, eval_loss, eval_full_loss
export circ_Z, circ_Zn, circ_Zsum, conv_Ry, conv_Ry2, conv_SU4, Rx_layer, Ry_layer, CNOT_layer, circ_HEA, circ_phase_flip, circ_hypergraph_state
export circ_swap_test, swap_test, circ_destructive_swap_test, destructive_swap_test, entanglement_difference, overlap
export build_QCNN, build_QNN, test_model, train_test_model

export W, S, Usp, eval_Usp, loss, pcp, block_encode2, QSVT_square

end