### Imports ###

using Pkg
Pkg.activate("./")
using QuantumPrimer
using Yao
using YaoPlots
using Plots
default(lw=2, ms=5, palette=:Set2_8)
using LinearAlgebra
using Random
using Optimisers
using JLD2

function printt(x)
    show(stdout, "text/plain", x)
    println()
end

function optimize_hea_identity(model::AbstractModel; iters::Int=500)
    n = model.n
    psi = rand_state(n)
    circ = model.circ
    params = model.params
    model.params = 2pi * rand(nparameters(model.circ))
    dispatch!(circ, params)
    opt = Optimisers.setup(Optimisers.ADAM(0.1), params)
    loss_track = []
    loss = 1 + expect(circ', psi |> circ)
    push!(loss_track, loss)
    for _ = 1:iters
        _, grads = expect'(circ', psi => circ)
        Optimisers.update!(opt, params, grads)
        dispatch!(circ, params)
        loss = 1 + expect(circ', psi |> circ)
        push!(loss_track, loss)
        if real(loss) < 1e-15
            println("Breaking; Reached loss < 1e-15")
            break
        end
    end
    return circ, loss_track
end

n = 4
ansatz = circ_HEA
depth = 1
circ = build_QNN(n, depth, ansatz=ansatz)
model = GeneralModel(n=n, circ=circ, ansatz=ansatz)
initialize_params(model);
# YaoPlots.plot(circ)

circ, loss_track = optimize_hea_identity(model)

circ |> YaoPlots.plot
circ' |> YaoPlots.plot

psi_test = rand_state(n)
psi_test ≈ (psi_test |> circ)

fidelity(psi_test, psi_test |> circ)

state(psi_test)
state(psi_test |> circ)

Plots.plot(loss_track)