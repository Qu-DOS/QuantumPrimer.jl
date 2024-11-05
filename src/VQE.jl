export circ_vqe

"""
    circ_vqe(H::AbstractMatrix; ansatz=circ_HEA::Function, depth::Union{Symbol, Int}=:nothing, iters::Int=500, verbose::Bool=false)

Prepare a quantum circuit to approximate the ground state of the Hamiltonian H, using the Variational Quantum Eigensolver (VQE) approach.

# Arguments
- `graph::Graph`: The graph from which the modified Laplacian matrix is derived.

# Keyword Arguments
- `depth::Union{Symbol, Int}=:nothing`: The depth of the hardware-efficient ansatz (HEA) used in the circuit. If set to `:nothing`, the depth is automatically determined based on the logarithm of the number of nodes in the graph.
- `iters::Int=500`: The number of iterations for the VQE optimization process.
- `verbose::Bool=false`: If `true`, prints the energy distance at every 50th iteration and at the beginning.

# Returns
- `QuantumCircuit`: The optimized quantum circuit that approximates the ground state of the modified Laplacian matrix.
"""
function circ_vqe(H::AbstractMatrix; ansatz=circ_HEA::Function, depth::Union{Symbol, Int}=:nothing, iters::Int=500, verbose::Bool=false)
    dim = size(H)[1]
    depth == :nothing ? depth=Int(ceil(log2(dim))) : depth=depth
    n_q = Int(ceil(log2(dim)))
    dim_extended = nextpow(2, dim)
    dim_extended == 1 ? dim_extended = 2 : nothing
    H_extended = zeros(ComplexF64, dim_extended, dim_extended)
    H_extended[1:dim, 1:dim] = H
    eigs = eigvals(H_extended)
    circ = chain(n_q)
    for _ = 1:depth
        push!(circ, ansatz(n_q))
    end
    params = rand(nparameters(circ))
    dispatch!(circ, params)
    opt = Optimisers.setup(Optimisers.ADAM(0.1), params)
    loss_track = []
    loss = expect(matblock(H_extended), zero_state(n_q) |> circ)
    push!(loss_track, loss)
    energy_distance = loss - eigs[1]
    verbose ? println("Iteration 0: energy distance = $(energy_distance)") : nothing
    for i = 1:iters
        _, grads = expect'(matblock(H_extended), zero_state(n_q) => circ)
        Optimisers.update!(opt, params, grads)
        dispatch!(circ, params)
        loss = expect(matblock(H_extended), zero_state(n_q) |> circ)
        push!(loss_track, loss)
        energy_distance = loss - eigs[1]
        (verbose && i % 50 == 0) ? println("Iteration $(i): energy distance = $(energy_distance)") : nothing
        if real(energy_distance) < 1e-15
            println("Breaking at iteration $(i). Reached energy distance = $(energy_distance)")
            break
        end
    end
    return circ, loss_track
end