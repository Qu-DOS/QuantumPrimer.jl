export AbstractGraph,
       Graph,
       SignedGraph,
       sort_lexicographic,
       create_line_graph,
       create_interaction_matrix,
       create_hamiltonian_matrix,
       from_string_to_matrix,
       ρ_gibbs,
       create_random_signed_graph,
       find_negative_eigs,
       find_degeneracy_gs,
       create_purified_state,
       classify_state_degeneracy,
       classify_state_negative_spectrum,
       classify_state_frustration_index,
       find_cycles,
       get_neighbors,
       dfs_find_cycles!,
       evaluate_cycles_signs,
       count_unbalanced_cycles,
       find_frustration_index,
       plot_signed_graph,
       balance_signed_graph,
       reduce_frustration_signed_graph

"""
    AbstractGraph

An abstract type for representing graphs.
"""
abstract type AbstractGraph end

"""
    Graph{T<:Vector{Vector{TT}} where TT<:Integer}

A mutable struct representing a graph.

# Fields
- `l::T`: A vector of vectors representing the adjacency list of the graph.

# Constructor
- `Graph(l)`: Creates a `Graph` instance with the given adjacency list `l`.
"""
mutable struct Graph{T<:Vector{Vector{TT}} where TT<:Integer} <: AbstractGraph
    l::T
    Graph(l) = new{typeof(l)}(sort_lexicographic(l))
end

"""
    SignedGraph{T<:Vector{Vector{T}} where T<:Integer, TT<:Vector{TT} where TT<:Integer}

A mutable struct representing a signed graph.

# Fields
- `l::T`: A vector of vectors representing the adjacency list of the graph.
- `s::TT`: A vector of integers representing the signs of the edges.

# Constructor
- `SignedGraph(l, s)`: Creates a `SignedGraph` instance with the given adjacency list `l` and edge signs `s`.
"""
mutable struct SignedGraph{T<:Vector{Vector{T}} where T<:Integer, TT<:Vector{TT} where TT<:Integer} <: AbstractGraph
    l::T
    s::TT
    SignedGraph(l, s) = new{typeof(l), typeof(s)}(sort_lexicographic(l, s)...)
end

"""
    sort_lexicographic(vec::Vector{Vector{Int}}) -> Vector{Vector{Int}}

Sorts a vector of vectors lexicographically and removes duplicates.

# Arguments
- `vec::Vector{Vector{Int}}`: The vector of vectors to be sorted.

# Returns
- `Vector{Vector{Int}}`: The sorted and unique vector of vectors.
"""
function sort_lexicographic(vec::Vector{Vector{Int}})
    return unique(sort(sort([sort(sub_vec) for sub_vec in vec]), by=length))
end

"""
    sort_lexicographic(vec_l::Vector{Vector{Int}}, vec_s::Vector{Int}) -> Tuple{Vector{Vector{Int}}, Vector{Int}}

Sorts a vector of vectors lexicographically and a corresponding vector of integers.

# Arguments
- `vec_l::Vector{Vector{Int}}`: The vector of vectors to be sorted.
- `vec_s::Vector{Int}`: The vector of integers to be sorted according to `vec_l`.

# Returns
- `Tuple{Vector{Vector{Int}}, Vector{Int}}`: The sorted vectors.
"""
function sort_lexicographic(vec_l::Vector{Vector{Int}}, vec_s::Vector{Int})
    res_l = sort(sort([sort(sub_vec) for sub_vec in vec_l]), by=length)
    perm = indexin(res_l, [sort(sub_vec) for sub_vec in vec_l])
    res_s = vec_s[perm]
    return res_l, res_s
end

"""
    sort_lexicographic(arr::Array) -> Array

Sorts an array of arrays lexicographically.

# Arguments
- `arr::Array`: The array of arrays to be sorted.

# Returns
- `Array`: The sorted array of arrays.
"""
function sort_lexicographic(arr::Array)
    res = []
    for i in 1:maximum([length(x) for x in arr])
        push!(res, sort(sort(filter(x->length(x)==i, arr))...))
    end
    return res
end

"""
    create_line_graph(G::AbstractGraph) -> Graph

Creates the line graph of a given graph.

# Arguments
- `G::AbstractGraph`: The input graph.

# Returns
- `Graph`: The line graph of the input graph.

# Throws
- An error if the input graph has no edges.
"""
function create_line_graph(G::AbstractGraph)
    filter(x->length(x)==2, G.l) == [] ? throw("The input graph has no edges.") : nothing
    G_edges = filter(x->length(x)==2, G.l)
    res = [[i] for i in eachindex(G_edges)]
    tuples = [(G_edges[i], i) for i in eachindex(G_edges)]
    for i in 1:length(G_edges)
        for j in i+1:length(G_edges)
            tmp = sort(vcat(tuples[i][1], tuples[j][1]))
            tmp != unique(tmp) ? push!(res, [tuples[i][2], tuples[j][2]]) : nothing
        end
    end    
    return Graph(res)
end

"""
    create_line_graph(G::SignedGraph) -> SignedGraph

Creates the line graph of a given signed graph.

# Arguments
- `G::SignedGraph`: The input signed graph.

# Returns
- `SignedGraph`: The line graph of the input signed graph.

# Throws
- An error if the input graph has no edges.
"""
function create_line_graph(G::SignedGraph)
    filter(x->length(x)==2, G.l) == [] ? error("The input graph has no edges.") : nothing
    G_edges = filter(x->length(x)==2, G.l)
    res_l = [[i] for i in 1:length(G_edges)] # add vertices for each edge
    res_s = zeros(Int, length(res_l))
    shift = findfirst(x->x!=0, G.s) - 1 # find the first non-zero element in the sign vector
    tuples = [(G_edges[i], i, G.s[i+shift]) for i in eachindex(G_edges)]
    for i in 1:length(G_edges)
        for j in i+1:length(G_edges)
            tmp = sort(vcat(tuples[i][1], tuples[j][1]))
            if tmp != unique(tmp) # if the two edges share a vertex
                push!(res_l, [tuples[i][2], tuples[j][2]])
                push!(res_s, tuples[i][3] * tuples[j][3])
            end
        end
    end    
    return SignedGraph(res_l, res_s)
end

"""
    create_interaction_matrix(G::SignedGraph) -> Matrix{Int}

Creates the interaction matrix of a given signed graph.

# Arguments
- `G::SignedGraph`: The input signed graph.

# Returns
- `Matrix{Int}`: The interaction matrix of the input signed graph.
"""
function create_interaction_matrix(G::SignedGraph)
    G_edges = filter(x->length(x)==2, G.l)
    if isnothing(findfirst(x->x!=0, G.s))
        shift = 1
    else
        shift = findfirst(x->x!=0, G.s) - 1 # find the first non-zero element in the sign vector
    end
    A = zeros(Int, shift, shift)
    for i in eachindex(G_edges)
        # println([G_edges[i][1], G_edges[i][2]])
        A[G_edges[i][1], G_edges[i][2]] = G.s[i+shift]
    end
    return A+A'
end

"""
    create_hamiltonian_matrix(A::Array{Int, 2}) -> Matrix{ComplexF64}

Creates the Hamiltonian matrix for a given interaction matrix.

# Arguments
- `A::Array{Int, 2}`: The interaction matrix.

# Returns
- `Matrix{ComplexF64}`: The Hamiltonian matrix.
"""
function create_hamiltonian_matrix(A::Array{Int, 2})
    M_I2 = Matrix(I2)
    M_X = Matrix(X)
    M_Y = Matrix(Y)
    M_Z = Matrix(Z)
    n = size(A)[1]
    if n == 0
        return Matrix{Int}(undef, 0, 0)
    end
    res = zeros(ComplexF64, 2^n, 2^n)
    for i in 1:n
        for j in i+1:n
            A[i, j] == 0 ? continue : nothing
            P = [ii == i || ii == j ? M_Z : M_I2 for ii in 1:n]
            res += A[i, j] * from_string_to_matrix(P)
        end
    end
    return -res
end

"""
    from_string_to_matrix(vec::AbstractArray) -> Matrix{ComplexF64}

Converts a vector of matrices (developed initially for Paulis) to a single matrix using the Kronecker product.

# Arguments
- `vec::AbstractArray`: The vector of matrices.

# Returns
- `Matrix{ComplexF64}`: The resulting matrix.
"""
function from_string_to_matrix(vec::AbstractArray)
    return Kronecker.kronecker(reverse(vec)...) # using kronecker from Kronecker.jl
end

"""
    ρ_gibbs(β::Real, matrix::AbstractMatrix) -> Matrix{ComplexF64}

Computes the Gibbs state for a given Hamiltonian matrix at inverse temperature β.

# Arguments
- `β::Real`: The inverse temperature.
- `matrix::AbstractMatrix`: The Hamiltonian matrix.

# Returns
- `Matrix{ComplexF64}`: The Gibbs state.
"""
function ρ_gibbs(β::Real, matrix::AbstractMatrix)
    ite_exp = exp(matrix * (-β))
    ρ_evol = ite_exp / tr(ite_exp)
    return ρ_evol
end

"""
    create_random_signed_graph(n::Int) -> SignedGraph

Creates a random signed graph with `n` vertices.

# Arguments
- `n::Int`: The number of vertices.

# Returns
- `SignedGraph`: The resulting random signed graph.
"""
function create_random_signed_graph(n::Int)
    res_l = [[i] for i in 1:n]
    res_s = zeros(Int, n)
    for i in 1:n
        for j in i+1:n
            if rand() < 0.5
                push!(res_l, [i, j])
                push!(res_s, rand([-1, 1]))
            end
        end
    end
    return SignedGraph(res_l, res_s)
end

"""
    find_negative_eigs(matrix::AbstractMatrix) -> Int

Finds the number of negative eigenvalues of a given matrix.

# Arguments
- `matrix::AbstractMatrix`: The input matrix.

# Returns
- `Int`: The number of negative eigenvalues.
"""
function find_negative_eigs(matrix::AbstractMatrix)
    eigs = eigvals(matrix)
    return count(x -> x < 0, eigs)
end

"""
    find_negative_eigs(G::SignedGraph) -> Int

Finds the number of negative eigenvalues of the Hamiltonian matrix of a given signed graph.

# Arguments
- `G::SignedGraph`: The input signed graph.

# Returns
- `Int`: The number of negative eigenvalues.
"""
function find_negative_eigs(G::SignedGraph)
    return find_negative_eigs(create_hamiltonian_matrix(create_interaction_matrix(G)))
end

"""
    find_degeneracy_gs(matrix::AbstractMatrix) -> Int

Finds the degeneracy of the ground state of a given matrix.

# Arguments
- `matrix::AbstractMatrix`: The input matrix.

# Returns
- `Int`: The degeneracy of the ground state.
"""
function find_degeneracy_gs(matrix::AbstractMatrix)
    if size(matrix) == (0, 0)
        return 0
    end
    eigs = eigvals(matrix)
    smallest_eig = minimum(eigs)
    return count(x -> x == smallest_eig, eigs)
end

"""
    find_degeneracy_gs(G::SignedGraph) -> Int

Finds the degeneracy of the ground state of the Hamiltonian matrix of a given signed graph.

# Arguments
- `G::SignedGraph`: The input signed graph.

# Returns
- `Int`: The degeneracy of the ground state.
"""
function find_degeneracy_gs(G::SignedGraph)
    return find_degeneracy_gs(create_hamiltonian_matrix(create_interaction_matrix(G)))
end

"""
    create_purified_state(β::Real, G::SignedGraph) -> ArrayReg

Creates a purified quantum state for a given signed graph at inverse temperature β.

# Arguments
- `β::Real`: The inverse temperature.
- `G::SignedGraph`: The input signed graph.

# Returns
- `ArrayReg`: The purified quantum state.
"""
function create_purified_state(β::Real, G::SignedGraph)
    return purify(DensityMatrix(ρ_gibbs(β, create_hamiltonian_matrix(create_interaction_matrix(G)))))
end

"""
    classify_state_degeneracy(n::Int, label::Int; max_iter::Int=1000, use_line_graph::Bool=false) -> SignedGraph

Classifies a random signed graph based on the degeneracy of its ground state.

# Arguments
- `n::Int`: The number of vertices.
- `label::Int`: The label indicating the desired degeneracy classification (1 for high degeneracy, -1 for low degeneracy).
- `max_iter::Int`: The maximum number of iterations to attempt, default is 1000.
- `use_line_graph::Bool`: Whether to use the line graph of the random signed graph, default is false.

# Returns
- `SignedGraph`: The classified random signed graph.

# Throws
- An error if a suitable graph cannot be generated within the maximum number of iterations.
"""
function classify_state_degeneracy(n::Int, label::Int; max_iter::Int=1000, use_line_graph::Bool=false)
    rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
    iter = 0
    if label == 1
        while find_degeneracy_gs(rand_graph) <= 3 && iter < max_iter
            rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
            iter += 1
        end
    elseif label == -1
        while find_degeneracy_gs(rand_graph) > 3 && iter < max_iter
            rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
            iter += 1
        end
    end
    iter == max_iter ? error("Could not generate state") : return rand_graph
end

"""
    classify_state_negative_spectrum(n::Int, label::Int; max_iter::Int=1000, use_line_graph::Bool=false) -> SignedGraph

Classifies a random signed graph based on the number of negative eigenvalues in its Hamiltonian matrix.

# Arguments
- `n::Int`: The number of vertices.
- `label::Int`: The label indicating the desired classification (1 for more than 3 negative eigenvalues, -1 for 3 or fewer negative eigenvalues).
- `max_iter::Int`: The maximum number of iterations to attempt, default is 1000.
- `use_line_graph::Bool`: Whether to use the line graph of the random signed graph, default is false.

# Returns
- `SignedGraph`: The classified random signed graph.

# Throws
- An error if a suitable graph cannot be generated within the maximum number of iterations.
"""
function classify_state_negative_spectrum(n::Int, label::Int; max_iter::Int=1000, use_line_graph::Bool=false)
    rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
    iter = 0
    if label == 1
        while find_negative_eigs(rand_graph) <= 3 && iter < max_iter
            rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
            iter += 1
        end
    elseif label == -1
        while find_negative_eigs(rand_graph) > 3 && iter < max_iter
            rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
            iter += 1
        end
    end
    iter == max_iter ? error("Could not generate state") : return rand_graph
end

"""
    classify_state_frustration_index(n::Int, threshold::Int, label::Int; max_iter::Int=10000, use_line_graph::Bool=false) -> SignedGraph

Classifies a random signed graph based on its frustration index.

# Arguments
- `n::Int`: The number of vertices.
- `threshold::Int`: The threshold for the frustration index.
- `label::Int`: The label indicating the desired classification (1 for frustration index above the threshold, -1 for frustration index below or equal to the threshold).
- `max_iter::Int`: The maximum number of iterations to attempt, default is 10000.
- `use_line_graph::Bool`: Whether to use the line graph of the random signed graph, default is false.

# Returns
- `SignedGraph`: The classified random signed graph.

# Throws
- An error if a suitable graph cannot be generated within the maximum number of iterations.
"""
function classify_state_frustration_index(n::Int, threshold::Int, label::Int; max_iter::Int=10000, use_line_graph::Bool=false)
    rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
    iter = 0
    if label == 1
        while find_frustration_index(rand_graph)[1] <= threshold && iter < max_iter
            rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
            iter += 1
        end
    elseif label == -1
        while find_frustration_index(rand_graph)[1] > threshold && iter < max_iter
            rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
            iter += 1
        end
    end
    iter == max_iter ? error("Could not generate state") : return rand_graph
end

"""
    get_neighbors(G::SignedGraph, vertex::Int) -> Vector{Int}

Gets the neighbors of a given vertex in a signed graph.

# Arguments
- `G::SignedGraph`: The input signed graph.
- `vertex::Int`: The vertex for which to find neighbors.

# Returns
- `Vector{Int}`: A vector of neighboring vertex indices.
"""
function get_neighbors(G::SignedGraph, vertex::Int)
    neighbors = Vector{Int}()
    G_edges = filter(x->length(x)==2, G.l)
    for edge in G_edges
        if edge[1] == vertex
            push!(neighbors, edge[2])
        elseif edge[2] == vertex
            push!(neighbors, edge[1])
        end
    end
    return neighbors
end

"""
    dfs_find_cycles!(G::SignedGraph, current_vertex::Int, start_vertex::Int, visited::BitVector, path::Vector{Int}, edge_path::Vector{Vector{Int}}, cycles::Set{Tuple{Vector{Int}, Vector{Vector{Int}}}})

Performs a depth-first search to find all cycles in a signed graph.

# Arguments
- `G::SignedGraph`: The input signed graph.
- `current_vertex::Int`: The current vertex in the DFS.
- `start_vertex::Int`: The starting vertex for the DFS.
- `visited::BitVector`: A bit vector indicating visited vertices.
- `path::Vector{Int}`: The current path in the DFS.
- `edge_path::Vector{Vector{Int}}`: The current edge path in the DFS.
- `cycles::Set{Tuple{Vector{Int}, Vector{Vector{Int}}}}`: The set of found cycles.
"""
function dfs_find_cycles!(G::SignedGraph, current_vertex::Int, start_vertex::Int, visited::BitVector, path::Vector{Int}, edge_path::Vector{Vector{Int}}, cycles::Set{Tuple{Vector{Int}, Vector{Vector{Int}}}})
    push!(path, current_vertex)
    visited[current_vertex] = true
    for neighbor in get_neighbors(G, current_vertex)
        edge = [current_vertex, neighbor]
        if neighbor == start_vertex && length(path) > 2
            push!(edge_path, edge)
            push!(cycles, (sort(copy(path)), sort_lexicographic(copy(edge_path))))
            pop!(edge_path)
        elseif !visited[neighbor]
            push!(edge_path, edge)
            dfs_find_cycles!(G, neighbor, start_vertex, visited, path, edge_path, cycles)
            pop!(edge_path)
        end
    end
    pop!(path)
    visited[current_vertex] = false
end

"""
    find_cycles(G::SignedGraph) -> Set{Tuple{Vector{Int}, Vector{Vector{Int}}}}

Finds all cycles in a given signed graph using depth-first search.

# Arguments
- `G::SignedGraph`: The input signed graph.

# Returns
- `Set{Tuple{Vector{Int}, Vector{Vector{Int}}}}`: A set of cycles, each represented as a tuple of vertex indices and edge paths.
"""
function find_cycles(G::SignedGraph)
    n = maximum([edge[end] for edge in G.l])  # Assumes lexicographic ordering of edges
    visited = falses(n)
    cycles = Set{Tuple{Vector{Int}, Vector{Vector{Int}}}}()
    for start_vertex in 1:n
        dfs_find_cycles!(G, start_vertex, start_vertex, visited, Int[], Vector{Int}[], cycles)
        fill!(visited, false)  # Reset visited after each full DFS
    end
    return unique(cycles)
end

"""
    evaluate_cycles_signs(G::SignedGraph) -> Vector{Int}

Evaluates the product of signs for the cycles in a given signed graph.

# Arguments
- `G::SignedGraph`: The input signed graph.

# Returns
- `Vector{Int}`: A vector of signs for each cycle.
"""
function evaluate_cycles_signs(G::SignedGraph)
    A = create_interaction_matrix(G)
    cycles = find_cycles(G)
    signs = Int[]
    for cycle in cycles
        sign = 1
        for ele in cycle[2]
            sign *= A[ele[1], ele[2]]
        end
        push!(signs, sign)
    end
    return signs
end

"""
    count_unbalanced_cycles(G::SignedGraph) -> Int

Counts the number of unbalanced cycles in a given signed graph.

# Arguments
- `G::SignedGraph`: The input signed graph.

# Returns
- `Int`: The number of unbalanced cycles.
"""
function count_unbalanced_cycles(G::SignedGraph)
    signs = evaluate_cycles_signs(G)
    return count(x -> x == -1, signs)
end

"""
    find_frustration_index(G::SignedGraph) -> Tuple{Int, Vector{Int}}

Finds the frustration index of a given signed graph by changing the sign of the edges and checking the number of unbalanced cycles.

# Arguments
- `G::SignedGraph`: The input signed graph.

# Returns
- `Tuple{Int, Vector{Int}}`: The frustration index and the corresponding sign changes.
"""
function find_frustration_index(G::SignedGraph)
    n = length(filter(x->length(x)==1, G.l))
    n_edges = length(filter(x->length(x)==2, G.l))
    frustration_index = n_edges
    sign_changes = ones(Int, n_edges)
    for i in 0:2^n_edges
        bin_i = digits(i, base=2, pad=n_edges)
        new_s = zeros(Int, n+n_edges)
        for j in 1:n_edges
            if bin_i[j] == 0
                new_s[n+j] = G.s[n+j]
            else
                new_s[n+j] = -G.s[n+j]
            end
        end
        new_G = SignedGraph(G.l, new_s)
        if count_unbalanced_cycles(new_G) == 0
            n_changes = sum(bin_i)
            if frustration_index > n_changes
                frustration_index = n_changes
                sign_changes = bin_i
            end
        end
        if frustration_index == 0
            sign_changes = zeros(Int, n_edges)
            break
        end
    end
    sign_changes = [i == 1 ? -1 : 1 for i in sign_changes]
    return frustration_index, sign_changes
end

"""
    plot_signed_graph(graph::SignedGraph) -> Tuple{SimpleGraph, Vector{Tuple{Float64, Float64}}, Matrix{Symbol}}

Plots a signed graph by creating a simple graph representation, a dataset of vertex positions, and an edge color matrix.

# Arguments
- `graph::SignedGraph`: The input signed graph.

# Returns
- `Tuple{SimpleGraph, Vector{Tuple{Float64, Float64}}, Matrix{Symbol}}`: A tuple containing the simple graph, the dataset of vertex positions, and the edge color matrix.
"""
function plot_signed_graph(graph::SignedGraph)
    n = length(filter(x->length(x)==1, graph.l))
    edgecolor_mat = fill(:empty, n, n)
    for (idx, edge) in enumerate(filter(x->length(x)==2, graph.l))
        edgecolor_mat[edge[1], edge[2]] = graph.s[n+idx] == 1 ? :black : :red
    end
    dataset = [(1 / 2 * cos(ϕ), 1 / 2 * sin(ϕ)) for ϕ in 0:2π/n:2π][1:end-1]
    simple_graph = SimpleGraph(n)
    for i in filter(x -> length(x) == 2, graph.l)
        add_edge!(simple_graph, i[1], i[2])
    end
    return simple_graph, dataset, edgecolor_mat
end

"""
    balance_signed_graph(graph::SignedGraph) -> SignedGraph

Balances a signed graph by adjusting the signs of the edges to make the frustration index zero.

# Arguments
- `graph::SignedGraph`: The input signed graph.

# Returns
- `SignedGraph`: The balanced signed graph.
"""
function balance_signed_graph(graph::SignedGraph)
    _, sign_changes = find_frustration_index(graph)
    padding = [i for i=1:length(filter(x->length(x)==1, graph.l))]
    return SignedGraph(graph.l, graph.s .* vcat(padding, sign_changes))
end

"""
    reduce_frustration_signed_graph(graph::SignedGraph, target::Int) -> SignedGraph

Reduce the frustration index of a signed graph by cycling through the .

# Arguments
- `graph::SignedGraph`: The input signed graph.
- `target::Int`: The number of sign changes to reduce the frustration index.

# Returns
- `SignedGraph`: The signed graph with reduced frustration index.
"""
function reduce_frustration_signed_graph(graph::SignedGraph, target::Int)
    frustration_index, sign_changes = find_frustration_index(graph)
    change_idx = findall(x -> x == -1, sign_changes)
    binary_vectors = [digits(i, base=2, pad=frustration_index).* 2 .- 1 for i in 0:(2^frustration_index - 1)]
    filter!(x -> count(i -> i==-1, x) == target, binary_vectors)
    if !isempty(binary_vectors)
        rand_idx = rand(binary_vectors)
        sign_changes[change_idx] = rand_idx
        padding = [i for i=1:length(filter(x->length(x)==1, graph.l))]
        return SignedGraph(graph.l, graph.s .* vcat(padding, sign_changes))
    else
        error("The number of sign changes requested is greater than the frustration index")
    end
end