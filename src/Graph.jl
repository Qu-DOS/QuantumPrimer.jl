abstract type AbstractGraph end

mutable struct Graph{T<:Vector{Vector{TT}} where TT<:Integer} <: AbstractGraph
    l::T
    Graph(l) = new{typeof(l)}(sort_lexicographic(l))
end

mutable struct SignedGraph{T<:Vector{Vector{T}} where T<:Integer, TT<:Vector{TT} where TT<:Integer} <: AbstractGraph
    l::T
    s::TT
    SignedGraph(l, s) = new{typeof(l), typeof(s)}(sort_lexicographic(l, s)...)
end

function sort_lexicographic(vec::Vector{Vector{Int}})
    return unique(sort(sort([sort(sub_vec) for sub_vec in vec]), by=length))
end

function sort_lexicographic(vec_l::Vector{Vector{Int}}, vec_s::Vector{Int})
    res_l = sort(sort([sort(sub_vec) for sub_vec in vec_l]), by=length)
    perm = indexin(res_l, [sort(sub_vec) for sub_vec in vec_l])
    res_s = vec_s[perm]
    return res_l, res_s
end

function sort_lexicographic(arr::Array)
    res = []
    for i in 1:maximum([length(x) for x in arr])
        push!(res, sort(sort(filter(x->length(x)==i, arr))...))
    end
    return res
end

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

M_I2 = Matrix(I2)
M_X = Matrix(X)
M_Y = Matrix(Y)
M_Z = Matrix(Z)
function create_hamiltonian_matrix(A::Array{Int, 2})
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
    return - res
end

function from_string_to_matrix(vec::AbstractArray)
    return Kronecker.kronecker(reverse(vec)...) # using kronecker from Kronecker.jl
end

function ρ_gibbs(β, matrix::AbstractMatrix)
    ite_exp = exp(matrix * (-β))
    ρ_evol = ite_exp/tr(ite_exp)
    return ρ_evol
end

function create_random_signed_graph(n)
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

function find_negative_eigs(matrix::AbstractMatrix)
    eigs = eigvals(matrix)
    return count(x -> x < 0, eigs)
end

function find_negative_eigs(G::SignedGraph)
    return find_negative_eigs(create_hamiltonian_matrix(create_interaction_matrix(G)))
end

function find_degeneracy_gs(matrix::AbstractMatrix)
    if size(matrix) == (0, 0)
        return 0
    end
    eigs = eigvals(matrix)
    # eigs = sort(real(diag(matrix)))
    smallest_eig = minimum(eigs)
    return count(x -> x == smallest_eig, eigs)
end

function find_degeneracy_gs(G::SignedGraph)
    return find_degeneracy_gs(create_hamiltonian_matrix(create_interaction_matrix(G)))
end

function create_purified_state(β, G::SignedGraph)
    return purify(DensityMatrix(ρ_gibbs(β, create_hamiltonian_matrix(create_interaction_matrix(G)))))
end

function classify_state_degeneracy(n, label; max_iter=1000, use_line_graph=false)
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

function classify_state_negative_spectrum(n, label; max_iter=1000, use_line_graph=false)
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

function classify_state_frustration_number(n, threshold, label; max_iter=10000, use_line_graph=false)
    rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
    iter = 0
    if label == 1
        while find_frustration_number(rand_graph)[1] <= threshold && iter < max_iter
            rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
            iter += 1
        end
    elseif label == -1
        while find_frustration_number(rand_graph)[1] > threshold && iter < max_iter
            rand_graph = use_line_graph ? create_line_graph(create_random_signed_graph(n)) : create_random_signed_graph(n)
            iter += 1
        end
    end
    iter == max_iter ? error("Could not generate state") : return rand_graph
end

function find_cycles(G::SignedGraph)
    A = create_interaction_matrix(G)
    n = size(A)[1]
    cycles = []
    for i in 1:n
        for j in i+1:n
            if A[i, j] != 0
                for k in j+1:n
                    if A[j, k] != 0 && A[i, k] != 0
                        push!(cycles, [i, j, k])
                    end
                end
            end
        end
    end
    return cycles
end

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

# evaluate the product of signes for the cycles in the graph
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

# count unbalanced cycles in the graph
function count_unbalanced_cycles(G::SignedGraph)
    signs = evaluate_cycles_signs(G)
    return count(x -> x == -1, signs)
end

# find the frustration number of the graph by changing the sign of the edges (all possible combinations of changes) and checking the number of unbalanced cycles
function find_frustration_number(G::SignedGraph)
    n = length(filter(x->length(x)==1, G.l))
    n_edges = length(filter(x->length(x)==2, G.l))
    frustration_number = n_edges
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
            if frustration_number > n_changes
                frustration_number = n_changes
                sign_changes = bin_i
            end
        end
        if frustration_number == 0
            sign_changes = zeros(Int, n_edges)
            break
        end
        # frustration_number == 0 ? break : nothing
    end
    # frustration_number == 0 ? sign_changes = zeros(Int, n_edges) : nothing
    sign_changes = [i == 1 ? -1 : 1 for i in sign_changes]
    return frustration_number, sign_changes
end

function plot_signed_graph(graph::SignedGraph)
    n = length(filter(x->length(x)==1, graph.l))
    edgecolor_mat = fill(:empty, n, n)
    for (idx, edge) in enumerate(filter(x->length(x)==2, graph.l))
        edgecolor_mat[edge[1], edge[2]] = graph.s[n+idx] == 1 ? :black : :red
    end
    # edgecolor_mat = zeros(Int, n, n)
    # for (idx, edge) in enumerate(filter(x->length(x)==2, graph.l))
    #     edgecolor_mat[edge[1], edge[2]] = graph.s[n+idx] == 1 ? 3 : 2
    # end
    # dataset = [(rand(), rand()) for i in 1:n]
    dataset = [(1 / 2 * cos(ϕ), 1 / 2 * sin(ϕ)) for ϕ in 0:2π/n:2π][1:end-1]
    simple_graph = SimpleGraph(n)
    for i in filter(x -> length(x) == 2, graph.l)
        add_edge!(simple_graph, i[1], i[2])
    end
    return simple_graph, dataset, edgecolor_mat
end

function balance_signed_graph(graph::SignedGraph)
    return SignedGraph(graph.l, graph.s .* vcat([i for i=1:length(filter(x->length(x)==1, graph.l))], find_frustration_number(graph)[2]))
end