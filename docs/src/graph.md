# Graph documentation

## Structure
```@docs
AbstractGraph
Graph
SignedGraph
sort_lexicographic
```

## Line graph
```@docs
create_line_graph
balance_signed_graph
```

## Mapping to Hamiltonian thermal state
```@docs
create_interaction_matrix
create_hamiltonian_matrix
from_string_to_matrix
ρ_gibbs
create_purified_state
```

## Classification functions
```@docs
find_negative_eigs
find_degeneracy_gs
classify_state_degeneracy
classify_state_negative_spectrum
classify_state_frustration_index
```

## Frustration index DFS
```@docs
find_cycles
get_neighbors
dfs_find_cycles!
evaluate_cycles_signs
count_unbalanced_cycles
find_frustration_index
```

## Utility graph function
```@docs
create_random_signed_graph
plot_signed_graph
```