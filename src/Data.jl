"""
    struct Data{S <: Vector{SS} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real}

Structure representing input data for the model.

## Fields
- `s::S`: Input states.
- `l::L`: Labels.

"""
struct Data{S <: Vector{SS} where SS <: ArrayReg, L <: Vector{LL} where LL <: Real}
    s::S
    l::L
end