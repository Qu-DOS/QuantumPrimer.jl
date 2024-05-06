abstract type Params end

Base.@kwdef mutable struct GenericParams{NN<:Integer, CC<:ChainBlock, TT<:AbstractVector{T} where T<:Real} <:Params
    n::NN
    circ::CC
    params::TT = Float64[]
end

Base.@kwdef mutable struct InvariantParams{NN<:Integer, CC<:ChainBlock, TT<:AbstractVector{T} where T<:Real} <:Params
    n::NN
    circ::CC
    params::TT = Float64[]
end

#initial params (no translational invariance). Take in a seed for reproducibility
function initialize_params(p::GenericParams)
    !ispow2(p.n) ? error("The register dimension has to be a power of 2") : nothing
    n_params = nparameters(p.circ)
    p.params = 2pi * rand(n_params)
end

#initial params (no translational invariance). Take in a seed for reproducibility
function initialize_params(p::InvariantParams)
    !ispow2(p.n) ? error("The register dimension has to be a power of 2") : nothing
    n_params = trunc(Int, 2*log2(p.n)) # extended vector dim is trunc(Int, 2*sum([2^i for i in 1:log2(n)]))
    p.params = 2pi * rand(n_params)
end

# expand unique translational invariant parameters to full vector
function expand_params(p::InvariantParams)
    n_params = length(p.params) # extended vector dim is trunc(Int, 2*sum([2^i for i in 1:log2(n)]))
    p_rev = reverse(p.params)
    return vcat([reverse(repeat(p_rev[1+2i:2+2i], 2*2^i)) for i in (n_params÷2)-1:-1:0]...)
end

# reduce vector of parameters to unique set (sum of corresponding parameters in the expanded version)
# used in full_grad evaluation
function reduce_params(n, params)
    layers = [trunc(Int, 2*sum([2^jj for jj in 1:ii])) for ii in 0:log2(n)]
    res = []
    p_rev = reverse(params)
    for i in 1:trunc(Int, log2(n))
        tmp = p_rev[1+layers[i]:2:layers[i+1]]
        push!(res, sum(tmp/length(tmp)))
        tmp = p_rev[2+layers[i]:2:layers[i+1]]
        push!(res, sum(tmp/length(tmp)))
    end
    return reverse(res)
end