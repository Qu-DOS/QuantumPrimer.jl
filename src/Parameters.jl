abstract type Params end

struct GenericParams{TT<:AbstractVector{T} where {T<:Real}} <: Params
    params::TT
end

struct InvariantParams{TT<:AbstractVector{T} where {T<:Real}} <: Params
    params::TT
end

#initial params (no translational invariance). Take in a seed for reproducibility
function initialize_params(n; invariant=false)
    if !ispow2(n)
        error("The register dimension has to be a power of 2")
    end
    if invariant
        n_params = trunc(Int, 2*log2(n)) # extended vector dim is trunc(Int, 2*sum([2^i for i in 1:log2(n)]))
    else
        n_params = nparameters(build_QCNN(n))
    end
    return 2pi*rand(n_params)
end

# expand unique translational invariant parameters to full vector
function expand_params(params)
    n_params = length(params) # extended vector dim is trunc(Int, 2*sum([2^i for i in 1:log2(n)]))
    params_rev = reverse(params)
    return vcat([reverse(repeat(params_rev[1+2i:2+2i], 2*2^i)) for i in (n_params÷2)-1:-1:0]...)
end

# reduce vector of parameters to unique set (sum of corresponding parameters in the expanded version)
# used in full_grad evaluation
function reduce_params(n, params)
    layers = [trunc(Int, 2*sum([2^jj for jj in 1:ii])) for ii in 0:log2(n)]
    res = []
    params_rev = reverse(params)
    for i in 1:trunc(Int, log2(n))
        tmp = params_rev[1+layers[i]:2:layers[i+1]]
        push!(res, sum(tmp/length(tmp)))
        tmp = params_rev[2+layers[i]:2:layers[i+1]]
        push!(res, sum(tmp/length(tmp)))
    end
    return reverse(res)
end