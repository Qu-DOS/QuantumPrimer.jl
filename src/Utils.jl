export haar_random_unitary,
       circ_haar_random_unitary,
       wasserstein_distance,
       KL_divergence,
       skewness

function haar_random_unitary(n::Int) # mezzadri2007generate math-ph/0609050
    U = randn(ComplexF64, n, n) + im*randn(ComplexF64, n, n)
    Z = qr(U)
    D = diagm(0 => diag(Z.R)./abs.(diag(Z.R)))
    return Z.Q * D
end

function circ_haar_random_unitary(n::Int)
    U = haar_random_unitary(2^n)
    circ = chain(n)
    push!(circ, matblock(U))
    return circ
end

function wasserstein_distance(p::Vector{Float64}, q::Vector{Float64})
    if abs(sum(p) - 1.0) > 1e-6 || abs(sum(q) - 1.0) > 1e-6
        error("Input vectors must be valid probability distributions.")
    end
    cdf_p = cumsum(p)
    cdf_q = cumsum(q)
    distance = sum(abs.(cdf_p .- cdf_q))
    return distance
end

function KL_divergence(p::Vector{Float64}, q::Vector{Float64})
    if abs(sum(p) - 1.0) > 1e-6 || abs(sum(q) - 1.0) > 1e-6
        error("Input vectors must be valid probability distributions.")
    end
    length(p) == length(q) || error("Input vectors must have the same length.")
    for i in eachindex(p)
        if q[i] == 0 && p[i] != 0
            error("p must be 0 whenever q is 0 for the KL divergence to be well-defined.")
        end
    end
    return sum(p .* log.(p ./ q))
end

function skewness(vec::Vector{Float64})
    n = length(vec)
    mean_v = mean(vec)
    std_v = std(vec)
    skewness = (n / ((n-1)*(n-2))) * sum(((x - mean_v) / std_v)^3 for x in vec)
    return skewness
end