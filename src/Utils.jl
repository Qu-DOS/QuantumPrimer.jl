export haar_random_unitary,
       circ_haar_random_unitary,
       wasserstein_distance

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