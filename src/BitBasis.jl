function density_matrix_from_vector(states::Vector; coeffs=:nothing)
    register = []
    for ele in states
        push!(register, ArrayReg(bit_literal(ele...)))
    end
    if coeffs!=:nothing
        return density_matrix(sum(register.*coeffs) |> normalize!)
    else
        return density_matrix(sum(register) |> normalize!)
    end
end

function register_from_vector(states::Vector; coeffs=:nothing)
    register = []
    for ele in states
        push!(register, ArrayReg(bit_literal(ele...)))
    end
    if coeffs!=:nothing
        coeffs /= sqrt(sum(coeffs.^2))
        return sum(register.*coeffs)
    else
        return sum(register./sqrt(length(states)))
    end
end

function create_bit_basis(n, simpl)
    res = []
    for ele in simpl
        tmp = zeros(Int64, n)
        for i in 1:n
            i in ele ? tmp[i]=1 : nothing
        end
        push!(res, tmp)
    end
    return res
end

function create_bit_basis(n, k, simpl)
    return filter(j -> (sum(j)==k+1), create_bit_basis(n, simpl))
end

function create_bit_basis_rand(n, k, simpl; superposition=false)
    basis = create_bit_basis(n, k, simpl)
    return superposition ? rand(basis, rand(1:length(basis))) : rand(basis, 1)
end