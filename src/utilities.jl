export dense_similar,
    similar_named_tuple,
    spindigits!,
    spindigits,
    binary_to_spin,
    spin_to_binary,
    LogRange,
    nonzero_triu,
    decimal_to_binary,
    decimal_to_spin,
    drop_target_sparsity

dense_similar(A::AbstractArray, args...) = similar(A, args...)
dense_similar(A::AbstractSparseMatrix, args...) = similar(Array(A), args...)

similar_named_tuple(a::NamedTuple) = NamedTuple{keys(a)}(map(k -> similar(a[k]), keys(a)))

function spindigits!(a::AbstractVector{<:Integer}, n::Integer)
    pad = length(a)

    n < 0 && throw(ArgumentError("n must be non-negative"))
    n < 2^pad || throw(ArgumentError("n must be less than 2^pad"))

    i = typeof(a)(0:pad-1)
    a .= ifelse.((n .>> i) .& 1 .== 1, 1, -1)

    return a
end

function spindigits(n::Integer, pad::Integer)
    a = Vector{Int8}(undef, pad)
    spindigits!(a, n)
    return a
end

function decimal_to_binary(n::Integer, pad::Integer)
    (0 <= n && log2(n) < pad) || throw(ArgumentError("n must be non-negative and less than 2^pad"))
    digits = zeros(Int8, pad)
    for i in 1:pad
        digits[i] = n % 2
        n ÷= 2
    end
    return digits
end

decimal_to_spin(n::Integer, pad::Integer) = 2 .* decimal_to_binary(n, pad) .- 1

function binary_to_spin(
    W::AbstractMatrix{T},
    bias::Union{Nothing,<:AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    isnothing(bias) && (bias = zeros(T, size(W, 1)))

    J = W ./ 4
    c = (bias .+ sum(W, dims = 2)[:]) / 2

    return J, c
end

function spin_to_binary(J::AbstractMatrix{T}, c::Union{Nothing,<:AbstractVector{T}} = nothing) where {T<:AbstractFloat}
    isnothing(c) && (c = zeros(T, size(J, 1)))

    W = 4 .* J
    bias = 2 .* c .- 4 .* sum(J, dims = 2)[:]

    return W, bias
end

LogRange(start, stop, length) = exp10.(LinRange(log10(start), log10(stop), length))

function nonzero_triu(A::Matrix; skip_zeros = true)
    n = size(A, 1)
    coo = Vector{Tuple{Int,Int}}(undef, n * (n - 1) ÷ 2)
    compact = Vector{eltype(A)}(undef, n * (n - 1) ÷ 2)

    count = 1
    for col in 1:n
        for row in 1:col-1
            val = A[row, col]
            val == 0 && skip_zeros && continue
            coo[count] = (row, col)
            compact[count] = val
            count += 1
        end
    end

    coo = coo[1:count-1]
    compact = compact[1:count-1]

    return coo, compact
end

function nonzero_triu(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    numnz = nnz(A) ÷ 2
    coo = Vector{Tuple{Ti,Ti}}(undef, numnz)
    compact = Vector{Tv}(undef, numnz)

    count = 1
    @inbounds for col in 1:size(A, 2)
        for k in A.colptr[col]:(A.colptr[col+1]-1)
            val = rowvals(A)[k]
            val >= col && continue
            coo[count] = (val, col)
            compact[count] = A.nzval[k]
            count += 1
        end
    end

    return coo, compact
end

function _drop_target_sparsity(W, target, alpha_bounds = (0.0, 1.0), depth = 1; max_depth = 10, abstol = 1e-6)
    tentative_alpha = sum(alpha_bounds) / 2

    pattern = abs.(W) .> tentative_alpha

    for i in axes(W, 1)
        if !any(pattern[i, :])
            min_idx = partialsortperm(W[i, :], 2, by = abs)
            pattern[i, min_idx] = pattern[min_idx, i] = true
        end
    end

    actual_sparsity = (1 - count(!, pattern) / length(pattern))

    if abs(actual_sparsity - target) < abstol
        return pattern
    elseif actual_sparsity < target && depth < max_depth
        return _drop_target_sparsity(W, target, (alpha_bounds[1], tentative_alpha), depth + 1, max_depth = max_depth)
    elseif actual_sparsity > target && depth < max_depth
        return _drop_target_sparsity(W, target, (tentative_alpha, alpha_bounds[2]), depth + 1, max_depth = max_depth)
    else
        return pattern
    end
end

function drop_target_sparsity(W::AbstractMatrix, target_sparsity::Real; max_depth = 30)
    pattern = _drop_target_sparsity(W, target_sparsity, max_depth = max_depth)
    return W .* pattern
end
