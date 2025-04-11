export dense_similar,
    similar_named_tuple,
    conf_int2spin!,
    conf_int2spin,
    binary_to_spin,
    spin_to_binary,
    nonzero_triu,
    drop_target_sparsity

@doc raw"""
    function dense_similar(A::AbstractArray, args...)
    function dense_similar(A::AbstractSparseMatrix, args...)

Create a dense array similar to the input `A`.

Eventual additional arguments are passed to the `similar` function.

# Example
```jldoctest
A = sprand(Float32, 10, 10, 0.1)
B = dense_similar(A)
println(size(B)) 
println(eltype(B))
println(typeof(B))

# output

(10, 10)
Float32
Matrix{Float32}
```
"""
dense_similar(A::AbstractArray, args...) = similar(A, args...)
dense_similar(A::AbstractSparseMatrix, args...) = similar(Array(A), args...)

@doc raw"""
    function similar_named_tuple(a::NamedTuple)

Create a new NamedTuple by recursively applying `similar` to each element of the input `a`.

# Example
```jldoctest
a = (x = rand(3), y = rand(Float32, 2, 2))
b = similar_named_tuple(a)
println(typeof(b) == typeof(a))
println(size(b.x) == size(a.x))
println(size(b.y) == size(a.y))

# output

true
true
true
```
"""
similar_named_tuple(a::NamedTuple) = NamedTuple{keys(a)}(map(k -> similar(a[k]), keys(a)))

@doc raw"""
    function conf_int2spin!(a::AbstractVector{<:Integer}, n::Integer)
    
Fill the vector `a` with the binary representation of the integer `n` as ``1`` and ``-1``.

See also [`conf_int2spin`](@ref QuboSolver.conf_int2spin)

# Example
```jldoctest
a = Vector{Int8}(undef, 4)
conf_int2spin!(a, 5)
println(a) 

# output

Int8[1, -1, 1, -1]
```
"""
function conf_int2spin!(a::AbstractVector{<:Integer}, n::Integer)
    pad = length(a)

    n < 0 && throw(ArgumentError("n must be non-negative"))
    log2(n) < pad || throw(ArgumentError("n must be less than 2^pad"))

    i = typeof(a)(0:pad-1)
    a .= ifelse.((n .>> i) .& 1 .== 1, 1, -1)

    return a
end

@doc raw"""
    function conf_int2spin(n::Integer, pad::Integer)

Create an n-long vector containing the binary representation of the integer `n` as ``1`` and ``-1``.

See also [`conf_int2spin!`](@ref QuboSolver.conf_int2spin!)

# Example
```jldoctest
a = conf_int2spin(5, 7)
println(a) 

# output

Int8[1, -1, 1, -1, -1, -1, -1]
```
"""
function conf_int2spin(n::Integer, pad::Integer)
    a = Vector{Int8}(undef, pad)
    conf_int2spin!(a, n)
    return a
end

@doc raw"""
    function binary_to_spin(
        W::AbstractMatrix{T}, 
        bias::Union{Nothing,<:AbstractVector{T}} = nothing
    ) where {T<:AbstractFloat}

Convert a QUBO matrix `W` and an optional bias vector `bias` from binary to spin representation.

# Returns
- `J::AbstractMatrix{T}`: The spin coupling matrix.
- `c::Union{Nothing,<:AbstractVector{T}}`: The bias vector.

# Example
```jldoctest
W = [0.0 1.0; 1.0 0.0]
bias = [1.0, 0.0]
J, c = binary_to_spin(W, bias)
println(J)
println(c)

# output

[0.0 0.25; 0.25 0.0]
[1.0, 0.5]
```
"""
function binary_to_spin(
    W::AbstractMatrix{T},
    bias::Union{Nothing,<:AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    isnothing(bias) && (bias = zeros(T, size(W, 1)))

    J = W ./ 4
    c = (bias .+ sum(W; dims = 2)[:]) / 2

    return J, c
end

@doc raw"""
    function spin_to_binary(
        J::AbstractMatrix{T}, 
        c::Union{Nothing,<:AbstractVector{T}} = nothing
    ) where {T<:AbstractFloat}

Convert a QUBO matrix `J` and an optional bias vector `c` from spin to binary representation.

# Returns
- `W::AbstractMatrix{T}`: The binary coupling matrix.
- `bias::Union{Nothing,<:AbstractVector{T}}`: The bias vector.

# Example
```jldoctest
J = [0.0 0.25; 0.25 0.0]
c = [1.0, 0.5]
W, bias = spin_to_binary(J, c)
println(W)
println(bias)

# output

[0.0 1.0; 1.0 0.0]
[1.0, 0.0]
```
"""
function spin_to_binary(
    J::AbstractMatrix{T},
    c::Union{Nothing,<:AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    isnothing(c) && (c = zeros(T, size(J, 1)))

    W = 4 .* J
    bias = 2 .* c .- 4 .* sum(J; dims = 2)[:]

    return W, bias
end

@doc raw"""
    function LogRange(start::Real, stop::Real, length::Int)

Create a log-spaced range of values between `start` and `stop` with `length` elements.
"""
LogRange(start, stop, length) = exp10.(LinRange(log10(start), log10(stop), length))

@doc raw"""
    function nonzero_triu(A::AbstractMatrix; skip_zeros = true)

Extract the coordinates and values of the non-zero elements in the upper triangular part of a matrix `A`.

# Arguments
- `A::AbstractMatrix`: The input matrix.
- `skip_zeros::Bool`: If true, skip zero values (default: true).

# Returns
- `coo::Vector{Tuple{Int,Int}}`: A vector of tuples representing the coordinates of the non-zero elements.
- `compact::Vector{eltype(A)}`: A vector of the non-zero values.

# Example
```jldoctest
W = [0.0 1.0 0.0; 1.0 0.0 2.0; 0.0 2.0 0.0]
coo, compact = nonzero_triu(W)
println(coo)
println(compact)

# output

[(1, 2), (2, 3)]
[1.0, 2.0]
```
"""
function nonzero_triu(A::Matrix; skip_zeros = true)
    n = size(A, 1)
    coo = Vector{Tuple{Int,Int}}(undef, n * (n - 1) ÷ 2)
    compact = Vector{eltype(A)}(undef, n * (n - 1) ÷ 2)

    count = 1
    for col ∈ 1:n
        for row ∈ 1:col-1
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

@doc raw"""
    function nonzero_triu(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Extract the coordinates and values of the non-zero elements in the upper triangular part of a sparse matrix `A`.

The diagonal elements are not included.
"""
function nonzero_triu(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    numnz = (nnz(A) - nnz(diag(A))) ÷ 2
    coo = Vector{Tuple{Ti,Ti}}(undef, numnz)
    compact = Vector{Tv}(undef, numnz)

    count = 1
    @inbounds for col ∈ 1:size(A, 2)
        for k ∈ A.colptr[col]:(A.colptr[col+1]-1)
            val = rowvals(A)[k]
            val >= col && continue
            coo[count] = (val, col)
            compact[count] = A.nzval[k]
            count += 1
        end
    end

    return coo, compact
end

function _drop_target_sparsity(
    W,
    target,
    alpha_bounds = (0.0, 1.0),
    depth = 1;
    max_depth = 10,
    abstol = 1e-6,
)
    tentative_alpha = sum(alpha_bounds) / 2

    pattern = abs.(W) .> tentative_alpha

    for i ∈ axes(W, 1)
        if !any(pattern[i, :])
            min_idx = partialsortperm(W[i, :], 2; by = abs)
            pattern[i, min_idx] = pattern[min_idx, i] = true
        end
    end

    actual_sparsity = (1 - count(!, pattern) / length(pattern))

    if abs(actual_sparsity - target) < abstol
        return pattern
    elseif actual_sparsity < target && depth < max_depth
        return _drop_target_sparsity(
            W,
            target,
            (alpha_bounds[1], tentative_alpha),
            depth + 1;
            max_depth = max_depth,
        )
    elseif actual_sparsity > target && depth < max_depth
        return _drop_target_sparsity(
            W,
            target,
            (tentative_alpha, alpha_bounds[2]),
            depth + 1;
            max_depth = max_depth,
        )
    else
        return pattern
    end
end

@doc raw"""
    function drop_target_sparsity(
        W::AbstractMatrix, 
        target_sparsity::Real; 
        max_depth = 30
    )

Drop the smallest elements of the matrix `W` until the target sparsity is reached.

# Arguments
- `W::AbstractMatrix`: The input matrix.
- `target_sparsity::Real`: The target sparsity level (between 0 and 1).
- `max_depth::Int`: The maximum recursion depth (default: 30).

# Returns
A copy of `W` with the smallest elements dropped to reach the target sparsity.

# Example
```jldoctest; setup = :(using Random; Random.seed!(1234))
A = randn(1000, 1000)
target_sparsity = 0.34
B = sparse(drop_target_sparsity(A, target_sparsity; max_depth = 50))
println(round(nnz(B)/length(B), digits = 2))

# output

0.34
```
"""
function drop_target_sparsity(W::AbstractMatrix, target_sparsity::Real; max_depth = 30)
    pattern = _drop_target_sparsity(W, target_sparsity; max_depth = max_depth)
    return W .* pattern
end
