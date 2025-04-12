export EdwardsAnderson, SherringtonKirkpatrick, Chimera

@doc raw"""
    abstract type QuboProblemClass end

Abstract type representing a class of QUBO problems.
"""
abstract type QuboProblemClass end

@doc raw"""
    struct SherringtonKirkpatrick <: QuboProblemClass end

An instance of `QuboProblemClass` representing the Sherrington-Kirkpatrick model.
"""
struct SherringtonKirkpatrick <: QuboProblemClass end

@doc raw"""
    struct EdwardsAnderson <: QuboProblemClass end

An instance of `QuboProblemClass` representing the Edwards-Anderson model.
"""
struct EdwardsAnderson <: QuboProblemClass end

@doc raw"""
    struct Chimera <: QuboProblemClass end

An instance of `QuboProblemClass` representing the Chimera model.
"""
struct Chimera <: QuboProblemClass end

@doc raw"""
    function rand(
        ::SherringtonKirkpatrick, 
        N::Int; 
        rng::AbstractRNG = Random.GLOBAL_RNG,
        eltype::Type = Float64
    )

Generate a random QUBO matrix for the Sherrington-Kirkpatrick model.

# Arguments

  - `N::Int`: Number of variables.
  - `rng::AbstractRNG`: Random number generator (default: `Random.GLOBAL_RNG`).
  - `eltype::Type`: Element type of the matrix elements (default: `Float64`).

# Example

```julia
W = rand(SherringtonKirkpatrick(), 4)
println(size(W))
println(all(diag(W) .== 0))
println(W == transpose(W))

# output

(4, 4)
true
true
```
"""
function Base.rand(
    ::SherringtonKirkpatrick,
    N::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    eltype::Type = Float64,
)
    W = randn(rng, eltype, N, N)
    W = triu(W, 1) + transpose(triu(W, 1))
    W ./= 2sqrt(N)
    return W
end

@doc raw"""
    function rand(
        ::EdwardsAnderson, 
        N_side::Int; 
        rng::AbstractRNG = Random.GLOBAL_RNG,
        sparse::Bool = false,
        eltype::Type = Float64
    )

Generate a random QUBO matrix for the 3D Edwards-Anderson model with open boundary conditions.

# Arguments

  - `N_side::Int`: Side length of the cubic lattice.
  - `rng::AbstractRNG`: Random number generator (default: `Random.GLOBAL_RNG`).
  - `sparse::Bool`: If true, generate a sparse matrix (default: `false`).
  - `eltype::Type`: Element type of the matrix elements (default: `Float64`).

# Example

```julia
W = rand(EdwardsAnderson(), 4; sparse = true) # 4x4x4 lattice
println(size(W))
println(typeof(W))
println(all(diag(W) .== 0))
println(W == transpose(W))

# output

(64, 64)
SparseMatrixCSC{Float64, Int64}
true
true
```
"""
function Base.rand(
    ::EdwardsAnderson,
    N_side::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    sparse::Bool = false,
    eltype::Type = Float64,
)
    N = N_side^3
    W = sparse ? spzeros(eltype, N, N) : zeros(eltype, N, N)
    a = reshape(1:N, (N_side, N_side, N_side))
    ind = 1:N_side
    ind = circshift(ind, -1)
    b = reshape(a[ind, :, :], N, 1)
    c = reshape(a[:, ind, :], N, 1)
    d = reshape(a[:, :, ind], N, 1)
    a = collect(1:N)
    fb = CartesianIndex.(a, b)
    fc = CartesianIndex.(a, c)
    fd = CartesianIndex.(a, d)
    W[fb] = randn(rng, N, 1)
    W[fc] = randn(rng, N, 1)
    W[fd] = randn(rng, N, 1)
    W = triu(W, 1) + transpose(triu(W, 1))
    W ./= 2
    return W
end

@doc raw"""
    function rand(
        ::Chimera, 
        N_rows::Int, 
        N_cols::Int, 
        N_spin_layer::Int = 4; 
        rng::AbstractRNG = Random.GLOBAL_RNG,
        sparse::Bool = false,
        eltype::Type = Float64
    )

Generate a random QUBO matrix for the Chimera model.

# Arguments

  - `N_rows::Int`: Number of rows in the Chimera lattice.
  - `N_cols::Int`: Number of columns in the Chimera lattice.
  - `N_spin_layer::Int`: Number of spins per unit cell (default: 4).
  - `rng::AbstractRNG`: Random number generator (default: `Random.GLOBAL_RNG`).
  - `sparse::Bool`: If true, generate a sparse matrix (default: `false`).
  - `eltype::Type`: Element type of the matrix elements (default: `Float64`).

# Example

```julia
W = rand(Chimera(), 4, 4, 4; sparse = true) # 4x4x(4x2) lattice
println(size(W))
println(typeof(W))
println(all(diag(W) .== 0))
println(W == transpose(W))

# output

(128, 128)
SparseMatrixCSC{Float64, Int64}
true
true
```
"""
function Base.rand(
    ::Chimera,
    N_rows::Int,
    N_cols::Int,
    N_spin_layer::Int = 4;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    sparse::Bool = false,
    eltype::Type = Float64,
)
    N_spins = N_rows * N_cols * 2N_spin_layer
    W = spzeros(eltype, N_spins, N_spins)

    row_coupling_idxs = diagind(N_rows * N_cols, N_rows * N_cols, N_cols)
    col_coupling_idxs = diagind(N_rows * N_cols, N_rows * N_cols, 1)
    decouple_idxs = diagind(N_rows - 1, N_rows - 1)

    for idx ∈ 1:2N_spin_layer:N_spins
        @views W[idx:(idx+N_spin_layer-1), (idx+N_spin_layer):(idx+2N_spin_layer-1)] .=
            randn(rng, N_spin_layer, N_spin_layer)
    end

    if N_rows > 1
        for offset ∈ 0:(N_spin_layer-1)
            @views W[(1+offset):2N_spin_layer:N_spins, (1+offset):2N_spin_layer:N_spins][row_coupling_idxs] .=
                randn(rng, (N_rows - 1) * N_cols)
        end
    end

    if N_cols > 1
        for offset ∈ 0:(N_spin_layer-1)
            @views W[
                (1+offset+N_spin_layer):2N_spin_layer:N_spins,
                (1+offset+N_spin_layer):2N_spin_layer:N_spins,
            ][col_coupling_idxs] .= randn(rng, N_rows * N_cols - 1)
            @views W[
                (offset+2N_spin_layer*N_cols+1-N_spin_layer):(2N_spin_layer*N_cols):(N_spins-N_spin_layer),
                (offset+2N_spin_layer*N_cols-N_spin_layer+1+2N_spin_layer):(2N_spin_layer*N_cols):(N_spins-N_spin_layer),
            ][decouple_idxs] .= 0
        end
    end

    dropzeros!(W)
    W .+= W'

    W ./= 2sqrt(N_spin_layer)

    return sparse ? W : Matrix(W)
end
