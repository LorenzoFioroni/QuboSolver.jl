export EdwardsAnderson, SherringtonKirkpatrick, Chimera


abstract type QuboProblemClass end

struct SherringtonKirkpatrick <: QuboProblemClass end
struct EdwardsAnderson <: QuboProblemClass end
struct Chimera <: QuboProblemClass end

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
        @views W[idx:idx+N_spin_layer-1, idx+N_spin_layer:idx+2N_spin_layer-1] .=
            randn(rng, N_spin_layer, N_spin_layer)
    end

    if N_rows > 1
        for offset ∈ 0:N_spin_layer-1
            @views W[1+offset:2N_spin_layer:N_spins, 1+offset:2N_spin_layer:N_spins][row_coupling_idxs] .=
                randn(rng, (N_rows - 1) * N_cols)
        end
    end

    if N_cols > 1
        for offset ∈ 0:N_spin_layer-1
            @views W[
                1+offset+N_spin_layer:2N_spin_layer:N_spins,
                1+offset+N_spin_layer:2N_spin_layer:N_spins,
            ][col_coupling_idxs] .= randn(rng, N_rows * N_cols - 1)
            @views W[
                offset+2N_spin_layer*N_cols+1-N_spin_layer:2N_spin_layer*N_cols:N_spins-N_spin_layer,
                offset+2N_spin_layer*N_cols-N_spin_layer+1+2N_spin_layer:2N_spin_layer*N_cols:N_spins-N_spin_layer,
            ][decouple_idxs] .= 0
        end
    end

    dropzeros!(W)
    W .+= W'

    W ./= 2sqrt(N_spin_layer)

    return sparse ? W : Matrix(W)
end
