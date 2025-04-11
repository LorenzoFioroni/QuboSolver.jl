@testitem "random > sherrington-kirkpatrick" begin
    using StableRNGs

    N = 100
    rng = StableRNG(1234)
    W = rand(SherringtonKirkpatrick(), N; rng = rng, eltype = Float32)
    avg = sum(W) / (N^2 - N)
    stdev = sqrt(sum((W .- avg) .^ 2) / (N^2 - N))
    @test size(W) == (N, N)
    @test typeof(W) == Matrix{Float32}
    @test all(isfinite.(W))
    @test W ≈ W'
    @test all(idx -> iszero(W[idx]), diagind(W))
    @test isapprox(avg, 0.0; atol = 5e-3)
    @test isapprox(stdev, 1 / 2√N; atol = 5e-3)

    # Test that the random number generator is used
    rng1 = StableRNG(1234)
    rng2 = StableRNG(1234)
    rng3 = StableRNG(5678)
    W1 = rand(SherringtonKirkpatrick(), N; rng = rng1, eltype = Float32)
    W2 = rand(SherringtonKirkpatrick(), N; rng = rng2, eltype = Float32)
    W3 = rand(SherringtonKirkpatrick(), N; rng = rng3, eltype = Float32)
    @test W1 == W2
    @test W1 != W3
end

@testitem "random > edwards-anderson" begin
    using StableRNGs

    # Test dense matrix type, shape and pattern
    L = 3
    N = L^3
    rng = StableRNG(1234)
    W = rand(EdwardsAnderson(), L; rng = rng, eltype = Float32, sparse = false)
    check_fun =
        ((i, j),) ->
            (
                abs(((i - 1) % L) - ((j - 1) % L)) +
                abs((((i - 1) ÷ L) % L) - (((j - 1) ÷ L) % L)) +
                abs(((i - 1) ÷ (L^2)) - ((j - 1) ÷ (L^2))) == 1
            ) == (W[i, j] != 0.0)
    @test size(W) == (N, N)
    @test typeof(W) == Matrix{Float32}
    @test all(isfinite.(W))
    @test W ≈ W'
    @test all(idx -> iszero(W[idx]), diagind(W))
    @test all(check_fun, Iterators.product(1:N, 1:N))

    # Test dense matrix statistics
    L = 12
    N = L^3
    N_nnz = 6 * (L^3 + 3L^2 - 8L - 12)
    rng = StableRNG(1234)
    W = rand(EdwardsAnderson(), L; rng = rng, eltype = Float32, sparse = false)
    avg = sum(W) / N_nnz
    stdev = sqrt(sum((W .- avg) .^ 2) / N_nnz)
    @test isapprox(avg, 0.0; atol = 5e-2)
    @test isapprox(stdev, 0.5; atol = 5e-2)

    # Test sparse matrix type, shape and pattern
    L = 3
    N = L^3
    N_nnz = 6 * (L^3 + 3L^2 - 8L - 12)
    rng = StableRNG(1234)
    W = rand(EdwardsAnderson(), L; rng = rng, eltype = Float32, sparse = true)
    check_fun =
        ((i, j),) ->
            (
                abs(((i - 1) % L) - ((j - 1) % L)) +
                abs((((i - 1) ÷ L) % L) - (((j - 1) ÷ L) % L)) +
                abs(((i - 1) ÷ (L^2)) - ((j - 1) ÷ (L^2))) == 1
            ) == (W[i, j] != 0.0)
    @test size(W) == (N, N)
    @test typeof(W) == SparseMatrixCSC{Float32,Int}
    @test all(isfinite.(W))
    @test W ≈ W'
    @test all(idx -> iszero(W[idx]), diagind(W))
    @test all(check_fun, Iterators.product(1:N, 1:N))
    @test nnz(W) == N_nnz
    @test all(!iszero(W.nzval))

    # Test sparse matrix statistics
    L = 12
    N = L^3
    N_nnz = 6 * (L^3 + 3L^2 - 8L - 12)
    rng = StableRNG(1234)
    W = rand(EdwardsAnderson(), L; rng = rng, eltype = Float32, sparse = true)
    avg = sum(W) / N_nnz
    stdev = sqrt(sum((W .- avg) .^ 2) / N_nnz)
    @test isapprox(avg, 0.0; atol = 5e-2)
    @test isapprox(stdev, 0.5; atol = 5e-2)

    # Test that the random number generator is used for dense matrices
    rng1 = StableRNG(1234)
    rng2 = StableRNG(1234)
    rng3 = StableRNG(5678)
    W1 = rand(EdwardsAnderson(), L; rng = rng1, eltype = Float32, sparse = false)
    W2 = rand(EdwardsAnderson(), L; rng = rng2, eltype = Float32, sparse = false)
    W3 = rand(EdwardsAnderson(), L; rng = rng3, eltype = Float32, sparse = false)
    @test W1 == W2
    @test W1 != W3

    # Test that the random number generator is used for sparse matrices
    rng1 = StableRNG(1234)
    rng2 = StableRNG(1234)
    rng3 = StableRNG(5678)
    W1 = rand(EdwardsAnderson(), L; rng = rng1, eltype = Float32, sparse = true)
    W2 = rand(EdwardsAnderson(), L; rng = rng2, eltype = Float32, sparse = true)
    W3 = rand(EdwardsAnderson(), L; rng = rng3, eltype = Float32, sparse = true)
    @test W1 == W2
    @test W1 != W3
end

@testitem "random > chimera" begin
    using StableRNGs

    # Test dense matrix type and shape
    N_rows = 4
    N_cols = 5
    N_spin_layer = 6
    N = N_rows * N_cols * 2N_spin_layer
    rng = StableRNG(1234)
    W = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng,
        eltype = Float32,
        sparse = false,
    )
    @test size(W) == (N, N)
    @test typeof(W) == Matrix{Float32}
    @test all(isfinite.(W))
    @test W ≈ W'
    @test all(idx -> iszero(W[idx]), diagind(W))

    # Test dense matrix statistics
    N_rows = 6
    N_cols = 5
    N_spin_layer = 30
    N = N_rows * N_cols * 2N_spin_layer
    N_nnz = 2N_spin_layer * (N_cols * N_rows * (N_spin_layer + 2) - N_rows - N_cols)
    rng = StableRNG(1234)
    W = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng,
        eltype = Float32,
        sparse = false,
    )
    avg = sum(W) / N_nnz
    stdev = sqrt(sum((W .- avg) .^ 2) / N_nnz)
    @test isapprox(avg, 0.0; atol = 5e-2)
    @test isapprox(stdev, 1 / 2√N_spin_layer; atol = 5e-2)

    # Test sparse matrix type and shape
    N_rows = 4
    N_cols = 5
    N_spin_layer = 6
    N = N_rows * N_cols * 2N_spin_layer
    N_nnz = 2N_spin_layer * (N_cols * N_rows * (N_spin_layer + 2) - N_rows - N_cols)
    rng = StableRNG(1234)
    W = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng,
        eltype = Float32,
        sparse = true,
    )
    @test size(W) == (N, N)
    @test typeof(W) == SparseMatrixCSC{Float32,Int}
    @test all(isfinite.(W))
    @test W ≈ W'
    @test all(idx -> iszero(W[idx]), diagind(W))
    @test nnz(W) == N_nnz
    @test all(!iszero(W.nzval))

    # Test sparse matrix statistics
    N_rows = 6
    N_cols = 5
    N_spin_layer = 30
    N = N_rows * N_cols * 2N_spin_layer
    N_nnz = 2N_spin_layer * (N_cols * N_rows * (N_spin_layer + 2) - N_rows - N_cols)
    rng = StableRNG(1234)
    W = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng,
        eltype = Float32,
        sparse = true,
    )
    avg = sum(W.nzval) / N_nnz
    stdev = sqrt(sum((W.nzval .- avg) .^ 2) / N_nnz)
    @test isapprox(avg, 0.0; atol = 5e-2)
    @test isapprox(stdev, 1 / 2√N_spin_layer; atol = 5e-2)

    # Test that the random number generator is used for dense matrices
    rng1 = StableRNG(1234)
    rng2 = StableRNG(1234)
    rng3 = StableRNG(5678)
    W1 = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng1,
        eltype = Float32,
        sparse = false,
    )
    W2 = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng2,
        eltype = Float32,
        sparse = false,
    )
    W3 = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng3,
        eltype = Float32,
        sparse = false,
    )
    @test W1 == W2
    @test W1 != W3

    # Test that the random number generator is used for sparse matrices
    rng1 = StableRNG(1234)
    rng2 = StableRNG(1234)
    rng3 = StableRNG(5678)
    W1 = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng1,
        eltype = Float32,
        sparse = true,
    )
    W2 = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng2,
        eltype = Float32,
        sparse = true,
    )
    W3 = rand(
        Chimera(),
        N_rows,
        N_cols,
        N_spin_layer;
        rng = rng3,
        eltype = Float32,
        sparse = true,
    )
    @test W1 == W2
    @test W1 != W3
end
