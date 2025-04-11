@testitem "utilities > dense similar" begin
    # Test with dense matrix
    A = rand(Float32, 10, 10)
    B = dense_similar(A)
    @test size(B) == size(A)
    @test eltype(B) == eltype(A)
    @test typeof(B) == Matrix{Float32}

    # Test with sparse matrix
    A = sprand(Float32, 10, 10, 0.1)
    B = dense_similar(A)
    @test size(B) == size(A)
    @test eltype(B) == eltype(A)
    @test typeof(B) == Matrix{Float32}

    # Test with additional arguments and dense matrix
    A = rand(Float32, 10, 10)
    B = dense_similar(A, 2, 2)
    @test size(B) == (2, 2)
    @test eltype(B) == eltype(A)
    @test typeof(B) == Matrix{Float32}

    A = rand(Float32, 10, 10)
    B = dense_similar(A, ComplexF32, (3, 3))
    @test size(B) == (3, 3)
    @test eltype(B) == ComplexF32
    @test typeof(B) == Matrix{ComplexF32}

    # Test with additional arguments and sparse matrix
    A = sprand(Float32, 10, 10, 0.1)
    B = dense_similar(A, 2, 2)
    @test size(B) == (2, 2)
    @test eltype(B) == eltype(A)
    @test typeof(B) == Matrix{Float32}

    A = sprand(Float32, 10, 10, 0.1)
    B = dense_similar(A, ComplexF32, (3, 3))
    @test size(B) == (3, 3)
    @test eltype(B) == ComplexF32
    @test typeof(B) == Matrix{ComplexF32}
end

@testitem "utilities > similar named tuple" begin
    a = (x = rand(3), y = rand(Float32, 2, 2), z = sprand(Float32, 5, 5, 0.5))
    b = similar_named_tuple(a)
    @test typeof(b) == typeof(a)
    for k ∈ (:x, :y, :z)
        @test size(b[k]) == size(a[k])
        @test eltype(b[k]) == eltype(a[k])
    end
end

@testitem "utilities > integer to spin representation" begin
    # Test allocating function
    a = conf_int2spin(5, 7)
    @test a == Int8[1, -1, 1, -1, -1, -1, -1]

    # Test in-place function
    a = Vector{Int8}(undef, 7)
    conf_int2spin!(a, 5)
    @test a == Int8[1, -1, 1, -1, -1, -1, -1]
end

@testitem "utilities > binary to spin problem" begin
    W = [0.0 1.0; 1.0 0.0]
    bias = [1.0, 0.0]
    J, c = binary_to_spin(W, bias)
    @test J == [0.0 0.25; 0.25 0.0]
    @test c == [1.0, 0.5]
end

@testitem "utilities > spin to binary problem" begin
    J = [0.0 0.25; 0.25 0.0]
    c = [1.0, 0.5]
    W, bias = spin_to_binary(J, c)
    @test W == [0.0 1.0; 1.0 0.0]
    @test bias == [1.0, 0.0]
end

@testitem "utilities > logrange" begin
    range = QuboSolver.LogRange(0.1, 1000, 5)
    @test range ≈ [0.1, 1.0, 10.0, 100.0, 1000.0]
end

@testitem "utilities > nonzero upper triangular" begin
    # Test dense matrix skipping zero elements
    A = [1.0 2.0 0.0; 2.0 0.0 3.0; 0.0 3.0 4.0]
    coo, compact_A = nonzero_triu(A)
    @test coo == [(1, 2), (2, 3)]
    @test compact_A == [2.0, 3.0]

    # Test dense matrix keeping zero elements
    A = [1.0 2.0 0.0; 2.0 0.0 3.0; 0.0 3.0 4.0]
    coo, compact_A = nonzero_triu(A; skip_zeros = false)
    @test coo == [(1, 2), (1, 3), (2, 3)]
    @test compact_A == [2.0, 0.0, 3.0]

    # Test sparse matrix
    A = sparse([1.0 2.0 0.0; 2.0 0.0 3.0; 0.0 3.0 4.0])
    coo, compact_A = nonzero_triu(A)
    @test coo == [(1, 2), (2, 3)]
    @test compact_A == [2.0, 3.0]
end

@testitem "utilities > drop target sparsity" begin
    A = randn(1000, 1000)
    target_sparsity = 0.35
    B = drop_target_sparsity(A, target_sparsity; max_depth = 50)
    @test size(B) == size(A)
    @test eltype(B) == eltype(A)
    @test typeof(B) == typeof(A)
    @test isapprox(nnz(sparse(B)) / length(B), target_sparsity, atol = 0.005)

    # At least one element per row
    A = randn(10, 10)
    target_sparsity = 1e-5
    B = drop_target_sparsity(A, target_sparsity; max_depth = 50)
    @test !any(all.(eachrow(B .== 0)))

    # Partial result when max_depth is reached
    A = randn(1000, 1000)
    target_sparsity = 0.23
    B = drop_target_sparsity(A, target_sparsity; max_depth = 2)
    @test nnz(sparse(B)) / length(B) < 1.0
end
