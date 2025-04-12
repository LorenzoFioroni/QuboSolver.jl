@testitem "brute force solver" begin
    using QuboSolver.Solvers.BruteForce

    @test BruteForce_solver <: AbstractSolver

    # Test without bias term
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    problem = QuboProblem(W)
    solver = BruteForce_solver()
    solution = solve!(problem, solver)
    @test solution isa Solution{Float64,BruteForce_solver}
    @test solution.energy isa Float64
    @test solution.configuration isa Vector{Int8}
    @test solution.energy ≈ -sum(abs, W)
    @test solution.configuration == [1, -1, 1, -1, 1] .* solution.configuration[1] # Force same sign as solution
    @test solution.solver === solver
    @test solution.metadata[:runtime] isa Float64
    @test 0.0 < solution.metadata[:runtime] < 60
    @test solution.metadata[:nthreads] == Threads.nthreads()

    # Test with bias term
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 0.1
    problem = QuboProblem(W, bias)
    solver = BruteForce_solver()
    solution = solve!(problem, solver)
    @test solution isa Solution{Float64,BruteForce_solver}
    @test solution.energy isa Float64
    @test solution.configuration isa Vector{Int8}
    @test solution.energy ≈ -sum(abs, W) - sum(bias)
    @test solution.configuration == [1, -1, 1, -1, 1] .* (-1)^(idx - 1) # spin idx is going to be +1
    @test solution.solver === solver
    @test solution.metadata[:runtime] isa Float64
    @test 0.0 < solution.metadata[:runtime] < 60
    @test solution.metadata[:nthreads] == Threads.nthreads()
end
