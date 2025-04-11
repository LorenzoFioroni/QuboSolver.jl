@testitem "problem > constructor without bias" begin
    W = [0.0 1.0; 1.0 0.0]
    problem = QuboProblem(W)
    @test problem.W == W
    @test problem.has_bias == false
    @test problem.c === nothing
    @test problem.N == 2
    @test problem.solutions == Solution[]
end

@testitem "problem > constructor with bias" begin
    W = [0.0 1.0; 1.0 0.0]
    bias = [0.0, 1.0]
    problem = QuboProblem(W, bias)
    @test problem.W == W
    @test problem.has_bias == true
    @test problem.c == bias
    @test problem.N == 2
    @test problem.solutions == Solution[]
end

@testitem "problem > constructor errors" begin
    # Test for non-square matrix
    W = [0.0 1.0 1.0; 1.0 0.0 1.0]
    @test_throws ArgumentError QuboProblem(W)

    # Test for non-symmetric matrix
    W = [0.0 1.0; 0.0 0.0]
    @test_throws ArgumentError QuboProblem(W)

    # Test for non-zero diagonal
    W = [1.0 1.0; 1.0 0.0]
    @test_throws ArgumentError QuboProblem(W)

    # Test for bias vector length mismatch
    W = [0.0 1.0; 1.0 0.0]
    bias = [0.5]
    @test_throws ArgumentError QuboProblem(W, bias)
end

@testitem "problem > energy computation" begin
    # Test for Int8 configuration
    W = [0.0 1.0; 1.0 0.0]
    bias = [0.0, 1.0]
    problem = QuboProblem(W, bias)
    configuration = Int8[1, -1]
    energy = get_energy(problem, configuration)
    @test energy == 3.0

    # Test for non-integer configuration
    W = [0.0 1.0; 1.0 0.0]
    bias = [0.0, 1.0]
    problem = QuboProblem(W, bias)
    configuration = [1.0, -1.0]
    energy = get_energy(problem, configuration)
    @test energy == 3.0

    # Test for non +1/-1 configuration
    configuration = [1.0, 0.0]
    @test_throws ArgumentError get_energy(problem, configuration)
end

@testitem "problem > adding solutions" begin
    struct TestSolver <: AbstractSolver end

    # Solution instance
    W = [0.0 1.0; 1.0 0.0]
    problem = QuboProblem(W)
    solution = Solution(2.0, [1, -1], TestSolver())
    add_solution(problem, solution)
    @test length(problem.solutions) == 1
    @test problem.solutions[1] == solution

    # Integer configuration
    W = [0.0 1.0; 1.0 0.0]
    problem = QuboProblem(W)
    configuration = Int8[1, -1]
    solution = add_solution(problem, configuration, TestSolver())
    @test length(problem.solutions) == 1
    @test problem.solutions[1] == solution

    # Non-integer configuration
    W = [0.0 1.0; 1.0 0.0]
    problem = QuboProblem(W)
    configuration = [1.0, -1.0]
    solution = add_solution(problem, configuration, TestSolver())
    @test length(problem.solutions) == 1
    @test problem.solutions[1] == solution
end

@testitem "problem > abstract solver" begin
    struct TestSolver <: AbstractSolver end
    W = [0.0 1.0; 1.0 0.0]
    problem = QuboProblem(W)
    @test_throws ErrorException solve!(problem, TestSolver())
end
