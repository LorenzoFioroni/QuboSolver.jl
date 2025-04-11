@testitem "solution > constructor" begin
    struct TestSolver <: AbstractSolver end
    # Test with Int8 configuration
    energy = 0.5
    configuration = Int8[1, -1, 1]
    metadata = Dict(:time => 0.1)
    solution = Solution(energy, configuration, TestSolver(); time = 0.1)
    @test solution.energy == energy
    @test solution.configuration == configuration
    @test solution.solver == TestSolver()
    @test solution.metadata == metadata

    # Test with non-integer configuration
    energy = 0.5
    configuration = [1.0, -1.0, 1.0]
    metadata = Dict(:time => 0.1)
    solution = Solution(energy, configuration, TestSolver(); time = 0.1)
    @test solution.energy == energy
    @test solution.configuration == Int8.(configuration)
    @test solution.solver == TestSolver()
    @test solution.metadata == metadata
end

@testitem "solution > constructor errors" begin
    struct TestSolver <: AbstractSolver end
    # Test for non +1/-1 configuration
    energy = 0.5
    configuration = Int8[1, 0, -1]
    @test_throws ArgumentError Solution(energy, configuration, TestSolver())
end

@testitem "solution > metadata access" begin
    struct TestSolver <: AbstractSolver end
    energy = 0.5
    configuration = Int8[1, -1, 1]
    solution = Solution(energy, configuration, TestSolver(); time = 0.1, iterations = 10)
    @test solution[:time] == 0.1
    @test solution[:iterations] == 10
    @test solution["time"] == 0.1
    @test solution["iterations"] == 10
end

@testitem "solution > displaying" begin
    struct TestSolver <: AbstractSolver end
    energy = 0.5
    configuration = Int8[1, -1, 1]
    solution = Solution(energy, configuration, TestSolver(); time = 0.1)
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))
    expected_output = "🟦🟨🟦 - Energy: 0.5 - Solver: TestSolver - Metadata count: 1"
    @test occursin(expected_output, output)
end

@testitem "solution > equality" begin
    struct TestSolver <: AbstractSolver end
    energy = 0.5
    configuration = Int8[1, -1, 1]
    sol1 = Solution(energy, configuration, TestSolver(); time = 0.1)
    sol2 = Solution(energy, configuration, TestSolver(); time = 0.1)
    sol3 = Solution(0.6, configuration, TestSolver(); time = 0.1)
    @test isequal(sol1, sol2)
    @test !isequal(sol1, sol3)
end
