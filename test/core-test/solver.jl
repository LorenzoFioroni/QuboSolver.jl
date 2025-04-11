@testitem "solver" begin
    struct TestSolver <: AbstractSolver end

    @test TestSolver <: AbstractSolver

    io = IOBuffer()
    show(io, TestSolver())
    output = String(take!(io))
    expected_output = "Solver: TestSolver"
    @test occursin(expected_output, output)
end
