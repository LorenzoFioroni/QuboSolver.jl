@testitem "tamc solver > method file" tags = [:needsthirdparty] begin
    using QuboSolver.Solvers.PTICM

    MCS = 350
    warmup_fraction = 0.1
    betas = [0.1, 0.2, 0.3]
    lo_num_beta = 2
    threads = 2
    num_replica_chains = 3
    file = PTICM.write_method_file(
        MCS,
        warmup_fraction,
        betas,
        lo_num_beta,
        threads,
        num_replica_chains,
    )
    try
        @test file isa String
        @test isfile(file)
        file_content = read(file, String)
        @test occursin("num_sweeps: $MCS", file_content)
        @test occursin("warmup_fraction: $warmup_fraction", file_content)
        @test occursin(
            "  beta:\n    Arr:\n" * join(["    - $beta\n" for beta ∈ betas]),
            file_content,
        )
        @test occursin("icm: true", file_content)
        @test occursin("lo_num_beta: $lo_num_beta", file_content)
        @test occursin("threads: $threads", file_content)
        @test occursin("num_replica_chains: $num_replica_chains", file_content)
    finally
        rm(file)
    end
end

@testitem "tamc solver > instance file" tags = [:needsthirdparty] begin
    using QuboSolver.Solvers.PTICM

    N = 5
    W = [0.0 1.0 2.0; 1.0 0.0 3.0; 2.0 3.0 0.0]
    bias = [4.0, 5.0, 6.0]
    problem = QuboProblem(W, bias)
    file = PTICM.write_instance_file(problem)
    try
        @test file isa String
        @test isfile(file)
        file_content = read(file, String)
        expected_output = join([
            "0 1 -1.0\n",
            "0 2 -2.0\n",
            "1 2 -3.0\n",
            "0 0 4.0\n",
            "1 1 5.0\n",
            "2 2 6.0\n",
        ])
        @test file_content == expected_output
    finally
        rm(file)
    end
end

@testitem "tamc solver > solve" tags = [:needsthirdparty] begin
    using QuboSolver.Solvers.PTICM
    using StableRNGs
    import Suppressor: @capture_err

    @test PTICM_solver <: AbstractSolver

    # Test result
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = PTICM_solver()
    sol = solve!(problem, solver; MCS = 10_000)
    @test sol isa Solution{Float64,PTICM_solver}
    @test sol.energy isa Float64
    @test sol.configuration isa Vector{Int8}
    @test sol.energy ≈ -sum(abs, W) - sum(bias)
    @test sol.configuration == [1, -1, 1, -1, 1] .* (-1)^(idx - 1) # spin idx is going to be +1
    @test sol.solver === solver
    @test sol.metadata[:runtime] isa Float64

    # Test warmup_fraction bounds
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = PTICM_solver()
    @test_throws ArgumentError solve!(problem, solver, MCS = 10_000, warmup_fraction = -0.1)
    @test_throws ArgumentError solve!(problem, solver, MCS = 10_000, warmup_fraction = 1.1)

    # Test MCS bounds
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = PTICM_solver()
    @test_throws ArgumentError solve!(problem, solver, MCS = 0)

    # Test non-empty betas
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = PTICM_solver()
    @test_throws ArgumentError solve!(problem, solver, betas = Float64[])

    # Test lo_num_beta bounds
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = PTICM_solver()
    @test_throws ArgumentError solve!(
        problem,
        solver,
        betas = [0.1, 0.2, 0.3],
        lo_num_beta = 0,
    )
    @test_throws ArgumentError solve!(
        problem,
        solver,
        betas = [0.1, 0.2, 0.3],
        lo_num_beta = 4,
    )
end
