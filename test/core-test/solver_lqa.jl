@testitem "variational meanfield solver > input pre-processing" begin
    using QuboSolver.Solvers.LQA
    using StableRNGs

    N = 10
    rng = StableRNG(1234)
    θ = 2 .* rand(rng, N) .- 1
    z = similar(θ)
    LQA.preprocess_input!(z, θ)
    @test all(θ .≈ π / 2 .* tanh.(z))
end

@testitem "variational meanfield solver > trigonometric" begin
    using QuboSolver.Solvers.LQA
    using StableRNGs

    N = 10
    rng = StableRNG(1234)
    θ = 2 .* rand(rng, N) .- 1
    z = similar(θ)
    LQA.preprocess_input!(z, θ)

    # Test the allocating function
    sines, cosines = LQA.compute_trig(z)
    @test all(sines .≈ sin.(θ))
    @test all(cosines .≈ cos.(θ))

    # Test the non-allocating function
    sines = similar(θ)
    cosines = similar(θ)
    LQA.compute_trig!(sines, cosines, z)
    @test all(sines .≈ sin.(θ))
    @test all(cosines .≈ cos.(θ))
end

@testitem "variational meanfield solver > schedule" begin
    using QuboSolver.Solvers.LQA

    # Test the ease function
    times = 0.0:0.1:1.0
    ease = LQA.ease.(times)
    @test all(x -> 0.0 <= x <= 1.0, ease)
    @test ease[1] == 0.0
    @test ease[end] == 1.0

    # Test the transverse-field schedule
    times = 0.0:0.1:1.0
    schedule = LQA.a.(times)
    @test all(x -> 0.0 <= x <= 1.0, schedule)
    @test schedule[1] == 1.0
    @test schedule[end] == 0.0

    # Test the Ising term schedule
    times = 0.0:0.1:1.0
    schedule = LQA.b.(times)
    @test all(x -> 0.0 <= x <= 1.0, schedule)
    @test schedule[1] == 0.0
    @test schedule[end] == 1.0
end

@testitem "variational meanfield solver > loss function" begin
    using QuboSolver.Solvers.LQA
    using StableRNGs
    import QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = rand(rng, N)
    problem = QuboProblem(W, bias)
    t = 0.4
    tf = 1.7

    # State definition 
    θ = 2 .* rand(rng, N) .- 1
    z = similar(θ)
    LQA.preprocess_input!(z, θ)
    sines, cosines = LQA.compute_trig(z)

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    ψ0 = mapreduce(
        i -> cos(θ[i] / 2) * ket_plus + sin(θ[i] / 2) * ket_minus,
        QuantumToolbox.kron,
        1:N,
    )
    H_tf = sum(op_on_ax(sigma_x, i) for i ∈ 1:N)
    H_ising =
        sum(W[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j) +
        sum(bias[i] * op_on_ax(sigma_z, i) for i ∈ 1:N)
    H = LQA.a(t) * tf * H_tf + LQA.b(t) * H_ising
    numerical_loss = -QuantumToolbox.expect(H, ψ0)

    # Test the loss function
    loss = LQA.loss(problem, sines, cosines, t, tf)
    @test loss isa Float64
    @test loss ≈ numerical_loss
end

@testitem "variational meanfield solver > gradient" begin
    using QuboSolver.Solvers.LQA
    using StableRNGs
    import QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = rand(rng, N)
    problem = QuboProblem(W, bias)
    t = 0.4
    tf = 1.7
    Δ = 1e-8

    # State definition 
    θ = 2 .* rand(rng, N) .- 1
    z = similar(θ)
    LQA.preprocess_input!(z, θ)
    sines, cosines = LQA.compute_trig(z)

    # Numerical gradient
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    loss =
        (z) -> begin
            ψ0 = mapreduce(
                i ->
                    cos(π / 4 * tanh(z[i])) * ket_plus +
                    sin(π / 4 * tanh(z[i])) * ket_minus,
                QuantumToolbox.kron,
                1:N,
            )
            H_tf = sum(op_on_ax(sigma_x, i) for i ∈ 1:N)
            H_ising =
                sum(W[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j) +
                sum(bias[i] * op_on_ax(sigma_z, i) for i ∈ 1:N)
            H = LQA.a(t) * tf * H_tf + LQA.b(t) * H_ising
            numerical_loss = -QuantumToolbox.expect(H, ψ0)
            return numerical_loss
        end
    numerical_gradient = similar(z)
    z_test = deepcopy(z)
    for i ∈ 1:N
        z_test[i] = z[i] + Δ
        l1 = loss(z_test)
        z_test[i] = z[i] - Δ
        l2 = loss(z_test)
        numerical_gradient[i] = (l1 - l2) / (2 * Δ)
        z_test[i] = z[i]
    end

    # Test the gradient function
    dz = similar(z)
    LQA.grad!(dz, problem, z, sines, cosines, t, tf)
    @test dz isa Vector{Float64}
    @test all(isapprox.(dz, numerical_gradient, atol = 1e-8, rtol = 1e-5))
end

@testitem "variational meanfield solver > sign rounding" begin
    using QuboSolver.Solvers.LQA
    using StableRNGs
    import QuantumToolbox

    # Problem definition
    rng = StableRNG(1234)
    N = 10
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = rand(rng, N)
    problem = QuboProblem(W, bias)

    # State definition
    θ = 2 .* rand(rng, N) .- 1
    z = similar(θ)
    LQA.preprocess_input!(z, θ)

    # Numerical rounding
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_z = QuantumToolbox.sigmaz()
    numerical_configuration = map(1:N) do i
        ψ = cos(θ[i] / 2) * ket_plus + sin(θ[i] / 2) * ket_minus
        bit = sign(QuantumToolbox.expect(sigma_z, ψ))
        return Int8(bit == 0 ? 1 : bit)
    end

    # Test the rounding function
    configuration = LQA.round_configuration(problem, z, SignRounding())
    @test configuration isa Vector{Int8}
    @test all(x -> x == -1 || x == 1, configuration)
    @test all(configuration .== numerical_configuration)
end

@testitem "variational meanfield solver > sequential rounding" begin
    using QuboSolver.Solvers.LQA
    using StableRNGs
    import QuantumToolbox

    # Test bias term error
    rng = StableRNG(1234)
    N = 10
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = rand(rng, N)
    problem = QuboProblem(W, bias)
    θ = 2 .* rand(rng, N) .- 1
    z = similar(θ)
    LQA.preprocess_input!(z, θ)
    @test_throws ArgumentError LQA.round_configuration(problem, z, SequentialRounding())

    # Problem definition 
    rng = StableRNG(1234)
    N = 10
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    problem = QuboProblem(W)

    # State definition
    θ = 2 .* rand(rng, N) .- 1
    z = similar(θ)
    LQA.preprocess_input!(z, θ)

    # Numerical rounding
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_z = QuantumToolbox.sigmaz()
    numerical_configuration = map(1:N) do i
        ψ = cos(θ[i] / 2) * ket_plus + sin(θ[i] / 2) * ket_minus
        return QuantumToolbox.expect(sigma_z, ψ)
    end
    for i ∈ 1:N
        numerical_configuration[i] = sign(dot(W[:, i], numerical_configuration))
    end
    numerical_configuration = map(numerical_configuration) do x
        return Int8(x == 0 ? 1 : x)
    end

    # Test the rounding function
    configuration = LQA.round_configuration(problem, z, SequentialRounding())
    @test configuration isa Vector{Int8}
    @test all(x -> x == -1 || x == 1, configuration)
    @test all(configuration .== numerical_configuration)
end

@testitem "variational meanfield solver > solve" begin
    using QuboSolver.Solvers.LQA
    using StableRNGs
    import Suppressor: @capture_err

    @test LQA_solver <: AbstractSolver
    @test SignRounding <: LQA.RoundingMethod
    @test SequentialRounding <: LQA.RoundingMethod

    rng = StableRNG(1234)
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = LQA_solver()

    # Test the solve function
    sol = solve!(problem, solver; rng = rng, progressbar = false)
    @test sol isa Solution{Float64,LQA_solver}
    @test sol.energy isa Float64
    @test sol.configuration isa Vector{Int8}
    @test sol.energy ≈ -sum(abs, W) - sum(bias)
    @test all(x -> x == -1 || x == 1, sol.configuration)
    @test sol.configuration == [1, -1, 1, -1, 1] .* (-1)^(idx - 1) # spin idx is going to be +1
    @test sol.solver === solver

    # Test MCS bounds
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    @test_throws ArgumentError solve!(problem, solver, iterations = 1, progressbar = false)

    # Test progress bar suppression
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    rng = StableRNG(1234)
    output = @capture_err begin
        sol = solve!(problem, solver; rng = rng, progressbar = false)
    end
    expected_output = ""
    @test output == expected_output

    # Test progress bar
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    rng = StableRNG(1234)
    N_iterations = 100_000
    output = @capture_err begin
        sol = solve!(problem, solver; iterations = N_iterations, rng = rng, progressbar = true)
    end
    @test output != ""

    # Test multiple rounding methods
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    problem = QuboProblem(W)
    rng = StableRNG(1234)
    solutions = solve!(
        problem,
        solver;
        rng = rng,
        progressbar = false,
        rounding = (SignRounding(), SequentialRounding()),
    )
    @test all(x -> x isa Solution{Float64,LQA_solver}, solutions)

    # Test energy storage
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    rng = StableRNG(1234)
    N_iterations = 1000
    tf = 1.7
    sol = solve!(
        problem,
        solver;
        rng = rng,
        tf = 1.7,
        iterations = N_iterations,
        progressbar = false,
        save_energy = true,
    )
    @test sol.metadata[:energy] isa Vector{Float64}
    @test length(sol.metadata[:energy]) == N_iterations
    @test isapprox(sol.metadata[:energy][1], -tf * N, atol = 1e-3)

    # Test parameter storage
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    rng = StableRNG(1234)
    sol = solve!(problem, solver; rng = rng, progressbar = false, save_params = true)
    @test sol.metadata[:params] isa Vector{Float64}
    @test length(sol.metadata[:params]) == N

    # Test runtime measurement
    N = 5
    W = (-1) .^ ((0:(N-1))' .+ (0:(N-1))) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    rng = StableRNG(1234)
    N_iterations = 100_000
    solve!(problem, solver; rng = rng, iterations = N_iterations, progressbar = false) # warmup
    sol = solve!(problem, solver; rng = rng, iterations = N_iterations, progressbar = false)
    runtime = sol.metadata[:runtime]
    sol2 =
        solve!(problem, solver; rng = rng, iterations = 2N_iterations, progressbar = false)
    runtime2 = sol2.metadata[:runtime]
    sol3 = solve!(
        problem,
        solver;
        rng = rng,
        iterations = N_iterations,
        inner_iterations = 2,
        progressbar = false,
    )
    runtime3 = sol3.metadata[:runtime]
    @test isapprox(runtime2, 2 * runtime, rtol = 0.3)
    @test isapprox(runtime3, 2 * runtime, rtol = 0.3)
end
