@testitem "variational gcs solver > param type definition" begin
    using QuboSolver.Solvers.GCS

    for f ∈ (:r, :d, :g, :t, :p)
        @test fieldtype(GCS.ParamType{Float32}, f) == Vector{Float32}
    end
    @test fieldtype(GCS.ParamType{Float32}, :M) == Matrix{Float32}
end

@testitem "variational gcs solver > trig type definition" begin
    using QuboSolver.Solvers.GCS

    for f ∈ (:sin_p, :cos_p, :sin_t, :cos_t, :sin_r, :cos_r, :sin_d, :cos_d)
        @test fieldtype(GCS.TrigType{Float32}, f) == Vector{Float32}
    end
    @test fieldtype(GCS.TrigType{Float32}, :exp_g) == Vector{Complex{Float32}}
    @test fieldtype(GCS.TrigType{Float32}, :exp_M) == Matrix{Complex{Float32}}
end

@testitem "variational gcs solver > temp type definition" begin
    using QuboSolver.Solvers.GCS

    for f ∈ (:Sz, :dSz_r, :dSz_d, :Sx, :dSx_r, :dSx_d, :dSx_g)
        @test fieldtype(GCS.TempType{Float32}, f) ==
              @NamedTuple{a1::Vector{Complex{Float32}}, a0::Vector{Float32}}
    end
    @test fieldtype(GCS.TempType{Float32}, :dSz_g) ==
          @NamedTuple{a1::Vector{Complex{Float32}}}
    for f ∈ (:psi, :dpsi_t, :dpsi_p)
        @test fieldtype(GCS.TempType{Float32}, f) == Matrix{Complex{Float32}}
    end
end

@testitem "variational gcs solver > thread temp type definition" begin
    using QuboSolver.Solvers.GCS

    for f ∈ (
        :psi_ket,
        :psi_ket_dM1,
        :psi_ket_dM2,
        :dpsi_t_ket,
        :dpsi_t_bra,
        :dpsi_p_ket,
        :dpsi_p_bra,
    )
        @test fieldtype(GCS.ThreadTempType{Float32}, f) == Matrix{Complex{Float32}}
    end
    for f ∈ (
        :mat_el,
        :mat_el_psi_dpsi_t,
        :mat_el_dpsi_t_psi,
        :mat_el_psi_dpsi_p,
        :mat_el_dpsi_p_psi,
        :mat_el_dM1,
        :mat_el_dM2,
    )
        @test fieldtype(GCS.ThreadTempType{Float32}, f) == Vector{Complex{Float32}}
    end
end

@testitem "variational gcs solver > prod and count nonzeros" begin
    using QuboSolver.Solvers.GCS

    arr = collect(Float64, 1:10)
    arr2 = collect(Float64, 1:10)
    arr3 = collect(Float64, 1:10)
    arr2[4] = 0.0
    arr3[4] = 0.0
    arr3[7] = 0.0

    @test GCS.prod_and_count_zeros(arr) == (factorial(10), -1)
    @test GCS.prod_and_count_zeros(arr2) == (factorial(10) / 4, 4)
    @test GCS.prod_and_count_zeros(arr3) == (0.0, -2)
end

@testitem "variational gcs solver > setup initial configuration" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs

    # Test sampled configuration
    N = 10
    rng1 = StableRNG(1234)
    rng2 = StableRNG(1234)
    params1 = GCS._setup_initial_conf(nothing, N, rng1, Float32)
    params2 = GCS._setup_initial_conf(nothing, N, rng2, Float32)
    @test params1 isa GCS.ParamType{Float32}
    @test params1 == params2
    for f ∈ (:r, :d, :g, :t, :p)
        @test size(params1[f]) == (N,)
    end
    @test size(params1[:M]) == (N, N)

    # Test provided configuration
    N = 10
    rng = StableRNG(1234)
    original_params = GCS._setup_initial_conf(nothing, N, rng, Float32)
    params = GCS._setup_initial_conf(original_params, N, rng, Float32)
    @test params isa GCS.ParamType{Float32}
    @test params == original_params
    @test params !== original_params
end

@testitem "variational gcs solver > trig type" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs

    # Test allocation
    N = 10
    rng = StableRNG(1234)
    params = GCS._setup_initial_conf(nothing, N, rng, Float32)
    trig = GCS.alloc_trig(params)
    @test trig isa GCS.TrigType{Float32}
    for f ∈ (:sin_p, :cos_p, :sin_t, :cos_t, :sin_r, :cos_r, :sin_d, :cos_d, :exp_g)
        @test size(trig[f]) == (N,)
    end
    @test size(trig.exp_M) == (N, N)

    # Test non-allocating computation
    N = 10
    rng = StableRNG(1234)
    params = GCS._setup_initial_conf(nothing, N, rng, Float32)
    trig = GCS.alloc_trig(params)
    GCS.compute_trig!(trig, params)
    @test trig.sin_t == sin.(params.t ./ 2)
    @test trig.cos_t == cos.(params.t ./ 2)
    @test trig.sin_r == sin.(2 .* params.r)
    @test trig.cos_r == cos.(2 .* params.r)
    @test trig.sin_d == sin.(params.d)
    @test trig.cos_d == cos.(params.d)
    @test trig.sin_p == sin.(params.p)
    @test trig.cos_p == cos.(params.p)
    @test trig.exp_g == exp.(1im .* params.g)
    @test trig.exp_M == exp.(0.25im .* params.M)

    # Test allocating computation
    N = 10
    rng = StableRNG(1234)
    params = GCS._setup_initial_conf(nothing, N, rng, Float32)
    trig = GCS.alloc_trig(params)
    GCS.compute_trig!(trig, params)
    trig2 = GCS.compute_trig(params)
    @test trig2 == trig
end

@testitem "variational gcs solver > temp type" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs

    # Test allocation
    N = 10
    rng = StableRNG(1234)
    params = GCS._setup_initial_conf(nothing, N, rng, Float32)
    temporal = GCS.alloc_temporal(params)
    @test temporal isa GCS.TempType{Float32}
    for f ∈ (:Sz, :dSz_r, :dSz_d, :Sx, :dSx_r, :dSx_d, :dSx_g)
        @test size(temporal[f].a0) == (N,)
        @test size(temporal[f].a1) == (N,)
    end
    @test size(temporal.dSz_g.a1) == (N,)
    for f ∈ (:psi, :dpsi_t, :dpsi_p)
        @test size(temporal[f]) == (N, 2)
    end

    # Test non-allocating computation
    N = 10
    rng = StableRNG(1234)
    params = GCS._setup_initial_conf(nothing, N, rng, Float32)
    trig = GCS.compute_trig(params)
    temporal = GCS.alloc_temporal(params)
    GCS.compute_temporal!(temporal, trig; grad = true)
    @test all(
        isapprox.(
            temporal.Sz.a1,
            trig.cos_d .* trig.exp_g .*
            (trig.sin_d .* (1 .- trig.cos_r) .+ trig.sin_r * im),
        ),
    )
    @test all(isapprox.(temporal.Sz.a0, trig.sin_d .^ 2 .* (1 .- trig.cos_r) .+ trig.cos_r))
    @test all(
        isapprox.(
            temporal.Sx.a1,
            trig.cos_d .^ 2 .* (1 .- trig.cos_r) .* real.(trig.exp_g) .* trig.exp_g .+
            trig.cos_r .- trig.sin_r .* trig.sin_d * im,
        ),
    )
    @test all(
        isapprox.(
            temporal.Sx.a0,
            trig.cos_d .* (
                trig.sin_d .* real.(trig.exp_g) .* (1 .- trig.cos_r) .+
                trig.sin_r .* imag(trig.exp_g)
            ),
        ),
    )
    @test all(
        isapprox.(
            temporal.psi[:, 1],
            ((trig.cos_t) .+ (trig.sin_t .* (trig.cos_p .+ trig.sin_p * im))) ./ sqrt(2),
        ),
    )
    @test all(
        isapprox.(
            temporal.psi[:, 2],
            ((trig.cos_t) .- (trig.sin_t .* (trig.cos_p .+ trig.sin_p * im))) ./ sqrt(2),
        ),
    )
    @test all(
        isapprox.(
            temporal.dSz_r.a1,
            2 .* trig.cos_d .* trig.exp_g .* (trig.sin_r .* trig.sin_d .+ trig.cos_r * im),
        ),
    )
    @test all(isapprox.(temporal.dSz_r.a0, 2 .* trig.sin_r .* (trig.sin_d .^ 2 .- 1)))
    @test all(
        isapprox.(
            temporal.dSz_d.a1,
            (trig.cos_d .^ 2 .- trig.sin_d .^ 2) .* trig.exp_g .* (1 .- trig.cos_r) .-
            trig.sin_r .* trig.sin_d .* trig.exp_g * im,
        ),
    )
    @test all(
        isapprox.(temporal.dSz_d.a0, 2 .* trig.sin_d .* trig.cos_d .* (1 .- trig.cos_r)),
    )
    @test all(
        isapprox.(
            temporal.dSz_g.a1,
            trig.cos_d .* (
                trig.sin_d .* (1 .- trig.cos_r) .* trig.exp_g * im .-
                trig.sin_r .* trig.exp_g
            ),
        ),
    )
    @test all(
        isapprox.(
            temporal.dSx_r.a1,
            2 .* trig.sin_r .* trig.cos_d .^ 2 .* real.(trig.exp_g) .* trig.exp_g .-
            2 .* trig.sin_r .- 2 .* trig.cos_r .* trig.sin_d * im,
        ),
    )
    @test all(
        isapprox.(
            temporal.dSx_r.a0,
            2 .* trig.cos_d .* (
                trig.sin_r .* trig.sin_d .* real.(trig.exp_g) .+
                trig.cos_r .* imag.(trig.exp_g)
            ),
        ),
    )
    @test all(
        isapprox.(
            temporal.dSx_d.a1,
            -2 .* trig.sin_d .* trig.cos_d .* (1 .- trig.cos_r) .* real.(trig.exp_g) .*
            trig.exp_g .- trig.sin_r .* trig.cos_d * im,
        ),
    )
    @test all(
        isapprox.(
            temporal.dSx_d.a0,
            (trig.cos_d .^ 2 .- trig.sin_d .^ 2) .* real.(trig.exp_g) .*
            (1 .- trig.cos_r) .- trig.sin_r .* trig.sin_d .* imag.(trig.exp_g),
        ),
    )
    @test all(
        isapprox.(
            temporal.dSx_g.a1,
            trig.exp_g .^ 2 .* trig.cos_d .^ 2 .* (1 .- trig.cos_r) * im,
        ),
    )
    @test all(
        isapprox.(
            temporal.dSx_g.a0,
            -trig.sin_d .* trig.cos_d .* (1 .- trig.cos_r) .* imag.(trig.exp_g) .+
            trig.sin_r .* trig.cos_d .* real.(trig.exp_g),
        ),
    )
    @test all(
        isapprox.(
            temporal.dpsi_t[:, 1],
            (
                (-0.5 .* trig.sin_t) .+
                (0.5 .* trig.cos_t .* (trig.cos_p .+ trig.sin_p * im))
            ) ./ sqrt(2),
        ),
    )
    @test all(
        isapprox.(
            temporal.dpsi_t[:, 2],
            (
                (-0.5 .* trig.sin_t) .-
                (0.5 .* trig.cos_t .* (trig.cos_p .+ trig.sin_p * im))
            ) ./ sqrt(2),
        ),
    )
    @test all(
        isapprox.(
            temporal.dpsi_p[:, 1],
            (trig.sin_t .* (-trig.sin_p .+ trig.cos_p * im)) ./ sqrt(2),
        ),
    )
    @test all(
        isapprox.(
            temporal.dpsi_p[:, 2],
            -(trig.sin_t .* (-trig.sin_p .+ trig.cos_p * im)) ./ sqrt(2),
        ),
    )
end

@testitem "variational gcs solver > thread temp type" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs

    N = 10
    rng = StableRNG(1234)
    params = GCS._setup_initial_conf(nothing, N, rng, Float32)
    thread_temp = GCS.alloc_thread_temporal(params)
    @test thread_temp isa Vector{GCS.ThreadTempType{Float32}}
    @test length(thread_temp) == Threads.nthreads()
    for ttemp ∈ thread_temp
        for f ∈ (
            :psi_ket,
            :psi_ket_dM1,
            :psi_ket_dM2,
            :dpsi_t_ket,
            :dpsi_t_bra,
            :dpsi_p_ket,
            :dpsi_p_bra,
        )
            @test size(ttemp[f]) == (N, 2)
        end
        for f ∈ (
            :mat_el,
            :mat_el_psi_dpsi_t,
            :mat_el_dpsi_t_psi,
            :mat_el_psi_dpsi_p,
            :mat_el_dpsi_p_psi,
            :mat_el_dM1,
            :mat_el_dM2,
        )
            @test size(ttemp[f]) == (N,)
        end
    end
end

@testitem "variational gcs solver > sum aligned nametuple" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs

    # Test with two namedtuples
    N = 10
    rng = StableRNG(1234)
    orig_a = GCS._setup_initial_conf(nothing, N, rng, Float64)
    a = deepcopy(orig_a)
    b = GCS._setup_initial_conf(nothing, N, rng, Float64)
    GCS._sum_aligned_namedtuples!(a, b, 2.0)
    for f ∈ keys(a)
        @test all(a[f] .≈ orig_a[f] .+ 2.0 .* b[f])
    end

    # Test with three namedtuples
    N = 10
    rng = StableRNG(1234)
    a = GCS._setup_initial_conf(nothing, N, rng, Float64)
    b = GCS._setup_initial_conf(nothing, N, rng, Float64)
    c = GCS._setup_initial_conf(nothing, N, rng, Float64)
    GCS._sum_aligned_namedtuples!(a, b, 2.0, c, 0.5)
    for f ∈ keys(a)
        @test all(a[f] .≈ 2.0 .* b[f] .+ 0.5 .* c[f])
    end
end

@testitem "variational gcs solver > symmetrize M" begin
    using QuboSolver.Solvers.GCS

    M = rand(Float32, 10, 10)
    M[diagind(M)] .= 0.0
    orig_M = deepcopy(M)
    GCS._symmetrize_M!(M)
    @test M ≈ orig_M + orig_M'
end

@testitem "variational gcs solver > schedule" begin
    using QuboSolver.Solvers.GCS

    # Test the ease function
    times = 0.0:0.1:1.0
    ease = GCS.ease.(times)
    @test all(x -> 0.0 <= x <= 1.0, ease)
    @test ease[1] == 0.0
    @test ease[end] == 1.0

    # Test the transverse-field schedule
    times = 0.0:0.1:1.0
    schedule = GCS.a.(times)
    @test all(x -> 0.0 <= x <= 1.0, schedule)
    @test schedule[1] == 1.0
    @test schedule[end] == 0.0

    # Test the Ising term schedule
    times = 0.0:0.1:1.0
    schedule = GCS.b.(times)
    @test all(x -> 0.0 <= x <= 1.0, schedule)
    @test schedule[1] == 0.0
    @test schedule[end] == 1.0
end

@testitem "variational gcs solver > loss x" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    x_multithread_params =
        collect(enumerate(Iterators.partition(1:N, cld(N, Threads.nthreads()))))
    x_loss = zeros(Float64, N)

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    rot_x = r .* cos.(δ) .* cos.(Γ)
    rot_y = r .* cos.(δ) .* sin.(Γ)
    rot_z = r .* sin.(δ)
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)
    GCS.compute_trig!(trig, params)
    GCS.compute_temporal!(temporal, trig)

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    ψ0 = mapreduce(
        i -> cos(θ[i] / 2) * ket_plus + exp(1im * ϕ[i]) * sin(θ[i] / 2) * ket_minus,
        QuantumToolbox.kron,
        1:N,
    )
    V = exp(-1im / 16 * sum(M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j))
    ψ1 = V * ψ0
    U = exp(
        -1im * sum(
            op_on_ax(rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z, i)
            for i ∈ 1:N
        ),
    )
    ψ2 = U * ψ1
    numerical_loss = map(i -> QuantumToolbox.expect(op_on_ax(sigma_x, i), ψ2), 1:N)

    # Test loss function
    GCS.loss_x(x_loss, trig, temporal, thread_temp, x_multithread_params)
    @test all(isapprox.(x_loss, numerical_loss, atol = 1e-8))
end

@testitem "variational gcs solver > loss z" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    x_multithread_params =
        collect(enumerate(Iterators.partition(1:N, cld(N, Threads.nthreads()))))
    z_loss = zeros(Float64, N)

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    rot_x = r .* cos.(δ) .* cos.(Γ)
    rot_y = r .* cos.(δ) .* sin.(Γ)
    rot_z = r .* sin.(δ)
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)
    GCS.compute_trig!(trig, params)
    GCS.compute_temporal!(temporal, trig)

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    ψ0 = mapreduce(
        i -> cos(θ[i] / 2) * ket_plus + exp(1im * ϕ[i]) * sin(θ[i] / 2) * ket_minus,
        QuantumToolbox.kron,
        1:N,
    )
    V = exp(-1im / 16 * sum(M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j))
    ψ1 = V * ψ0
    U = exp(
        -1im * sum(
            op_on_ax(rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z, i)
            for i ∈ 1:N
        ),
    )
    ψ2 = U * ψ1
    numerical_loss = map(i -> QuantumToolbox.expect(op_on_ax(sigma_z, i), ψ2), 1:N)

    # Test loss function
    GCS.loss_z(z_loss, trig, temporal, thread_temp, x_multithread_params)
    @test all(isapprox.(z_loss, numerical_loss, atol = 1e-8))
end

@testitem "variational gcs solver > loss zz" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = rand(rng, N)
    problem = QuboProblem(W, bias)
    compact_idx, compact_W = nonzero_triu(W)
    _params = enumerate(zip(compact_idx, compact_W))
    zz_multithread_params = collect(
        enumerate(Iterators.partition(_params, cld(length(_params), Threads.nthreads()))),
    )
    zz_loss = zeros(Float64, length(compact_idx))

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    rot_x = r .* cos.(δ) .* cos.(Γ)
    rot_y = r .* cos.(δ) .* sin.(Γ)
    rot_z = r .* sin.(δ)
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)
    GCS.compute_trig!(trig, params)
    GCS.compute_temporal!(temporal, trig)

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    ψ0 = mapreduce(
        i -> cos(θ[i] / 2) * ket_plus + exp(1im * ϕ[i]) * sin(θ[i] / 2) * ket_minus,
        QuantumToolbox.kron,
        1:N,
    )
    V = exp(-1im / 16 * sum(M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j))
    ψ1 = V * ψ0
    U = exp(
        -1im * sum(
            op_on_ax(rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z, i)
            for i ∈ 1:N
        ),
    )
    ψ2 = U * ψ1
    numerical_loss = map(
        ((i, j),) ->
            QuantumToolbox.expect(op_on_ax(sigma_z, i) * op_on_ax(sigma_z, j), ψ2),
        compact_idx,
    )

    # Test loss function
    GCS.loss_zz(zz_loss, trig, temporal, thread_temp, zz_multithread_params)
    @test all(isapprox.(zz_loss, numerical_loss, atol = 1e-8))
end

@testitem "variational gcs solver > loss" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = rand(rng, N)
    problem = QuboProblem(W, bias)
    compact_idx, compact_W = nonzero_triu(W)
    _params = enumerate(zip(compact_idx, compact_W))
    x_multithread_params =
        collect(enumerate(Iterators.partition(1:N, cld(N, Threads.nthreads()))))
    zz_multithread_params = collect(
        enumerate(Iterators.partition(_params, cld(length(_params), Threads.nthreads()))),
    )
    x_loss = zeros(Float64, N)
    zz_loss = zeros(Float64, length(compact_idx))
    t = 0.4
    tf = 1.7

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    rot_x = r .* cos.(δ) .* cos.(Γ)
    rot_y = r .* cos.(δ) .* sin.(Γ)
    rot_z = r .* sin.(δ)
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    GCS.compute_trig!(trig, params)
    GCS.compute_temporal!(temporal, trig)
    thread_temp = GCS.alloc_thread_temporal(params)

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    ψ0 = mapreduce(
        i -> cos(θ[i] / 2) * ket_plus + exp(1im * ϕ[i]) * sin(θ[i] / 2) * ket_minus,
        QuantumToolbox.kron,
        1:N,
    )
    V = exp(-1im / 16 * sum(M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j))
    ψ1 = V * ψ0
    U = exp(
        -1im * sum(
            op_on_ax(rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z, i)
            for i ∈ 1:N
        ),
    )
    ψ2 = U * ψ1
    H_tf = sum(op_on_ax(sigma_x, i) for i ∈ 1:N)
    H_ising =
        sum(W[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j) +
        sum(bias[i] * op_on_ax(sigma_z, i) for i ∈ 1:N)
    H = GCS.a(t) * tf * H_tf + GCS.b(t) * H_ising
    numerical_loss = -QuantumToolbox.expect(H, ψ2)

    # Test loss function
    loss = GCS.loss(
        params,
        t,
        tf,
        trig,
        temporal,
        thread_temp,
        zz_multithread_params,
        x_multithread_params,
        compact_W,
        problem.c,
        zz_loss,
        x_loss,
    )
    @test loss isa Float64
    @test loss ≈ numerical_loss
end

@testitem "variational gcs solver > gradient x" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    x_multithread_params =
        collect(enumerate(Iterators.partition(1:N, cld(N, Threads.nthreads()))))
    x_loss = zeros(Float64, N)
    l = zeros(Float64, 1)
    Δ = 1e-7

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)
    GCS.compute_trig!(trig, params)
    GCS.compute_temporal!(temporal, trig)
    dz_x = [similar_named_tuple(params) for _ ∈ 1:Threads.nthreads()]
    for i ∈ 1:Threads.nthreads()
        for f ∈ (:t, :p, :r, :d, :g, :M)
            dz_x[i][f] .= 0.0
        end
    end

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    loss =
        (params, matel = false) -> begin
            ψ0 = mapreduce(
                i ->
                    cos(params.t[i] / 2) * ket_plus +
                    exp(1im * params.p[i]) * sin(params.t[i] / 2) * ket_minus,
                QuantumToolbox.kron,
                1:N,
            )
            V = exp(
                -1im / 16 * sum(
                    params.M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j
                ),
            )
            ψ1 = V * ψ0
            rot_x = params.r .* cos.(params.d) .* cos.(params.g)
            rot_y = params.r .* cos.(params.d) .* sin.(params.g)
            rot_z = params.r .* sin.(params.d)
            U = exp(
                -1im * sum(
                    op_on_ax(
                        rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z,
                        i,
                    ) for i ∈ 1:N
                ),
            )
            ψ2 = U * ψ1
            losses =
                map(i -> real(QuantumToolbox.expect(op_on_ax(sigma_x, i), ψ2)), 1:N)
            return matel ? (sum(losses), losses) : sum(losses)
        end
    _, numerical_loss = loss(params, true)
    numerical_gradient = GCS.ParamType{Float64}((
        r = zeros(Float64, N),
        d = zeros(Float64, N),
        g = zeros(Float64, N),
        t = zeros(Float64, N),
        p = zeros(Float64, N),
        M = zeros(Float64, N, N),
    ))
    for f ∈ (:t, :p, :r, :d, :g)
        for i ∈ 1:N
            original_value = params[f][i]
            params[f][i] = original_value + Δ
            l1 = loss(params)
            params[f][i] = original_value - Δ
            l2 = loss(params)
            params[f][i] = original_value
            numerical_gradient[f][i] = (l1 - l2) / (2 * Δ)
        end
    end
    for i ∈ 1:N
        for j ∈ i+1:N
            original_value = params.M[i, j]
            params.M[i, j] = params.M[j, i] = original_value + Δ
            l1 = loss(params)
            params.M[i, j] = params.M[j, i] = original_value - Δ
            l2 = loss(params)
            params.M[i, j] = params.M[j, i] = original_value
            numerical_gradient.M[i, j] = (numerical_gradient.M[j, i] = (l1 - l2) / (2 * Δ))
        end
    end

    # Test loss function
    GCS.loss_x_and_grad!(x_loss, dz_x, trig, temporal, thread_temp, x_multithread_params)
    @inbounds for ii ∈ 2:Threads.nthreads()
        GCS._sum_aligned_namedtuples!(dz_x[1], dz_x[ii], 1.0)
    end
    @test all(isapprox.(x_loss, numerical_loss, atol = 1e-8))
    for f ∈ (:t, :p, :r, :d, :g)
        @test all(isapprox.(numerical_gradient[f], dz_x[1][f], atol = 1e-8, rtol = 1e-5))
    end
end

@testitem "variational gcs solver > gradient z" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = randn(rng, N)
    problem = QuboProblem(W, bias)
    z_multithread_params = collect(
        enumerate(Iterators.partition(zip(1:N, problem.c), cld(N, Threads.nthreads()))),
    )
    z_loss = zeros(Float64, N)
    l = zeros(Float64, 1)
    Δ = 1e-7

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)
    GCS.compute_trig!(trig, params)
    GCS.compute_temporal!(temporal, trig)
    dz_z = [similar_named_tuple(params) for _ ∈ 1:Threads.nthreads()]
    for i ∈ 1:Threads.nthreads()
        for f ∈ (:t, :p, :r, :d, :g, :M)
            dz_z[i][f] .= 0.0
        end
    end

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    loss =
        (params, matel = false) -> begin
            ψ0 = mapreduce(
                i ->
                    cos(params.t[i] / 2) * ket_plus +
                    exp(1im * params.p[i]) * sin(params.t[i] / 2) * ket_minus,
                QuantumToolbox.kron,
                1:N,
            )
            V = exp(
                -1im / 16 * sum(
                    params.M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j
                ),
            )
            ψ1 = V * ψ0
            rot_x = params.r .* cos.(params.d) .* cos.(params.g)
            rot_y = params.r .* cos.(params.d) .* sin.(params.g)
            rot_z = params.r .* sin.(params.d)
            U = exp(
                -1im * sum(
                    op_on_ax(
                        rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z,
                        i,
                    ) for i ∈ 1:N
                ),
            )
            ψ2 = U * ψ1
            losses =
                map(i -> real(QuantumToolbox.expect(op_on_ax(sigma_z, i), ψ2)), 1:N)
            return matel ? (sum(problem.c .* losses), losses) : sum(problem.c .* losses)
        end
    _, numerical_loss = loss(params, true)
    numerical_gradient = GCS.ParamType{Float64}((
        r = zeros(Float64, N),
        d = zeros(Float64, N),
        g = zeros(Float64, N),
        t = zeros(Float64, N),
        p = zeros(Float64, N),
        M = zeros(Float64, N, N),
    ))
    for f ∈ (:t, :p, :r, :d, :g)
        for i ∈ 1:N
            original_value = params[f][i]
            params[f][i] = original_value + Δ
            l1 = loss(params)
            params[f][i] = original_value - Δ
            l2 = loss(params)
            params[f][i] = original_value
            numerical_gradient[f][i] = (l1 - l2) / (2 * Δ)
        end
    end
    for i ∈ 1:N
        for j ∈ i+1:N
            original_value = params.M[i, j]
            params.M[i, j] = params.M[j, i] = original_value + Δ
            l1 = loss(params)
            params.M[i, j] = params.M[j, i] = original_value - Δ
            l2 = loss(params)
            params.M[i, j] = params.M[j, i] = original_value
            numerical_gradient.M[i, j] = (numerical_gradient.M[j, i] = (l1 - l2) / (2 * Δ))
        end
    end

    # Test loss function
    GCS.loss_z_and_grad!(z_loss, dz_z, trig, temporal, thread_temp, z_multithread_params)
    @inbounds for ii ∈ 2:Threads.nthreads()
        GCS._sum_aligned_namedtuples!(dz_z[1], dz_z[ii], 1.0)
    end
    @test all(isapprox.(z_loss, numerical_loss, atol = 1e-8))
    for f ∈ (:t, :p, :r, :d, :g)
        @test all(isapprox.(numerical_gradient[f], dz_z[1][f], atol = 1e-8, rtol = 1e-5))
    end
end

@testitem "variational gcs solver > gradient z" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = randn(rng, N)
    problem = QuboProblem(W, bias)
    compact_idx, compact_W = nonzero_triu(problem.W)
    _params = enumerate(zip(compact_idx, compact_W))
    zz_multithread_params = collect(
        enumerate(Iterators.partition(_params, cld(length(_params), Threads.nthreads()))),
    )
    zz_loss = zeros(Float64, length(compact_idx))
    l = zeros(Float64, 1)
    Δ = 1e-7

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)
    GCS.compute_trig!(trig, params)
    GCS.compute_temporal!(temporal, trig)
    dz_zz = [similar_named_tuple(params) for _ ∈ 1:Threads.nthreads()]
    for i ∈ 1:Threads.nthreads()
        for f ∈ (:t, :p, :r, :d, :g, :M)
            dz_zz[i][f] .= 0.0
        end
    end

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    loss =
        (params, matel = false) -> begin
            ψ0 = mapreduce(
                i ->
                    cos(params.t[i] / 2) * ket_plus +
                    exp(1im * params.p[i]) * sin(params.t[i] / 2) * ket_minus,
                QuantumToolbox.kron,
                1:N,
            )
            V = exp(
                -1im / 16 * sum(
                    params.M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j
                ),
            )
            ψ1 = V * ψ0
            rot_x = params.r .* cos.(params.d) .* cos.(params.g)
            rot_y = params.r .* cos.(params.d) .* sin.(params.g)
            rot_z = params.r .* sin.(params.d)
            U = exp(
                -1im * sum(
                    op_on_ax(
                        rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z,
                        i,
                    ) for i ∈ 1:N
                ),
            )
            ψ2 = U * ψ1
            losses = map(
                ((i, j),) -> real(
                    QuantumToolbox.expect(op_on_ax(sigma_z, i) * op_on_ax(sigma_z, j), ψ2),
                ),
                compact_idx,
            )
            return if matel
                (sum(2 .* compact_W .* losses), losses)
            else
                sum(2 .* compact_W .* losses)
            end
        end
    _, numerical_loss = loss(params, true)
    numerical_gradient = GCS.ParamType{Float64}((
        r = zeros(Float64, N),
        d = zeros(Float64, N),
        g = zeros(Float64, N),
        t = zeros(Float64, N),
        p = zeros(Float64, N),
        M = zeros(Float64, N, N),
    ))
    for f ∈ (:t, :p, :r, :d, :g)
        for i ∈ 1:N
            original_value = params[f][i]
            params[f][i] = original_value + Δ
            l1 = loss(params)
            params[f][i] = original_value - Δ
            l2 = loss(params)
            params[f][i] = original_value
            numerical_gradient[f][i] = (l1 - l2) / (2 * Δ)
        end
    end
    for i ∈ 1:N
        for j ∈ i+1:N
            original_value = params.M[i, j]
            params.M[i, j] = params.M[j, i] = original_value + Δ
            l1 = loss(params)
            params.M[i, j] = params.M[j, i] = original_value - Δ
            l2 = loss(params)
            params.M[i, j] = params.M[j, i] = original_value
            numerical_gradient.M[i, j] = (numerical_gradient.M[j, i] = (l1 - l2) / (2 * Δ))
        end
    end

    # Test loss function
    GCS.loss_zz_and_grad!(
        zz_loss,
        dz_zz,
        trig,
        temporal,
        thread_temp,
        zz_multithread_params,
    )
    @inbounds for ii ∈ 2:Threads.nthreads()
        GCS._sum_aligned_namedtuples!(dz_zz[1], dz_zz[ii], 1.0)
    end
    @test all(isapprox.(zz_loss, numerical_loss, atol = 1e-8))
    for f ∈ (:t, :p, :r, :d, :g)
        @test all(isapprox.(numerical_gradient[f], dz_zz[1][f], atol = 1e-8, rtol = 1e-5))
    end
end

@testitem "variational gcs solver > gradient" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    N = 4
    rng = StableRNG(1234)
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = randn(rng, N)
    problem = QuboProblem(W, bias)
    compact_idx, compact_W = nonzero_triu(problem.W)
    _params = enumerate(zip(compact_idx, compact_W))
    x_multithread_params =
        collect(enumerate(Iterators.partition(1:N, cld(N, Threads.nthreads()))))
    z_multithread_params = collect(
        enumerate(Iterators.partition(zip(1:N, problem.c), cld(N, Threads.nthreads()))),
    )
    zz_multithread_params = collect(
        enumerate(Iterators.partition(_params, cld(length(_params), Threads.nthreads()))),
    )
    x_loss = zeros(Float64, N)
    zz_loss = zeros(Float64, length(compact_idx))
    l = zeros(Float64, 1)
    Δ = 1e-7
    t = 0.4
    tf = 1.7

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)
    dz_zz = [similar_named_tuple(params) for _ ∈ 1:Threads.nthreads()]
    dz_x = [similar_named_tuple(params) for _ ∈ 1:Threads.nthreads()]
    dz = similar_named_tuple(params)

    # Numerical loss
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    H_tf = sum(op_on_ax(sigma_x, i) for i ∈ 1:N)
    H_ising =
        sum(W[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j) +
        sum(bias[i] * op_on_ax(sigma_z, i) for i ∈ 1:N)
    H = GCS.a(t) * tf * H_tf + GCS.b(t) * H_ising
    loss =
        (params, matel = false) -> begin
            ψ0 = mapreduce(
                i ->
                    cos(params.t[i] / 2) * ket_plus +
                    exp(1im * params.p[i]) * sin(params.t[i] / 2) * ket_minus,
                QuantumToolbox.kron,
                1:N,
            )
            V = exp(
                -1im / 16 * sum(
                    params.M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j
                ),
            )
            ψ1 = V * ψ0
            rot_x = params.r .* cos.(params.d) .* cos.(params.g)
            rot_y = params.r .* cos.(params.d) .* sin.(params.g)
            rot_z = params.r .* sin.(params.d)
            U = exp(
                -1im * sum(
                    op_on_ax(
                        rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z,
                        i,
                    ) for i ∈ 1:N
                ),
            )
            ψ2 = U * ψ1
            return -real(QuantumToolbox.expect(H, ψ2))
        end
    numerical_loss = loss(params)
    numerical_gradient = GCS.ParamType{Float64}((
        r = zeros(Float64, N),
        d = zeros(Float64, N),
        g = zeros(Float64, N),
        t = zeros(Float64, N),
        p = zeros(Float64, N),
        M = zeros(Float64, N, N),
    ))
    for f ∈ (:t, :p, :r, :d, :g)
        for i ∈ 1:N
            original_value = params[f][i]
            params[f][i] = original_value + Δ
            l1 = loss(params)
            params[f][i] = original_value - Δ
            l2 = loss(params)
            params[f][i] = original_value
            numerical_gradient[f][i] = (l1 - l2) / (2 * Δ)
        end
    end
    for i ∈ 1:N
        for j ∈ i+1:N
            original_value = params.M[i, j]
            params.M[i, j] = params.M[j, i] = original_value + Δ
            l1 = loss(params)
            params.M[i, j] = params.M[j, i] = original_value - Δ
            l2 = loss(params)
            params.M[i, j] = params.M[j, i] = original_value
            numerical_gradient.M[i, j] = (numerical_gradient.M[j, i] = (l1 - l2) / (2 * Δ))
        end
    end

    # Test loss function
    GCS.loss_and_grad!(
        l,
        dz,
        params,
        t,
        tf,
        trig,
        temporal,
        thread_temp,
        zz_multithread_params,
        x_multithread_params,
        z_multithread_params,
        compact_W,
        bias,
        zz_loss,
        x_loss,
        dz_zz,
        dz_x,
    )
    @test l[1] ≈ numerical_loss
    for f ∈ (:t, :p, :r, :d, :g)
        @test all(isapprox.(numerical_gradient[f], dz[f], atol = 1e-8, rtol = 1e-5))
    end
end

@testitem "variational gcs solver > sign rounding" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    rng = StableRNG(1234)
    N = 10
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = rand(rng, N)
    problem = QuboProblem(W, bias)
    x_multithread_params =
        collect(enumerate(Iterators.partition(1:N, cld(N, Threads.nthreads()))))

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)

    # Numerical rounding
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    ψ0 = mapreduce(
        i ->
            cos(params.t[i] / 2) * ket_plus +
            exp(1im * params.p[i]) * sin(params.t[i] / 2) * ket_minus,
        QuantumToolbox.kron,
        1:N,
    )
    V = exp(
        -1im / 16 * sum(params.M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j),
    )
    ψ1 = V * ψ0
    rot_x = params.r .* cos.(params.d) .* cos.(params.g)
    rot_y = params.r .* cos.(params.d) .* sin.(params.g)
    rot_z = params.r .* sin.(params.d)
    U = exp(
        -1im * sum(
            op_on_ax(rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z, i)
            for i ∈ 1:N
        ),
    )
    ψ2 = U * ψ1
    numerical_configuration = map(1:N) do i
        bit = sign(QuantumToolbox.expect(op_on_ax(sigma_z, i), ψ2))
        return Int8(bit == 0 ? 1 : bit)
    end

    # Test the rounding function
    configuration = GCS.round_configuration(
        problem,
        params,
        SignRounding(),
        trig,
        temporal,
        thread_temp,
        x_multithread_params,
    )
    @test configuration isa Vector{Int8}
    @test all(x -> x == -1 || x == 1, configuration)
    @test all(configuration .== numerical_configuration)
end

@testitem "variational gcs solver > sequential rounding" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    using QuantumToolbox

    # Problem definition
    rng = StableRNG(1234)
    N = 10
    W = rand(SherringtonKirkpatrick(), N; rng = rng)
    bias = rand(rng, N)
    problem = QuboProblem(W)
    problem_with_bias = QuboProblem(W, bias)
    x_multithread_params =
        collect(enumerate(Iterators.partition(1:N, cld(N, Threads.nthreads()))))

    # State definition
    θ = π * rand(rng, Float64, N)
    ϕ = 2π * rand(rng, Float64, N)
    M = randn(rng, Float64, N, N)
    M = triu(M, 1) + triu(M, 1)'
    r = rand(rng, Float64, N)
    δ = π * rand(rng, Float64, N)
    Γ = 2π * rand(rng, Float64, N)
    params = GCS.ParamType{Float64}((r = r, d = δ, g = Γ, t = θ, p = ϕ, M = M))
    trig = GCS.alloc_trig(params)
    temporal = GCS.alloc_temporal(params)
    thread_temp = GCS.alloc_thread_temporal(params)

    # Numerical rounding
    ket_zero = QuantumToolbox.basis(2, 0)
    ket_one = QuantumToolbox.basis(2, 1)
    ket_plus = (ket_zero + ket_one) / sqrt(2)
    ket_minus = (ket_zero - ket_one) / sqrt(2)
    sigma_x = QuantumToolbox.sigmax()
    sigma_y = QuantumToolbox.sigmay()
    sigma_z = QuantumToolbox.sigmaz()
    id = QuantumToolbox.qeye(2)
    op_on_ax = (op, ax) -> mapreduce(i -> i == ax ? op : id, QuantumToolbox.kron, 1:N)
    zz_on_ax =
        (ax1, ax2) ->
            mapreduce(i -> i == ax1 || i == ax2 ? sigma_z : id, QuantumToolbox.kron, 1:N)
    ψ0 = mapreduce(
        i ->
            cos(params.t[i] / 2) * ket_plus +
            exp(1im * params.p[i]) * sin(params.t[i] / 2) * ket_minus,
        QuantumToolbox.kron,
        1:N,
    )
    V = exp(
        -1im / 16 * sum(params.M[i, j] * zz_on_ax(i, j) for i ∈ 1:N for j ∈ 1:N if i != j),
    )
    ψ1 = V * ψ0
    rot_x = params.r .* cos.(params.d) .* cos.(params.g)
    rot_y = params.r .* cos.(params.d) .* sin.(params.g)
    rot_z = params.r .* sin.(params.d)
    U = exp(
        -1im * sum(
            op_on_ax(rot_x[i] * sigma_x + rot_y[i] * sigma_y + rot_z[i] * sigma_z, i)
            for i ∈ 1:N
        ),
    )
    ψ2 = U * ψ1
    numerical_configuration = map(i -> QuantumToolbox.expect(op_on_ax(sigma_z, i), ψ2), 1:N)
    for i ∈ 1:N
        numerical_configuration[i] = sign(dot(W[:, i], numerical_configuration))
    end
    numerical_configuration = map(numerical_configuration) do x
        return Int8(x == 0 ? 1 : x)
    end

    # Test the rounding function
    @test_throws ArgumentError GCS.round_configuration(
        problem_with_bias,
        params,
        SequentialRounding(),
        trig,
        temporal,
        thread_temp,
        x_multithread_params,
    )
    configuration = GCS.round_configuration(
        problem,
        params,
        SequentialRounding(),
        trig,
        temporal,
        thread_temp,
        x_multithread_params,
    )
    @test configuration isa Vector{Int8}
    @test all(x -> x == -1 || x == 1, configuration)
    @test all(configuration .== numerical_configuration)
end

@testitem "variational gcs solver > solve" begin
    using QuboSolver.Solvers.GCS
    using StableRNGs
    import Suppressor: @capture_err

    @test GCS_solver <: AbstractSolver
    @test SignRounding <: GCS.RoundingMethod
    @test SequentialRounding <: GCS.RoundingMethod

    rng = StableRNG(1234)
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = GCS_solver()

    # Test the solve function
    sol = solve!(problem, solver; rng = rng, progressbar = false)
    @test sol isa Solution{Float64,GCS_solver}
    @test sol.energy isa Float64
    @test sol.configuration isa Vector{Int8}
    @test sol.energy ≈ -sum(abs, W) - sum(bias)
    @test all(x -> x == -1 || x == 1, sol.configuration)
    @test sol.configuration == [1, -1, 1, -1, 1] .* (-1)^(idx - 1) # spin idx is going to be +1
    @test sol.solver === solver

    # Test MCS bounds
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    @test_throws ArgumentError solve!(problem, solver, iterations = 1, progressbar = false)

    # Test progress bar suppression
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
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
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W)#, bias)
    rng = StableRNG(1234)
    N_iterations = 100_000
    output = @capture_err begin
        sol = solve!(problem, solver; iterations = N_iterations, rng = rng, progressbar = true)
    end
    @test output != ""

    # Test multiple rounding methods
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
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
    @test all(x -> x isa Solution{Float64,GCS_solver}, solutions)

    # Test energy storage
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
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
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W)#, bias)
    rng = StableRNG(1234)
    sol = solve!(problem, solver; rng = rng, progressbar = false, save_params = true)
    @test sol.metadata[:params] isa GCS.ParamType{Float64}
    for f ∈ (:r, :d, :g, :p, :t)
        @test length(sol.metadata[:params][f]) == N
    end
    @test size(sol.metadata[:params].M) == (N, N)

    # Test runtime measurement
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    rng = StableRNG(1234)
    N_iterations = 100_000
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
