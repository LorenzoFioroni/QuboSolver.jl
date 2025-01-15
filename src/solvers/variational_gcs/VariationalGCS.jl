module VariationalGCS

using ..QuboSolver
using Random
using ProgressMeter
using Optimisers

export VariationalGCSSolver, solve!

struct VariationalGCSSolver <: AbstractSolver end

abstract type RoundingMethod end
struct SignRounding <: RoundingMethod end
struct SequentialRounding <: RoundingMethod end
struct QuantumRelaxAndRound <: RoundingMethod end

const ParamType{T} = @NamedTuple{t::Vector{T}, p::Vector{T}, M::Matrix{T}, r::Vector{T}, d::Vector{T}, g::Vector{T}}
const TrigType{T} = @NamedTuple{
    sin_t::Vector{T},
    cos_t::Vector{T},
    sin_p::Vector{T},
    cos_p::Vector{T},
    sin_d::Vector{T},
    cos_d::Vector{T},
    exp_g::Vector{Complex{T}},
    sin_r::Vector{T},
    cos_r::Vector{T},
    exp_M::Matrix{Complex{T}},
}
const TempType{T} = @NamedTuple{
    Sz::@NamedTuple{a1::Vector{Complex{T}}, a0::Vector{T}},
    dSz_r::@NamedTuple{a1::Vector{Complex{T}}, a0::Vector{T}},
    dSz_d::@NamedTuple{a1::Vector{Complex{T}}, a0::Vector{T}},
    dSz_g::@NamedTuple{a1::Vector{Complex{T}}},
    Sx::@NamedTuple{a1::Vector{Complex{T}}, a0::Vector{T}},
    dSx_r::@NamedTuple{a1::Vector{Complex{T}}, a0::Vector{T}},
    dSx_d::@NamedTuple{a1::Vector{Complex{T}}, a0::Vector{T}},
    dSx_g::@NamedTuple{a1::Vector{Complex{T}}, a0::Vector{T}},
    psi::Matrix{Complex{T}},
    dpsi_t::Matrix{Complex{T}},
    dpsi_p::Matrix{Complex{T}},
}
const ThreadTempType{T} = @NamedTuple{
    psi_ket::Matrix{Complex{T}},
    psi_ket_dM1::Matrix{Complex{T}},
    psi_ket_dM2::Matrix{Complex{T}},
    dpsi_t_ket::Matrix{Complex{T}},
    dpsi_t_bra::Matrix{Complex{T}},
    dpsi_p_ket::Matrix{Complex{T}},
    dpsi_p_bra::Matrix{Complex{T}},
    mat_el::Vector{Complex{T}},
    mat_el_psi_dpsi_t::Vector{Complex{T}},
    mat_el_dpsi_t_psi::Vector{Complex{T}},
    mat_el_psi_dpsi_p::Vector{Complex{T}},
    mat_el_dpsi_p_psi::Vector{Complex{T}},
    mat_el_dM1::Vector{Complex{T}},
    mat_el_dM2::Vector{Complex{T}},
}

function prod_and_count_zeros(x::Vector{T}) where {T}
    p = one(T)
    c = 0
    nnzidx = -1
    @inbounds @simd for i in eachindex(x)
        if iszero(x[i])
            c == 1 && return (zero(T), -2)
            c += 1
            nnzidx = i
        else
            p *= x[i]
        end
    end
    return (p, nnzidx)
end

function alloc_trig(z::ParamType{T}) where {T}
    return (
        sin_t = similar(z.t),
        cos_t = similar(z.t),
        sin_p = similar(z.p),
        cos_p = similar(z.p),
        sin_d = similar(z.d),
        cos_d = similar(z.d),
        exp_g = similar(z.g, Complex{T}),
        sin_r = similar(z.r),
        cos_r = similar(z.r),
        exp_M = similar(z.M, Complex{T}),
    )
end

function compute_trig!(trig::TrigType{T}, z::ParamType{T}) where {T}
    trig.sin_t .= sin.(z.t ./ 2)
    trig.cos_t .= cos.(z.t ./ 2)
    trig.sin_p .= sin.(z.p)
    trig.cos_p .= cos.(z.p)
    trig.sin_d .= sin.(z.d)
    trig.cos_d .= cos.(z.d)
    trig.exp_g .= exp.(1im .* z.g)
    trig.sin_r .= sin.(2 .* z.r)
    trig.cos_r .= cos.(2 .* z.r)
    trig.exp_M .= exp.(-0.25im .* z.M)
    return
end

function compute_trig(z::ParamType{T}) where {T}
    trig = alloc_trig(z)
    compute_trig!(trig, z)
    return trig
end

function alloc_temporal(z::ParamType{T}) where {T}
    N = length(z.t)
    return (
        Sz = (a1 = Vector{Complex{T}}(undef, N), a0 = Vector{T}(undef, N)),
        dSz_r = (a1 = Vector{Complex{T}}(undef, N), a0 = Vector{T}(undef, N)),
        dSz_d = (a1 = Vector{Complex{T}}(undef, N), a0 = Vector{T}(undef, N)),
        dSz_g = (a1 = Vector{Complex{T}}(undef, N),),
        Sx = (a1 = Vector{Complex{T}}(undef, N), a0 = Vector{T}(undef, N)),
        dSx_r = (a1 = Vector{Complex{T}}(undef, N), a0 = Vector{T}(undef, N)),
        dSx_d = (a1 = Vector{Complex{T}}(undef, N), a0 = Vector{T}(undef, N)),
        dSx_g = (a1 = Vector{Complex{T}}(undef, N), a0 = Vector{T}(undef, N)),
        psi = Matrix{Complex{T}}(undef, N, 2),
        dpsi_t = Matrix{Complex{T}}(undef, N, 2),
        dpsi_p = Matrix{Complex{T}}(undef, N, 2),
    )
end

function alloc_thread_temporal(z::ParamType{T}) where {T}
    N = length(z.t)
    return [
        (
            psi_ket = Matrix{Complex{T}}(undef, N, 2),
            psi_ket_dM1 = Matrix{Complex{T}}(undef, N, 2),
            psi_ket_dM2 = Matrix{Complex{T}}(undef, N, 2),
            dpsi_t_ket = Matrix{Complex{T}}(undef, N, 2),
            dpsi_t_bra = Matrix{Complex{T}}(undef, N, 2),
            dpsi_p_ket = Matrix{Complex{T}}(undef, N, 2),
            dpsi_p_bra = Matrix{Complex{T}}(undef, N, 2),
            mat_el = Vector{Complex{T}}(undef, N),
            mat_el_psi_dpsi_t = Vector{Complex{T}}(undef, N),
            mat_el_dpsi_t_psi = Vector{Complex{T}}(undef, N),
            mat_el_psi_dpsi_p = Vector{Complex{T}}(undef, N),
            mat_el_dpsi_p_psi = Vector{Complex{T}}(undef, N),
            mat_el_dM1 = Vector{Complex{T}}(undef, N),
            mat_el_dM2 = Vector{Complex{T}}(undef, N),
        ) for _ in 1:Threads.nthreads()
    ]
end

function compute_temporal!(temporal::TempType{T}, trig::TrigType{T}; grad = true) where {T}
    @inbounds begin
        @. temporal.Sz.a1 = trig.cos_d * trig.exp_g * (trig.sin_d * (1 - trig.cos_r) + trig.sin_r * im)
        @. temporal.Sz.a0 = trig.sin_d .^ 2 * (1 - trig.cos_r) + trig.cos_r

        @. temporal.Sx.a1 =
            trig.cos_d .^ 2 * (1 - trig.cos_r) * real.(trig.exp_g) * trig.exp_g + trig.cos_r -
            trig.sin_r * trig.sin_d * im
        @. temporal.Sx.a0 =
            trig.cos_d * (trig.sin_d * real.(trig.exp_g) * (1 - trig.cos_r) + trig.sin_r * imag(trig.exp_g))

        @. temporal.psi[:, 1] = ((trig.cos_t) + (trig.sin_t * (trig.cos_p + trig.sin_p * im))) / sqrt(2)
        @. temporal.psi[:, 2] = ((trig.cos_t) - (trig.sin_t * (trig.cos_p + trig.sin_p * im))) / sqrt(2)

        !grad && return

        @. temporal.dSz_r.a1 = 2 * trig.cos_d * trig.exp_g * (trig.sin_r * trig.sin_d + trig.cos_r * im)
        @. temporal.dSz_r.a0 = 2 * trig.sin_r * (trig.sin_d .^ 2 - 1)

        @. temporal.dSz_d.a1 =
            (trig.cos_d .^ 2 - trig.sin_d .^ 2) * trig.exp_g * (1 - trig.cos_r) -
            trig.sin_r * trig.sin_d * trig.exp_g * im
        @. temporal.dSz_d.a0 = 2 * trig.sin_d * trig.cos_d * (1 - trig.cos_r)

        @. temporal.dSz_g.a1 = trig.cos_d * (trig.sin_d * (1 - trig.cos_r) * trig.exp_g * im - trig.sin_r * trig.exp_g)

        @. temporal.dSx_r.a1 =
            2 * trig.sin_r * trig.cos_d .^ 2 * real.(trig.exp_g) * trig.exp_g - 2 * trig.sin_r -
            2 * trig.cos_r * trig.sin_d * im
        @. temporal.dSx_r.a0 =
            2 * trig.cos_d * (trig.sin_r * trig.sin_d * real.(trig.exp_g) + trig.cos_r * imag.(trig.exp_g))

        @. temporal.dSx_d.a1 =
            -2 * trig.sin_d * trig.cos_d * (1 - trig.cos_r) * real.(trig.exp_g) * trig.exp_g -
            trig.sin_r * trig.cos_d * im
        @. temporal.dSx_d.a0 =
            (trig.cos_d .^ 2 - trig.sin_d .^ 2) * real.(trig.exp_g) * (1 - trig.cos_r) -
            trig.sin_r * trig.sin_d * imag.(trig.exp_g)

        @. temporal.dSx_g.a1 = trig.exp_g .^ 2 * trig.cos_d .^ 2 * (1 - trig.cos_r) * im
        @. temporal.dSx_g.a0 =
            -trig.sin_d * trig.cos_d * (1 - trig.cos_r) * imag.(trig.exp_g) +
            trig.sin_r * trig.cos_d * real.(trig.exp_g)

        @. temporal.dpsi_t[:, 1] = ((-0.5 * trig.sin_t) + (0.5 * trig.cos_t * (trig.cos_p + trig.sin_p * im))) / sqrt(2)
        @. temporal.dpsi_t[:, 2] = ((-0.5 * trig.sin_t) - (0.5 * trig.cos_t * (trig.cos_p + trig.sin_p * im))) / sqrt(2)
        @. temporal.dpsi_p[:, 1] = (trig.sin_t * (-trig.sin_p + trig.cos_p * im)) / sqrt(2)
        @. temporal.dpsi_p[:, 2] = -(trig.sin_t * (-trig.sin_p + trig.cos_p * im)) / sqrt(2)
    end
    return nothing
end

function _sum_aligned_namedtuples!(a::ParamType{T}, b::ParamType{T}, b_coeff::T) where {T}
    a.t .+= b_coeff .* b.t
    a.p .+= b_coeff .* b.p
    a.M .+= b_coeff .* b.M
    a.r .+= b_coeff .* b.r
    a.d .+= b_coeff .* b.d
    a.g .+= b_coeff .* b.g
    return nothing
end

function _sum_aligned_namedtuples!(a::ParamType{T}, b::ParamType{T}, b_coeff::T, c::ParamType{T}, c_coeff::T) where {T}
    a.t .= b_coeff .* b.t .+ c_coeff .* c.t
    a.p .= b_coeff .* b.p .+ c_coeff .* c.p
    a.M .= b_coeff .* b.M .+ c_coeff .* c.M
    a.r .= b_coeff .* b.r .+ c_coeff .* c.r
    a.d .= b_coeff .* b.d .+ c_coeff .* c.d
    a.g .= b_coeff .* b.g .+ c_coeff .* c.g
    return nothing
end

function _symmetrize_M!(M::AbstractMatrix{T}) where {T}
    @inbounds for i in axes(M, 1)
        @simd for j in (i+1):size(M, 2)
            tmp = M[i, j]
            M[i, j] += M[j, i]
            M[j, i] += tmp
        end
    end
    return
end

function _setup_initial_conf(::Nothing, N, rng, T)
    return (
        t = T(1e-3) .* randn(rng, T, N),
        p = T(2π) .* rand(rng, T, N),
        M = zeros(T, N, N),
        r = zeros(T, N),
        d = zeros(T, N),
        g = zeros(T, N),
    )
end

_setup_initial_conf(initial_conf::ParamType{T}, _, _, _) where {T} = deepcopy(initial_conf)

ease(x::AbstractFloat) = acos(1 - 2x) / π
a(t::AbstractFloat) = 1 - ease(t)
b(t::AbstractFloat) = ease(t)

function loss_x(
    x_loss::Vector{T},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) in x_multithread_params
        for j in chunk
            @inbounds begin
                # alpha = 1
                thread_temporal[tid].psi_ket .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                Pj = prod(thread_temporal[tid].mat_el)
                x_loss[j] = 2real(temporal.Sx.a1[j] * Pj)

                # alpha = 3
                thread_temporal[tid].psi_ket .= temporal.psi

                thread_temporal[tid].psi_ket[j, 2] *= -1.0

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                Pj = prod(thread_temporal[tid].mat_el)
                x_loss[j] += real(temporal.Sx.a0[j] * Pj)
            end
        end
    end
end

function loss_z(
    z_loss::Vector{T},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) in x_multithread_params
        for j in chunk
            @inbounds begin
                # alpha = 1
                thread_temporal[tid].psi_ket .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                Pj = prod(thread_temporal[tid].mat_el)
                z_loss[j] = 2real(temporal.Sz.a1[j] * Pj)

                # alpha = 3
                thread_temporal[tid].psi_ket .= temporal.psi

                thread_temporal[tid].psi_ket[j, 2] *= -1.0

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                Pj = prod(thread_temporal[tid].mat_el)
                z_loss[j] += real(temporal.Sz.a0[j] * Pj)
            end
        end
    end
end

function loss_zz(
    zz_loss::Vector{T},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    zz_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) in zz_multithread_params
        for (lin_ind, ((j, k), w)) in chunk
            @inbounds begin
                # alpha = 1, beta = 1
                thread_temporal[tid].psi_ket .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket[k, 2] = thread_temporal[tid].psi_ket[k, 1]
                thread_temporal[tid].psi_ket[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                P_jk = prod(thread_temporal[tid].mat_el)
                zz_loss[lin_ind] = 2real(temporal.Sz.a1[j] * temporal.Sz.a1[k] * P_jk)

                # alpha = 1, beta = 2
                thread_temporal[tid].psi_ket .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].psi_ket[:, 2] .*= trig.exp_M[:, k]
                thread_temporal[tid].psi_ket[k, 1] = thread_temporal[tid].psi_ket[k, 2]
                thread_temporal[tid].psi_ket[k, 2] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                P_jk = prod(thread_temporal[tid].mat_el)
                zz_loss[lin_ind] += 2real(temporal.Sz.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)

                # alpha = 1, beta = 3
                thread_temporal[tid].psi_ket .= temporal.psi

                thread_temporal[tid].psi_ket[k, 2] *= -1.0
                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                P_jk = prod(thread_temporal[tid].mat_el)
                zz_loss[lin_ind] += 2real(temporal.Sz.a1[j] * temporal.Sz.a0[k] * P_jk)

                # alpha = 3, beta = 1
                thread_temporal[tid].psi_ket .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket[k, 2] = thread_temporal[tid].psi_ket[k, 1]
                thread_temporal[tid].psi_ket[k, 1] = 0.0 + 0.0 * im
                thread_temporal[tid].psi_ket[j, 2] *= -1.0

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                P_jk = prod(thread_temporal[tid].mat_el)
                zz_loss[lin_ind] += 2real(temporal.Sz.a0[j] * temporal.Sz.a1[k] * P_jk)

                # alpha = 3, beta = 3
                thread_temporal[tid].psi_ket .= temporal.psi

                thread_temporal[tid].psi_ket[k, 2] *= -1.0
                thread_temporal[tid].psi_ket[j, 2] *= -1.0

                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)

                P_jk = prod(thread_temporal[tid].mat_el)
                zz_loss[lin_ind] += real(temporal.Sz.a0[j] * temporal.Sz.a0[k] * P_jk)
            end
        end
    end

    return
end

function loss(
    z::ParamType{T},
    time::AbstractFloat,
    tf::AbstractFloat,
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    zz_multithread_params,
    x_multithread_params,
    compact_W,
    zz_loss::Vector{T},
    x_loss::Vector{T},
) where {T}
    compute_trig!(trig, z)
    compute_temporal!(temporal, trig)

    zz_loss .= zero(T)
    x_loss .= zero(T)

    loss_zz(zz_loss, trig, temporal, thread_temporal, zz_multithread_params)
    loss_x(x_loss, trig, temporal, thread_temporal, x_multithread_params)

    zz_loss .*= 2 .* compact_W

    return -tf * a(time) * sum(x_loss) - b(time) * sum(zz_loss)
end

function loss_x_and_grad!(
    x_loss::Vector{T},
    dz_x::Vector{ParamType{T}},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) in x_multithread_params
        for j in chunk
            @inbounds begin
                # alpha = 1
                thread_temporal[tid].psi_ket .= temporal.psi
                thread_temporal[tid].dpsi_t_ket .= temporal.dpsi_t
                thread_temporal[tid].dpsi_t_bra .= conj.(temporal.dpsi_t)
                thread_temporal[tid].dpsi_p_ket .= temporal.dpsi_p
                thread_temporal[tid].dpsi_p_bra .= conj.(temporal.dpsi_p)
                thread_temporal[tid].psi_ket_dM1 .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_t_ket[j, 2] = thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] = thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*= -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*= 0.25im .* conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM1[j, 2] = thread_temporal[tid].psi_ket_dM1[j, 1]
                thread_temporal[tid].psi_ket_dM1[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_t, thread_temporal[tid].dpsi_t_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_t_psi, thread_temporal[tid].dpsi_t_bra)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_p, thread_temporal[tid].dpsi_p_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_p_psi, thread_temporal[tid].dpsi_p_bra)
                sum!(thread_temporal[tid].mat_el_dM1, thread_temporal[tid].psi_ket_dM1)

                Pj, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    x_loss[j] += 2real(temporal.Sx.a1[j] * Pj)
                    dz_x[tid].r[j] += 2real(temporal.dSx_r.a1[j] * Pj)
                    dz_x[tid].d[j] += 2real(temporal.dSx_d.a1[j] * Pj)
                    dz_x[tid].g[j] += 2real(temporal.dSx_g.a1[j] * Pj)
                    @views @. dz_x[tid].t +=
                        2real(
                            temporal.Sx.a1[j] *
                            Pj *
                            (thread_temporal[tid].mat_el_dpsi_t_psi + thread_temporal[tid].mat_el_psi_dpsi_t) /
                            thread_temporal[tid].mat_el,
                        )

                    @views @. dz_x[tid].p +=
                        2real(
                            temporal.Sx.a1[j] *
                            Pj *
                            (thread_temporal[tid].mat_el_dpsi_p_psi + thread_temporal[tid].mat_el_psi_dpsi_p) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_x[tid].M[:, j] +=
                        2real(temporal.Sx.a1[j] * Pj * thread_temporal[tid].mat_el_dM1 / thread_temporal[tid].mat_el)
                elseif nnz_idx != -2
                    dz_x[tid].t[nnz_idx] +=
                        2real(
                            temporal.Sx.a1[j] *
                            Pj *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                            ),
                        )

                    dz_x[tid].p[nnz_idx] +=
                        2real(
                            temporal.Sx.a1[j] *
                            Pj *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_p[nnz_idx]
                            ),
                        )
                    dz_x[tid].M[nnz_idx, j] += 2real(temporal.Sx.a1[j] * Pj * thread_temporal[tid].mat_el_dM1[nnz_idx])
                end

                # alpha = 3
                thread_temporal[tid].psi_ket .= temporal.psi
                thread_temporal[tid].dpsi_t_ket .= temporal.dpsi_t
                thread_temporal[tid].dpsi_t_bra .= conj.(temporal.dpsi_t)
                thread_temporal[tid].dpsi_p_ket .= temporal.dpsi_p
                thread_temporal[tid].dpsi_p_bra .= conj.(temporal.dpsi_p)

                thread_temporal[tid].psi_ket[j, 2] *= -1.0

                thread_temporal[tid].dpsi_t_ket[j, 2] *= -1.0

                thread_temporal[tid].dpsi_p_ket[j, 2] *= -1.0

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_t, thread_temporal[tid].dpsi_t_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_t_psi, thread_temporal[tid].dpsi_t_bra)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_p, thread_temporal[tid].dpsi_p_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_p_psi, thread_temporal[tid].dpsi_p_bra)

                Pj, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    x_loss[j] += real(temporal.Sx.a0[j] * Pj)
                    dz_x[tid].r[j] += real(temporal.dSx_r.a0[j] * Pj)
                    dz_x[tid].d[j] += real(temporal.dSx_d.a0[j] * Pj)
                    dz_x[tid].g[j] += real(temporal.dSx_g.a0[j] * Pj)
                    @views @. dz_x[tid].t += real(
                        temporal.Sx.a0[j] *
                        Pj *
                        (thread_temporal[tid].mat_el_dpsi_t_psi + thread_temporal[tid].mat_el_psi_dpsi_t) /
                        thread_temporal[tid].mat_el,
                    )
                    @views @. dz_x[tid].p += real(
                        temporal.Sx.a0[j] *
                        Pj *
                        (thread_temporal[tid].mat_el_dpsi_p_psi + thread_temporal[tid].mat_el_psi_dpsi_p) /
                        thread_temporal[tid].mat_el,
                    )
                elseif nnz_idx != -2
                    dz_x[tid].t[nnz_idx] += real(
                        temporal.Sx.a0[j] *
                        Pj *
                        (
                            thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                            thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                        ),
                    )
                    dz_x[tid].p[nnz_idx] += real(
                        temporal.Sx.a0[j] *
                        Pj *
                        (
                            thread_temporal[tid].mat_el_dpsi_p_psi[nnz_idx] +
                            thread_temporal[tid].mat_el_psi_dpsi_p[nnz_idx]
                        ),
                    )
                end
            end
        end
    end

    return
end

function loss_zz_and_grad!(
    zz_loss::Vector{T},
    dz_zz::Vector{ParamType{T}},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    zz_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) in zz_multithread_params
        for (lin_ind, ((j, k), w)) in chunk
            @inbounds begin
                dz_M_view_j = view(dz_zz[tid].M, :, j)
                dz_M_view_k = view(dz_zz[tid].M, :, k)

                # alpha = 1, beta = 1
                thread_temporal[tid].psi_ket .= temporal.psi
                thread_temporal[tid].dpsi_t_ket .= temporal.dpsi_t
                thread_temporal[tid].dpsi_t_bra .= conj.(temporal.dpsi_t)
                thread_temporal[tid].dpsi_p_ket .= temporal.dpsi_p
                thread_temporal[tid].dpsi_p_bra .= conj.(temporal.dpsi_p)
                thread_temporal[tid].psi_ket_dM1 .= temporal.psi
                thread_temporal[tid].psi_ket_dM2 .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket[k, 2] = thread_temporal[tid].psi_ket[k, 1]
                thread_temporal[tid].psi_ket[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].dpsi_t_ket[k, 2] = thread_temporal[tid].dpsi_t_ket[k, 1]
                thread_temporal[tid].dpsi_t_ket[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_t_ket[j, 2] = thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].dpsi_p_ket[k, 2] = thread_temporal[tid].dpsi_p_ket[k, 1]
                thread_temporal[tid].dpsi_p_ket[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] = thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket_dM1[k, 2] = thread_temporal[tid].psi_ket_dM1[k, 1]
                thread_temporal[tid].psi_ket_dM1[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*= -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*= 0.25im .* conj.(trig.exp_M[:, j])

                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*= -0.25im .* trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*= 0.25im .* conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM2[j, 2] = thread_temporal[tid].psi_ket_dM2[j, 1]
                thread_temporal[tid].psi_ket_dM2[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM2 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_t, thread_temporal[tid].dpsi_t_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_t_psi, thread_temporal[tid].dpsi_t_bra)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_p, thread_temporal[tid].dpsi_p_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_p_psi, thread_temporal[tid].dpsi_p_bra)
                sum!(thread_temporal[tid].mat_el_dM1, thread_temporal[tid].psi_ket_dM1)
                sum!(thread_temporal[tid].mat_el_dM2, thread_temporal[tid].psi_ket_dM2)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += 2real(temporal.Sz.a1[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].r[j] += 2w * 2real(temporal.dSz_r.a1[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].r[k] += 2w * 2real(temporal.Sz.a1[j] * temporal.dSz_r.a1[k] * P_jk)
                    dz_zz[tid].d[j] += 2w * 2real(temporal.dSz_d.a1[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].d[k] += 2w * 2real(temporal.Sz.a1[j] * temporal.dSz_d.a1[k] * P_jk)
                    dz_zz[tid].g[j] += 2w * 2real(temporal.dSz_g.a1[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].g[k] += 2w * 2real(temporal.Sz.a1[j] * temporal.dSz_g.a1[k] * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_t_psi + thread_temporal[tid].mat_el_psi_dpsi_t) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_p_psi + thread_temporal[tid].mat_el_psi_dpsi_p) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_j +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] * temporal.Sz.a1[k] * P_jk * thread_temporal[tid].mat_el_dM1 /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_k +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] * temporal.Sz.a1[k] * P_jk * thread_temporal[tid].mat_el_dM2 /
                            thread_temporal[tid].mat_el,
                        )
                elseif nnz_idx != -2
                    dz_zz[tid].t[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                            ),
                        )
                    dz_zz[tid].p[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_p[nnz_idx]
                            ),
                        )
                    dz_M_view_j[nnz_idx] +=
                        2w *
                        2real(temporal.Sz.a1[j] * temporal.Sz.a1[k] * P_jk * thread_temporal[tid].mat_el_dM1[nnz_idx])
                    dz_M_view_k[nnz_idx] +=
                        2w *
                        2real(temporal.Sz.a1[j] * temporal.Sz.a1[k] * P_jk * thread_temporal[tid].mat_el_dM2[nnz_idx])
                end

                # alpha = 1, beta = 2
                thread_temporal[tid].psi_ket .= temporal.psi
                thread_temporal[tid].dpsi_t_ket .= temporal.dpsi_t
                thread_temporal[tid].dpsi_t_bra .= conj.(temporal.dpsi_t)
                thread_temporal[tid].dpsi_p_ket .= temporal.dpsi_p
                thread_temporal[tid].dpsi_p_bra .= conj.(temporal.dpsi_p)
                thread_temporal[tid].psi_ket_dM1 .= temporal.psi
                thread_temporal[tid].psi_ket_dM2 .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].psi_ket[:, 2] .*= trig.exp_M[:, k]
                thread_temporal[tid].psi_ket[k, 1] = thread_temporal[tid].psi_ket[k, 2]
                thread_temporal[tid].psi_ket[k, 2] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= trig.exp_M[:, k]
                thread_temporal[tid].dpsi_t_ket[k, 1] = thread_temporal[tid].dpsi_t_ket[k, 2]
                thread_temporal[tid].dpsi_t_ket[k, 2] = 0.0 + 0.0 * im
                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_t_ket[j, 2] = thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= trig.exp_M[:, k]
                thread_temporal[tid].dpsi_p_ket[k, 1] = thread_temporal[tid].dpsi_p_ket[k, 2]
                thread_temporal[tid].dpsi_p_ket[k, 2] = 0.0 + 0.0 * im
                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] = thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*= conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*= trig.exp_M[:, k]
                thread_temporal[tid].psi_ket_dM1[k, 1] = thread_temporal[tid].psi_ket_dM1[k, 2]
                thread_temporal[tid].psi_ket_dM1[k, 2] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*= -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*= 0.25im .* conj.(trig.exp_M[:, j])

                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*= 0.25im .* conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*= -0.25im .* trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM2[j, 2] = thread_temporal[tid].psi_ket_dM2[j, 1]
                thread_temporal[tid].psi_ket_dM2[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM2 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_t, thread_temporal[tid].dpsi_t_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_t_psi, thread_temporal[tid].dpsi_t_bra)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_p, thread_temporal[tid].dpsi_p_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_p_psi, thread_temporal[tid].dpsi_p_bra)
                sum!(thread_temporal[tid].mat_el_dM1, thread_temporal[tid].psi_ket_dM1)
                sum!(thread_temporal[tid].mat_el_dM2, thread_temporal[tid].psi_ket_dM2)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += 2real(temporal.Sz.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)
                    dz_zz[tid].r[j] += 2w * 2real(temporal.dSz_r.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)
                    dz_zz[tid].r[k] += 2w * 2real(temporal.Sz.a1[j] * conj(temporal.dSz_r.a1[k]) * P_jk)
                    dz_zz[tid].d[j] += 2w * 2real(temporal.dSz_d.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)
                    dz_zz[tid].d[k] += 2w * 2real(temporal.Sz.a1[j] * conj(temporal.dSz_d.a1[k]) * P_jk)
                    dz_zz[tid].g[j] += 2w * 2real(temporal.dSz_g.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)
                    dz_zz[tid].g[k] += 2w * 2real(temporal.Sz.a1[j] * conj(temporal.dSz_g.a1[k]) * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_t_psi + thread_temporal[tid].mat_el_psi_dpsi_t) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_p_psi + thread_temporal[tid].mat_el_psi_dpsi_p) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_j +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] * conj(temporal.Sz.a1[k]) * P_jk * thread_temporal[tid].mat_el_dM1 /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_k +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] * conj(temporal.Sz.a1[k]) * P_jk * thread_temporal[tid].mat_el_dM2 /
                            thread_temporal[tid].mat_el,
                        )
                elseif nnz_idx != -2
                    dz_zz[tid].t[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                            ),
                        )
                    dz_zz[tid].p[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_p[nnz_idx]
                            ),
                        )
                    dz_M_view_j[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            thread_temporal[tid].mat_el_dM1[nnz_idx],
                        )
                    dz_M_view_k[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            thread_temporal[tid].mat_el_dM2[nnz_idx],
                        )
                end

                # alpha = 1, beta = 3
                thread_temporal[tid].psi_ket .= temporal.psi
                thread_temporal[tid].dpsi_t_ket .= temporal.dpsi_t
                thread_temporal[tid].dpsi_t_bra .= conj.(temporal.dpsi_t)
                thread_temporal[tid].dpsi_p_ket .= temporal.dpsi_p
                thread_temporal[tid].dpsi_p_bra .= conj.(temporal.dpsi_p)
                thread_temporal[tid].psi_ket_dM1 .= temporal.psi
                thread_temporal[tid].psi_ket_dM2 .= temporal.psi

                thread_temporal[tid].psi_ket[k, 2] *= -1.0
                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket[j, 2] = thread_temporal[tid].psi_ket[j, 1]
                thread_temporal[tid].psi_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_t_ket[k, 2] *= -1.0
                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_t_ket[j, 2] = thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_p_ket[k, 2] *= -1.0
                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] = thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].psi_ket_dM1[k, 2] *= -1.0
                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*= -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*= 0.25im .* conj.(trig.exp_M[:, j])

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_t, thread_temporal[tid].dpsi_t_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_t_psi, thread_temporal[tid].dpsi_t_bra)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_p, thread_temporal[tid].dpsi_p_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_p_psi, thread_temporal[tid].dpsi_p_bra)
                sum!(thread_temporal[tid].mat_el_dM1, thread_temporal[tid].psi_ket_dM1)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += 2real(temporal.Sz.a1[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].r[j] += 2w * 2real(temporal.dSz_r.a1[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].r[k] += 2w * 2real(temporal.Sz.a1[j] * temporal.dSz_r.a0[k] * P_jk)
                    dz_zz[tid].d[j] += 2w * 2real(temporal.dSz_d.a1[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].d[k] += 2w * 2real(temporal.Sz.a1[j] * temporal.dSz_d.a0[k] * P_jk)
                    dz_zz[tid].g[j] += 2w * 2real(temporal.dSz_g.a1[j] * temporal.Sz.a0[k] * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_t_psi + thread_temporal[tid].mat_el_psi_dpsi_t) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_p_psi + thread_temporal[tid].mat_el_psi_dpsi_p) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_j +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] * temporal.Sz.a0[k] * P_jk * thread_temporal[tid].mat_el_dM1 /
                            thread_temporal[tid].mat_el,
                        )
                elseif nnz_idx != -2
                    dz_zz[tid].t[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                            ),
                        )
                    dz_zz[tid].p[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_p[nnz_idx]
                            ),
                        )
                    dz_M_view_j[nnz_idx] +=
                        2w *
                        2real(temporal.Sz.a1[j] * temporal.Sz.a0[k] * P_jk * thread_temporal[tid].mat_el_dM1[nnz_idx])
                end

                # alpha = 3, beta = 1
                thread_temporal[tid].psi_ket .= temporal.psi
                thread_temporal[tid].dpsi_t_ket .= temporal.dpsi_t
                thread_temporal[tid].dpsi_t_bra .= conj.(temporal.dpsi_t)
                thread_temporal[tid].dpsi_p_ket .= temporal.dpsi_p
                thread_temporal[tid].dpsi_p_bra .= conj.(temporal.dpsi_p)
                thread_temporal[tid].psi_ket_dM1 .= temporal.psi
                thread_temporal[tid].psi_ket_dM2 .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket[k, 2] = thread_temporal[tid].psi_ket[k, 1]
                thread_temporal[tid].psi_ket[k, 1] = 0.0 + 0.0 * im
                thread_temporal[tid].psi_ket[j, 2] *= -1.0

                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].dpsi_t_ket[k, 2] = thread_temporal[tid].dpsi_t_ket[k, 1]
                thread_temporal[tid].dpsi_t_ket[k, 1] = 0.0 + 0.0 * im
                thread_temporal[tid].dpsi_t_ket[j, 2] *= -1.0

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].dpsi_p_ket[k, 2] = thread_temporal[tid].dpsi_p_ket[k, 1]
                thread_temporal[tid].dpsi_p_ket[k, 1] = 0.0 + 0.0 * im
                thread_temporal[tid].dpsi_p_ket[j, 2] *= -1.0

                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*= -0.25im .* trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*= 0.25im .* conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket_dM2[j, 2] *= -1.0

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM2 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_t, thread_temporal[tid].dpsi_t_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_t_psi, thread_temporal[tid].dpsi_t_bra)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_p, thread_temporal[tid].dpsi_p_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_p_psi, thread_temporal[tid].dpsi_p_bra)
                sum!(thread_temporal[tid].mat_el_dM2, thread_temporal[tid].psi_ket_dM2)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += 2real(temporal.Sz.a0[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].r[j] += 2w * 2real(temporal.dSz_r.a0[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].r[k] += 2w * 2real(temporal.Sz.a0[j] * temporal.dSz_r.a1[k] * P_jk)
                    dz_zz[tid].d[j] += 2w * 2real(temporal.dSz_d.a0[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].d[k] += 2w * 2real(temporal.Sz.a0[j] * temporal.dSz_d.a1[k] * P_jk)
                    dz_zz[tid].g[k] += 2w * 2real(temporal.Sz.a0[j] * temporal.dSz_g.a1[k] * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w *
                        2real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_t_psi + thread_temporal[tid].mat_el_psi_dpsi_t) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w *
                        2real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_p_psi + thread_temporal[tid].mat_el_psi_dpsi_p) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_k +=
                        2w *
                        2real(
                            temporal.Sz.a0[j] * temporal.Sz.a1[k] * P_jk * thread_temporal[tid].mat_el_dM2 /
                            thread_temporal[tid].mat_el,
                        )
                elseif nnz_idx != -2
                    dz_zz[tid].t[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                            ),
                        )
                    dz_zz[tid].p[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_p[nnz_idx]
                            ),
                        )
                    dz_M_view_k[nnz_idx] +=
                        2w *
                        2real(temporal.Sz.a0[j] * temporal.Sz.a1[k] * P_jk * thread_temporal[tid].mat_el_dM2[nnz_idx])
                end

                # alpha = 3, beta = 3
                thread_temporal[tid].psi_ket .= temporal.psi
                thread_temporal[tid].dpsi_t_ket .= temporal.dpsi_t
                thread_temporal[tid].dpsi_t_bra .= conj.(temporal.dpsi_t)
                thread_temporal[tid].dpsi_p_ket .= temporal.dpsi_p
                thread_temporal[tid].dpsi_p_bra .= conj.(temporal.dpsi_p)

                thread_temporal[tid].psi_ket[k, 2] *= -1.0
                thread_temporal[tid].psi_ket[j, 2] *= -1.0

                thread_temporal[tid].dpsi_t_ket[k, 2] *= -1.0
                thread_temporal[tid].dpsi_t_ket[j, 2] *= -1.0

                thread_temporal[tid].dpsi_p_ket[k, 2] *= -1.0
                thread_temporal[tid].dpsi_p_ket[j, 2] *= -1.0

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_t, thread_temporal[tid].dpsi_t_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_t_psi, thread_temporal[tid].dpsi_t_bra)
                sum!(thread_temporal[tid].mat_el_psi_dpsi_p, thread_temporal[tid].dpsi_p_ket)
                sum!(thread_temporal[tid].mat_el_dpsi_p_psi, thread_temporal[tid].dpsi_p_bra)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += real(temporal.Sz.a0[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].r[j] += 2w * real(temporal.dSz_r.a0[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].r[k] += 2w * real(temporal.Sz.a0[j] * temporal.dSz_r.a0[k] * P_jk)
                    dz_zz[tid].d[j] += 2w * real(temporal.dSz_d.a0[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].d[k] += 2w * real(temporal.Sz.a0[j] * temporal.dSz_d.a0[k] * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w * real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_t_psi + thread_temporal[tid].mat_el_psi_dpsi_t) /
                            thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w * real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (thread_temporal[tid].mat_el_dpsi_p_psi + thread_temporal[tid].mat_el_psi_dpsi_p) /
                            thread_temporal[tid].mat_el,
                        )
                elseif nnz_idx != -2
                    dz_zz[tid].t[nnz_idx] +=
                        2w * real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                            ),
                        )
                    dz_zz[tid].p[nnz_idx] +=
                        2w * real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_p[nnz_idx]
                            ),
                        )
                end
            end
        end
    end

    return
end

function loss_and_grad!(
    l::Vector{T},
    dz::ParamType{T},
    z::ParamType{T},
    time::AbstractFloat,
    tf::AbstractFloat,
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    zz_multithread_params,
    x_multithread_params,
    compact_W::Vector{T},
    zz_loss::Vector{T},
    x_loss::Vector{T},
    dz_zz::Vector{ParamType{T}},
    dz_x::Vector{ParamType{T}},
) where {T}
    compute_trig!(trig, z)
    compute_temporal!(temporal, trig)

    @inbounds for i in 1:Threads.nthreads()
        for k in keys(dz_zz[i])
            dz_zz[i][k] .= zero(T)
            dz_x[i][k] .= zero(T)
        end
    end

    zz_loss .= zero(T)
    x_loss .= zero(T)

    loss_zz_and_grad!(zz_loss, dz_zz, trig, temporal, thread_temporal, zz_multithread_params)
    loss_x_and_grad!(x_loss, dz_x, trig, temporal, thread_temporal, x_multithread_params)

    zz_loss .*= 2 .* compact_W
    l .= -tf * a(time) * sum(x_loss) - b(time) * sum(zz_loss)

    @inbounds for ii in 2:Threads.nthreads()
        _sum_aligned_namedtuples!(dz_zz[1], dz_zz[ii], one(T))
        _sum_aligned_namedtuples!(dz_x[1], dz_x[ii], one(T))
    end

    _sum_aligned_namedtuples!(dz, dz_x[1], -tf * a(time), dz_zz[1], -b(time))

    _symmetrize_M!(dz.M)
    dz.M[diagind(dz.M)] .= zero(T)

    return
end

function round_configuration(
    ::QuboProblem,
    z::ParamType{T},
    ::SignRounding,
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    z_loss = zero(z.t)
    mean_z = similar(z.t)

    compute_trig!(trig, z)
    compute_temporal!(temporal, trig)
    loss_z(z_loss, trig, temporal, thread_temporal, x_multithread_params)

    sum!(mean_z, z_loss)

    conf = Int8.(sign.(mean_z))
    conf[conf.==0] .= 1

    return conf
end

function round_configuration(
    problem::QuboProblem,
    z::ParamType{T},
    ::SequentialRounding,
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    z_loss = zero(z.t)
    mean_z = similar(z.t)

    compute_trig!(trig, z)
    compute_temporal!(temporal, trig)
    loss_z(z_loss, trig, temporal, thread_temporal, x_multithread_params)

    sum!(mean_z, z_loss)

    for i in 1:problem.N
        mean_z[i] = sign(dot(problem.W[:, i], mean_z))
    end
    conf = Int8.(mean_z)

    conf[conf.==0] .= 1

    return conf
end

function QuboSolver.solve!(
    problem::QuboProblem{T,TW,Tc},
    solver::VariationalGCSSolver;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    initial_conf::Union{ParamType{T},Nothing} = nothing,
    iterations::Int = 1000,
    inner_iterations::Int = 1,
    tf::T = one(T),
    rounding::Union{RoundingMethod,Tuple{Vararg{RoundingMethod}}} = SignRounding(),
    save_params::Bool = false,
    save_energy::Bool = false,
    opt::Optimisers.AbstractRule = Adam(0.05),
    progressbar::Bool = true,
) where {T<:AbstractFloat,TW<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
    problem.has_bias && throw(ArgumentError("Bias term not supported yet"))

    iterations < 2 && throw(ArgumentError("Number of iterations must be at least 2"))

    z = _setup_initial_conf(initial_conf, problem.N, rng, T)

    compact_idx, compact_W = nonzero_triu(problem.W)
    _params = enumerate(zip(compact_idx, compact_W))
    zz_multithread_params = collect(enumerate(Iterators.partition(_params, cld(length(_params), Threads.nthreads()))))
    x_multithread_params = collect(enumerate(Iterators.partition(1:problem.N, cld(problem.N, Threads.nthreads()))))

    trig = alloc_trig(z)
    temporal = alloc_temporal(z)
    thread_temporal = alloc_thread_temporal(z)
    zz_loss = Vector{T}(undef, length(compact_W))
    x_loss = Vector{T}(undef, problem.N)
    dz_zz = [similar_named_tuple(z) for _ in 1:Threads.nthreads()]
    dz_x = [similar_named_tuple(z) for _ in 1:Threads.nthreads()]

    l = Vector{T}(undef, 1)
    dz = similar_named_tuple(z)

    state_tree = Optimisers.setup(opt, z)

    save_energy && (energy = Vector{T}(undef, iterations))

    initial_time = time()

    @showprogress showspeed = true enabled = progressbar for it in 1:iterations
        time = (it - 1) / (iterations - 1)
        for _ in 1:inner_iterations
            loss_and_grad!(
                l,
                dz,
                z,
                time,
                tf,
                trig,
                temporal,
                thread_temporal,
                zz_multithread_params,
                x_multithread_params,
                compact_W,
                zz_loss,
                x_loss,
                dz_zz,
                dz_x,
            )
            state_tree, z = Optimisers.update(state_tree, z, dz)
        end

        save_energy && (energy[it] = l[1])
    end

    delta_time = time() - initial_time

    metadata = (runtime = delta_time,)
    save_params && (metadata = merge(metadata, (params = z,)))
    save_energy && (metadata = merge(metadata, (energy = energy,)))

    if isa(rounding, RoundingMethod)
        return add_solution(
            problem,
            round_configuration(problem, z, rounding, trig, temporal, thread_temporal, x_multithread_params),
            solver;
            metadata...,
        )
    else
        return map(
            round_method -> add_solution(
                problem,
                round_configuration(problem, z, round_method, trig, temporal, thread_temporal, x_multithread_params),
                solver;
                metadata...,
            ),
            rounding,
        )
    end
end

end
