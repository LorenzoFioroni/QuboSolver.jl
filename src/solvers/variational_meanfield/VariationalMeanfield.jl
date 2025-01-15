module VariationalMeanfield

using ..QuboSolver
using Random
using Optimisers
using ProgressMeter

export VariationalMeanfieldSolver, solve!

struct VariationalMeanfieldSolver <: AbstractSolver end

abstract type RoundingMethod end
struct SignRounding <: RoundingMethod end
struct SequentialRounding <: RoundingMethod end

preprocess_input!(z::Vector{T}, initial_conf::Vector{T}) where {T<:AbstractFloat} = z .= atanh.(2 ./ pi .* initial_conf)

function compute_trig!(sines::Vector{T}, cosines::Vector{T}, z::Vector{T}) where {T<:AbstractFloat}
    sines .= sin.(pi / 2 * tanh.(z))
    cosines .= cos.(pi / 2 * tanh.(z))
    return nothing
end

function compute_trig(z::AbstractVector{<:AbstractFloat})
    sines = similar(z)
    cosines = similar(z)
    compute_trig!(sines, cosines, z)
    return sines, cosines
end

ease(x::AbstractFloat) = acos(1 - 2x) / π
a(t::AbstractFloat) = 1 - ease(t)
b(t::AbstractFloat) = ease(t)

function loss(problem::QuboProblem, sines::Vector{T}, cosines::Vector{T}, time::T, tf::T) where {T<:AbstractFloat}
    E = -tf * a(time) * sum(cosines) - b(time) * dot(sines, problem.W * sines)
    problem.has_bias && (E -= b(time) * dot(problem.c, sines))
    return E
end

function grad!(
    dz::Vector{T},
    problem::QuboProblem,
    z::Vector{T},
    sines::Vector{T},
    cosines::Vector{T},
    time::Real,
    tf::Real,
) where {T<:AbstractFloat}
    @. dz = pi / 2 * ((tf * a(time) * sines - b(time) * (2 * ($*)(problem.W, sines) * cosines)) * (1 - tanh(z)^2))

    if problem.has_bias
        @. dz -= pi / 2 * b(time) * problem.c * cosines * (1 - tanh(z)^2)
    end
end

function round_configuration(::QuboProblem, z::AbstractVector{<:AbstractFloat}, ::SignRounding)
    conf = similar(z, Int8)
    conf .= Int8.(sign.(sin.(pi / 2 .* tanh.(z))))

    conf[conf.==0] .= 1
    return conf
end

function round_configuration(problem::QuboProblem, z::AbstractVector{<:AbstractFloat}, ::SequentialRounding)
    problem.has_bias && throw(ArgumentError("Sequential rounding is not supported for problems with bias terms"))

    mean_z = sin.(pi / 2 .* tanh.(z))
    for i in 1:problem.N
        mean_z[i] = sign(dot(problem.W[:, i], mean_z))
    end
    conf = Int8.(mean_z)

    conf[conf.==0] .= 1

    return conf
end

function QuboSolver.solve!(
    problem::QuboProblem{T,Tw,Tc},
    solver::VariationalMeanfieldSolver;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    initial_conf::Union{AbstractVector{T},Nothing} = nothing,
    iterations::Int = 1000,
    inner_iterations::Int = 1,
    opt::Optimisers.AbstractRule = Adam(),
    tf::T = one(T),
    rounding::Union{RoundingMethod,Tuple{Vararg{RoundingMethod}}} = SignRounding(),
    save_params::Bool = false,
    save_energy::Bool = false,
    progressbar::Bool = true,
) where {T<:AbstractFloat,Tw<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
    iterations < 2 && throw(ArgumentError("Number of iterations must be at least 2"))

    if isnothing(initial_conf)
        initial_conf = dense_similar(problem.W, problem.N)
        randn!(rng, initial_conf)
        initial_conf .*= 0.001
    end

    z = similar(initial_conf)
    preprocess_input!(z, initial_conf)

    dz = similar(z)
    sines, cosines = compute_trig(z)

    state_tree = Optimisers.setup(opt, z)

    save_energy && (energy = Vector{T}(undef, iterations))

    initial_time = time()

    @showprogress showspeed = true enabled = progressbar for it in 1:iterations
        time = (it - 1) / (iterations - 1)
        for _ in 1:inner_iterations
            grad!(dz, problem, z, sines, cosines, time, tf)
            state_tree, z = Optimisers.update(state_tree, z, dz)
            compute_trig!(sines, cosines, z)
        end

        save_energy && (energy[it] = loss(problem, sines, cosines, time, tf))
    end

    delta_time = time() - initial_time

    metadata = (runtime = delta_time,)
    save_params && (metadata = merge(metadata, (params = z,)))
    save_energy && (metadata = merge(metadata, (energy = energy,)))

    if isa(rounding, RoundingMethod)
        return add_solution(problem, round_configuration(problem, z, rounding), solver; metadata...)
    else
        return map(
            round_method -> add_solution(problem, round_configuration(problem, z, round_method), solver; metadata...),
            rounding,
        )
    end
end

end
