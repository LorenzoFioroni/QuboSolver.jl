module GCS

using ..QuboSolver
using Random
using ProgressMeter
using Optimisers

export GCS_solver, solve!, SignRounding, SequentialRounding

@doc raw"""
    struct GCS_solver <: AbstractSolver end

Variational solver for QUBO problems using GCS states.

Use the GCS algorithm [fioroniEntanglementassisted2025](@cite) to solve the given [`QuboProblem`](@ref QuboSolver.QuboProblem). 
The analytical form of the GCS states used is
```math
\ket{ψ} = \mathcal{U}(x) \mathcal{V}(M) \ket{θ,φ}.
```
The operator ``\mathcal{U}(x)`` is a product of single-spin rotations on all particles. The vector
``x`` determines the rotation axis and angle for each spin. For our implementation, we parametrize
it in spherical coordinates as 
```math
x_i = \begin{bmatrix} r_i \cos(δ_i) \cos(Γ_i) \\ r_i \cos(δ_i) \sin(Γ_i) \\ r_i \sin(δ_i) \end{bmatrix}.
```
The operator ``\mathcal{V}(M)`` entangles the qubits via correlated ``σ_z\,σ_z`` rotations. Specifically,
```math
\mathcal{V}(M) = \exp(-i \sum_{i,j} M_{ij} σ_z^{(i)} σ_z^{(j)}),
```
where ``M`` is a coupling matrix assumed to be symmetric and with zero diagonal.
Finally, ``\ket{θ,φ}`` is the product state 
```math
\ket{θ,φ} = ⊗_i \left(\cos(θ_i/2) \ket{0} + \sin(θ_i/2) e^{i φ_i} \ket{1}\right).
```
GCS states are thus obtained by applying an entangling operator to a product state, and then
applying an additional rotation operator to each qubit.

!!! tip 
    For more information on the GCS algorithm, see
    [https://arxiv.org/abs/2501.09078](https://arxiv.org/abs/2501.09078).

!!! warning
    To use this solver, you need to explicitly import the `GCS` module in your code:
    ```julia
    using QuboSolver.Solvers.GCS
    ```
"""
struct GCS_solver <: AbstractSolver end

@doc raw"""
    abstract type RoundingMethod end

Abstract type for a discretization method.

Concrete types should be defined for each discretization method implemented, and they should 
define the `round_configuration` function.
"""
abstract type RoundingMethod end

@doc raw"""
    struct SignRounding <: RoundingMethod end

Discretization method that discretizes the quantum state to a binary configuration setting ``b_i = \text{sign}(σ_z)``.
"""
struct SignRounding <: RoundingMethod end

@doc raw"""
    struct SequentialRounding <: RoundingMethod end

Discretization method that implements the sequential rounding algorithm presented in [https://doi.org/10.1137/20M132016X](https://doi.org/10.1137/20M132016X).
"""
struct SequentialRounding <: RoundingMethod end

@doc raw"""
    const ParamType{T} = @NamedTuple{
        t::Vector{T},
        p::Vector{T},
        M::Matrix{T},
        r::Vector{T},
        d::Vector{T},
        g::Vector{T},
    }

Named tuple containing the parameters of the GCS state.

- `t`: vector of angles ``θ``.
- `p`: vector of angles ``φ``.
- `M`: matrix of the couplings.
- `r`: vector of magnitudes ``r``.
- `d`: vector of angles ``δ``.
- `g`: vector of angles ``Γ``.
"""
const ParamType{T} = @NamedTuple{
    t::Vector{T},
    p::Vector{T},
    M::Matrix{T},
    r::Vector{T},
    d::Vector{T},
    g::Vector{T},
}

@doc raw"""
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

Named tuple containing the trigonometric functions of the parameters of the GCS state.

This is used to avoid recomputing the trigonometric functions multiple times.

- `sin_t`: vector of sine of angles ``θ``.
- `cos_t`: vector of cosine of angles ``θ``.
- `sin_p`: vector of sine of angles ``φ``.
- `cos_p`: vector of cosine of angles ``φ``.
- `sin_d`: vector of sine of angles ``δ``.
- `cos_d`: vector of cosine of angles ``δ``.
- `exp_g`: vector of complex exponentials of angles ``Γ``.
- `sin_r`: vector of sine of angles ``r``.
- `cos_r`: vector of cosine of angles ``r``.
- `exp_M`: matrix of complex exponentials of the couplings.
"""
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

@doc raw"""
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

Named tuple containing temporal variables for the GCS calculations.

- `Sz`: named tuple containing the coefficients for the transformation of the ``σ_z`` operator 
    via the unitary conjugation ``\mathcal{U}^\dagger \sigma_z \mathcal{U}``. The fields `a1` 
    and `a0` contain the vectors of coefficients for the transformation of the ``σ_z`(i)`` operator
    and represent the prefactor of ``\sigma_+`` and ``\sigma_z`` respectively.
- `dSz_r`: named tuple containing the derivatives w.r.t. ``r`` of the coefficients for the 
    transformation of the ``σ_z`` operator via the unitary conjugation ``\mathcal{U}^\dagger \sigma_z \mathcal{U}``.
- `dSz_d`: named tuple containing the derivatives w.r.t. ``δ`` of the coefficients for the 
    transformation of the ``σ_z`` operator via the unitary conjugation ``\mathcal{U}^\dagger \sigma_z \mathcal{U}``.
- `dSz_g`: named tuple containing the derivatives w.r.t. ``Γ`` of the coefficients for the 
    transformation of the ``σ_z`` operator via the unitary conjugation ``\mathcal{U}^\dagger \sigma_z \mathcal{U}``.
- `Sx`: named tuple containing the coefficients for the transformation of the ``σ_x`` operator
    via the unitary conjugation ``\mathcal{U}^\dagger \sigma_x \mathcal{U}``. The fields `a1` 
    and `a0` contain the vectors of coefficients for the transformation of the ``σ_x``(i) operator
    and represent the prefactor of ``\sigma_+`` and ``\sigma_z`` respectively.
- `dSx_r`: named tuple containing the derivatives w.r.t. ``r`` of the coefficients for the
    transformation of the ``σ_x`` operator via the unitary conjugation ``\mathcal{U}^\dagger \sigma_x \mathcal{U}``.
- `dSx_d`: named tuple containing the derivatives w.r.t. ``δ`` of the coefficients for the
    transformation of the ``σ_x`` operator via the unitary conjugation ``\mathcal{U}^\dagger \sigma_x \mathcal{U}``.
- `dSx_g`: named tuple containing the derivatives w.r.t. ``Γ`` of the coefficients for the
    transformation of the ``σ_x`` operator via the unitary conjugation ``\mathcal{U}^\dagger \sigma_x \mathcal{U}``.
- `psi`: matrix storing the coefficients of the state ``\ket{θ,φ}``. The first column contains 
    the amplitudes of the ``\ket{0}`` state and the second column contains the amplitudes of the 
    ``\ket{1}`` state.
- `dpsi_t`: matrix storing the derivatives w.r.t. ``θ`` of the coefficients of the state ``\ket{θ,φ}``. 
- `dpsi_p`: matrix storing the derivatives w.r.t. ``φ`` of the coefficients of the state ``\ket{θ,φ}``.
"""
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

@doc raw"""
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

Named tuple containing the temporal variables for the GCS calculations for each thread.

- `psi_ket`: matrix storing the coefficients of the state ``\ket{θ,φ}``. The first column contains 
    the amplitudes of the ``\ket{0}`` state and the second column contains the amplitudes of the 
    ``\ket{1}`` state.
- `psi_ket_dM1`: matrix storing the coefficients of the state, used to compute the gradient w.r.t. 
    to the matrix M.
- `psi_ket_dM2`: matrix storing the coefficients of the state, used to compute the gradient w.r.t. 
    to the matrix M.
- `dpsi_t_ket`: matrix storing the coefficients of the state, used to compute the gradient w.r.t. 
    to the angles ``θ``.
- `dpsi_t_bra`: matrix storing the coefficients of the state, used to compute the gradient w.r.t.
    to the angles ``θ``.
- `dpsi_p_ket`: matrix storing the coefficients of the state, used to compute the gradient w.r.t.
    to the angles ``φ``.
- `dpsi_p_bra`: matrix storing the coefficients of the state, used to compute the gradient w.r.t.
    to the angles ``φ``.
- `mat_el`: vector storing the single-spin matrix elements for the computed expectation values.
- `mat_el_psi_dpsi_t`: vector storing the single-spin matrix elements for the computed expectation 
    values, used to compute the gradient w.r.t. the angles ``θ``.
- `mat_el_dpsi_t_psi`: vector storing the single-spin matrix elements for the computed expectation
    values, used to compute the gradient w.r.t. the angles ``θ``.
- `mat_el_psi_dpsi_p`: vector storing the single-spin matrix elements for the computed expectation
    values, used to compute the gradient w.r.t. the angles ``φ``.
- `mat_el_dpsi_p_psi`: vector storing the single-spin matrix elements for the computed expectation
    values, used to compute the gradient w.r.t. the angles ``φ``.
- `mat_el_dM1`: vector storing the single-spin matrix elements for the computed expectation
    values, used to compute the gradient w.r.t. the matrix M.
- `mat_el_dM2`: vector storing the single-spin matrix elements for the computed expectation
    values, used to compute the gradient w.r.t. the matrix M.
"""
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

@doc raw"""
    function prod_and_count_zeros(x::Vector{T}) where {T}

Compute the product of the elements of the vector `x` and count the number of zeros.

If the vector `x` does not contain any zeros, the function returns the product of its elements 
and the flag `-1`.
If the vector `x` contains one zero, the function returns the product of its non-zero elements 
and the index of the zero element.
If the vector `x` contains more than one zero, the function returns `0` and the flag`-2`.
"""
function prod_and_count_zeros(x::Vector{T}) where {T}
    p = one(T)
    c = 0
    nnzidx = -1
    @inbounds @simd for i ∈ eachindex(x)
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

@doc raw"""
    function alloc_trig(z::ParamType{T}) where {T}

Allocate the named tuple containing the trigonometric functions of the parameters of the GCS state.
"""
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

@doc raw"""
    function compute_trig!(trig::TrigType{T}, z::ParamType{T}) where {T}

Compute the trigonometric functions of the parameters of the GCS state and store them in the
named tuple `trig` in place.

See also [`compute_trig`](#ref).
"""
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

@doc raw"""
    function compute_trig(z::ParamType{T}) where {T}

Compute the trigonometric functions of the parameters of the GCS state and store them in a new
    named tuple `trig`.

See also [`compute_trig!`](#ref).
"""
function compute_trig(z::ParamType{T}) where {T}
    trig = alloc_trig(z)
    compute_trig!(trig, z)
    return trig
end

@doc raw"""
    function alloc_temporal(z::ParamType{T}) where {T}

Allocate the named tuple containing the temporal variables for the GCS calculations.
"""
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

@doc raw"""
    function alloc_thread_temporal(z::ParamType{T}) where {T}

Allocate the named tuples containing the temporal variables for the GCS calculations for each 
of the available threads.
"""
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
        ) for _ ∈ 1:Threads.nthreads()
    ]
end

@doc raw"""
    function compute_temporal!(temporal::TempType{T}, trig::TrigType{T}; grad = true) where {T}

Compute the temporal variables for the GCS calculations and store them in place in the named
tuple `temporal`.
"""
function compute_temporal!(temporal::TempType{T}, trig::TrigType{T}; grad = true) where {T}
    @inbounds begin
        @. temporal.Sz.a1 =
            trig.cos_d * trig.exp_g * (trig.sin_d * (1 - trig.cos_r) + trig.sin_r * im)
        @. temporal.Sz.a0 = trig.sin_d .^ 2 * (1 - trig.cos_r) + trig.cos_r

        @. temporal.Sx.a1 =
            trig.cos_d .^ 2 * (1 - trig.cos_r) * real.(trig.exp_g) * trig.exp_g +
            trig.cos_r - trig.sin_r * trig.sin_d * im
        @. temporal.Sx.a0 =
            trig.cos_d * (
                trig.sin_d * real.(trig.exp_g) * (1 - trig.cos_r) +
                trig.sin_r * imag(trig.exp_g)
            )

        @. temporal.psi[:, 1] =
            ((trig.cos_t) + (trig.sin_t * (trig.cos_p + trig.sin_p * im))) / sqrt(2)
        @. temporal.psi[:, 2] =
            ((trig.cos_t) - (trig.sin_t * (trig.cos_p + trig.sin_p * im))) / sqrt(2)

        !grad && return

        @. temporal.dSz_r.a1 =
            2 * trig.cos_d * trig.exp_g * (trig.sin_r * trig.sin_d + trig.cos_r * im)
        @. temporal.dSz_r.a0 = 2 * trig.sin_r * (trig.sin_d .^ 2 - 1)

        @. temporal.dSz_d.a1 =
            (trig.cos_d .^ 2 - trig.sin_d .^ 2) * trig.exp_g * (1 - trig.cos_r) -
            trig.sin_r * trig.sin_d * trig.exp_g * im
        @. temporal.dSz_d.a0 = 2 * trig.sin_d * trig.cos_d * (1 - trig.cos_r)

        @. temporal.dSz_g.a1 =
            trig.cos_d *
            (trig.sin_d * (1 - trig.cos_r) * trig.exp_g * im - trig.sin_r * trig.exp_g)

        @. temporal.dSx_r.a1 =
            2 * trig.sin_r * trig.cos_d .^ 2 * real.(trig.exp_g) * trig.exp_g -
            2 * trig.sin_r - 2 * trig.cos_r * trig.sin_d * im
        @. temporal.dSx_r.a0 =
            2 *
            trig.cos_d *
            (trig.sin_r * trig.sin_d * real.(trig.exp_g) + trig.cos_r * imag.(trig.exp_g))

        @. temporal.dSx_d.a1 =
            -2 *
            trig.sin_d *
            trig.cos_d *
            (1 - trig.cos_r) *
            real.(trig.exp_g) *
            trig.exp_g - trig.sin_r * trig.cos_d * im
        @. temporal.dSx_d.a0 =
            (trig.cos_d .^ 2 - trig.sin_d .^ 2) * real.(trig.exp_g) * (1 - trig.cos_r) -
            trig.sin_r * trig.sin_d * imag.(trig.exp_g)

        @. temporal.dSx_g.a1 = trig.exp_g .^ 2 * trig.cos_d .^ 2 * (1 - trig.cos_r) * im
        @. temporal.dSx_g.a0 =
            -trig.sin_d * trig.cos_d * (1 - trig.cos_r) * imag.(trig.exp_g) +
            trig.sin_r * trig.cos_d * real.(trig.exp_g)

        @. temporal.dpsi_t[:, 1] =
            ((-0.5 * trig.sin_t) + (0.5 * trig.cos_t * (trig.cos_p + trig.sin_p * im))) /
            sqrt(2)
        @. temporal.dpsi_t[:, 2] =
            ((-0.5 * trig.sin_t) - (0.5 * trig.cos_t * (trig.cos_p + trig.sin_p * im))) /
            sqrt(2)
        @. temporal.dpsi_p[:, 1] = (trig.sin_t * (-trig.sin_p + trig.cos_p * im)) / sqrt(2)
        @. temporal.dpsi_p[:, 2] = -(trig.sin_t * (-trig.sin_p + trig.cos_p * im)) / sqrt(2)
    end
    return nothing
end

@doc raw"""
    function _sum_aligned_namedtuples!(a::ParamType{T}, b::ParamType{T}, b_coeff::T) where {T}

Sum two ParamType named tuples and store the result in place in the first one. The second one is
multiplied by the coefficient `b_coeff` before being added to the first one.
"""
function _sum_aligned_namedtuples!(a::ParamType{T}, b::ParamType{T}, b_coeff::T) where {T}
    a.t .+= b_coeff .* b.t
    a.p .+= b_coeff .* b.p
    a.M .+= b_coeff .* b.M
    a.r .+= b_coeff .* b.r
    a.d .+= b_coeff .* b.d
    a.g .+= b_coeff .* b.g
    return nothing
end

@doc raw"""
    function _sum_aligned_namedtuples!(a::ParamType{T}, b::ParamType{T}, b_coeff::T, c::ParamType{T}, c_coeff::T) where {T}

Sum two ParamType named tuples `b` and `c` and store the result in place in the ParamType `a`. 
Two coefficients `b_coeff` and `c_coeff` are used to scale the contributions of `b` and `c` respectively.
"""
function _sum_aligned_namedtuples!(
    a::ParamType{T},
    b::ParamType{T},
    b_coeff::T,
    c::ParamType{T},
    c_coeff::T,
) where {T}
    a.t .= b_coeff .* b.t .+ c_coeff .* c.t
    a.p .= b_coeff .* b.p .+ c_coeff .* c.p
    a.M .= b_coeff .* b.M .+ c_coeff .* c.M
    a.r .= b_coeff .* b.r .+ c_coeff .* c.r
    a.d .= b_coeff .* b.d .+ c_coeff .* c.d
    a.g .= b_coeff .* b.g .+ c_coeff .* c.g
    return nothing
end

@doc raw"""
    function _symmetrize_M!(M::AbstractMatrix{T}) where {T}

Symmetrize the matrix `M` in place summing the upper and lower triangular parts. The matrix is 
assumed to be square.
"""
function _symmetrize_M!(M::AbstractMatrix{T}) where {T}
    @inbounds for i ∈ axes(M, 1)
        @simd for j ∈ (i+1):size(M, 2)
            tmp = M[i, j]
            M[i, j] += M[j, i]
            M[j, i] += tmp
        end
    end
    return
end

@doc raw"""
    function _setup_initial_conf(::Nothing, N::Int, rng::AbstractRNG, T::Type{T}) where {T}

Initialize the parameters of the GCS state close to the `\ket{+}` state.

## Arguments
- `N`: number of qubits.
- `rng`: random number generator.
- `T`: type of the parameters.
"""
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

@doc raw"""
    function _setup_initial_conf(initial_conf::ParamType{T}, _, _, _) where {T}

Copy the initial configuration `initial_conf` to avoid modifying the original one.
"""
_setup_initial_conf(initial_conf::ParamType{T}, _, _, _) where {T} = deepcopy(initial_conf)

@doc raw"""
    ease(x::AbstractFloat) = acos(1 - 2x) / π

Easing function for the time variable `x ∈ [0, 1]`.
"""
ease(x::AbstractFloat) = acos(1 - 2x) / π

@doc raw"""
    a(t::AbstractFloat) = 1 - ease(t)
    
Time-dependent coefficient for the transverse field term in the loss function.
"""
a(t::AbstractFloat) = 1 - ease(t)

@doc raw"""
    b(t::AbstractFloat) = ease(t)

Time-dependent coefficient for the Ising term in the loss function.
"""
b(t::AbstractFloat) = ease(t)

@doc raw"""
    function loss_x(
        x_loss::Vector{T},
        trig::TrigType{T},
        temporal::TempType{T},
        thread_temporal::Vector{ThreadTempType{T}},
        x_multithread_params,
    ) where {T}

compute the transverse field term of the loss function

## Arguments
- `x_loss`: vector to store the expectation values `⟨σ_x^{(i)}⟩`.
- `trig`: trigonometric functions of the parameters of the GCS state (assumed to be precomputed).
- `temporal`: temporal variables for the GCS calculations (assumed to be precomputed).
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `x_multithread_params`: partitioning of the qubits for the multithreaded computation.
"""
function loss_x(
    x_loss::Vector{T},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) ∈ x_multithread_params
        for j ∈ chunk
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

@doc raw"""
    function loss_z(
        z_loss::Vector{T},
        trig::TrigType{T},
        temporal::TempType{T},
        thread_temporal::Vector{ThreadTempType{T}},
        x_multithread_params,
    ) where {T}

Compute the expectation value of the ``\sigma_z`` operator for each qubit in the GCS state.

## Arguments
- `z_loss`: vector to store the expectation values `⟨σ_z^{(i)}⟩`.
- `trig`: trigonometric functions of the parameters of the GCS state (assumed to be precomputed).
- `temporal`: temporal variables for the GCS calculations (assumed to be precomputed).
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `x_multithread_params`: partitioning of the qubits for the multithreaded computation.
"""
function loss_z(
    z_loss::Vector{T},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) ∈ x_multithread_params
        for j ∈ chunk
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

@doc raw"""
    function loss_zz(
        zz_loss::Vector{T},
        trig::TrigType{T},
        temporal::TempType{T},
        thread_temporal::Vector{ThreadTempType{T}},
        zz_multithread_params,
    ) where {T}
Compute the expectation value of the ``\sigma_z^{(i)} \sigma_z^{(j)}`` operator for each pair of 
qubits in the GCS state.

## Arguments
- `zz_loss`: vector to store the expectation values `⟨σ_z^{(i)} σ_z^{(j)}⟩` (only the elements with 
    `i < j` corresponding to a non-zero coefficient in the QUBO problem are computed).
- `trig`: trigonometric functions of the parameters of the GCS state (assumed to be precomputed).
- `temporal`: temporal variables for the GCS calculations (assumed to be precomputed).
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `zz_multithread_params`: partitioning of the pairs of qubits for the multithreaded computation. 
    Only the elements with `i < j` corresponding to a non-zero coefficient in the QUBO problem are computed.
"""
function loss_zz(
    zz_loss::Vector{T},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    zz_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) ∈ zz_multithread_params
        for (lin_ind, ((j, k), w)) ∈ chunk
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
                zz_loss[lin_ind] +=
                    2real(temporal.Sz.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)

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

@doc raw"""
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

Compute the loss function for the GCS state.

## Arguments
- `z`: parameters of the GCS state.
- `time`: time identifying the current Hamiltonian.
- `tf`: transverse field strength.
- `trig`: trigonometric functions of the parameters of the GCS state.
- `temporal`: temporal variables for the GCS calculations.
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `zz_multithread_params`: partitioning of the pairs of qubits for the multithreaded computation. 
    Only the elements with `i < j` corresponding to a non-zero coefficient in the QUBO problem are computed.
- `x_multithread_params`: partitioning of the qubits for the multithreaded computation.
- `compact_W`: compact representation of the QUBO matrix. Only the elements with `i < j` 
    corresponding to a non-zero coefficient in the QUBO problem are stored.
- `bias`: bias vector of the QUBO problem. If the problem does not have a bias, this argument
    should be `nothing`, otherwise it should be a vector of the same length as the number of variables.
- `zz_loss`: vector to store the expectation values `⟨σ_z^{(i)} σ_z^{(j)}⟩` (only the elements with
    `i < j` corresponding to a non-zero coefficient in the QUBO problem are computed).
- `x_loss`: vector to store the expectation values `⟨σ_x^{(i)}⟩`.
"""
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
    bias::Union{Nothing,Vector{T}},
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

    loss = -tf * a(time) * sum(x_loss) - b(time) * sum(zz_loss)
    if !isnothing(bias)
        loss_z(x_loss, trig, temporal, thread_temporal, x_multithread_params)
        loss -= b(time) * dot(bias, x_loss)
    end
    return loss
end

@doc raw"""
    function loss_x_and_grad!(
        x_loss::Vector{T},
        dz_x::Vector{ParamType{T}},
        trig::TrigType{T},
        temporal::TempType{T},
        thread_temporal::Vector{ThreadTempType{T}},
        x_multithread_params,
    ) where {T}

Compute the transverse field term of the loss function and its gradient with respect to the
parameters of the GCS state.

## Arguments
- `x_loss`: vector to store the expectation values `⟨σ_x^{(i)}⟩`.
- `dz_x`: vector to store the gradients of the loss function with respect to the parameters of 
    the GCS state. Each thread has its own gradient ParamType to avoid race conditions.
- `trig`: trigonometric functions of the parameters of the GCS state (assumed to be precomputed).
- `temporal`: temporal variables for the GCS calculations (assumed to be precomputed).
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `x_multithread_params`: partitioning of the qubits for the multithreaded computation.
"""
function loss_x_and_grad!(
    x_loss::Vector{T},
    dz_x::Vector{ParamType{T}},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) ∈ x_multithread_params
        for j ∈ chunk
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
                thread_temporal[tid].dpsi_t_ket[j, 2] =
                    thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] =
                    thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*=
                    -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*=
                    0.25im .* conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM1[j, 2] =
                    thread_temporal[tid].psi_ket_dM1[j, 1]
                thread_temporal[tid].psi_ket_dM1[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )
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
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_t
                            ) / thread_temporal[tid].mat_el,
                        )

                    @views @. dz_x[tid].p +=
                        2real(
                            temporal.Sx.a1[j] *
                            Pj *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_p
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_x[tid].M[:, j] +=
                        2real(
                            temporal.Sx.a1[j] * Pj * thread_temporal[tid].mat_el_dM1 /
                            thread_temporal[tid].mat_el,
                        )
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
                    dz_x[tid].M[nnz_idx, j] +=
                        2real(
                            temporal.Sx.a1[j] *
                            Pj *
                            thread_temporal[tid].mat_el_dM1[nnz_idx],
                        )
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
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )

                Pj, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    x_loss[j] += real(temporal.Sx.a0[j] * Pj)
                    dz_x[tid].r[j] += real(temporal.dSx_r.a0[j] * Pj)
                    dz_x[tid].d[j] += real(temporal.dSx_d.a0[j] * Pj)
                    dz_x[tid].g[j] += real(temporal.dSx_g.a0[j] * Pj)
                    @views @. dz_x[tid].t += real(
                        temporal.Sx.a0[j] *
                        Pj *
                        (
                            thread_temporal[tid].mat_el_dpsi_t_psi +
                            thread_temporal[tid].mat_el_psi_dpsi_t
                        ) / thread_temporal[tid].mat_el,
                    )
                    @views @. dz_x[tid].p += real(
                        temporal.Sx.a0[j] *
                        Pj *
                        (
                            thread_temporal[tid].mat_el_dpsi_p_psi +
                            thread_temporal[tid].mat_el_psi_dpsi_p
                        ) / thread_temporal[tid].mat_el,
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

@doc raw"""
    function loss_z_and_grad!(
        z_loss::Vector{T},
        dz_z::Vector{ParamType{T}},
        trig::TrigType{T},
        temporal::TempType{T},
        thread_temporal::Vector{ThreadTempType{T}},
        z_multithread_params,
    ) where {T}

Compute the bias field term of the loss function and its gradient with respect to the parameters 
of the GCS state.

## Arguments
- `z_loss`: vector to store the expectation values `⟨σ_z^{(i)}⟩`.
- `dz_z`: vector to store the gradients of the loss function with respect to the parameters of 
    the GCS state. Each thread has its own gradient ParamType to avoid race conditions.
- `trig`: trigonometric functions of the parameters of the GCS state (assumed to be precomputed).
- `temporal`: temporal variables for the GCS calculations (assumed to be precomputed).
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `z_multithread_params`: partitioning of the qubits and bias terms for the multithreaded computation
"""
function loss_z_and_grad!(
    z_loss::Vector{T},
    dz_x::Vector{ParamType{T}},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    z_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) ∈ z_multithread_params
        for (j, c) ∈ chunk
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
                thread_temporal[tid].dpsi_t_ket[j, 2] =
                    thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] =
                    thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*=
                    -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*=
                    0.25im .* conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM1[j, 2] =
                    thread_temporal[tid].psi_ket_dM1[j, 1]
                thread_temporal[tid].psi_ket_dM1[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )
                sum!(thread_temporal[tid].mat_el_dM1, thread_temporal[tid].psi_ket_dM1)

                Pj, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    z_loss[j] += 2real(temporal.Sz.a1[j] * Pj)
                    dz_x[tid].r[j] += c * 2real(temporal.dSz_r.a1[j] * Pj)
                    dz_x[tid].d[j] += c * 2real(temporal.dSz_d.a1[j] * Pj)
                    dz_x[tid].g[j] += c * 2real(temporal.dSz_g.a1[j] * Pj)
                    @views @. dz_x[tid].t +=
                        c *
                        2real(
                            temporal.Sz.a1[j] *
                            Pj *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_t
                            ) / thread_temporal[tid].mat_el,
                        )

                    @views @. dz_x[tid].p +=
                        c *
                        2real(
                            temporal.Sz.a1[j] *
                            Pj *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_p
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_x[tid].M[:, j] +=
                        c *
                        2real(
                            temporal.Sz.a1[j] * Pj * thread_temporal[tid].mat_el_dM1 /
                            thread_temporal[tid].mat_el,
                        )
                elseif nnz_idx != -2
                    dz_x[tid].t[nnz_idx] +=
                        c *
                        2real(
                            temporal.Sz.a1[j] *
                            Pj *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                            ),
                        )

                    dz_x[tid].p[nnz_idx] +=
                        c *
                        2real(
                            temporal.Sz.a1[j] *
                            Pj *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi[nnz_idx] +
                                thread_temporal[tid].mat_el_psi_dpsi_p[nnz_idx]
                            ),
                        )
                    dz_x[tid].M[nnz_idx, j] +=
                        c *
                        2real(
                            temporal.Sz.a1[j] *
                            Pj *
                            thread_temporal[tid].mat_el_dM1[nnz_idx],
                        )
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
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )

                Pj, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    z_loss[j] += real(temporal.Sz.a0[j] * Pj)
                    dz_x[tid].r[j] += c * real(temporal.dSz_r.a0[j] * Pj)
                    dz_x[tid].d[j] += c * real(temporal.dSz_d.a0[j] * Pj)
                    @views @. dz_x[tid].t += real(
                        c *
                        temporal.Sz.a0[j] *
                        Pj *
                        (
                            thread_temporal[tid].mat_el_dpsi_t_psi +
                            thread_temporal[tid].mat_el_psi_dpsi_t
                        ) / thread_temporal[tid].mat_el,
                    )
                    @views @. dz_x[tid].p += real(
                        c *
                        temporal.Sz.a0[j] *
                        Pj *
                        (
                            thread_temporal[tid].mat_el_dpsi_p_psi +
                            thread_temporal[tid].mat_el_psi_dpsi_p
                        ) / thread_temporal[tid].mat_el,
                    )
                elseif nnz_idx != -2
                    dz_x[tid].t[nnz_idx] += real(
                        c *
                        temporal.Sz.a0[j] *
                        Pj *
                        (
                            thread_temporal[tid].mat_el_dpsi_t_psi[nnz_idx] +
                            thread_temporal[tid].mat_el_psi_dpsi_t[nnz_idx]
                        ),
                    )
                    dz_x[tid].p[nnz_idx] += real(
                        c *
                        temporal.Sz.a0[j] *
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

@doc raw"""
    function loss_zz_and_grad!(
        zz_loss::Vector{T},
        dz_zz::Vector{ParamType{T}},
        trig::TrigType{T},
        temporal::TempType{T},
        thread_temporal::Vector{ThreadTempType{T}},
        zz_multithread_params,
    ) where {T}

Compute the expectation value of the ``\sigma_z^{(i)} \sigma_z^{(j)}`` operator for each pair of
qubits in the GCS state and its gradient with respect to the parameters of the GCS state.

## Arguments
- `zz_loss`: vector to store the expectation values `⟨σ_z^{(i)} σ_z^{(j)}⟩` (only the elements with
    `i < j` corresponding to a non-zero coefficient in the QUBO problem are computed).
- `dz_zz`: vector to store the gradients of the loss function with respect to the parameters of
    the GCS state. Each thread has its own gradient ParamType to avoid race conditions.
- `trig`: trigonometric functions of the parameters of the GCS state (assumed to be precomputed).
- `temporal`: temporal variables for the GCS calculations (assumed to be precomputed).
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `zz_multithread_params`: partitioning of the pairs of qubits for the multithreaded computation. 
    Only the elements with `i < j` corresponding to a non-zero coefficient in the QUBO problem are computed.
"""
function loss_zz_and_grad!(
    zz_loss::Vector{T},
    dz_zz::Vector{ParamType{T}},
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    zz_multithread_params,
) where {T}
    Threads.@threads for (tid, chunk) ∈ zz_multithread_params
        for (lin_ind, ((j, k), w)) ∈ chunk
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
                thread_temporal[tid].dpsi_t_ket[k, 2] =
                    thread_temporal[tid].dpsi_t_ket[k, 1]
                thread_temporal[tid].dpsi_t_ket[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_t_ket[j, 2] =
                    thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].dpsi_p_ket[k, 2] =
                    thread_temporal[tid].dpsi_p_ket[k, 1]
                thread_temporal[tid].dpsi_p_ket[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] =
                    thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket_dM1[k, 2] =
                    thread_temporal[tid].psi_ket_dM1[k, 1]
                thread_temporal[tid].psi_ket_dM1[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*=
                    -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*=
                    0.25im .* conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM1[j, 2] =
                    thread_temporal[tid].psi_ket_dM1[j, 1]
                thread_temporal[tid].psi_ket_dM1[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*=
                    -0.25im .* trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*=
                    0.25im .* conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket_dM2[k, 2] =
                    thread_temporal[tid].psi_ket_dM2[k, 1]
                thread_temporal[tid].psi_ket_dM2[k, 1] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM2[j, 2] =
                    thread_temporal[tid].psi_ket_dM2[j, 1]
                thread_temporal[tid].psi_ket_dM2[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM2 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )
                sum!(thread_temporal[tid].mat_el_dM1, thread_temporal[tid].psi_ket_dM1)
                sum!(thread_temporal[tid].mat_el_dM2, thread_temporal[tid].psi_ket_dM2)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += 2real(temporal.Sz.a1[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].r[j] +=
                        2w * 2real(temporal.dSz_r.a1[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].r[k] +=
                        2w * 2real(temporal.Sz.a1[j] * temporal.dSz_r.a1[k] * P_jk)
                    dz_zz[tid].d[j] +=
                        2w * 2real(temporal.dSz_d.a1[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].d[k] +=
                        2w * 2real(temporal.Sz.a1[j] * temporal.dSz_d.a1[k] * P_jk)
                    dz_zz[tid].g[j] +=
                        2w * 2real(temporal.dSz_g.a1[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].g[k] +=
                        2w * 2real(temporal.Sz.a1[j] * temporal.dSz_g.a1[k] * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_t
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_p
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_j +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            thread_temporal[tid].mat_el_dM1 / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_k +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            thread_temporal[tid].mat_el_dM2 / thread_temporal[tid].mat_el,
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
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            thread_temporal[tid].mat_el_dM1[nnz_idx],
                        )
                    dz_M_view_k[nnz_idx] +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            thread_temporal[tid].mat_el_dM2[nnz_idx],
                        )
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
                thread_temporal[tid].dpsi_t_ket[k, 1] =
                    thread_temporal[tid].dpsi_t_ket[k, 2]
                thread_temporal[tid].dpsi_t_ket[k, 2] = 0.0 + 0.0 * im
                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_t_ket[j, 2] =
                    thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= trig.exp_M[:, k]
                thread_temporal[tid].dpsi_p_ket[k, 1] =
                    thread_temporal[tid].dpsi_p_ket[k, 2]
                thread_temporal[tid].dpsi_p_ket[k, 2] = 0.0 + 0.0 * im
                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] =
                    thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*= conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*= trig.exp_M[:, k]
                thread_temporal[tid].psi_ket_dM1[k, 1] =
                    thread_temporal[tid].psi_ket_dM1[k, 2]
                thread_temporal[tid].psi_ket_dM1[k, 2] = 0.0 + 0.0 * im
                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*=
                    -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*=
                    0.25im .* conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM1[j, 2] =
                    thread_temporal[tid].psi_ket_dM1[j, 1]
                thread_temporal[tid].psi_ket_dM1[j, 1] = 0.0 + 0.0 * im

                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*=
                    0.25im .* conj.(trig.exp_M[:, k])
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*=
                    -0.25im .* trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].psi_ket_dM2[j, 2] =
                    thread_temporal[tid].psi_ket_dM2[j, 1]
                thread_temporal[tid].psi_ket_dM2[j, 1] = 0.0 + 0.0 * im
                thread_temporal[tid].psi_ket_dM2[j, 2] =
                    thread_temporal[tid].psi_ket_dM2[j, 1]
                thread_temporal[tid].psi_ket_dM2[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM2 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )
                sum!(thread_temporal[tid].mat_el_dM1, thread_temporal[tid].psi_ket_dM1)
                sum!(thread_temporal[tid].mat_el_dM2, thread_temporal[tid].psi_ket_dM2)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] +=
                        2real(temporal.Sz.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)
                    dz_zz[tid].r[j] +=
                        2w * 2real(temporal.dSz_r.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)
                    dz_zz[tid].r[k] +=
                        2w * 2real(temporal.Sz.a1[j] * conj(temporal.dSz_r.a1[k]) * P_jk)
                    dz_zz[tid].d[j] +=
                        2w * 2real(temporal.dSz_d.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)
                    dz_zz[tid].d[k] +=
                        2w * 2real(temporal.Sz.a1[j] * conj(temporal.dSz_d.a1[k]) * P_jk)
                    dz_zz[tid].g[j] +=
                        2w * 2real(temporal.dSz_g.a1[j] * conj(temporal.Sz.a1[k]) * P_jk)
                    dz_zz[tid].g[k] +=
                        2w * 2real(temporal.Sz.a1[j] * conj(temporal.dSz_g.a1[k]) * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_t
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_p
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_j +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            thread_temporal[tid].mat_el_dM1 / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_k +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            conj(temporal.Sz.a1[k]) *
                            P_jk *
                            thread_temporal[tid].mat_el_dM2 / thread_temporal[tid].mat_el,
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
                thread_temporal[tid].dpsi_t_ket[j, 2] =
                    thread_temporal[tid].dpsi_t_ket[j, 1]
                thread_temporal[tid].dpsi_t_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].dpsi_p_ket[k, 2] *= -1.0
                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, j]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, j])
                thread_temporal[tid].dpsi_p_ket[j, 2] =
                    thread_temporal[tid].dpsi_p_ket[j, 1]
                thread_temporal[tid].dpsi_p_ket[j, 1] = 0.0 + 0.0 * im

                thread_temporal[tid].psi_ket_dM1[k, 2] *= -1.0
                @views thread_temporal[tid].psi_ket_dM1[:, 1] .*=
                    -0.25im .* trig.exp_M[:, j]
                @views thread_temporal[tid].psi_ket_dM1[:, 2] .*=
                    0.25im .* conj.(trig.exp_M[:, j])

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM1 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )
                sum!(thread_temporal[tid].mat_el_dM1, thread_temporal[tid].psi_ket_dM1)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += 2real(temporal.Sz.a1[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].r[j] +=
                        2w * 2real(temporal.dSz_r.a1[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].r[k] +=
                        2w * 2real(temporal.Sz.a1[j] * temporal.dSz_r.a0[k] * P_jk)
                    dz_zz[tid].d[j] +=
                        2w * 2real(temporal.dSz_d.a1[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].d[k] +=
                        2w * 2real(temporal.Sz.a1[j] * temporal.dSz_d.a0[k] * P_jk)
                    dz_zz[tid].g[j] +=
                        2w * 2real(temporal.dSz_g.a1[j] * temporal.Sz.a0[k] * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_t
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_p
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_j +=
                        2w *
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            thread_temporal[tid].mat_el_dM1 / thread_temporal[tid].mat_el,
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
                        2real(
                            temporal.Sz.a1[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            thread_temporal[tid].mat_el_dM1[nnz_idx],
                        )
                end

                # alpha = 3, beta = 1
                thread_temporal[tid].psi_ket .= temporal.psi
                thread_temporal[tid].dpsi_t_ket .= temporal.dpsi_t
                thread_temporal[tid].dpsi_t_bra .= conj.(temporal.dpsi_t)
                thread_temporal[tid].dpsi_p_ket .= temporal.dpsi_p
                thread_temporal[tid].dpsi_p_bra .= conj.(temporal.dpsi_p)
                thread_temporal[tid].psi_ket_dM2 .= temporal.psi

                @views thread_temporal[tid].psi_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket[k, 2] = thread_temporal[tid].psi_ket[k, 1]
                thread_temporal[tid].psi_ket[k, 1] = 0.0 + 0.0 * im
                thread_temporal[tid].psi_ket[j, 2] *= -1.0

                @views thread_temporal[tid].dpsi_t_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].dpsi_t_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].dpsi_t_ket[k, 2] =
                    thread_temporal[tid].dpsi_t_ket[k, 1]
                thread_temporal[tid].dpsi_t_ket[k, 1] = 0.0 + 0.0 * im
                thread_temporal[tid].dpsi_t_ket[j, 2] *= -1.0

                @views thread_temporal[tid].dpsi_p_ket[:, 1] .*= trig.exp_M[:, k]
                @views thread_temporal[tid].dpsi_p_ket[:, 2] .*= conj.(trig.exp_M[:, k])
                thread_temporal[tid].dpsi_p_ket[k, 2] =
                    thread_temporal[tid].dpsi_p_ket[k, 1]
                thread_temporal[tid].dpsi_p_ket[k, 1] = 0.0 + 0.0 * im
                thread_temporal[tid].dpsi_p_ket[j, 2] *= -1.0

                @views thread_temporal[tid].psi_ket_dM2[:, 1] .*=
                    -0.25im .* trig.exp_M[:, k]
                @views thread_temporal[tid].psi_ket_dM2[:, 2] .*=
                    0.25im .* conj.(trig.exp_M[:, k])
                thread_temporal[tid].psi_ket_dM2[j, 2] *= -1.0

                thread_temporal[tid].dpsi_t_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].dpsi_p_bra .*= thread_temporal[tid].psi_ket
                thread_temporal[tid].psi_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_t_ket .*= conj.(temporal.psi)
                thread_temporal[tid].dpsi_p_ket .*= conj.(temporal.psi)
                thread_temporal[tid].psi_ket_dM2 .*= conj.(temporal.psi)

                sum!(thread_temporal[tid].mat_el, thread_temporal[tid].psi_ket)
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )
                sum!(thread_temporal[tid].mat_el_dM2, thread_temporal[tid].psi_ket_dM2)

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += 2real(temporal.Sz.a0[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].r[j] +=
                        2w * 2real(temporal.dSz_r.a0[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].r[k] +=
                        2w * 2real(temporal.Sz.a0[j] * temporal.dSz_r.a1[k] * P_jk)
                    dz_zz[tid].d[j] +=
                        2w * 2real(temporal.dSz_d.a0[j] * temporal.Sz.a1[k] * P_jk)
                    dz_zz[tid].d[k] +=
                        2w * 2real(temporal.Sz.a0[j] * temporal.dSz_d.a1[k] * P_jk)
                    dz_zz[tid].g[k] +=
                        2w * 2real(temporal.Sz.a0[j] * temporal.dSz_g.a1[k] * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w *
                        2real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_t
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w *
                        2real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_p
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_M_view_k +=
                        2w *
                        2real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            thread_temporal[tid].mat_el_dM2 / thread_temporal[tid].mat_el,
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
                        2real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a1[k] *
                            P_jk *
                            thread_temporal[tid].mat_el_dM2[nnz_idx],
                        )
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
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_t,
                    thread_temporal[tid].dpsi_t_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_t_psi,
                    thread_temporal[tid].dpsi_t_bra,
                )
                sum!(
                    thread_temporal[tid].mat_el_psi_dpsi_p,
                    thread_temporal[tid].dpsi_p_ket,
                )
                sum!(
                    thread_temporal[tid].mat_el_dpsi_p_psi,
                    thread_temporal[tid].dpsi_p_bra,
                )

                P_jk, nnz_idx = prod_and_count_zeros(thread_temporal[tid].mat_el)
                if nnz_idx == -1
                    zz_loss[lin_ind] += real(temporal.Sz.a0[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].r[j] +=
                        2w * real(temporal.dSz_r.a0[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].r[k] +=
                        2w * real(temporal.Sz.a0[j] * temporal.dSz_r.a0[k] * P_jk)
                    dz_zz[tid].d[j] +=
                        2w * real(temporal.dSz_d.a0[j] * temporal.Sz.a0[k] * P_jk)
                    dz_zz[tid].d[k] +=
                        2w * real(temporal.Sz.a0[j] * temporal.dSz_d.a0[k] * P_jk)
                    @views @. dz_zz[tid].t +=
                        2w * real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_t_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_t
                            ) / thread_temporal[tid].mat_el,
                        )
                    @views @. dz_zz[tid].p +=
                        2w * real(
                            temporal.Sz.a0[j] *
                            temporal.Sz.a0[k] *
                            P_jk *
                            (
                                thread_temporal[tid].mat_el_dpsi_p_psi +
                                thread_temporal[tid].mat_el_psi_dpsi_p
                            ) / thread_temporal[tid].mat_el,
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

@doc raw"""
    loss_and_grad!(
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
        z_multithread_params,
        compact_W::Vector{T},
        zz_loss::Vector{T},
        x_loss::Vector{T},
        dz_zz::Vector{ParamType{T}},
        dz_x::Vector{ParamType{T}},
    )

Compute the loss and gradients for the Qubo problem.

## Arguments
- `l`: one-element vector to store the loss inplace.
- `dz`: ParamType to store the gradients inplace.
- `z`: ParamType with the current configuration.
- `time`: time identifying the current Hamiltonian
- `tf`: transverse field strength
- `trig`: trigonometric functions of the parameters of the GCS state.
- `temporal`: temporal variables for the GCS calculations.
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `zz_multithread_params`: partitioning of the pairs of qubits for the multithreaded computation. 
    Only the elements with `i < j` corresponding to a non-zero coefficient in the QUBO problem are computed.
- `x_multithread_params`: partitioning of the qubits for the multithreaded computation.
- `z_multithread_params`: partitioning of the qubits and bias terms for the multithreaded computation.
- `compact_W`: compact representation of the QUBO matrix. Only the elements with `i < j` 
    corresponding to a non-zero coefficient in the QUBO problem are stored.
- `bias`: bias vector of the QUBO problem. If the problem does not have a bias, this argument
    should be `nothing`, otherwise it should be a vector of the same length as the number of variables.
- `zz_loss`: vector to store the expectation values `⟨σ_z^{(i)} σ_z^{(j)}⟩` (only the elements with
    `i < j` corresponding to a non-zero coefficient in the QUBO problem are computed).
- `x_loss`: vector to store the expectation values `⟨σ_x^{(i)}⟩`.
- `dz_zz`: vector to store the gradients of the loss function with respect to the parameters of
    the GCS state. Each thread has its own gradient ParamType to avoid race conditions.
- `dz_x`: vector to store the gradients of the loss function with respect to the parameters of
    the GCS state. Each thread has its own gradient ParamType to avoid race conditions.
"""
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
    z_multithread_params,
    compact_W::Vector{T},
    bias::Union{Nothing,Vector{T}},
    zz_loss::Vector{T},
    x_loss::Vector{T},
    dz_zz::Vector{ParamType{T}},
    dz_x::Vector{ParamType{T}},
) where {T}
    compute_trig!(trig, z)
    compute_temporal!(temporal, trig)

    @inbounds for i ∈ 1:Threads.nthreads()
        for k ∈ keys(dz_zz[i])
            dz_zz[i][k] .= zero(T)
            dz_x[i][k] .= zero(T)
        end
    end

    zz_loss .= zero(T)
    x_loss .= zero(T)

    loss_zz_and_grad!(
        zz_loss,
        dz_zz,
        trig,
        temporal,
        thread_temporal,
        zz_multithread_params,
    )
    loss_x_and_grad!(x_loss, dz_x, trig, temporal, thread_temporal, x_multithread_params)

    zz_loss .*= 2 .* compact_W
    l .= -tf * a(time) * sum(x_loss) - b(time) * sum(zz_loss)

    @inbounds for ii ∈ 2:Threads.nthreads()
        _sum_aligned_namedtuples!(dz_zz[1], dz_zz[ii], one(T))
        _sum_aligned_namedtuples!(dz_x[1], dz_x[ii], one(T))
    end

    _sum_aligned_namedtuples!(dz, dz_x[1], -tf * a(time), dz_zz[1], -b(time))

    if !isnothing(bias)
        x_loss .= zero(T)
        @inbounds for i ∈ 1:Threads.nthreads()
            for k ∈ keys(dz_x[i])
                dz_x[i][k] .= zero(T)
            end
        end

        loss_z_and_grad!(
            x_loss,
            dz_x,
            trig,
            temporal,
            thread_temporal,
            z_multithread_params,
        )
        @inbounds for ii ∈ 2:Threads.nthreads()
            _sum_aligned_namedtuples!(dz_x[1], dz_x[ii], one(T))
        end

        l .-= b(time) * dot(bias, x_loss)
        _sum_aligned_namedtuples!(dz, dz_x[1], -b(time))
    end

    _symmetrize_M!(dz.M)
    dz.M[diagind(dz.M)] .= zero(T)

    return
end

@doc raw"""
    round_configuration(
        problem::QuboProblem,
        z::ParamType{T},
        ::SignRounding,
        trig::TrigType{T},
        temporal::TempType{T},
        thread_temporal::Vector{ThreadTempType{T}},
        x_multithread_params,
    )

Get a classical configuration from the GCS state using the SignRounding method.

## Arguments
- `problem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) object.
- `z`: ParamType with the current configuration.
- `trig`: trigonometric functions of the parameters of the GCS state.
- `temporal`: temporal variables for the GCS calculations.
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `x_multithread_params`: partitioning of the qubits for the multithreaded computation.
"""
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

@doc raw"""
    round_configuration(
        problem::QuboProblem,
        z::ParamType{T},
        ::SequentialRounding,
        trig::TrigType{T},
        temporal::TempType{T},
        thread_temporal::Vector{ThreadTempType{T}},
        x_multithread_params,
    )

Get a classical configuration from the GCS state using the SequentialRounding method.

## Arguments
- `problem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) object.
- `z`: ParamType with the current configuration.
- `trig`: trigonometric functions of the parameters of the GCS state.
- `temporal`: temporal variables for the GCS calculations.
- `thread_temporal`: temporal variables for the GCS calculations for each thread.
- `x_multithread_params`: partitioning of the qubits for the multithreaded computation.
"""
function round_configuration(
    problem::QuboProblem,
    z::ParamType{T},
    ::SequentialRounding,
    trig::TrigType{T},
    temporal::TempType{T},
    thread_temporal::Vector{ThreadTempType{T}},
    x_multithread_params,
) where {T}
    problem.has_bias && throw(
        ArgumentError("Sequential rounding is not supported for problems with bias terms"),
    )

    z_loss = zero(z.t)
    mean_z = similar(z.t)

    compute_trig!(trig, z)
    compute_temporal!(temporal, trig)
    loss_z(z_loss, trig, temporal, thread_temporal, x_multithread_params)

    sum!(mean_z, z_loss)

    for i ∈ 1:problem.N
        mean_z[i] = sign(dot(problem.W[:, i], mean_z))
    end
    conf = Int8.(mean_z)

    conf[conf.==0] .= 1

    return conf
end

@doc raw"""
    function solve!(
        problem::QuboProblem{T,TW,Tc},
        solver::GCS_solver;
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
    )

Solve the QUBO problem using the Variational GCS method [fioroniEntanglementassisted2025](@cite).

!!! tip 
    For more information on the GCS algorithm, see
    [https://arxiv.org/abs/2501.09078](https://arxiv.org/abs/2501.09078).

!!! warning
    To use this solver, you need to explicitly import the `GCS` module in your code:
    ```julia
    using QuboSolver.Solvers.GCS
    ```

# Arguments
- `problem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) object.
- `solver`: [`GCS_solver`](@ref QuboSolver.Solvers.GCS.GCS_solver) object.
- `rng`: random number generator (default: `Random.GLOBAL_RNG`).
- `initial_conf`: initial parameter configuration (default: `nothing` for random initialization).
- `iterations`: number of time steps (default: `1000`).
- `inner_iterations`: number of gradient descent steps each time step (default: `1`).
- `tf`: transverse field strength (default: `1.0`).
- `rounding`: Rounding method used to obtain the classical configuration from the GCS
    state (default: [`SignRounding`](@ref QuboSolver.Solvers.GCS.SignRounding)). Accepts both a single method or a tuple of methods.
- `save_params`: whether to store the final parameters of the GCS state (default: `false`).
- `save_energy`: whether to store the variational energy during the optimization (default: `false`).
- `opt`: optimizer for the gradient descent (default: `Adam(0.05)`).
- `progressbar`: whether to show a progress bar (default: `true`).

# Returns
The optimal solution found by the solver. Metadata include the runtime as `runtime`, the final 
parameters of the GCS state as `params` (if `save_params` is `true`), and the variational energy 
during the optimization as `energy` (if `save_energy` is `true`).

# Example
```jldoctest; setup = :(using Random; Random.seed!(11))
using QuboSolver.Solvers.GCS

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, GCS_solver(); save_params=true, save_energy=true, progressbar=false)

# output

🟦🟦 - Energy: -3.0 - Solver: GCS_solver - Metadata count: 3
```
"""
function QuboSolver.solve!(
    problem::QuboProblem{T,TW,Tc},
    solver::GCS_solver;
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
    # problem.has_bias && throw(ArgumentError("Bias term not supported yet"))

    iterations < 2 && throw(ArgumentError("Number of iterations must be at least 2"))

    z = _setup_initial_conf(initial_conf, problem.N, rng, T)

    compact_idx, compact_W = nonzero_triu(problem.W)
    _params = enumerate(zip(compact_idx, compact_W))
    zz_multithread_params = collect(
        enumerate(Iterators.partition(_params, cld(length(_params), Threads.nthreads()))),
    )
    x_multithread_params = collect(
        enumerate(Iterators.partition(1:problem.N, cld(problem.N, Threads.nthreads()))),
    )
    z_multithread_params = if problem.has_bias
        collect(
            enumerate(
                Iterators.partition(
                    zip(1:problem.N, problem.c),
                    cld(problem.N, Threads.nthreads()),
                ),
            ),
        )
    else
        nothing
    end

    trig = alloc_trig(z)
    temporal = alloc_temporal(z)
    thread_temporal = alloc_thread_temporal(z)
    zz_loss = Vector{T}(undef, length(compact_W))
    x_loss = Vector{T}(undef, problem.N)
    dz_zz = [similar_named_tuple(z) for _ ∈ 1:Threads.nthreads()]
    dz_x = [similar_named_tuple(z) for _ ∈ 1:Threads.nthreads()]

    l = Vector{T}(undef, 1)
    dz = similar_named_tuple(z)

    state_tree = Optimisers.setup(opt, z)

    save_energy && (energy = Vector{T}(undef, iterations))

    initial_time = time()

    pbar = Progress(iterations; showspeed = true, enabled = progressbar)
    for it ∈ 1:iterations
        time = (it - 1) / (iterations - 1)
        for _ ∈ 1:inner_iterations
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
                z_multithread_params,
                compact_W,
                problem.c,
                zz_loss,
                x_loss,
                dz_zz,
                dz_x,
            )
            state_tree, z = Optimisers.update(state_tree, z, dz)
        end

        save_energy && (energy[it] = l[1])
        next!(pbar)
    end

    delta_time = time() - initial_time

    metadata = (runtime = delta_time,)
    save_params && (metadata = merge(metadata, (params = z,)))
    save_energy && (metadata = merge(metadata, (energy = energy,)))

    if isa(rounding, RoundingMethod)
        return add_solution(
            problem,
            round_configuration(
                problem,
                z,
                rounding,
                trig,
                temporal,
                thread_temporal,
                x_multithread_params,
            ),
            solver;
            metadata...,
        )
    else
        return map(
            round_method -> add_solution(
                problem,
                round_configuration(
                    problem,
                    z,
                    round_method,
                    trig,
                    temporal,
                    thread_temporal,
                    x_multithread_params,
                ),
                solver;
                metadata...,
            ),
            rounding,
        )
    end
end

end
