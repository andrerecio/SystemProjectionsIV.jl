# ============================================================================
# Utility Functions
# ============================================================================

# Source: adapted from gragusa MacroEconometricsTools.jl, vendored under
# `docs/ext/gragusa/MacroEconometricsTools.jl`. Lag, matrix, and companion-form
# helpers are reimplemented here (rather than depending on the upstream git
# package) and lightly adapted to this package's NaN-over-missing convention.

# ============================================================================
# Lag operations (replaces ShiftedArrays dependency)
# ============================================================================

"""
    lag(x::AbstractVector, n::Int; default=NaN)

Create lagged version of vector `x` by `n` periods.

# Arguments
- `x::AbstractVector`: Input vector
- `n::Int`: Number of lags (positive for lags, negative for leads)
- `default`: Value to fill for unavailable observations (default: NaN)

# Examples
```julia
x = [1, 2, 3, 4, 5]
lag(x, 1)  # [NaN, 1, 2, 3, 4]
lag(x, 2)  # [NaN, NaN, 1, 2, 3]
```
"""
function lag(x::AbstractVector{T}, n::Int; default = T(NaN)) where {T <: AbstractFloat}
    len = length(x)
    if n == 0
        return copy(x)
    elseif n > 0  # Positive lag
        result = Vector{T}(undef, len)
        result[1:n] .= default
        result[(n + 1):end] .= @view x[1:(end - n)]
        return result
    else  # Negative lag (lead)
        n_abs = abs(n)
        result = Vector{T}(undef, len)
        result[(end - n_abs + 1):end] .= default
        result[1:(end - n_abs)] .= @view x[(n_abs + 1):end]
        return result
    end
end

function lag(X::AbstractMatrix{T}, n::Int; default = T(NaN)) where {T <: AbstractFloat}
    return hcat([lag(col, n; default = default) for col in eachcol(X)]...)
end

# Data-space analogue of gragusa's `lags` formula term
# (docs/ext/gragusa/Regress.jl/src/utils/formula.jl, exported by LocalProjections.jl):
# `lags(x, n)` there expands to the columns lag(x,1), …, lag(x,n).
"""
    lags(x::AbstractVecOrMat, p::Int; default=NaN)

Matrix of the first `p` lags of `x`: columns `[lag(x,1) … lag(x,p)]`, `T × (n_vars·p)`
(lag-major column order for matrix input). The first `p` rows are padded with
`default` (`NaN`), so trim the burn-in rows from all series before passing the
result as controls to [`spiv`](@ref):

```julia
X = lags(x, p)[(p + 1):end, :]   # together with y[(p+1):end], etc.
```
"""
function lags(
        x::AbstractVecOrMat{T}, p::Int; default = T(NaN)) where {T <: AbstractFloat}
    p ≥ 1 || throw(ArgumentError("p must be ≥ 1, got $p"))
    return hcat([lag(x, n; default = default) for n in 1:p]...)
end

"""
    create_lags(X::AbstractMatrix{T}, p::Int) where T

Create matrix of lagged values for VAR estimation.

# Arguments
- `X::Matrix{T}`: Data matrix (T × n_vars)
- `p::Int`: Number of lags

# Returns
- Matrix of size (T × (1 + n_vars * p)) with intercept and lags
"""
function create_lags(X::AbstractMatrix{T}, p::Int) where {T <: AbstractFloat}
    n_obs, n_vars = size(X)
    n_cols = 1 + n_vars * p

    # Preallocate with concrete type
    lagged = Matrix{T}(undef, n_obs, n_cols)

    # Intercept
    lagged[:, 1] .= one(T)

    # Lags
    for lag_num in 1:p
        for var_idx in 1:n_vars
            col_idx = 1 + (lag_num - 1) * n_vars + var_idx
            lagged[:, col_idx] = lag(view(X, :, var_idx), lag_num; default = T(NaN))
        end
    end

    return lagged
end

"""
    create_lags!(dest::AbstractMatrix, X::AbstractMatrix, p::Int, include_intercept::Bool=true)

In-place creation of lagged matrix for VAR estimation.

# Arguments
- `dest::Matrix`: Pre-allocated destination matrix
- `X::Matrix`: Source data matrix
- `p::Int`: Number of lags
- `include_intercept::Bool`: Whether to include intercept column
"""
function create_lags!(
        dest::AbstractMatrix{T},
        X::AbstractMatrix{T},
        p::Int,
        include_intercept::Bool = true
) where {T}
    n_obs, n_vars = size(X)
    offset = include_intercept ? 1 : 0

    # Intercept
    if include_intercept
        fill!(view(dest, :, 1), one(T))
    end

    # Create lags
    for lag_num in 1:p
        for var_idx in 1:n_vars
            col_idx = offset + (lag_num - 1) * n_vars + var_idx
            for t in (lag_num + 1):n_obs
                dest[t, col_idx] = X[t - lag_num, var_idx]
            end
        end
    end

    # Set first p rows to NaN (unusable due to lags)
    dest[1:p, :] .= NaN

    return dest
end

# ============================================================================
# Matrix utilities
# ============================================================================

"""
    duplication_matrix(n::Int)

Magnus-Neudecker duplication matrix D_n that satisfies vec(A) = D_n * vech(A).
"""
function duplication_matrix(n::Int)
    n² = n * n
    n_vech = n * (n + 1) ÷ 2

    D = zeros(n², n_vech)

    vech_idx = 1
    for j in 1:n
        for i in j:n
            vec_idx_ij = (j - 1) * n + i
            vec_idx_ji = (i - 1) * n + j
            D[vec_idx_ij, vech_idx] = 1.0
            if i != j
                D[vec_idx_ji, vech_idx] = 1.0
            end
            vech_idx += 1
        end
    end

    return D
end

"""
    elimination_matrix(n::Int)

Magnus-Neudecker elimination matrix L_n that satisfies vech(A) = L_n * vec(A).
"""
function elimination_matrix(n::Int)
    n² = n * n
    n_vech = n * (n + 1) ÷ 2

    L = zeros(n_vech, n²)

    vech_idx = 1
    for j in 1:n
        for i in j:n
            vec_idx = (j - 1) * n + i
            L[vech_idx, vec_idx] = 1.0
            vech_idx += 1
        end
    end

    return L
end

"""
    commutation_matrix(m::Int, n::Int)

Commutation matrix K_{m,n} that satisfies vec(A') = K_{m,n} * vec(A) for A ∈ ℝ^{m×n}.
"""
function commutation_matrix(m::Int, n::Int)
    mn = m * n
    K = zeros(mn, mn)

    for i in 1:m
        for j in 1:n
            # Position in vec(A)
            vec_idx = (j - 1) * m + i
            # Position in vec(A')
            vec_t_idx = (i - 1) * n + j
            K[vec_t_idx, vec_idx] = 1.0
        end
    end

    return K
end

# ============================================================================
# Companion form utilities
# ============================================================================

"""
    companion_form(A::Array{T,3}) where T

Build companion form matrix from VAR lag coefficients.

# Arguments
- `A::Array{T,3}`: Lag coefficient array (n_vars, n_vars, n_lags)

# Returns
- `F::Matrix{T}`: Companion matrix (n_vars*n_lags × n_vars*n_lags)
"""
function companion_form(A::Array{T, 3}) where {T}
    n_vars, _, n_lags = size(A)
    n = n_vars * n_lags

    F = zeros(T, n, n)

    # Top block: lag coefficients
    for lag in 1:n_lags
        row_range = 1:n_vars
        col_range = ((lag - 1) * n_vars + 1):(lag * n_vars)
        F[row_range, col_range] .= view(A, :, :, lag)
    end

    # Identity blocks below
    if n_lags > 1
        for i in 1:(n_lags - 1)
            row_start = n_vars * i + 1
            row_end = n_vars * (i + 1)
            col_start = n_vars * (i - 1) + 1
            col_end = n_vars * i
            F[row_start:row_end, col_start:col_end] .= I(n_vars)
        end
    end

    return F
end
