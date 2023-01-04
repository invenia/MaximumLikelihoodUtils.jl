
function set_ω(X, df, μ, Σ)
    """ E-step """
    d = size(X, 1)
    δ = invquad(Σ, X .- μ)
    ω = (df + d) ./ (df .+ δ)
    return ω
end


function set_μ_Σ(X, ω, μ_old, Σ_old)
    """ M-step """
    p, n = size(X)
    μ = sum(ω .* ColVecs(X)) / sum(ω)
    diff = ColVecs(X) .- Ref(μ_old)
    Σ = zeros(size(Σ_old))
    for (df, w) in zip(diff, ω)
        BLAS.gemm!('N', 'T', w, df, df, 1.0, Σ)
    end
    return reshape(μ, p, 1), PDMat(Hermitian(Σ / n + 1e-8I))
end


function set_df(ω, p)
    return find_zero(df -> objective_df(df, ω, p), (0.01, 500.0), Bisection())
end


function objective_df(df, ω, p)
    return (
        -digamma(df / 2)
        + log(df / 2)
        + mean(log.(ω) .- ω)
        + 1.0
        + digamma((df + p) / 2)
        - log((df + p) / 2)
    )
end


function fit_mle_tdist_em(X; df=100.0, learn_df=true, dims=1, max_iters=1000, tol=1e-4, verbose=false)
    """
    Maximum likelihood estimation (MLE) for the multivariate-t distribution using expectation-maximisation (EM)
    Based on the MCECM algorithm given in Section 5 of https://www3.stat.sinica.edu.tw/statistica/oldpdf/a5n12.pdf
    """
    @assert dims <= ndims(X)
    @assert dims <= 2
    if ndims(X) == 1
        X = (dims == 1) ? reshape(X, length(X), 1) : reshape(X, 1, length(X))
    end
    X = (dims == 1) ? transpose(X) : X  # from here on, we use dims=2
    # initialise parameters using the "method of moments":
    _df = max(df, 2.1)
    if size(X, 1) > 1
        μ, Σ = mean(X, dims=2), PDMat(Hermitian(cov(X, dims=2, corrected=true) * (_df - 2.0) / _df + 1e-8I))
    else
        μ, σ = mean_and_std(X, corrected=true)
        μ = reshape([μ], 1, 1)  # location
        Σ = reshape([σ ^ 2 * (_df - 2.0) / _df], 1, 1)  # scale matrix
    end
    if verbose
        @printf("iter %2d, NLL: %1.4f \n", 0, -mvt_log_lik(df, μ, Σ, X))
    end
    diff = 99
    i = 1
    # run the iterative EM algorithm
    while diff > tol && i < max_iters
        μ_old, Σ_old = μ, Σ
        ω = set_ω(X, df, μ, Σ)  # E-step
        μ, Σ = set_μ_Σ(X, ω, μ, Σ)  # M-step
        if learn_df
            # ω = set_ω(X, df, μ, Σ)  # TODO: is this step needed? paper says so but doesn't seem to make a difference
            df = set_df(ω, size(X, 1))
        end
        diff = mean(abs.(μ - μ_old)) + mean(abs.(Σ - Σ_old))
        if verbose
            @printf("iter %2d, NLL: %1.4f, diff: %1.4f \n", i, -mvt_log_lik(df, μ, Σ, X), diff)
        end
        i = i + 1
    end
    return MvTDist(df, vec(μ), Σ)
end
