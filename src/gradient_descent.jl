
function initialise_θ(df, X)
    """ Initialise the parameters using the method of moments """
    μ = mean(X, dims=2)
    Σ = positive_definite((df-2)/df * cov(X, dims=2))
    return (μ = μ, Σ = Σ)
end


function fit_mle_tdist_graddesc(X, df; dims=1, max_iters=1000, verbose=true)
    @assert dims <= ndims(X)
    @assert dims <= 2
    if ndims(X) == 1
        X = (dims == 1) ? reshape(X, length(X), 1) : reshape(X, 1, length(X))
    end
    X = (dims == 1) ? transpose(X) : X  # from here on, we use dims=2

    # Define the objective
    obj(θ) = -mvt_log_lik(df, θ.μ, θ.Σ, X)

    θ = initialise_θ(df, X)
    θ_flat, unpack = value_flatten(θ)

    # Optimise the parameters using gradient descent:
    res = optimize(
        obj ∘ unpack,
        θ -> only(Zygote.gradient(obj ∘ unpack, θ)),
        θ_flat,
        LBFGS(
            alphaguess = LineSearches.InitialStatic(scaled=true),
            linesearch = LineSearches.BackTracking(),
        ),
        Optim.Options(
            iterations = max_iters,
            show_trace = verbose
        );
        inplace=false,
    )

    θ_mle = unpack(res.minimizer)
    return MvTDist(df, vec(θ_mle.μ), θ_mle.Σ)
end
