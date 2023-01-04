using Distributions
using LinearAlgebra
using MaximumLikelihoodUtils

df = 3.0;
dim = 20;
μ_true = randn(dim);
S = randn(dim, dim);
Σ_true = S * S' + 1e-6I;
d_true = MvTDist(df, μ_true, Σ_true);
N = 200;
X = rand(d_true, N);

df_guess = 10.0;

d_em = fit_mle_tdist_em(X; df=df_guess, dims=2, verbose=true)

d_gd = fit_mle_tdist_graddesc(X, df_guess; dims=2, verbose=true)
