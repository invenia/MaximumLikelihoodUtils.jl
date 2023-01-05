using Distributions
using Distributions: PDMat
using LinearAlgebra
using MaximumLikelihoodUtils
using MaximumLikelihoodUtils: ColVecs

dim = 20;
feat_dim = 10;
# note: for polynomial regression (order > 1) the features are no longer multivariate-t
# distributed. This leads us to generally underestimate the degrees of freedom parameter.
polynomial_order = 3;
fdim = polynomial_order * feat_dim;

# feature map for polynomial regression
ϕ(x) = vcat(map(_x -> [_x^i for i in 1:polynomial_order], x)...)

df = 3.0;

# sample some feature data
μ_feat = randn(feat_dim);
S = 0.2randn(feat_dim, feat_dim);
Σ_feat = S * S' + 1e-6I;
d_feat = MvTDist(df, μ_feat, Σ_feat);
N = 1000;
X = rand(d_feat, N);  # inputs
ϕX = hcat(ϕ.(ColVecs(X))...);  # feature-mapped inputs

# build true model
A = 0.2randn(dim, fdim);
b = randn(dim);
_C = randn(dim, dim);
C = _C * _C' + 1e-6I;

# sample some observations
# d_true = MvTDist.(Ref(df), ColVecs(A * X .+ b), Ref(PDMat(C)));
d_true = MvTDist.(Ref(df), ColVecs(A * ϕX .+ b), Ref(PDMat(C)));
Y = hcat(rand.(d_true)...);

# concatenate all data (features + observations)
# D = vcat(X, Y);
D = vcat(ϕX, Y);

# compute maximum likelihood solution
d_em = fit_mle(MvTDist, D)

# solve system of linear eqns to obtain model parameters (https://www.overleaf.com/project/6343ead19cfce1b523979b2e)
A_em = d_em.Σ[fdim+1:end, 1:fdim] / d_em.Σ[1:fdim, 1:fdim]
b_em = d_em.μ[fdim+1:end] - A_em * d_em.μ[1:fdim]
C_em = d_em.Σ[fdim+1:end, fdim+1:end] - A_em * d_em.Σ[1:fdim, fdim+1:end]
