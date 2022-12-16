module MaximumLikelihoodUtils

using Distributions
using Distributions: Hermitian, PDMat, BLAS, invquad
using DistributionsAD
using KernelFunctions: RowVecs, ColVecs  # TODO: remove dependency on KernelFunctions
using LinearAlgebra
using LineSearches
using Optim
using ParameterHandling
using Printf
using Zygote

export fit_mle_tdist_em, fit_mle_tdist_graddesc


function mvt_log_lik(df, loc, scale, X)
    """ evaluate the log likelihood of the data under a multivariate-t distribution """
    d = MvTDist(df, vec(loc), scale)
    return mean(logpdf(d, X))
end


include("expectation_maximisation.jl")
include("gradient_descent.jl")

end
