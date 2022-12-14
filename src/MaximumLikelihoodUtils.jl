module MaximumLikelihoodUtils

using Distributions
using Distributions: Hermitian, PDMat, BLAS, invquad
using KernelFunctions: RowVecs, ColVecs  # TODO: remove dependency on KernelFunctions
using LinearAlgebra
using Printf

export fit_mle_tdist

include("expectation_maximisation.jl")

end
