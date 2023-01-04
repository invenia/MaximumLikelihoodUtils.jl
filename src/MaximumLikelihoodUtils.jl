module MaximumLikelihoodUtils

using Distributions
using Distributions: Hermitian, PDMat, BLAS, invquad, GenericMvTDist
using KernelFunctions: RowVecs, ColVecs  # TODO: remove dependency on KernelFunctions
using LinearAlgebra
using Printf
using Roots: find_zero, Bisection
using SpecialFunctions: digamma

export fit_mle

include("expectation_maximisation.jl")

end
