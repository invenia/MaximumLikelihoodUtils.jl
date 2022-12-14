using MaximumLikelihoodUtils
using Documenter

DocMeta.setdocmeta!(MaximumLikelihoodUtils, :DocTestSetup, :(using MaximumLikelihoodUtils); recursive=true)

makedocs(;
    modules=[MaximumLikelihoodUtils],
    authors="Invenia Technical Computing Corporation",
    repo="https://github.com/invenia/MaximumLikelihoodUtils.jl/blob/{commit}{path}#{line}",
    sitename="MaximumLikelihoodUtils.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://invenia.github.io/MaximumLikelihoodUtils.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    checkdocs=:exports,
    strict=true,
)

deploydocs(;
    repo="github.com/invenia/MaximumLikelihoodUtils.jl",
    devbranch="main",
)
