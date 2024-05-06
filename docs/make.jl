using qcomp
using Documenter

DocMeta.setdocmeta!(qcomp, :DocTestSetup, :(using qcomp); recursive=true)

makedocs(;
    modules=[qcomp],
    authors="Stefano Scali <scali.stefano@gmail.com> and contributors",
    sitename="qcomp.jl",
    format=Documenter.HTML(;
        canonical="https://Qu-DOS.github.io/qcomp.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Qu-DOS/qcomp.jl",
    devbranch="main",
)
