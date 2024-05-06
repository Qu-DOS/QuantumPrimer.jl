using QComp
using Documenter

DocMeta.setdocmeta!(QComp, :DocTestSetup, :(using QComp); recursive=true)

makedocs(;
    modules=[QComp],
    authors="Stefano Scali <scali.stefano@gmail.com> and contributors",
    sitename="QComp.jl",
    format=Documenter.HTML(;
        canonical="https://Qu-DOS.github.io/QComp.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Qu-DOS/QComp.jl",
    devbranch="main",
)
