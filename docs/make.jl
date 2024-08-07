using QuantumPrimer
using Documenter

DocMeta.setdocmeta!(QuantumPrimer, :DocTestSetup, :(using QuantumPrimer); recursive=true)

makedocs(;
    modules=[QuantumPrimer],
    authors="Stefano Scali <scali.stefano@gmail.com> and contributors",
    sitename="QuantumPrimer.jl",
    format=Documenter.HTML(;
        canonical="https://Qu-DOS.github.io/QuantumPrimer.jl",
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "QNN" => "qnn.md",
        "QSP" => "qsp.md",
    ],
)

deploydocs(;
    repo="github.com/Qu-DOS/QuantumPrimer.jl",
    devbranch="main",
)
