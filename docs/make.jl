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
        "QCNN" => "qcnn.md",
        "Data" => "data.md",
        "Model" => "model.md",
        "Loss" => "loss.md",
        "Gradient" => "gradient.md",
        "Cost" => "cost.md",
        "Circuit" => "circuit.md",
        "Differencing" => "differencing.md",
        "Activation" => "activation.md",
        "Train and Test" => "traintest.md",
        "QSP" => "qsp.md",
        "Graph" => "graph.md",
        "Utils" => "utils.md"
    ],
)

deploydocs(;
    repo="github.com/Qu-DOS/QuantumPrimer.jl",
    devbranch="main",
)
