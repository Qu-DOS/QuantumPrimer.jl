[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Qu-DOS.github.io/QComp.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Qu-DOS.github.io/QComp.jl/dev/)
[![Build Status](https://github.com/Qu-DOS/QComp.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Qu-DOS/QComp.jl/actions/workflows/CI.yml?query=branch%3Amain)

# QComp.jl
A library to collect and standardize QML routines.

# Run minimal working examples
Find Julia notebooks of minimal working examples in the folder `notebooks`.

<!-- ## Online documentation -->
<!-- Check the online documentation at <a href="https://Qu-DOS.github.io/QComp.jl/dev/">this link</a>. -->

# Contributions
1. Open a branch with a name suggesting the feature that is being implemented, e.g. "qcnn-invariant"
2. Work on the branch to implement the feature
3. Once the feature is complete, open a pull request to merge with main and WAIT for someone to check the pull request
4. Do not enforce any merge, ever!

## Repo structure
* **.github/workflows**: contains the yml file to build the documentation and commit on the gh-pages branch
* **docs**: contains make.jl and index.md for the generation of documentation
* **runs**: contains run files which can be used as a template to run the code
* **notebooks**: contains Julia notebooks with minimal working examples and showcasing of applications
* **src**: contains the source code