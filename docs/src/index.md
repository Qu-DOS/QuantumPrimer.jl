# QuantumPrimer Documentation
Welcome to the documentation for the `QuantumPrimer` module.

## Overview
`QuantumPrimer` is a Julia module for quantum computing and quantum machine learning tasks. Mainly based on Yao.jl to provide a comprehensive toolkit for building quantum neural networks and various quantum algorithms.

## Dependencies
`QuantumPrimer` relies on the following Julia packages:

- `Yao`
- `Random`
- `Optimisers`
- `LinearAlgebra`
- `Statistics`
- `Combinatorics`
- `ForwardDiff`
- `Kronecker`
- `Graphs`

## Modules
- `QNN`: Implements Quantum Neural Networks.
- `QCNN`: Implements Quantum Convolutional Neural Networks.
- `Data`: Handles data preparation and manipulation for quantum machine learning tasks.
- `Model`: Defines models and architectures for quantum machine learning.
- `Loss`: Contains loss functions used in training quantum models.
- `Gradient`: Implements gradient calculation methods for quantum models.
- `Cost`: Provides cost functions for evaluating quantum models.
- `Circuit`: Provides functionalities for building and manipulating quantum circuits.
- `Differencing`: Provides methods for numerical differentiation.
- `Activation`: Contains various activation functions used in quantum neural networks.
- `TrainTest`: Handles training and testing of quantum models.
- `QSP`: Implements Quantum Signal Processing algorithms.
- `Graph`: Implements graph-based algorithms and structures.
- `Utils`: Contains utility functions and helpers.

## Usage
To use the `QuantumPrimer` package, include it in your Julia script:
```julia
include("../src/QuantumPrimer.jl")
using .QuantumPrimer
```