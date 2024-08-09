# Neuroblox

## About
### Overview
Neuroblox is designed for computational neuroscience and psychiatry applications. Our tools range from brain circuit simulations to control circuit system identification, bridging scales from spiking neurons to fMRI-derived circuits, interactions between the brain and other physiological systems, experimental optimization, and scientific machine learning.

### Features
Neuroblox is based on a library of modular computational building blocks (“blox”) in the form of systems of symbolic dynamic differential equations, which can be flexibly combined to describe large-scale brain dynamics. Our libraries of modular blox consist of individual neurons (Hodgkin-Huxley, IF, QIF, LIF, etc.), neural mass models (Jansen-Rit, Wilson-Cowan, Lauter-Breakspear, Next Generation, microcanonical circuits, etc.), and biomimetically-constrained control circuit elements.

Once a model is built, it can be simulated efficiently and used to fit electrophysiological and neuroimaging data. Moreover, the circuit behavior of multiple model variants can be investigated to aid in distinguishing between competing hypotheses.

### Interface
Users can interface with Neuroblox either via Julia code or using a simple drag-and-drop GUI designed to be intuitive to neuroscientists. Both interfaces allow researchers to automatically generate high-performance models from which one can run stimulations with parameters fit to experimental data.

### Performance
The Neuroblox back-end (Neuroblox.jl) is built using [Julia](https://julialang.org/), an open-source, high-level scripting language designed for high-performance in computation-intensive applications. Our benchmarks show greater than 100x increase in speed over neural mass model implementations using the Virtual Brain (Python) and similar packages in MATLAB.

### Implementation
Under the hood, we employ ModelingToolkit.jl to describe the dynamical behavior of blox as symbolic (stochastic/delay) differential equations. For parameter fitting of brain circuit dynamical models, we use Turing.jl to perform probabilistic modeling, including Hamilton-Monte-Carlo sampling and Automated Differentiation Variational Inference.

## Installation

Neuroblox requires a valid [Julia](https://julialang.org/downloads/) installation and [JuliaHub](https://juliahub.com/ui/Home) user account. Using Neuroblox via the GUI does *not* require any programming knowledge, but interested users can learn more about Julia [here](https://julialang.org/learning/).

To install Neuroblox.jl, first add the JuliaHubRegistry and then use the Julia package manager:

```julia
using Pkg
Pkg.add("PkgAuthentication")
using PkgAuthentication
PkgAuthentication.install("juliahub.com")
Pkg.Registry.add()
Pkg.add("Neuroblox")
```

## Licensing

Neuroblox is free for non-commerical and academic use. For full details of the license, please see
[the Neuroblox EULA](https://github.com/Neuroblox/NeurobloxEULA). For commercial use, get in contact
with sales@neuroblox.org.
