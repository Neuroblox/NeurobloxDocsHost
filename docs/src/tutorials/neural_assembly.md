# [Tutorial on building a neural assembly from bottom-up](@id neural_assembly_tutorial)

## Single spiking neuron from Hodgkin-Huxley model

Hodgkin-Huxley (HH) formalism to describe membrane potential of a single neuron

```math
    \begin{align}
    C_m\frac{dV}{dt} &= -g_L(V-V_L) - {g}_{Na}m^3h(V-V_{Na}) -{g}_Kn^4(V-V_K) + I_{in} - I_{syn} \\
    \frac{dm}{dt} &= \alpha_{m}(V)(1-m) + \beta_{m}(V)m \\ 
    \frac{dh}{dt} &= \alpha_{h}(V)(1-h) + \beta_{h}(V)h \\
    \frac{dn}{dt} &= \alpha_{n}(V)(1-n) + \beta_{n}(V)n 
    \end{align}
```

A single HH neuron can be simulated as follows

```@example neural_assembly
using Neuroblox
using MetaGraphs
using DifferentialEquations
using Random
using Plots
using Statistics

#create a single excitatory neuron with steady input current I_in = 0.5 microA/cm2
nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_bg=0.5)

#add a single neuron blox as a single node in a graph
g = MetaDiGraph()
add_blox!.(Ref(g), [nn1])
```

```@example neural_assembly
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
length(unknowns(sys))
```

To solve the system, we first create an Ordinary Differential Equation Problem and then solve it over the tspan of (0,1e) using a Vern7() solver.  The solution is saved every 0.1ms. The unit of time in Neuroblox is 1ms.

```@example neural_assembly
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)
plot(sol.t,sol[1,:],xlims=(0,1000),xlabel="time (ms)",ylabel="mV")
```