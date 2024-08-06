# [Getting Started with Neuroblox](@id neuroblox_example)

This tutorial will introduce you to simulating brain dynamics using Neuroblox.

## Example 1: Creating a Simple Neural Circuit

In this example, we'll create a simple oscillating circuit using two Wilson-Cowan neural mass models [1]. The Wilson-Cowan model is one of the most influential models in computational neuroscience [2], describing the dynamics of interactions between populations of excitatory and inhibitory neurons.

### The Wilson-Cowan Model

Each Wilson-Cowan neural mass is described by the following equations:

```math
\begin{align}
\nonumber
\frac{dE}{dt} &= \frac{-E}{\tau_E} + S_E(c_{EE}E - c_{IE}I + \eta\textstyle\sum{jcn})\\[10pt]
\nonumber
\frac{dI}{dt} &= \frac{-I}{\tau_I} + S_I\left(c_{EI}E - c_{II}I\right)
\end{align}
```

where $E$ and $I$ denote the activity levels of the excitatory and inhibitory populations, respectively. The terms $\frac{dE}{dt}$ and $\frac{dI}{dt}$ describe the rate of change of these activity levels over time. The parameters $\tau_E$ and $\tau_I$ are time constants analogous to membrane time constants in single neuron models, determining how quickly the excitatory and inhibitory populations respond to changes. The coefficients $c_{EE}$ and $c_{II}$ represent self-interaction (or feedback) within excitatory and inhibitory populations, while $c_{IE}$ and $c_{EI}$ represent the cross-interactions between the two populations. The term $\eta\sum{jcn}$ represents external input to the excitatory population from other brain regions or external stimuli, with $\eta$ acting as a scaling factor. While $S_E$ and $S_I$ are sigmoid functions that represent the responses of neuronal populations to input stimuli, defined as:

```math
S_k(x) = \frac{1}{1 + exp(-a_kx - \theta_k)}
```

where $a_k$ and $\theta_k$ determine the steepness and threshold of the response, respectively.

### Building the Circuit

Let's create an oscillating circuit by connecting two Wilson-Cowan neural masses:

```@example Wilson-Cowan
using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

# Create two Wilson-Cowan blox
@named WC1 = WilsonCowan()
@named WC2 = WilsonCowan()

# Create a graph to represent our circuit
g = MetaDiGraph()
add_blox!.(Ref(g), [WC1, WC2])

# Define the connectivity between the neural masses
adj = [-1 7; 4 -1]
create_adjacency_edges!(g, adj)

```

Here, we've created two Wilson-Cowan Blox and connected them as nodes in a directed graph. The `adj` matrix defines the weighted edges between these nodes. Each entry `adj[i,j]` represents how the output of blox `j` influences the input of blox `i`:

- Diagonal elements (`adj[1,1]` and `adj[2,2]`): Self-connections, adding feedback to each blox.
- Off-diagonal elements (`adj[1,2]` and `adj[2,1]`): Inter-blox connections, determining how each blox influences the other.

By default, the output of each Wilson-Cowan blox is its excitatory activity (E). The negative self-connections (-1) provide inhibitory feedback, while the positive inter-blox connections (6) provide strong excitatory coupling. This setup creates an oscillatory dynamic between the two Wilson-Cowan units.

### Creating the Model

Now, let's build the complete model:

```@example Wilson-Cowan
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
```

This creates a differential equations system from our graph representation using ModelingToolkit and symbolically simplifies it for efficient computation.

### Simulating the Model

Finally, let's simulate our model. The following code creates and solves an `ODEProblem` for our system, simulating 100 time units of activity. In Neuroblox, the default time unit is milliseconds. We use `Rodas4`, a solver efficient for stiff problems. The solution is saved every 0.1 ms, allowing us to observe the detailed evolution of the system's behavior.

```@example Wilson-Cowan
prob = ODEProblem(sys, [], (0.0, 100), [])
sol = solve(prob, Rodas4(), saveat=0.1)
plot(sol)
```

[[1] Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical journal, 12(1), 1-24.](https://www.cell.com/biophysj/fulltext/S0006-3495(72)86068-5)

[[2] Destexhe, A., & Sejnowski, T. J. (2009). The Wilson–Cowan model, 36 years later. Biological cybernetics, 101(1), 1-2.](https://link.springer.com/article/10.1007/s00422-009-0328-3)



## Example 2 : Building a Brain Circuit from literature using Neural Mass Models

In this example, we'll construct a model of Parkinson's disease using eight Jansen-Rit Neural Mass Models, based on the work of Liu et al. (2020) [1].

### The Jansen-Rit Neural Mass Model

The Jansen-Rit model [2] is another popular neural mass model that, like the Wilson-Cowan model from [Example 1](#example-1-creating-a-simple-neural-circuit), describes the average activity of neural populations. Each Jansen-Rit unit is defined by the following differential equations:

```math
\begin{align}
\frac{dx}{dt} &= y-\frac{2}{\tau}x \\[10pt]
\frac{dy}{dt} &= -\frac{x}{\tau^2} + \frac{H}{\tau} \left[2\lambda S(\textstyle\sum{jcn}) - \lambda\right]
\end{align}
```

where $x$ represents the average postsynaptic membrane potential of the neural population, $y$ is an auxiliary variable, $\tau$ is the membrane time constant, $H$ is the maximum postsynaptic potential amplitude, $\lambda$ determines the maximum firing rate, and $\sum{jcn}$ represents the sum of all synaptic inputs to the population. The sigmoid function $S(x)$ models the population's firing rate response to input and is defined as:

```math
S(x) = \frac{1}{1 + \text{exp}(-rx)}
```

where $r$ controls the steepness of the sigmoid, affecting the population's sensitivity to input.

### Setting Up the Model

Let's start by importing the necessary libraries and defining our neural masses:

```@example Jansen-Rit
using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

# Convert time units from seconds to milliseconds
τ_factor = 1000

# Define Jansen-Rit neural masses for different brain regions
@named Str = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3)
@named GPE = JansenRit(τ=0.04*τ_factor, cortical=false)  # all default subcortical except τ
@named STN = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=500, r=0.1)
@named GPI = JansenRit(cortical=false)  # default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=0.002*τ_factor, H=10/τ_factor, λ=20, r=5)
@named EI  = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=5, r=5)
@named PY  = JansenRit(cortical=true)  # default parameters cortical Jansen Rit blox
@named II  = JansenRit(τ=2.0*τ_factor, H=60/τ_factor, λ=5, r=5)

blox = [Str, GPE, STN, GPI, Th, EI, PY, II]
```

Here, we've created eight Jansen-Rit neural masses representing different brain regions involved in Parkinson's disease. The `τ_factor` is used to convert time units from seconds (as in the original paper) to milliseconds (Neuroblox's default time unit).

### Building the Circuit

Now, let's create a graph representing our brain circuit:

```@example Jansen-Rit
g = MetaDiGraph()
add_blox!.(Ref(g), blox)
```

We've created a MetaDiGraph and added our neural masses as nodes. Next, we'll define the connections between these nodes based on the known anatomy of the basal ganglia-thalamocortical circuit.

ModelingToolkit allows us to create parameters that can be passed into the equations symbolically:

```@example Jansen-Rit
# Define connection strength parameters
params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5
```

As an alternative to creating edges with an adjacency matrix (as shown in the previous example), here we demonstrate a different approach by adding edges one by one. In this case, we set the connections specified in Table 2 of Liu et al. [1], although we only implement a subset of the nodes and edges to describe a simplified version of the model:

```@example Jansen-Rit
# Add connections between brain regions
add_edge!(g, 2, 1, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 2, 2, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 2, 3, Dict(:weight => C_BG_Th))
add_edge!(g, 3, 2, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 3, 7, Dict(:weight => C_Cor_BG_Th))
add_edge!(g, 4, 2, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 4, 3, Dict(:weight => C_BG_Th))
add_edge!(g, 5, 4, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 6, 5, Dict(:weight => C_BG_Th_Cor))
add_edge!(g, 6, 7, Dict(:weight => 6*C_Cor))
add_edge!(g, 7, 6, Dict(:weight => 4.8*C_Cor))
add_edge!(g, 7, 8, Dict(:weight => -1.5*C_Cor))
add_edge!(g, 8, 7, Dict(:weight => 1.5*C_Cor))
add_edge!(g, 8, 8, Dict(:weight => 3.3*C_Cor))
add_edge!(g,1,1,:weight, -0.5*C_BG_Th)
add_edge!(g,1,2,:weight, C_BG_Th)
add_edge!(g,2,1,:weight, -0.5*C_BG_Th)
add_edge!(g,2,5,:weight, C_Cor_BG_Th)
add_edge!(g,3,1,:weight, -0.5*C_BG_Th)
add_edge!(g,3,2,:weight, C_BG_Th)
add_edge!(g,4,3,:weight, -0.5*C_BG_Th)
add_edge!(g,4,4,:weight, C_BG_Th_Cor)
```

### Creating the Model

Let's build the complete model:

```@example Jansen-Rit
@named final_system = system_from_graph(g)
final_system_sys = structural_simplify(final_system)
```

This creates a differential equations system from our graph representation using ModelingToolkit and symbolically simplifies it for efficient computation.

### Simulating the Model

Lastly, we create the `ODEProblem` for our system, select an algorithm, in this case `Tsit5()`, and simulate 1 second of brain activity.

```@example Jansen-Rit
sim_dur = 1000.0 # Simulate for 1 second
prob = ODEProblem(final_system_sys, [], (0.0, sim_dur))
sol = solve(prob, Tsit5(), saveat=1)
plot(sol)
```

[[1] Liu, C., Zhou, C., Wang, J., Fietkiewicz, C., & Loparo, K. A. (2020). The role of coupling connections in a model of the cortico-basal ganglia-thalamocortical neural loop for the generation of beta oscillations. Neural Networks, 123, 381-392.](https://doi.org/10.1016/j.neunet.2019.12.021)