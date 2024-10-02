# Getting Started

# In this tutorial, you'll learn how to install and use some basic structures from `Neuroblox`.

# Installing Neuroblox

# To install `Neuroblox.jl`, we first add the `JuliaHubRegistry` and then use the Julia package manager:


using Pkg

Pkg.add("PkgAuthentication")

using PkgAuthentication

PkgAuthentication.install("juliahub.com")
Pkg.Registry.add()
Pkg.add("Neuroblox")


# How Neuroblox Works?

# When using Neuroblox, we usually don't have to model our neurons as we did in the last tutorial. In fact, Neuroblox has in-built functions that make our lives much easier, mainly when dealing with more complex system (for example, when involving several neurons).

# How to model two Izhikevich neurons

# Now that we are using `Neuroblox.jl`, we can model an Izhikevich neuron pretty straightforwardly using the `IzhikevichNeuron()` function. It can receive parameters as inputs, but if we don't have none, it'll just model a standard one. 

using Neuroblox, ModelingToolkit

@named izh1 = IzhikevichNeuron()
@named izh2 = IzhikevichNeuron()

# Now, we have two neurons and we need to connect them. For this, we'll use a library called `MetaGraphs`. We'll use it to create a graph and, with Neuroblox, we'll add `izh1` and `izh2` to it as vertices and we'll create weigheted directed edges connecting them in both directions. Here's how we do it:

using MetaGraphs
graph = MetaDiGraph()

# Easy, right? Now, we need to add the vertices to the graph. We do this using the `add_blox!.` function, from `Neuroblox` library. Here's how we use it:

add_blox!.(Ref(graph_being_modified)), [vertices_being_added])

# In this case, we would have

add_blox!.(Ref(graph), [izh1, izh2])

# Great, we have a graph with two vertices that represents two Izhikevich neurons. Now we need to connect them using the `add_edge!` function.

add_edge!(graph_being_modified, from_which_vertice, to_which_vertice, Dict(:weight => weight, :connection_rule => "basic"))

# When refering to these vertices, they'll be ordered in the order you used to indicate them when declaring the function. In this case, we have:

add_edge!(graph, 1, 2, Dict(:weight => 1.0, :connection_rule => "basic"))
add_edge!(graph, 1, 2, Dict(:weight => -0.5, :connection_rule => "basic"))

# Here, the weights are arbitrary; you can set then however fits better for you.

# Now, we are basically done. We need to transform our graph into an ODE system. In general, this would be quite tedious, since the ODE system that represents these interactions can get quite big (the ones for this specific system have 8 differential equations). 

# Fortunately, we can use the `system_from_graph()` function that does exactly what you think it does: creates a system of equations from a weighted directed graph.

@named sys = system_from_graph(graph)

# However, ODE solvers tend to be way less effective when dealing with differential-algebric systems. To make things work better, we can transform it into a regular differential system by using  `structural_simplify()`:

simplified_sys = structural_simplify(sys)

# Now the `simplified_sys` system should be a regular ODE system, which we know how to solve:

using DifferentialEquations, CairoMakie

tspan = (0.0, 100.0)

prob = ODEProblem(simplified_sys, [], tspan)
sol = solve(prob)

times = sol.t
values = sol[1, :]

fig, ax, plt = lines(times, values, color=:black, label="M.P (mV)")

ax.xlabel = "Time (ms)"
ax.ylabel = "Membrane Potential (mV)"
ax.title = "Izhikevich Neuron Model"
axislegend(ax)

display(fig)

# and here it is. You've just modelled a two neuron system!
