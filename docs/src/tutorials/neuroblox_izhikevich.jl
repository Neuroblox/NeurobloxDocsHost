using Pkg

Pkg.add("PkgAuthentication")

using PkgAuthentication

PkgAuthentication.install("juliahub.com")
Pkg.Registry.add()
Pkg.add("Neuroblox")

using Neuroblox, ModelingToolkit

@named izh1 = IzhikevichNeuron()
@named izh2 = IzhikevichNeuron()

using MetaGraphs
graph = MetaDiGraph()

add_blox!.(Ref(graph), [izh1, izh2])
add_edge!(graph, 1, 2, Dict(:weight => 1.0, :connection_rule => "basic"))
add_edge!(graph, 1, 2, Dict(:weight => -0.5, :connection_rule => "basic"))

@named sys = system_from_graph(graph)
simplified_sys = structural_simplify(sys)

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