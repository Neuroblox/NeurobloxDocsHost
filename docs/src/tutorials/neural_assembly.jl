# #Tutorial on bottom-up approach to building a neural assembly


# ### Single spiking neuron from Hodgkin-Huxley model

# Hodgkin-Huxley (HH) formalism to describe membrane potential of a single neuron

 #md ```math
 #md   \begin{align}
 #md   C_m\frac{dV}{dt} &= -g_L(V-V_L) - {g}_{Na}m^3h(V-V_{Na}) -{g}_Kn^4(V-V_K) + I_{in} - I_{syn} \\
 #md   \frac{dm}{dt} &= \alpha_{m}(V)(1-m) + \beta_{m}(V)m \\ 
 #md   \frac{dh}{dt} &= \alpha_{h}(V)(1-h) + \beta_{h}(V)h \\
 #md   \frac{dn}{dt} &= \alpha_{n}(V)(1-n) + \beta_{n}(V)n 
 #md   \end{align}
 #md ```

# A single HH neuron can be simulated as follows

#md ```@example neural_assembly
using Neuroblox
using MetaGraphs
using DifferentialEquations
using Random
using Plots
using Statistics

# define a single excitatory neuron 'blox' with steady input current I_in = 0.5 microA/cm2
nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_bg=0.5)

# add the single neuron 'blox' as a single node in a graph
g = MetaDiGraph()
add_blox!.(Ref(g), [nn1])
# md ```


#md ```@example neural_assembly
# create an ODESystem from the graph
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
length(unknowns(sys))
# md```

# To solve the system, we first create an Ordinary Differential Equation Problem and then solve it over the tspan of (0,1e) using a Vern7() solver.  The solution is saved every 0.1ms. The unit of time in Neuroblox is 1ms.

#md ```@example neural_assembly
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)
plot(sol.t,sol[1,:],xlims=(0,1000),xlabel="time (ms)",ylabel="mV")
# ```


# ### Connecting three neurons through synapses to make a small local circuit

# While creating a system of multiple components (neurons in this case), each component should be defined within the same namespace. So first
# we define a global namespace.
global_namespace=:g

# define three neurons, two excitatory and one inhibitory 

nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_bg=0.4,namespace=global_namespace)
nn2 = HHNeuronInhibBlox(name=Symbol("nrn2"), I_bg=0.1,namespace=global_namespace)
nn3 = HHNeuronExciBlox(name=Symbol("nrn3"), I_bg=1.4,namespace=global_namespace)
assembly = [nn1,nn2,nn3] 

# add the three neurons as nodes in a graph
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)

# connect the nodes with the edges (synapses in this case), with the synaptic weights specified as arguments
add_edge!(g, 1, 2, Dict(:weight => 1)) #connection from node 1 to node 2 (nn1 to nn2)
add_edge!(g, 2, 3, Dict(:weight => 0.2)) #connection from node 2 to node 3 (nn2 to nn3)

# create an ODESystem from the graph and then solve it using an ODE solver
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)

# plotting membrane voltage activity of all neurons
ss=convert(Array,sol) 
st=unknowns(sys)    #collecting all the state variables from the whole ODEsystem
vlist=Int64[] 
for ii = 1:length(st)     
    if contains(string(st[ii]), "V(t)")   #extracting only voltage variables
            push!(vlist,ii)
    end
end

# creating a matrix where every row represents the voltage time series of a single neuron.
V = ss[vlist,:]
	
# creating a matrix where every row represents the voltage time series of a single neuron, with an added constant. 
# This makes all the time timeseries stacked up in the plot.

V_stack=zeros(length(vlist),length(sol.t))  
for ii = 1:length(vlist)
    V_stack[ii,:] .= V[ii,:] .+ 200*(ii-1)   
end
plot(sol.t,V_stack[:,:]',color= "blue",label=false)


# ### Creating a lateral inhibition circuit (the "winner-take-all" circuit) in superficial cortical layer


N_exci = 5;

n_inh = HHNeuronInhibBlox(name = "inh", namespace=global_namespace, G_syn = 4.0, τ = 70) 
n_excis=[]
for i = 1:N_exci
    n_exci = HHNeuronExciBlox(
        name = Symbol("exci$i"),
        namespace=global_namespace, 
        G_syn = 3.0, 
        τ = 5,
        I_bg = 5*rand(), # behave differently if I_bg is array
    ) 

    push!(n_excis,n_exci)
end


g = MetaDiGraph()
add_blox!(g, n_inh)
for i in Base.OneTo(N_exci)
    add_blox!(g, n_excis[i])
    add_edge!(g, 1, i+1, :weight, 1.0)
    add_edge!(g, i+1, 1, :weight, 1.0)
end

@named sys = system_from_graph(g)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)

# plotting membrane voltage activity of all neurons
ss=convert(Array,sol) 
st=unknowns(sys)    #collecting all the state variables from the whole ODEsystem
vlist=Int64[] 
for ii = 1:length(st)     
    if contains(string(st[ii]), "V(t)")   #extracting only voltage variables
            push!(vlist,ii)
    end
end

# creating a matrix where every row represents the voltage time series of a single neuron.
V = ss[vlist,:]
	
# creating a matrix where every row represents the voltage time series of a single neuron, with an added constant. 
# This makes all the time timeseries stacked up in the plot.

V_stack=zeros(length(vlist),length(sol.t))  
for ii = 1:length(vlist)
    V_stack[ii,:] .= V[ii,:] .+ 200*(ii-1)   
end
plot(sol.t,V_stack[2:end,:]',color= "blue",label=false);
plot!(sol.t,V_stack[1,:],color= "red",label=false)