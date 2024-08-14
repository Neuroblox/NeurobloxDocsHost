# # Introduction
# Describe what the tutorial will do, e.g. we are implementing a decision-making circuit model from Wang 2002 [1]
# Show some graph of the model (eventually this could be a GUI screenshot with the circuit, even for code tutorials).

# # Model definition
## (Mention in comment why we are using every non-Neuroblox package)
using Neuroblox
using MetaGraphs ## use its MetaGraph type to build the circuit (actually we can avoid using Metagraph explicitly, Neuroblox can @reexport relevant structs)
using OrdinaryDiffEq ## to build the ODE problem and solve it, gain access to multiple solvers from this (AFAIK this is lighter than using DifferentialEquations)
using Distributions ## for statistical distributions 
using Plots ## for displaying plots (plotting recipes will be internal to Neuroblox in an extension and load when CairoMakie or Plots is imported by the user)

## Describe what the local variables you define are for
global_ns = :g ## global name for the circuit. All components should be inside this namespace.

tspan = (0, 1000) ## Simulation time span [ms]
spike_rate = 2.4 ## spikes / ms

f = 0.15 ## ratio of selective excitatory to non-selective excitatory neurons
N_E = 24 ## total number of excitatory neurons
N_I = Int(ceil(N_E / 4)) ## total number of inhibitory neurons
N_E_selective = Int(ceil(f * N_E)) ## number of selective excitatory neurons
N_E_nonselective = N_E - 2 * N_E_selective ## number of non-selective excitatory neurons

w₊ = 1.7 
w₋ = 1 - f * (w₊ - 1) / (1 - f)

## Use scaling factors for conductance parameters so that our abbreviated model 
## can exhibit the same competition behavior between the two selective excitatory populations
## as the larger model in Wang 2002 does.
exci_scaling_factor = 1600 / N_E
inh_scaling_factor = 400 / N_I

coherence = 0 # random dot motion coherence [%]
dt_spike_rate = 50 # update interval for the stimulus spike rate [ms]
μ_0 = 40e-3 # mean stimulus spike rate [spikes / ms]
ρ_A = ρ_B = μ_0 / 100
μ_A = μ_0 + ρ_A * coherence
μ_B = μ_0 + ρ_B * coherence 
σ = 4e-3 # standard deviation of stimulus spike rate [spikes / ms]

spike_rate_A = Normal(μ_A, σ) => dt_spike_rate # spike rate distribution for selective population A
spike_rate_B = Normal(μ_B, σ) => dt_spike_rate # spike rate distribution for selective population B

# Blox definitions
@named background_input = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns, N_trains=1);

@named stim_A = PoissonSpikeTrain(spike_rate_A, tspan; namespace = global_ns);
@named stim_B = PoissonSpikeTrain(spike_rate_B, tspan; namespace = global_ns);

@named n_A = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor);
@named n_B = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor) ;
@named n_ns = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_nonselective, weight = 1.0, exci_scaling_factor, inh_scaling_factor);
@named n_inh = LIFInhCircuitBlox(; namespace = global_ns, N_neurons = N_I, weight = 1.0, exci_scaling_factor, inh_scaling_factor);

bloxs = [background_input, stim_A, stim_B, n_A, n_B, n_ns, n_inh];
## This is a convenience step so that we can later add edges to the graph using the Blox names.
## (We should replace add_edge! with a nicer interface to avoid this eventually)
d = Dict(b => i for (i,b) in enumerate(bloxs)); 

g = MetaDiGraph()
add_blox!.(Ref(g), bloxs)

add_edge!(g, d[background_input], d[n_A], Dict(:weight => 1))
add_edge!(g, d[background_input], d[n_B], Dict(:weight => 1))
add_edge!(g, d[background_input], d[n_ns], Dict(:weight => 1))
add_edge!(g, d[background_input], d[n_inh], Dict(:weight => 1))

add_edge!(g, d[stim_A], d[n_A], Dict(:weight => 1))
add_edge!(g, d[stim_B], d[n_B], Dict(:weight => 1))

add_edge!(g, d[n_A], d[n_B], Dict(:weight => w₋))
add_edge!(g, d[n_A], d[n_ns], Dict(:weight => 1))
add_edge!(g, d[n_A], d[n_inh], Dict(:weight => 1))

add_edge!(g, d[n_B], d[n_A], Dict(:weight => w₋))
add_edge!(g, d[n_B], d[n_ns], Dict(:weight => 1))
add_edge!(g, d[n_B], d[n_inh], Dict(:weight => 1))

add_edge!(g, d[n_ns], d[n_A], Dict(:weight => w₋))
add_edge!(g, d[n_ns], d[n_B], Dict(:weight => w₋))
add_edge!(g, d[n_ns], d[n_inh ], Dict(:weight => 1))

add_edge!(g, d[n_inh], d[n_A], Dict(:weight => 1))
add_edge!(g, d[n_inh], d[n_B], Dict(:weight => 1))
add_edge!(g, d[n_inh], d[n_ns], Dict(:weight => 1))

## Build the ODE system from the model graph
@named sys = system_from_graph(g);
## Simplify the ODE system
sys_simpl = structural_simplify(sys);
## Build an ODE Problem object out of the system
prob = ODEProblem(sys_simpl, [], tspan);
## Solve the ODE Problem by choosing a solver
sol = solve(prob, Vern7(lazy=false))

# # Simulation results
# (Such plotting commands will be replaced with recipes)
plot(sol[1:end]; idxs = [n_A.odesystem.neuron1.V])
plot(sol[1:end]; idxs = [n_B.odesystem.neuron1.V])
plot(sol[1:end]; idxs = [n_inh.odesystem.neuron1.V])
# Briefly describe the results in a paragraph.
