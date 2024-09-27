using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using CairoMakie
# using Statistics
# using Plots
# using DSP
# using Peaks
# using SparseArrays
# using Random


# utility function to include in Neuroblox
get_system(blox) = blox.odesystem
# get_system(blox::AbstractBlox) = blox.odesystem


# Isolated MSN network in baseline condition
@named msn = Striatum_MSN_Adam();
sys = get_system(msn);
sys = structural_simplify(sys)

# check all the 100 neurons, each with its associated associated currents
unknowns(sys)
prob = SDEProblem(sys, [], (0.0, 5500.0), [])
sol = solve(prob, RKMil(); dt=0.05, saveat=0.05)

# plot voltage of a single neuron
plot(sol, idxs=1, axis = (xlabel = "time (ms)", ylabel = "membrane potential (mV)"))

# plot mean field
meanfield(msn, sol, axis = (xlabel = "time (ms)", ylabel = "membrane potential (mV)", title = "Mean Field"))

# get mean firing rate
spikes = detect_spikes(msn, sol; threshold=-55)
t, fr = mean_firing_rate(spikes, sol)

# raster plot
rasterplot(fig[1,3], msn, sol, threshold = -55.0, axis = (; title = "Neuron's Spikes - Mean Firing Rate: $(fr[1]) spikes/s"))

# power spectrum of the GABAa current

                        
fig = Figure(resolution = (1500, 600))
powerspectrumplot(fig[1,1], msn, sol; state = "G",
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 4),
                        beta_label_position = (22, 4),
                        gamma_label_position = (60, 4),
                        axis = (; title = "Periodogram with no window"))


powerspectrumplot(fig[1,2], msn, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 4),
                        beta_label_position = (22, 4),
                        gamma_label_position = (60, 4),
                        axis = (; title = "Welch's method with no Hanning window"))


# ens_prob = EnsembleProblem(prob)
# ens_sol = solve(ens_prob, RKMil(); dt=0.05, saveat=0.05, trajectories=5)

global_ns = :g
# Basal ganglia model in baseline condition
@named msn = Striatum_MSN_Adam(namespace=global_ns)
@named fsi = Striatum_FSI_Adam(namespace=global_ns)
@named gpe = GPe_Adam(namespace=global_ns)
@named stn = STN_Adam(namespace=global_ns)


assembly = [msn, fsi, gpe, stn]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)
add_edge!(g, 1, 3, Dict(:weight=> 2.5/33, :density=>0.33))
add_edge!(g, 2, 1, Dict(:weight=> 0.6/7.5, :density=>0.15))
add_edge!(g, 3, 4, Dict(:weight=> 0.3/4, :density=>0.05))
add_edge!(g, 4, 2, Dict(:weight=> 0.165/4, :density=>0.1))
# the fractions above represent ḡ_inh/number of presynaptic neurons

@named neuron_net = system_from_graph(g)
sys = structural_simplify(neuron_net)
prob = SDEProblem(sys, [], (0.0, 5500.0), [])
sol = solve(prob, RKMil(); dt=0.05, saveat=0.05)



fig = Figure(resolution = (1500, 600))
powerspectrumplot(fig[1,1], msn, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3),
                        axis = (; title = "MSN"))


powerspectrumplot(fig[1,2], fsi, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3),
                        axis = (; title = "FSI"))

powerspectrumplot(fig[1,3], gpe, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3),
                        axis = (; title = "GPe"))


powerspectrumplot(fig[1,4], stn, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3),
                        axis = (; title = "STN"))


fig




# Parkinson's condition

@named msn = Striatum_MSN_Adam(namespace=global_ns, I_bg = 1.2519*ones(100), G_M = 1.2)
@named fsi = Striatum_FSI_Adam(namespace=global_ns, I_bg = 4.511*ones(50), weight = 0.2, g_weight = 0.075)

assembly = [msn, fsi, gpe, stn]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)
ḡ_inh = 0.48
add_edge!(g, 2, 1, Dict(:weight=> ḡ_inh/7.5, :density=>0.15))
add_edge!(g, 1, 3, Dict(:weight=> 2.5/33, :density=>0.33))
add_edge!(g, 3, 4, Dict(:weight=> 0.3/4, :density=>0.05))
add_edge!(g, 4, 2, Dict(:weight=> 0.165/4, :density=>0.1))
# the fractions above represent ḡ_inh/number of presynaptic neurons

@named neuron_net = system_from_graph(g)
sys = structural_simplify(neuron_net)
prob = SDEProblem(sys, [], (0.0, 5500.0), [])
sol = solve(prob, RKMil(); dt=0.05, saveat=0.05)



powerspectrumplot(fig[2,1], msn, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3),
                        axis = (; title = "MSN"))


powerspectrumplot(fig[2,2], fsi, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3),
                        axis = (; title = "FSI"))

powerspectrumplot(fig[2,3], gpe, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3),
                        axis = (; title = "GPe"))


powerspectrumplot(fig[2,4], stn, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3),
                        axis = (; title = "STN"))


fig