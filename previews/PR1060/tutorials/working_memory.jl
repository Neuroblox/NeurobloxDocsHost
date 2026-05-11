# # Working Memory as Persistent Neural Activity

# ## Introduction

# Working memory is the brain's capacity to hold information "in mind" for seconds
# to minutes even after the triggering stimulus has been removed. A hallmark of
# working memory is **delay-period activity**: neurons in prefrontal and
# inferotemporal (IT) cortex maintain elevated firing rates throughout a memory
# delay, well after the sensory input has ceased (Fuster & Alexander 1971; Funahashi
# et al. 1989).

# Zipser (1991) [1] showed that a recurrent neural network (RNN) trained with
# **backpropagation through time (BPTT)** reproduces the heterogeneous delay-period
# firing patterns observed in IT cortex. The trained network stores information in
# **attractor states** — stable, self-sustaining activity patterns maintained by
# strong recurrent excitatory connections. Zipser, Kehoe, Littlewort & Fuster (1993)
# [2] extended this to spiking neurons, connecting the attractor mechanism to the
# temporal statistics of real IT recordings.

# > **Note — what is not implemented:** The full Zipser (1991) model requires training
# > a recurrent network with BPTT to learn weight matrices that support attractor states
# > for arbitrary input values. Neuroblox does not currently implement recurrent network
# > training algorithms. This tutorial focuses instead on the **dynamics** that underlie
# > working memory: the bistable attractor structure that arises in a Wilson-Cowan
# > neural mass when recurrent excitation is strong. The same mechanism is at work in
# > the trained Zipser networks.

# **In this tutorial you will learn to:**
# - Set `WilsonCowan` blox parameters to enter the bistable regime.
# - Observe two stable fixed points (spontaneous and persistent-activity states) from
#   different initial conditions.
# - Use `PulsesStimulus` to deliver a brief stimulus that loads information into the network.
# - Connect two competing `WilsonCowan` populations to show selective persistent activity.

# ## The Wilson-Cowan Model and Bistability

# A single excitatory-inhibitory (E-I) population can exhibit **bistability** —
# two stable equilibria — when recurrent self-excitation is strong. The
# `WilsonCowan` blox captures this with two coupled equations:

# ```math
# \tau_E \dot{E} = -E + \frac{1}{1 + e^{-a_E(c_{EE}E - c_{IE}I - \theta_E + \eta J_{in})}} \\
# \tau_I \dot{I} = -I + \frac{1}{1 + e^{-a_I(c_{EI}E - c_{II}I - \theta_I)}}
# ```

# When $c_{EE}$ is large enough, the E-nullcline folds back on itself, creating three
# intersections with the I-nullcline: two stable fixed points (low-activity and
# high-activity) separated by an unstable saddle. A brief input can push the system
# from the low state to the high state; the high state then persists indefinitely
# without further input — this is the computational signature of working memory.

# ## Setup

using Neuroblox
using OrdinaryDiffEqTsit5
using CairoMakie
using Test, ReferenceTests # hide

# ## Bistability: Two Stable Fixed Points

# We choose parameters that place the Wilson-Cowan model firmly in the bistable regime.
# The key requirement is `c_EE * a_E > 4` (the E-nullcline must fold), together with
# inhibitory parameters that constrain the high-activity state to a physiological range.

@graph g begin
    @nodes wc = WilsonCowan(;
        c_EE = 16.0, ## strong recurrent E→E excitation: creates the nullcline fold
        c_EI = 10.0, ## E→I drive: activates inhibition at high E
        c_IE = 6.0,  ## I→E suppression: limits the high-activity state
        c_II = 1.0,  ## weak I→I self-inhibition
        θ_E  = 3.5,  ## excitatory threshold: keeps the spontaneous state near zero
        θ_I  = 4.0,  ## inhibitory threshold
        a_E  = 1.2,  ## default excitatory sigmoid slope
        a_I  = 2.0   ## default inhibitory sigmoid slope
    )
end

tspan = (0.0, 200.0)  ## 200 ms

## Spontaneous (low-activity) initial condition — below the separatrix
prob_low  = ODEProblem(g, [wc.E => 0.05, wc.I => 0.02], tspan, [])
sol_low   = solve(prob_low,  Tsit5())

## Memory (high-activity) initial condition — above the separatrix
prob_high = ODEProblem(g, [wc.E => 0.85, wc.I => 0.80], tspan, [])
sol_high  = solve(prob_high, Tsit5())

fig1 = Figure(size=(800, 350))
ax1  = Axis(fig1[1,1]; xlabel="Time (ms)", ylabel="E (excitatory activity)",
            title="Bistability: two stable fixed points")
lines!(ax1, sol_low.t,  state_timeseries(wc, sol_low,  "E"); color=:steelblue, label="Spontaneous (E₀ = 0.05)")
lines!(ax1, sol_high.t, state_timeseries(wc, sol_high, "E"); color=:firebrick,  label="Active / memory (E₀ = 0.85)")
hlines!(ax1, [state_timeseries(wc, sol_low,  "E")[end]]; color=:steelblue, linestyle=:dash, alpha=0.5)
hlines!(ax1, [state_timeseries(wc, sol_high, "E")[end]]; color=:firebrick,  linestyle=:dash, alpha=0.5)
axislegend(ax1; position=:rc)
fig1
@test_reference "plots/wm_bistability.png" fig1 by=psnr_equality(24) # hide

# The two traces converge to different stable values from different starting points.
# The dashed lines mark the two stable fixed points. This bistability is the
# computational basis for working memory: each stable state can represent
# "item stored" vs. "nothing stored."

# ## Memory Loading with a Brief Stimulus

# Starting from the spontaneous state, a brief current pulse can push the system
# across the separatrix (unstable saddle), after which recurrent excitation
# maintains the high-activity state indefinitely. This is the neural correlate
# of *writing* an item into working memory.

# `PulsesStimulus` delivers a square-wave pulse at user-specified times:

@graph g_mem begin
    @nodes begin
        mem  = WilsonCowan(;
            c_EE=16.0, c_EI=10.0, c_IE=6.0, c_II=1.0, θ_E=3.5, θ_I=4.0
        )
        ## A single 50 ms loading pulse starting at t = 100 ms
        stim = PulsesStimulus(; baseline=0.0, pulse_amp=3.0, t_start=[100.0], pulse_width=50.0)
    end
    @connections begin
        stim => mem, (weight = 1.0)
    end
end

# > **Julia tip — named tuples in `@connections`:** The `(weight = 1.0)` syntax
# > creates a named tuple that sets connection parameters. More keyword–value pairs
# > can be added (e.g. `(weight = 1.0, synapse = ...)`) to override other defaults.

tspan_mem = (0.0, 400.0)
prob_mem  = ODEProblem(g_mem, [mem.E => 0.05, mem.I => 0.02], tspan_mem, [])
sol_mem   = solve(prob_mem, Tsit5())

E_mem = state_timeseries(mem, sol_mem, "E")

fig2 = Figure(size=(800, 400))
ax2  = Axis(fig2[1,1]; xlabel="Time (ms)", ylabel="E (excitatory activity)",
            title="Memory loading: brief stimulus → persistent activity")
lines!(ax2, sol_mem.t, E_mem; color=:firebrick)
## Shade the stimulus window
stim_start, stim_end = 100.0, 150.0
vspan!(ax2, stim_start, stim_end; color=(:orange, 0.3), label="Stimulus (50 ms)")
hlines!(ax2, [state_timeseries(wc, sol_high, "E")[end]]; color=:gray, linestyle=:dash,
        label="High fixed point")
axislegend(ax2; position=:rb)
fig2
@test_reference "plots/wm_memory_loading.png" fig2 by=psnr_equality(24) # hide

# During the orange window, the stimulus raises the effective excitatory drive above
# threshold. The system transitions to the high-activity state. After the stimulus
# ends, recurrent excitation sustains the activity — the network "remembers" that
# the stimulus was presented.

# ## Two Competing Memory Populations

# Real working memory circuits can hold one of several items selectively. Here we
# create two Wilson-Cowan populations, **A** and **B**, with mutual inhibitory
# connections: when one is active it suppresses the other. Each population receives
# its own independent loading pulse. Whichever is loaded first "wins" and suppresses
# the other.

## Same local parameters for both populations
wc_params = (c_EE=16.0, c_EI=10.0, c_IE=6.0, c_II=1.0, θ_E=3.5, θ_I=4.0)

@graph g_comp begin
    @nodes begin
        popA = WilsonCowan(; wc_params...)
        popB = WilsonCowan(; wc_params...)
        ## Population A is loaded at t = 80 ms; population B is loaded at t = 280 ms
        stimA = PulsesStimulus(; baseline=0.0, pulse_amp=3.0, t_start=[80.0],  pulse_width=50.0)
        stimB = PulsesStimulus(; baseline=0.0, pulse_amp=3.0, t_start=[280.0], pulse_width=50.0)
    end
    @connections begin
        stimA => popA, (weight = 1.0)
        stimB => popB, (weight = 1.0)
        ## Mutual inhibition: each population's E output suppresses the other.
        ## A negative weight on a BasicConnection means inhibitory input to jcn.
        ## Weight magnitude must exceed pulse_amp / E_high ≈ 3.0/0.8 = 3.75 so
        ## that the winner's suppression outweighs the loser's loading pulse.
        popA  => popB, (weight = -5.0)
        popB  => popA, (weight = -5.0)
    end
end

tspan_comp = (0.0, 600.0)
prob_comp  = ODEProblem(g_comp, [popA.E => 0.05, popA.I => 0.02,
                                  popB.E => 0.05, popB.I => 0.02], tspan_comp, [])
sol_comp   = solve(prob_comp, Tsit5())

E_A = state_timeseries(popA, sol_comp, "E")
E_B = state_timeseries(popB, sol_comp, "E")

fig3 = Figure(size=(800, 400))
ax3  = Axis(fig3[1,1]; xlabel="Time (ms)", ylabel="E (excitatory activity)",
            title="Two competing memory populations (mutual inhibition)")
lines!(ax3, sol_comp.t, E_A; color=:steelblue, label="Population A")
lines!(ax3, sol_comp.t, E_B; color=:firebrick,  label="Population B")
## Shade stimulus windows
vspan!(ax3,  80.0, 130.0; color=(:steelblue, 0.2), label="Stimulus A")
vspan!(ax3, 280.0, 330.0; color=(:firebrick,  0.2), label="Stimulus B")
axislegend(ax3; position=:rc)
fig3
@test_reference "plots/wm_competing_populations.png" fig3 by=psnr_equality(24) # hide

# Population A is loaded first (t = 80–130 ms) and maintains persistent activity.
# Its inhibitory output onto B prevents B from being loaded even when stimulus B
# arrives at t = 280 ms. This **winner-takes-all** competition is the neural
# correlate of selective working memory: only one item can be held at a time.

# > **Exercise:** Vary the amplitude of stimulus B (`pulse_amp` in `stimB`). How
# > strong does it need to be to overcome the inhibition from A and flip the memory?
# > This is an analogue of **memory updating** under strong new evidence.

# ## Summary

# This tutorial demonstrated:
# - How strong recurrent excitation (`c_EE = 16`) creates two stable fixed points in a
#   Wilson-Cowan E-I population.
# - How a brief `PulsesStimulus` stimulus loads information into the high-activity attractor.
# - How mutual inhibition between two populations implements a winner-takes-all
#   competition for working memory.

# The trained Zipser (1991) RNN produces these same attractor properties automatically
# through BPTT: its learned weights create multiple stable high-activity states, one per
# possible memorized value. Implementing RNN training in Neuroblox is a natural next step.

# ## References
# - [1] Zipser D. (1991). Recurrent network model of the neural mechanism of short-term
#   active memory. *Neural Computation*, 3(2):179–193.
#   https://doi.org/10.1162/neco.1991.3.2.179
# - [2] Zipser D, Kehoe B, Littlewort G, Fuster J. (1993). A spiking network model of
#   short-term active memory. *Journal of Neuroscience*, 13(8):3406–3420.
#   https://doi.org/10.1523/JNEUROSCI.13-08-03406.1993
# - [3] Wilson HR, Cowan JD. (1972). Excitatory and inhibitory interactions in localized
#   populations of model neurons. *Biophysical Journal*, 12(1):1–24.
