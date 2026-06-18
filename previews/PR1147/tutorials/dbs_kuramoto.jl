# # Synchronization and Deep Brain Stimulation in a Kuramoto Oscillator Network

# ## Introduction

# Deep brain stimulation (DBS) is a neurosurgical treatment for movement disorders such as Parkinson's disease.
# High-frequency electrical pulses (typically 130 Hz) are delivered through an implanted electrode, most commonly
# to the subthalamic nucleus (STN). One intriguing response is **evoked resonant neural activity (ERNA)**:
# oscillations in the local field potential that appear between DBS pulses and whose frequency and amplitude
# slowly drift over hundreds of seconds of sustained stimulation.

# In this tutorial we use the **Kuramoto model** [1] — a population of coupled phase oscillators — to explore
# how DBS affects synchrony in a neural population. This is based on the model introduced in Sermon et al. (2024) [2],
# which showed that the slow ERNA drift can be reproduced by coupling Kuramoto oscillators to a vesicle
# depletion model.

# > **Note — what is not yet implemented:** The full Sermon et al. (2024) model requires *vesicle depletion*
# > dynamics: three interacting pools of synaptic vesicles whose depletion and recovery under repeated DBS pulses
# > cause the effective coupling strength K to decrease slowly over time, explaining the long-term ERNA drift.
# > This component is **not yet implemented in Neuroblox**. The tutorial demonstrates the oscillator
# > synchronization and DBS-entrainment aspects of the model, which are already available.

# **In this tutorial you will learn to:**
# - Build a population of `KuramotoOscillator` blox and observe spontaneous synchronization.
# - Use `create_edges!` to specify all-to-all coupling.
# - Compute the **Kuramoto order parameter** $r(t) \in [0,1]$ as a measure of population synchrony.
# - Connect a `DBS` stimulus to the oscillator population and observe its effect on synchrony.

# ## The Kuramoto Model

# Each oscillator $i$ in a population of $N$ has a phase $\theta_i(t)$ and a natural angular frequency
# $\omega_i$. In Neuroblox, the dynamics are described by

# ```math
# \dot{\theta}_i = \omega_i + \frac{1}{N}\sum_{j=1}^N K_{ij}\sin(\theta_j - \theta_i) + \zeta \, \xi_i(t)
# ```

# where $K_{ij}$ is the coupling weight from oscillator $j$ to $i$, and $\xi_i(t)$ is white noise with
# amplitude $\zeta$. For all-to-all coupling with uniform strength, $K_{ij} = K/N$ for $j \neq i$.

# The degree of synchrony is captured by the **order parameter**:

# ```math
# r(t) = \left|\frac{1}{N}\sum_{j=1}^N e^{i\theta_j(t)}\right|
# ```

# $r = 1$ means fully synchronized (all phases equal), and $r = 0$ means fully incoherent.

# ## Setup

using Neuroblox
using StochasticDiffEq
using Random
using CairoMakie
using Statistics
using Test, ReferenceTests # hide

Random.seed!(42) ## Fix random seed for reproducibility

# Parameters based on Sermon et al. (2024). The natural angular frequency `ω = 249.0` (default) corresponds
# to a resonance frequency of about 40 Hz (ω ≈ 2π × 40 rad/s). The spread `σ_ω` introduces heterogeneity.

N     = 80      ## number of oscillators
K     = 0.6     ## all-to-all coupling strength: try values 0.0, 0.2, 0.6, 1.0 to explore the synchrony transition
ζ     = 0.5     ## noise amplitude
σ_ω   = 0.2     ## spread of natural frequencies (same units as ω); Kc ≈ 2σ_ω ≈ 0.4, so K = 0.6 is above the transition

## Draw individual natural frequencies from a normal distribution
ωs = 249.0 .+ σ_ω .* randn(N)

# > **Julia tip — broadcasting:** The `.+` and `.*` operators apply element-wise to arrays. So
# > `249.0 .+ σ_ω .* randn(N)` adds 249.0 to each element of the random vector.

# ## Building the Oscillator Population

# We create the population using an in-line loop inside `@graph`. The `include_noise=true` flag
# selects the stochastic (`SDE`) version of the oscillator, which requires `SDEProblem`.

@graph g begin
    @nodes blox = [KuramotoOscillator(; include_noise=true, ω=ωs[i], ζ=ζ) for i in 1:N]
end

# Now we specify the all-to-all coupling using `create_edges!`. The weight matrix has
# `K/N` off-diagonal and 0 on the diagonal (no self-coupling).

K_matrix = fill(K / N, N, N)
for i in 1:N
    K_matrix[i, i] = 0.0
end
create_edges!(g, K_matrix, blox, blox)

# ## Simulating Without DBS

tspan = (0.0, 1000.0)  ## 1 second simulation
prob  = SDEProblem(g, [], tspan, [])
## `EulerHeun()` is a fixed-step stochastic solver. The step size dt=0.1 ms is sufficient here.
sol   = solve(prob, EulerHeun(), dt=0.1, saveat=1.0);

# ## Computing the Order Parameter

# `state_timeseries(blox, sol, "θ")` returns the phase time series for each oscillator as a vector.
# We stack them into a matrix (time × N) to compute the order parameter at each time step.

θ_all   = hcat([state_timeseries(blox[i], sol, "θ") for i in 1:N]...)  ## shape: (T, N)
z       = mean(exp.(im .* θ_all); dims=2)    ## complex mean field
r_nodbs = vec(abs.(z))                        ## order parameter

fig = Figure(size=(800, 350))
ax  = Axis(fig[1,1]; xlabel="Time (ms)", ylabel="Order parameter r",
           title="Spontaneous synchronization  (K = $K,  N = $N)")
lines!(ax, sol.t, r_nodbs; color=:steelblue)
hlines!(ax, [mean(r_nodbs[end÷2:end])]; color=:black, linestyle=:dash,
        label="Mean r (steady state)")
ylims!(ax, 0, 1)
axislegend(ax; position=:rb)
fig
save(joinpath("../assets/", "dbs_no_stim.svg"), fig); # hide
@test_reference "plots/dbs_no_stim.png" fig by=psnr_equality(24) # hide

# The order parameter should rise from near 0 to a sustained value above 0.5 once K exceeds the
# critical coupling Kc ≈ 2σ_ω (the Kuramoto transition). Try setting K = 0.0 to see incoherence.

# ## Adding DBS

# `DBS` generates square-wave pulses at the specified frequency and amplitude. Neuroblox automatically
# connects it to `KuramotoOscillator` blox by shifting the instantaneous frequency of each oscillator on every pulse.

@graph g_dbs begin
    @nodes begin
        blox_dbs = [KuramotoOscillator(; include_noise=true, ω=ωs[i], ζ=ζ) for i in 1:N]
        ## High-frequency DBS at 130 Hz.  pulse_width=2.0 ms and amplitude=5.0 are chosen to
        ## demonstrate desynchronization; clinically, pulse widths are typically 0.06–0.5 ms.
        dbs = DBSStimulus(; frequency=130.0, amplitude=5.0, pulse_width=2.0)
    end
end

## Same all-to-all coupling
create_edges!(g_dbs, K_matrix, blox_dbs, blox_dbs)

## Connect DBS to every oscillator with spatially heterogeneous weights.
## A uniform DBS electric field (equal weights) would kick all phases identically,
## leaving phase differences — and therefore r — unchanged.  Heterogeneous weights
## (reflecting the uneven field spread around the electrode) give each oscillator a
## different effective frequency offset, broadening the distribution and desynchronizing
## the network when the DBS-induced spread pushes Kc above K.
dbs_weights = rand(N) .* 2   ## weights ∈ [0, 2], mean ≈ 1
for (i, osc) in enumerate(blox_dbs)
    add_connection!(g_dbs, dbs, osc, DefaultRule(weight=dbs_weights[i]))
end

prob_dbs = SDEProblem(g_dbs, [], tspan, [])
sol_dbs  = solve(prob_dbs, EulerHeun(), dt=0.1, saveat=1.0);

θ_dbs  = hcat([state_timeseries(blox_dbs[i], sol_dbs, "θ") for i in 1:N]...)
r_dbs  = vec(abs.(mean(exp.(im .* θ_dbs); dims=2)))

# ## Comparing Synchrony With and Without DBS

fig2 = Figure(size=(800, 400))
ax2  = Axis(fig2[1,1]; xlabel="Time (ms)", ylabel="Order parameter r",
            title="Effect of heterogeneous DBS (130 Hz, amplitude = 5.0) on synchrony")
lines!(ax2, sol.t,     r_nodbs; label="No DBS",  color=:steelblue)
lines!(ax2, sol_dbs.t, r_dbs;   label="With DBS (130 Hz)", color=:firebrick)
ylims!(ax2, 0, 1)
axislegend(ax2; position=:rb)
fig2
save(joinpath("../assets/", "dbs_with_stim.svg"), fig2); # hide
@test_reference "plots/dbs_with_stim.png" fig2 by=psnr_equality(24) # hide

# Because the DBS weights are heterogeneous, each oscillator receives a different frequency
# offset, effectively increasing the spread of natural frequencies.  When that spread pushes
# the critical coupling Kc above K, the network can no longer sustain synchrony and r falls
# back toward zero — demonstrating DBS-induced desynchronization.

# > **Exercise:** Vary the DBS frequency between 50 Hz and 200 Hz and observe how the synchrony
# > changes. What happens when the DBS frequency equals the natural oscillation frequency?
# > What happens when it is exactly half the natural frequency?

# ## Summary

# This tutorial demonstrated:
# - How a population of `KuramotoOscillator` blox self-organizes above the critical coupling Kc.
# - How `DBS` pulses modulate the collective phase dynamics.
# - How to measure synchrony via the Kuramoto order parameter computed from `state_timeseries`.

# The full Sermon et al. (2024) model additionally requires coupling to vesicle depletion dynamics
# (three pools: RRP, RP, RtP) that cause K to decay slowly over hundreds of seconds of DBS.
# This component is not yet available in Neuroblox but would be the natural next step to reproduce
# the long-term ERNA drift shown in Figure 2 of the paper.

# ## References
# - [1] Kuramoto Y. (1975). Self-entrainment of a population of coupled non-linear oscillators.
#   *Lecture Notes in Physics*, 39. Springer. https://doi.org/10.1007/BFb0013365
# - [2] Sermon JJ, Wiest C, Tan H, Denison T, Duchet B. (2024). Evoked resonant neural activity
#   long-term dynamics can be reproduced by a computational model with vesicle depletion.
#   *Neurobiology of Disease*, 199, 106565. https://doi.org/10.1016/j.nbd.2024.106565
