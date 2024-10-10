```@meta
EditURL = "spectral_DCM.jl"
```

````@example spectral_DCM
using Neuroblox

using LinearAlgebra
using Graphs
using DifferentialEquations
using DataFrames
using OrderedCollections
using CairoMakie
using ModelingToolkit

########## Assemble model ##########
````

In this tutorial we will define a circuit of three linear neuronal mass models, all driven by an Ornstein-Uhlenbeck process.
We will model fMRI data by a balloon model and BOLD signal on top.
After simulation of this simple model we will use spectral Dynamic Causal Modeling to infer some of the model parameters from the simulation time series.
Step 1: define the graph, add blocks
Step 2: simulate the model
Step 3: compute the cross spectral density
Step 4: setup the DCM
Step 5: estimate
Step 5: plot the results

## Define the model
We will define a model of 3 regions. This means first of all to define a graph.
To this graph we will add three linear neuronal mass models which constitute the (hidden) neuronal dynamics.
These constitute three nodes of the graph.
Next we will also need some input that stimulates the activity, we use simple Ornstein-Uhlenbeck blocks to create stochastic inputs.
One per region.
We want to simulate fMRI signals thus we will need to also add a BalloonModel per region.
Note that the Ornstein-Uhlenbeck block will feed into the linear neural mass which in turn will feed into the BalloonModel blox.
This needs to be represented by the way we define the edges.

````@example spectral_DCM
nr = 3             # number of regions
g = MetaDiGraph()
regions = []   # list of neural mass blocks to then connect them to each other with an adjacency matrix
````

Now add the different blocks to each region and connect the blocks within each region:

````@example spectral_DCM
for i = 1:nr
    region = LinearNeuralMass(;name=Symbol("r$(i)₊lm"))
    push!(regions, region)          # store neural mass model for connection of regions

    # add Ornstein-Uhlenbeck block as noisy input to the current region
    input = OUBlox(;name=Symbol("r$(i)₊ou"), σ=0.1)
    add_edge!(g, input => region; :weight => 1/16)   # Note that 1/16 is taken from SPM12, this stabilizes the balloon model simulation. Alternatively the noise of the Ornstein-Uhlenbeck block or the weight of the edge connecting neuronal activity and balloon model could be reduced to guarantee numerical stability.

    # simulate fMRI signal with BalloonModel which includes the BOLD signal on top of the balloon model dynamics
    measurement = BalloonModel(;name=Symbol("r$(i)₊bm"))
    add_edge!(g, region => measurement; :weight => 1.0)
end
````

Next we define the between-region connectivity matrix and make sure that it is diagonally dominant to guarantee numerical stability (see Gershgorin theorem).

````@example spectral_DCM
A_true = randn(nr, nr)
A_true -= diagm(map(a -> sum(abs, a), eachrow(A_true)))    # ensure diagonal dominance of matrix
for idx in CartesianIndices(A_true)
    add_edge!(g, regions[idx[1]] => regions[idx[2]]; :weight => A_true[idx[1], idx[2]])
end
````

finally we compose the simulation model

````@example spectral_DCM
@named simmodel = system_from_graph(g)
simmodel = structural_simplify(simmodel, split=false)
````

setup simulation of the model, time in seconds

````@example spectral_DCM
tspan = (0.0, 512.0)
prob = SDEProblem(simmodel, [], tspan)
dt = 2.0   # two seconds as measurement interval for fMRI
sol = solve(prob, saveat=dt);
nothing #hide
````

##Plot simulation results

plot bold signal time series

````@example spectral_DCM
idx_m = get_idx_tagged_vars(simmodel, "measurement")    # get index of bold signal
f = Figure()
ax = Axis(f[1, 1],
    title = "fMRI time series",
    xlabel = "Time [s]",
    ylabel = "BOLD",
)
lines!(ax, sol, idxs=idx_m)
f
# (Note to self: tosymbol(x; escape=false)
````

##Estimate and plot cross-spectrum

````@example spectral_DCM
dfsol = DataFrame(sol)
data = Matrix(dfsol[:, idx_m])
````

We compute the cross-spectral density by fitting a linear model of order `p` and then compute the csd analytically from the parameters of the multivariate autoregressive model

````@example spectral_DCM
p = 8
mar = mar_ml(data, p)   # maximum likelihood estimation of the MAR coefficients and noise covariance matrix
ns = size(data, 1)
freq = range(min(128, ns*dt)^-1, max(8, 2*dt)^-1, 32)
csd = mar2csd(mar, freq, dt^-1)
````

Now plot the cross-spectrum:

````@example spectral_DCM
fig = Figure(size=(1200, 800))
grid = fig[1, 1] = GridLayout()
for i = 1:nr
    for j = 1:nr
        ax = Axis(grid[i, j])
        lines!(ax, freq, real.(csd[:, i, j]))
        if (i==1) && (j==2)
            ax.title = "Cross spectral density"
        end
        if (i==3) && (j==2)
            ax.xlabel = "Frequency [Hz]"
        end
    end
end
fig
````

#Model inference

We will now assemble a new model that is used for fitting the previous simulations.
This procedure is similar to before with the difference that we will define global parameters and use tags such as [tunable=false/true] to define which parameters we will want to estimate.
Note that parameters are tunable by default.

````@example spectral_DCM
g = MetaDiGraph()
regions = []   # list of neural mass blocks to then connect them to each other with an adjacency matrix
````

The following parameters are shared accross regions, which is why we define them here.

````@example spectral_DCM
@parameters lnκ=0.0 [tunable=false] lnϵ=0.0 [tunable=false] lnτ=0.0 [tunable=false]   # lnκ: decay parameter for hemodynamics; lnϵ: ratio of intra- to extra-vascular components, lnτ: transit time scale
@parameters C=1/16 [tunable=false]   # note that C=1/16 is taken from SPM12 and stabilizes the balloon model simulation. See also comment above.

for i = 1:nr
    region = LinearNeuralMass(;name=Symbol("r$(i)₊lm"))
    push!(regions, region)
    input = ExternalInput(;name=Symbol("r$(i)₊ei"))
    add_edge!(g, input => region; :weight => C)

    # we assume fMRI signal and model them with a BalloonModel
    measurement = BalloonModel(;name=Symbol("r$(i)₊bm"), lnτ=lnτ, lnκ=lnκ, lnϵ=lnϵ)
    add_edge!(g, region => measurement; :weight => 1.0)
end

A_prior = 0.01*randn(nr, nr)
A_prior -= diagm(diag(A_prior))    # ensure diagonal dominance of matrix
````

Since we want to optimize these weights we turn them into symbolic parameters:
Add the symbolic weights to the edges and connect reegions.

````@example spectral_DCM
@parameters A[1:nr^2] = vec(A_prior) [tunable = true]
for (i, idx) in enumerate(CartesianIndices(A_prior))
    if idx[1] == idx[2]
        add_edge!(g, regions[idx[1]] => regions[idx[2]]; :weight => -exp(A[i])/2)  # -exp(A[i])/2: treatement of diagonal elements in SPM12 to make diagonal dominance (see Gershgorin Theorem) more likely but it is not guaranteed
    else
        add_edge!(g, regions[idx[2]] => regions[idx[1]]; :weight => A[i])
    end
end

@named fitmodel = system_from_graph(g)
fitmodel = structural_simplify(fitmodel, split=false)
````

#Setup DCM

````@example spectral_DCM
max_iter = 128            # maximum number of iterations
````

TODO: Place all this part into a helper function that defines the priors until `csdsetup`.
attribute initial conditions to states

````@example spectral_DCM
sts, idx_sts = get_dynamic_states(fitmodel)
idx_u = get_idx_tagged_vars(fitmodel, "ext_input")         # get index of external input state
idx_bold, s_bold = get_eqidx_tagged_vars(fitmodel, "measurement")  # get index of equation of bold state
````

the following step is needed if the model's Jacobian would give degenerate eigenvalues if expanded around 0 (which is the default expansion)

````@example spectral_DCM
perturbedfp = Dict(sts .=> abs.(0.001*rand(length(sts))))     # slight noise to avoid issues with Automatic Differentiation. TODO: find different solution, this is hacky.

modelparam = OrderedDict()
for par in tunable_parameters(fitmodel)
    modelparam[par] = Symbolics.getdefaultval(par)
end
np = length(modelparam)
indices = Dict(:dspars => collect(1:np))
````

Noise parameter mean

````@example spectral_DCM
modelparam[:lnα] = [0.0, 0.0];           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014
n = length(modelparam[:lnα]);
indices[:lnα] = collect(np+1:np+n);
np += n;
modelparam[:lnβ] = [0.0, 0.0];           # global observation noise, ln(β) as above
n = length(modelparam[:lnβ]);
indices[:lnβ] = collect(np+1:np+n);
np += n;
modelparam[:lnγ] = zeros(Float64, nr);   # region specific observation noise
indices[:lnγ] = collect(np+1:np+nr);
np += nr
indices[:u] = idx_u
indices[:m] = idx_bold
indices[:sts] = idx_sts
````

define prior variances

````@example spectral_DCM
paramvariance = copy(modelparam)
paramvariance[:lnα] = ones(Float64, length(modelparam[:lnα]))./64.0;
paramvariance[:lnβ] = ones(Float64, length(modelparam[:lnβ]))./64.0;
paramvariance[:lnγ] = ones(Float64, nr)./64.0;
for (k, v) in paramvariance
    if occursin("A[", string(k))
        paramvariance[k] = ones(length(v));
    elseif occursin("κ", string(k))
        paramvariance[k] = ones(length(v))./256.0;
    elseif occursin("ϵ", string(k))
        paramvariance[k] = 1/256.0;
    elseif occursin("τ", string(k))
        paramvariance[k] = 1/256.0;
    end
end

priors = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
hyperpriors = Dict(:Πλ_pr => 128.0*ones(1, 1),   # prior metaparameter precision, needs to be a matrix
                   :μλ_pr => [8.0]               # prior metaparameter mean, needs to be a vector
                  );

csdsetup = (mar_order = p, freq = freq, dt = dt);

(state, setup) = setup_sDCM(dfsol[:, [s[1:end-3] for s in String.(Symbol.(s_bold))]], fitmodel, perturbedfp, csdsetup, priors, hyperpriors, indices, modelparam, "fMRI");
nothing #hide
````

HACK: on machines with very small amounts of RAM, Julia can run out of stack space while compiling the code called in this loop
this should be rewritten to abuse the compiler less, but for now, an easy solution is just to run it with more allocated stack space.

````@example spectral_DCM
with_stack(f, n) = fetch(schedule(Task(f, n)))

with_stack(5_000_000) do  # 5MB of stack space
    for iter in 1:max_iter
        state.iter = iter
        run_sDCM_iteration!(state, setup)
        print("iteration: ", iter, " - F:", state.F[end] - state.F[2], " - dF predicted:", state.dF[end], "\n")
        if iter >= 4
            criterion = state.dF[end-3:end] .< setup.tolerance
            if all(criterion)
                print("convergence\n")
                break
            end
        end
    end
end
````

##Plot Results
Later place all into one figure using [Makie-style layouts](https://docs.makie.org/stable/tutorials/layout-tutorial)
Plot the free energy evolution:

````@example spectral_DCM
freeenergy(state)
````

Plot the estimated posterior of the effective connectivity and compare that to the true parameter values.
Bar hight are the posterior mean and error bars are the standard deviation of the posterior.

````@example spectral_DCM
ecbarplot(state, setup, A_true)
# TODO: need to implement a ecbarplot! to be able to integrate into one figure


using Literate
Literate.markdown("./docs/src/tutorials/spectral_DCM.jl", "./docs/src/tutorials/")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

