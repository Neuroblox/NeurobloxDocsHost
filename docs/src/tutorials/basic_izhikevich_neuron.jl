# How to create an ODE system and model an Izhikevich Neuron

# The Izhikevich model

# The Izhikevich model is described by the following two-dimensional system of ordinary differential equations:

# ```math
# \begin{align}
# \frac{dv}{dt} &= 0.04 v^2 + 5 v + 140 - u + I \\
# \frac{du}{dt} &= a(bv - u)
# \end{align}
# ```

# with the auxiliary after-spike resetting

# ```math
# \begin{align}
# \text{if }v \geq 30\text{ mV, then }\begin{cases}v\leftarrow c, u\leftarrow u + d.\end{cases}
# \end{align}
# ```

# In order to create this model in Julia, we first need to create an ODE system. We can do this by basically replicating what we do to a single equation, but adding a few more as below:

using ModelingToolkit: t_nounits as t, D_nounits as D
using DifferentialEquations

@variables v(t) u(t)
@parameters a b c d I

equation = [D(v) ~ 0.04 * v ^ 2 + 5 * v + 140 - u + I
           D(u) ~ a * (b * v - u)]

# Now, we need to create the event that generates the spikes. For this, we create a function continuous event to represent the discontinuity in our functions. Those should be written in the following format:

name_of_event = [[event_1] => [effect_1]
                [event_2] => [effect_2]
                ...
                [event_n] => [effect_n]]

# In our case, it'll be:

event = [[v ~ 30.0] => [u ~ u + d]
        [v ~ 30.0] => [v ~ c]]

# and now, using the `ODESystem` function, we'll create our system of ODE's. We need to declare our independent variable, dependent variables and parameters when creating this system, and if it exists (in this case, it does exist), we also need to declare our events:

@named izh_system = ODESystem(equation, t, [v, u], [a, b, c, d, I]; continuous_events = event)

# Finally, we define our parameters and solve our system:

# Those below can change; I'm using the parameters for a chattering dynamic
p = [a =>  0.02, b => 0.2, c => -50.0, d => 2.0, I => 10.0]

u0 = [v => -65.0, u => -13.0]

tspan = (0.0, 100.0)

izh_prob = ODEProblem(complete(izh_system), u0, tspan, p)
izh_sol = solve(izh_prob)

# Now that our problem is solved, we just have to plot it:

using CairoMakie

times = izh_sol.t
values = izh_sol[1, :]

fig, ax, plt = lines(times, values, color=:black, label="M.P (mV)")

ax.xlabel = "Time (ms)"
ax.ylabel = "Membrane Potential (mV)"
ax.title = "Izhikevich Neuron Model"
axislegend(ax)

display(fig)

# We're done! You've just plotted an Izhikevich neuron.