# [Julia Basics and how to model a single neuron](@id beginner_tutorial)

This initial tutorial will introduce Julia's basic structures and how to model the LIF and the Izhikevich models of neuronal spiking.

## Julia's basic structures

In Julia, we assign a variable simply by calling it by its name and assigning a value to it using the `=` symbol:

```@example beginner_tutorial
my_variable = 42
```

For basic math operations, we can just use the following operators:

```@example beginner_tutorial
sum = 3 + 2
product = 3 * 2
difference = 3 - 2
quotient = 3 / 2
power = 3 ^ 2 
modulus = 3 % 2
```

To define a function in Julia, we first write `function` followed by the function's name, its parameters, the commands and `end`:

```@example beginner_tutorial
function tutorial(parameter)
    # Your code goes here
    println("The input was $parameter.")
end
```

And we can simply call it by its name:
 
```@example beginner_tutorial
tutorial("Neuroblox")
#This should print "The input was Neuroblox."
```

## Plotting graphs

We can plot graphs in Julia using the `Plots.jl` package. To add this package we use the terminal:

```@example beginner_tutorial
using Pkg
Pkg.add("Plots")
# This will use the package manager to explicitely add the Plots.jl package

using Plots
# This will load the package
```

Inside this package, there are several built-in functions. First, we have to tell Plots.jl which backend we will be using for rendering plots.

In this tutorial we will be using `GR backend`, but there are several others and you can find them here: https://docs.juliaplots.org/latest/backends/

To call GR, we simply call it as the parameterless function `gr()`. After that, we need an interval and a data set to generate a graph. In general, we'll have something like this:

```@example beginner_tutorial
using Plots

# Set GR as the backend for Plots.jl
gr()

# Create some data (an interval and a random 10-element vector)
x = 1:10
y = rand(10)

# Plot the data
plot(x, y, label="Random Data", xlabel="X-axis", ylabel="Y-axis", title="Example")
```

You can try running the code above in your Julia terminal using your own data to see how it works.

## Solving differential equations

An essential package we'll be using is `DifferentialEquations.jl`, a library designed to solve various types of differential equations, such as ODEs, SDEs and PDEs.

To use the `DifferentialEquations` library, we need to define a differential equation as a function, solve it and then plot it (we know how to do the last part).

Another useful library we'll be using is `ModelingToolkit.jl`, a Julia package designed to make our lives easiear when dealing with complex models and performing tasks such as model simplification, parameter estimation and code generation.

Let's solve the ODE $\frac{du}{dt}=-a \cdot u$  using this package. First we need to define what are our variables and what are our parameters. For this, we'll use the `@variables`and `@parameters` macros from `ModelingToolkit`:

```@example beginner_tutorial
# Install and import these packages
using Pkg
Pkg.add("ModelingToolkit")
Pkg.add("DifferentialEquations")
using ModelingToolkit
using DifferentialEquations

# Define the time variable
@variables t

# Define the time-dependent variable u
@variables u(t)

# Define the parameter a
@parameters a
```

We also have to define our differential operator (it'll work as a function). We just need to declare it using the function `Differential(independent_variable)`:

```@example beginner_tutorial
# Define the differential operator 
D = Differential(t)
```

Finally, when using `ModelingToolkit`, the `~` symbol is used as the equal sign when writing an equation:

```@example beginner_tutorial
# Define the differential equation du/dt = -a * u
equation = D(u) ~ -a * u
```

Now, in order to build the ODE system, we use the `@named` macro to name the system and the `ODESystem` function:

```@example beginner_tutorial
# Build the ODE system and assign it a name
@named system = ODESystem(equation, t)
```

We also have to define our system's initial conditions and parameter values. We use the following syntax:

```@example beginner_tutorial
# Define the initial conditions and parameters
u0 = [u => 1.0]
p = [a => 2.0]
tspan = (0.0 , 5.0)
```

Finally, using the functions `ODEProblem` and `solve` from `DifferentialEquations` we convert our system to a numerical problem and solve it:

```@example beginner_tutorial
# Set up the problem
problem = ODEProblem(complete(system), u0, tspan, p)

# Solve the problem
solution = solve(problem)
```

Finally, we can graph the solution using what we've already learned:

```@example beginner_tutorial
using Plots

plot(solution, xlabel="Time", ylabel="u(t)", title="Solution to du/dt=-a*u")
```

## The Izhikevich model

The Izhikevich model is described by the following two-dimensional system of ordinary differential equations:

```math
\begin{align}
\frac{dv}{dt} &= 0.04 v^2 + 5 v + 140 - u + I \\
\frac{du}{dt} &= a(bv - u)
\end{align}
```

with the auxiliary after-spike resetting

```math
\begin{align}
\text{if }v \geq 30\text{ mV, then }\begin{cases}v\leftarrow c, u\leftarrow u + d.\end{cases}
\end{align}
```

In order to create this model in Julia, we first need to create an ODE using what we've already learned:

```@example beginner_tutorial 
using ModelingToolkit
using DifferentialEquations

@variables t v(t) u(t)
@parameters a b c d I
D = Differential(t)

equation = [D(v) ~ 0.04 * v ^ 2 + 5 * v + 140 - u + I
           D(u) ~ a * (b * v - u)]
```

Now, we need to create the event that generates the spikes. For this, we create a function continuous event to represent the discontinuity in our functions. Those should be written in the following format:

```julia
name_of_event = [[event_1] => [effect_1]
                [event_2] => [effect_2]
                ...
                [event_n] => [effect_n]]
```

In our case, it'll be:

```@example beginner_tutorial
event = [[v ~ 30.0] => [u ~ u + d]
        [v ~ 30.0] => [v ~ c]]
```

and now, using the `ODESystem` function, we'll create our system of ODE's. We need to declare our independent variable, dependent variables and parameters when creating this system, and if it exists (in this case, it does exist), we also need to declare our events:

```@example beginner_tutorial
@named izh_system = ODESystem(equation, t, [v, u], [a, b, c, d, I]; continuous_events = event)
```
 Finally, we define our parameters and solve our system:

 ```@example beginner_tutorial
 # Those below can change; I'm using the parameters for a chattering dynamic
p = [a =>  0.02, b => 0.2, c => -50.0, d => 2.0, I => 10.0]

u0 = [v => -65.0, u => -13.0]

tspan = (0.0, 100.0)

izh_prob = ODEProblem(complete(izh_system), u0, tspan, p)
izh_sol = solve(izh_prob)
```

Now that our problem is solved, we just have to plot it:

```@example beginner_tutorial
using Plots
gr()

# The "vars" below is used to plot only the "v" variable
plot(izh_sol, vars = [v], xlabel="Time (ms)", ylabel="Membrane Potential (mV)", title="Izhikevich Neuron Model")
```

We're done! You've just plotted a single neuron model.
