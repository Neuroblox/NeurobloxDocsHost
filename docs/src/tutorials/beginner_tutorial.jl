# [Julia Basics and how to model an ODE](@id beginner_tutorial)

# This initial tutorial will introduce Julia's basic structures and how to model an ordinary differential equation.

# Julia's basic structures

# In Julia, we assign a variable simply by calling it by its name and assigning a value to it using the `=` symbol:

my_variable = 42

# For basic math operations, we can just use the following operators:

sum = 3 + 2
product = 3 * 2
difference = 3 - 2
quotient = 3 / 2
power = 3 ^ 2 
modulus = 3 % 2

# To define a function in Julia, we first write `function` followed by the function's name, its parameters, the commands and `end`:

function tutorial(parameter)
    # Your code goes here
    println("The input was $parameter.")
end

# And we can simply call it by its name:
 
tutorial("Neuroblox")
# This should print "The input was Neuroblox."

# Plotting graphs

# We can plot graphs in Julia using the `CairoMakie.jl` package. To add this package we use the terminal:

using Pkg
Pkg.add("CairoMakie")
# This will use the package manager to explicitely add the CairoMakie.jl package

using CairoMakie
# This will load the package

# Notice that `CairoMakie` is only one of Makie's backends for rendering. There are a few other such as `GLMakie`, but for this tutorial, we're going to use this one. After loading the package, we need an interval and a data set to generate a graph. We'll create some random data for now:

using CairoMakie

# Create some data (an interval and a random 10-element vector)
x = 1:10    # This sets an interval
y = rand(10)    # This gives us random y-values

# To create the plot, we'll use the `lines` function. It can receive several inputs as arguments such as `color`, `linewidth`, `markersize` and many other. You can check all of them by yourself in Makie's official documentation webpage: https://docs.makie.org/stable/.

# The said function returns us a tuple containing the overall figure containing the plot, the axis within the figure and the actual line plot created. To plot the data created, we can simply do:

# Create the line plot with a few properties
fig, ax, plt = lines(x, y, color:=black, label="Random Data")
ax.xlabel = "X axis"
ax.ylabel = "Y axis"
ax.title = "Line Plot"
axislegend(ax)

# Save the plot as a PDF file (optional)
save("line_plot.pdf", fig)

# Display the plot
display(fig)

# You can try running the code above in your Julia terminal using your own data to see how it works.

# Solving differential equations

# An essential package we'll be using is `DifferentialEquations.jl`, a library designed to solve various types of differential equations, such as ODEs, SDEs and PDEs.

# To use the `DifferentialEquations` library, we need to define a differential equation as a function, solve it and then plot it (we know how to do the last part).

# Another useful library we'll be using is `ModelingToolkit.jl`, a Julia package designed to make our lives easiear when dealing with complex models and performing tasks such as model simplification, parameter estimation and code generation.

# Let's solve the ODE $\frac{du}{dt}=-a \cdot u$  using this package. First we need to define what are our variables and what are our parameters. For this, we'll use the `@variables`and `@parameters` macros from `ModelingToolkit`:

# Install and import these packages
using Pkg
Pkg.add("ModelingToolkit")
Pkg.add("DifferentialEquations")
using ModelingToolkit: t_nounits as t, D_nounits as D   # Define the time variable and the differentiation operator
using DifferentialEquations

# Define the time-dependent variable u
@variables u(t)

# Define the parameter a
@parameters a

# Also, when using `ModelingToolkit`, the `~` symbol is used as the equal sign when writing an equation:

# Define the differential equation du/dt = -a * u
equation = D(u) ~ -a * u

# Now, in order to build the ODE system, we use the `@named` macro to name the system and the `ODESystem` function:

# Build the ODE system and assign it a name
@named system = ODESystem(equation, t)

# We also have to define our system's initial conditions and parameter values. We use the following syntax:

# Define the initial conditions and parameters
u0 = [u => 1.0]
p = [a => 2.0]
tspan = (0.0 , 5.0)

# Finally, using the functions `ODEProblem` and `solve` from `DifferentialEquations` we convert our system to a numerical problem and solve it:

# Set up the problem
problem = ODEProblem(complete(system), u0, tspan, p)

# Solve the problem
solution = solve(problem)

# Now, we need to extract the time points and `u` values from the solution. To do this, we simply do:

times = solution.t

# The 1 here means we're extracting the values from the first variable 
values = sol[1, :]  

# Finally, we can plot the graph of this solution using what we've learned:

using CairoMakie

fig, ax, plt = lines(times, values, color=:black, label="u(t)")

ax.xlabel = "Time"
ax.ylabel = "u(t)"
ax.title = "Solution of the ODE"
axislegend(ax)

display(fig)

# Great! You just learned how to model your own ODE in Julia!