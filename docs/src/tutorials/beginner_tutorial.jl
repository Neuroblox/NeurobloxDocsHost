# Basic Julia commands #

my_variable = 42

sum = 3 + 2
product = 3 * 2
difference = 3 - 2
quotient = 3 / 2
power = 3 ^ 2 
modulus = 3 % 2

function tutorial(parameter)
    # Your code goes here
    println("The input was $parameter.")
end

tutorial("Neuroblox")
# This should print "The input was Neuroblox."

# ---------------------------------------------- #

# CairoMakie basic Tutorial #

using Pkg
Pkg.add("CairoMakie")
# This will use the package manager to explicitely add the CairoMakie.jl package

using CairoMakie
# This will load the package

# Create some data (an interval and a random 10-element vector)
x = 1:10    # This sets an interval
y = rand(10)    # This gives us random y-values

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

# ---------------------------------------------- #

# EDO solving Tutorial #

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

# Define the differential equation du/dt = -a * u
equation = D(u) ~ -a * u

# Build the ODE system and assign it a name
@named system = ODESystem(equation, t)

# Define the initial conditions and parameters
u0 = [u => 1.0]
p = [a => 2.0]
tspan = (0.0 , 5.0)

# Set up the problem
problem = ODEProblem(complete(system), u0, tspan, p)

# Solve the problem
solution = solve(problem)

times = solution.t

# The 1 here means we're extracting the values from the first variable 
values = sol[1, :]  

using CairoMakie

fig, ax, plt = lines(times, values, color=:black, label="u(t)")

ax.xlabel = "Time"
ax.ylabel = "u(t)"
ax.title = "Solution of the ODE"
axislegend(ax)

display(fig)

# ---------------------------------------------- #

