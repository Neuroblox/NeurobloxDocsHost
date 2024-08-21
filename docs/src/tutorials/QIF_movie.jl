using Neuroblox
using MetaGraphs
using DifferentialEquations
using Random
using Plots
using Statistics

qif1 = QIFNeuron(name=Symbol("qif1"), I_in=0.5)
g = MetaDiGraph()
add_blox!.(Ref(g), [qif1])

@named sys = system_from_graph(g)
sys = structural_simplify(sys)
length(unknowns(sys))

prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)
plot(sol.t,sol[1,:],xlims=(0,1000),xlabel="time (ms)",ylabel="mV")
p_old = prob.p[1]
p_old[10] = 0.5 # I_in
prob2 = remake(prob, p = p_old)
sol = solve(prob2, Vern7(), saveat=0.1)
I = 0:0.01:0.1
anim = @animate for i in I
    @show i
    qif1 = QIFNeuron(name=Symbol("qif1"), I_in=i)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [qif1])

    @named sys = system_from_graph(g)
    sys = structural_simplify(sys)
    prob_new = ODEProblem(sys, [], (0.0, 1000), [])
    sol_new = solve(prob_new, Vern7(), saveat=0.1)
    plot_qif = plot(sol_new.t,sol_new[1,:],xlims=(0,1000),xlabel="time (ms)",ylabel="mV",label="I_in=$(i)",legend=:top)
end
gif(anim, "anim_fps5.gif", fps = 5)