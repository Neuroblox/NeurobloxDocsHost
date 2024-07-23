# [Tutorial on building a neural assembly from bottom-up](@id neural_assembly_tutorial)

## Single spiking neuron from Hodgkin-Huxley model


```math
    \begin{align}
    C_m\frac{dV}{dt} &= -g_L(V-V_L) - \Bar{g}_{Na}m^3h(V-V_{Na}) -\Bar{g}_Kn^4(V-V_K) + I_{in} - I_{syn} \\
    \frac{dm}{dt} &= \alpha_{m}(V)(1-m) + \beta_{m}(V)m \\ 
    \frac{dh}{dt} &= \alpha_{h}(V)(1-h) + \beta_{h}(V)h \\
    \frac{dn}{dt} &= \alpha_{n}(V)(1-n) + \beta_{n}(V)n 
    \end{align}
```