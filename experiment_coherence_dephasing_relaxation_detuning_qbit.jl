using DifferentialEquations;
using LinearAlgebra;
using ArgParse;

s = ArgParseSettings()
@add_arg_table s begin
    "--detuning-value"
        help = "absolute value of detuning (linear frequency) in Hz. This supersedes the detuning_ratio flag if set"
        arg_type = Float64
    "--detuning-ratio"
        help = "detuning * T2 value. This can be superseded by the detuning-value if set."
        arg_type = Float64
        default = 0.1
    "--t1"
        help = "T1 time in us"
        arg_type = Int
        default = 70
    "--t2"
        help = "T2 time in us"
        arg_type = Int
        default = 50
    "--vx"
        help = "Starting vx state"
        arg_type = Float64
        default = 0.5
end
parsed_args = parse_args(ARGS, s);

# The setup for the experiment is:
# 1. Z-axis dephasing
# 2. Pure relaxation, i.e. relaxation to ground state
# The initial state is the bloch vector (1/sqrt(2),0,1/sqrt(2))
# The output is a Hamiltonian, with units in angular frequency. Rabi frequency
# is linear frequency, so divide by 2pi to get in terms of Rabi frequency.

# Experiment setup
h = 6.62607015 * 1e-34; # Planck's constant - Joule per Hertz
kb = 1.380649 * 1e-23; # Boltzmann's constant - Joule per Kelvin

# 1/t2 = 1/(2t1) + 1/t_phi.
# t_phi - corresponds to dephasing. Equal to 1/gamma
# t1 - corresponds to thermal relaxation.
t1us = parsed_args["t1"];
t2us = parsed_args["t2"];
t1 = t1us * 1e-6;
t2 = t2us * 1e-6;
dephasing_gamma = 0.5*((1/t2)-(1/(2*t1)));
thermal_gamma = 1 / t1;
@assert dephasing_gamma >= 0;

detuning_freq = parsed_args["detuning-value"] == nothing ? parsed_args["detuning-ratio"]/t2 : parsed_args["detuning-value"];

@enum SimulationType begin
    # For detuning experiments, the static field is assumed to be (detuning_freq)/2 * σz
    # in linear frequency units. With no dissipation and external control, this is
    # expected to give oscillations with time period (1/detuning_freq).
    detuned_free_decay = 1
    ideal_tracking_control = 2
    detuned_tracking_control = 3
end;

# Sampling rate is 2.4 giga samples per sec
sampling_rate = 2.4 * 1e9;
qbit_freq = 3877496000;
bath_temp = 50 * 10e-3;

# Simulation code
F4 = Matrix{ComplexF64}[];
push!(F4, [1 0;0 1]);
push!(F4, [0 1;1 0]);
push!(F4, [0 -1im;1im 0]);
push!(F4, [1 0;0 -1]);

matrix_to_coeff(matrix) = [real(tr(x' * matrix)) for x in F4];
coeff_to_matrix(coeff) = 0.5 * reduce(+, [x*y for (x,y) in zip(cat([1],coeff,dims=1), F4)])

commutator(x, y) = x*y - y*x;

function get_relaxation_coeff(temp, qbit_freq)
    # Returns the rate associated with relaxation.
    #
    # Assume qubit energy is -(1/2)ωσ. Then ground state population in Gibbs state is
    # 1/(1+exp(-βH)). The probability p of the excitation channel is simply the population
    # of excited state at equilibrium.
    βH = (h * qbit_freq) / (kb * temp);
    return 1/(1+exp(-βH))
end

function dissipator(v)
    dephasing_dissipator = -2 * dephasing_gamma * [v[1], v[2], 0];
    assume_pure_relaxation = true;
    relaxation_coefficient = assume_pure_relaxation ? 1 : get_relaxation_coeff(bath_temp, qbit_freq);
    thermal_dissipator = thermal_gamma * (-relaxation_coefficient * [v[1]/2, v[2]/2, v[3]-1] - (1-relaxation_coefficient) * [v[1]/2, v[2]/2, v[3]+1]);
    return dephasing_dissipator + thermal_dissipator;
end

function target(v)
    # Coherence magnitude = vx^2+vy^2;
    return v[1]^2 + v[2]^2;
end

function grad(v)
    return 2 * [v[1], v[2], 0];
end

function get_hamiltonian(v,p,t)
    # For conversion from Hamiltonian in Pauli basis to vectors, we assume
    # H = hx σx + hy σy + hz σz. This gives vec(h) = (hx, hy, hz). The resulting
    # differential equation is dv/dt = 2 h x v. Note that the detuning Hamiltonian
    # is detuning_freq/2 in linear frequency units.
    detuning_hamiltonian = [0,0,detuning_freq/2 * 2 * pi];
    # dephasing_hamiltonian(x) = -dephasing_gamma / x[3] * [x[2], -x[1], 0];
    # thermal_hamiltonian(x) = -thermal_gamma / (4 * x[3]) * [x[2], -x[1], 0];
    if p[1] == detuned_free_decay::SimulationType
        return detuning_hamiltonian;
    elseif p[1] == ideal_tracking_control::SimulationType
        dephasing_hamiltonian = -dephasing_gamma / v[3] * [v[2], -v[1], 0];
        thermal_hamiltonian = -thermal_gamma / (4 * v[3]) * [v[2], -v[1], 0];
        return dephasing_hamiltonian + thermal_hamiltonian;
    elseif p[1] == detuned_tracking_control::SimulationType
        # Use the ideal case hamiltonian at time t to apply here. The
        # ideal case hamiltonian is a function of the state in ideal case at time t.
        if t > p[2].t[end]
            return detuning_hamiltonian;
        end
        buff_v = p[2](t);
        dephasing_hamiltonian = -dephasing_gamma / buff_v[3] * [buff_v[2], -buff_v[1], 0];
        thermal_hamiltonian = -thermal_gamma / (4 * buff_v[3]) * [buff_v[2], -buff_v[1], 0];
        return dephasing_hamiltonian + thermal_hamiltonian + detuning_hamiltonian;
    end
    throw(Error("Undefined control sequence in get_hamiltonian: unexpected value of SimulationType"))
end

# Parameters p (tuple):
# 1. First element - is_ramsey_setup
# 2. Second element - ideal solution
function lindblad(v, p, t)
    hamiltonian = get_hamiltonian(v,p,t);
    if any(isnan, hamiltonian)
        return [Inf, Inf, Inf]
    end
    return 2 * cross(hamiltonian, v) + dissipator(v)
end

function get_hamiltonian_matrix(u, t, int)
    h_vec = get_hamiltonian(u, (ideal_tracking_control::SimulationType, nothing), 0)
    return h_vec[1] * F4[2] + h_vec[2] * F4[3] + h_vec[3] * F4[4]
end

function integral(solution)
    prev_time = 0
    rectangle_estimates = [solution.u[i] * (solution.t[i] - solution.t[i-1]) for i in range(2,length(solution))]
    return sum(rectangle_estimates)
end

is_ramsey_setup = false;
simulation_type = ideal_tracking_control::SimulationType;
if !is_ramsey_setup
    x = parsed_args["vx"];#0.5*sqrt(t2/t1);
    v = [x,0,sqrt(1-x^2)];
else
    simulation_type = detuned_free_decay::SimulationType;
    v = [1,0,0]
end
tend = 6*t2;

function solve_wrapper(starting_state, time_end, simulation_type, past_solution)
    verbose = true;
    if verbose
        print("Starting simulation for: $(simulation_type)\n")
        print("Initial state: $(starting_state)\n");
        print("T1, T2: $(round(t1*1e6))us, $(round(t2*1e6))us\n");
        print("Detuning frequency: $(round(detuning_freq)) Hz\n");
        print("\n");
    end
    abstol, reltol = 1e-8,1e-6;
    problem = ODEProblem(lindblad, starting_state, (0.0, time_end), (simulation_type, past_solution));
    solution = solve(problem, alg_hints=[:stiff], abstol=abstol, reltol=reltol);
    return solution
end

ideal_solution = solve_wrapper(v, tend, ideal_tracking_control::SimulationType, nothing);
detuned_solution = solve_wrapper(v, tend, detuned_tracking_control::SimulationType, ideal_solution);
ramsey_solution = solve_wrapper([1,0,0], tend, detuned_free_decay::SimulationType, nothing);

using Plots;
plotly();

# Graph attributes
linewidth = 3;
size = (1500,800);

# plot(ideal_solution, size=size, linewidth=linewidth, show=true,
#         title="T1=$(round(t1*1e6))us, T2=$(round(t2*1e6))us, Ideal",
#         xlabel="Time (s)",
#         label=["vx ideal" "vy ideal" "vz ideal"]);
# plot(ideal_solution.t, [target(u) for u in ideal_solution.u], show=true, ylim=(0,1), label="target ideal")
# plot(ideal_solution.t,
#     [[get_hamiltonian(u,(detuned_free_decay::SimulationType,nothing),0)[1] for u in ideal_solution.u],
#      [get_hamiltonian(u,(detuned_free_decay::SimulationType,nothing),0)[2] for u in ideal_solution.u],
#      [get_hamiltonian(u,(detuned_free_decay::SimulationType,nothing),0)[3] for u in ideal_solution.u]
#     ], show=true, label=["hx ideal" "hy ideal" "hz ideal"])

# graph = plot();
# plot!(graph, detuned_solution, size=size, linewidth=linewidth,
#     title="T1=$(round(t1*1e6))us, T2=$(round(t2*1e6))us, CM, Detuning=$(round(detuning_freq))Hz",
#     xlabel="Time (s)",
#     label=["vx" "vy" "vz"]);
# vx = parsed_args["vx"];
# if vx^2 > 0.5* t2us/t1us
#     alpha = sqrt(2 * t1us * (vx ^ 2)  / t2us - 1);
#     vz = sqrt(1 - vx^2);
#     tb = t1 * (1/alpha * (atan(1/alpha) + atan((2*vz-1)/alpha)) + 0.5 * log( ((2*vz-1)^2+alpha^2)/(1+alpha^2) ));
#     detune_ratio = detuning_freq*t2;
#     breakdown_vy = (1 - exp(-tb / t2)) * detune_ratio * vx;
#     predicted_vy = sqrt(breakdown_vy^2 + vx^2) * detune_ratio * exp(atan(breakdown_vy/vx)/detune_ratio-1);
    
#     plot!(graph, [tb], label="Breakdown", st=:vline);
#     # plot!(graph, detuned_solution.t, [predicted_vy for x in detuned_solution.t], label="Predicted vy max");
    
#     print("Breakdown time - $(tb), Expected max - $(predicted_vy)\n");
# end
# display(graph)
# plot(detuned_solution.t, [target(u) for u in detuned_solution.u], show=true, ylim=(0,1), label="target")

# plot(ramsey_solution, size=size, linewidth=linewidth, show=true,
#         title="T1=$(round(t1*1e6))us, T2=$(round(t2*1e6))us, Ramsey, Detuning=$(round(detuning_freq))Hz",
#         xlabel="Time (s)",
#         label=["vx" "vy" "vz"]);

# detuned_values = [detuned_solution(t)[2]/sqrt(t) for t in detuned_solution.t];
# ramsey_values = [ramsey_solution(t)[2]/sqrt(t) for t in ramsey_solution.t];
# plot(detuned_solution.t, detuned_values, label="Detuned", linewidth=linewidth, size=size);
# plot!(ramsey_solution.t, ramsey_values, label="Ramsey", linewidth=linewidth, size=size, show=true);

using NLopt, ForwardDiff;

function get_crude_estimate_for_max(diffEqSolution, is_positive_detuning)
    # Simply pick out the max from the solution array.
    prefactor = is_positive_detuning ? 1 : -1;
    return argmax(map(u->prefactor*u[2], diffEqSolution.u));
end

# Find the true max vy using the solution to the differential equation.
function max_vy_objective(t::Vector, grad::Vector, diffEqSolution, is_positive_detuning)
    prefactor = is_positive_detuning ? 1 : -1;
    if length(grad) > 0
           # use ForwardDiff for the gradient for vy.
           # using first(t) is supposed to be faster than t[1]
           grad[1] = prefactor * ForwardDiff.derivative((t)->diffEqSolution(first(t), idxs=2), first(t));
     end
     return prefactor * diffEqSolution(first(t), idxs=2)
end

# Find the maximum vy achieved in the differential equation solutions.
# This routine does not work when vy is constant (eg. - detuning is 0).
function get_max_vy(solution, is_positive_detuning)
    # This is a crude estimate - the solution has interpolations and take maximum values
    # away from this point.
    crude_argmax_vy = get_crude_estimate_for_max(solution, is_positive_detuning);
    
    # Get global maximum by improving upon the crude estimate.
    # Method - local maximisation around the crude estimate.
    opt = NLopt.Opt(:LD_MMA, 1); # Local derivative based MMA approach, only 1 optimisation variable.
    if solution.t[crude_argmax_vy] - 3e-6 > 0
        opt.lower_bounds = [solution.t[crude_argmax_vy] - 3e-6]; # lower bound on t
    else
        opt.lower_bounds = [0.0];
    end
    if solution.t[crude_argmax_vy] + 3e-6 > solution.t[end]
        opt.upper_bounds = [solution.t[end]];
    else
        opt.upper_bounds = [solution.t[crude_argmax_vy] + 3e-6]; # upper bound on t
    end
    opt.xtol_rel = 1e-6; # Relative tolerance for t
    opt.max_objective = (x,g) -> max_vy_objective(x,g,solution,is_positive_detuning);
    (max_vy,argmax_t,ret) = NLopt.optimize(opt, [solution.t[crude_argmax_vy]]); # best initial guess for t
    
    return (max_vy,argmax_t[1]);
end

function get_crude_estimate_for_vyst(diffEqSolution, is_positive_detuning)
    # Simply pick out the max from the solution array.
    prefactor = is_positive_detuning ? 1 : -1;
    return argmax(map(t->prefactor*diffEqSolution(t)[2]/sqrt(t), diffEqSolution.t[2:end]))+1;
end

# Find the true max vy using the solution to the differential equation.
function max_vyst_objective(t::Vector, grad::Vector, diffEqSolution, is_positive_detuning)
    prefactor = is_positive_detuning ? 1 : -1;
    vy = diffEqSolution(first(t), idxs=2);
    if length(grad) > 0
           # use ForwardDiff for the gradient for vy.
           # using first(t) is supposed to be faster than t[1]
           vy_dot = ForwardDiff.derivative((t)->diffEqSolution(first(t), idxs=2), first(t));
           grad[1] = prefactor * (vy_dot - 0.5*vy/first(t)) / sqrt(first(t));
     end
     return prefactor * vy / sqrt(first(t));
end

# Find the maximum vy achieved in the differential equation solutions.
# This routine does not work when vy is constant (eg. - detuning is 0).
function get_max_vyst(solution, is_positive_detuning)
    # Get global maximum by improving upon the crude estimate.
    crude_argmax = get_crude_estimate_for_vyst(solution, is_positive_detuning);
    
    # Method - local maximisation around the crude estimate.
    opt = NLopt.Opt(:LD_MMA, 1); # Local derivative based MMA approach, only 1 optimisation variable.
    if solution.t[crude_argmax] - 5e-6 > 0
        opt.lower_bounds = [solution.t[crude_argmax] - 5e-6]; # lower bound on t
    else
        opt.lower_bounds = [0.0];
    end
    if solution.t[crude_argmax] + 5e-6 > solution.t[end]
        opt.upper_bounds = [solution.t[end]];
    else
        opt.upper_bounds = [solution.t[crude_argmax] + 5e-6]; # upper bound on t
    end
    opt.xtol_rel = 1e-6; # Relative tolerance for t
    opt.max_objective = (x,g) -> max_vyst_objective(x,g,solution,is_positive_detuning);
    (max_vyst,argmax_t,ret) = NLopt.optimize(opt, [solution.t[crude_argmax]]); # best initial guess for t
    
    return (max_vyst,argmax_t[1]);
end

vy_r, tim = get_max_vy(ramsey_solution, true);
print("Ramsey - Value - $(vy_r), Argmax: $(tim) s. vs T2/2: $(tim * 2 / t2)\n");
vy, tim = get_max_vy(detuned_solution, true);
print("Stabilized - Value - $(vy), Argmax: $(tim) s.\n");
print("Ratio - $(vy/vy_r)\n");
# print("Ideal solution - area under v_y: ", integral(ideal_solution), "\n")
# print("Detuned solution - area under v_y: ", integral(detuned_solution), "\n")

# vx, vy = 0.4, 0.5;
# free_decay_solution = solve_wrapper([vx,vy,0], tend, detuned_free_decay::SimulationType, nothing);
# predicted_max = atan(detuning_freq * 2 * pi * t2)-atan(vy/vx) > 0 ? sqrt(vx^2+vy^2) * exp(-(atan(detuning_freq * 2 * pi * t2)-atan(vy/vx))/(detuning_freq * 2 * pi * t2)) * sin(atan(detuning_freq * 2 * pi * t2)) : vy;
# plot(free_decay_solution, size=size, linewidth=linewidth,
#         title="T1=$(round(t1*1e6))us, T2=$(round(t2*1e6))us, Free Decay, Detuning=$(round(detuning_freq))Hz",
#         xlabel="Time (s)",
#         label=["vx" "vy" "vz"]);
# plot!(free_decay_solution.t, [predicted_max for u in free_decay_solution.u], linewidth=linewidth, show=true,
#         size=size, label="Predicted max", linestyle=:dash, linecolor=:black,);

# using JLD2;
# @save "ideal_solution.jld2" ideal_solution
# @save "detuned_solution.jld2" detuned_solution
# detuned_hamiltonian = [[get_hamiltonian(u)[1] for u in ideal_solution.u], [get_hamiltonian(u)[2] for u in ideal_solution.u], [get_hamiltonian(u)[3] for u in ideal_solution.u]]





