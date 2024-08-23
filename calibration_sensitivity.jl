using DifferentialEquations;
using LinearAlgebra;
using ArgParse;

s = ArgParseSettings()
@add_arg_table s begin
    "--trial"
        help = "whether this is a trial run"
        action = :store_true
    "--single"
        help = "whether to run optimisation for single shot type experiments"
        action = :store_true
end
parsed_args = parse_args(ARGS, s);

# This program runs a parameter sweep for coherence preservation enhanced quantum sensing.
# The setup for the experiment is:
# 1. Z-axis dephasing
# 2. Pure relaxation, i.e. relaxation to ground state
# The initial state is the bloch vector (1/sqrt(2),0,1/sqrt(2))
# The output is a Hamiltonian, with units in angular frequency. Rabi frequency
# is linear frequency, so divide by 2pi to get in terms of Rabi frequency.

# Experiment setup
h = 6.62607015 * 1e-34; # Planck's constant - Joule per Hertz
kb = 1.380649 * 1e-23; # Boltzmann's constant - Joule per Kelvin

# Sampling rate is 2.4 giga samples per sec
sampling_rate = 2.4 * 1e9;
qbit_freq = 3877496000;
bath_temp = 50 * 10e-3;

SIMULATION_TYPE = 1;
DETUNING_FREQ = 2;
DEPHASING_GAMMA = 3;
THERMAL_GAMMA = 4;
IDEAL_TRAJECTORY = 5;

@enum Optimisation begin
	# For detuning experiments, the static field is assumed to be (detuning_freq)/2 * σz
	# in linear frequency units. With no dissipation and external control, this is
	# expected to give oscillations with time period (1/detuning_freq).
	single_shot = 1
    multiple_shot = 2
end;

@enum SimulationType begin
	# For detuning experiments, the static field is assumed to be (detuning_freq)/2 * σz
	# in linear frequency units. With no dissipation and external control, this is
	# expected to give oscillations with time period (1/detuning_freq).
	detuned_free_decay = 1
	ideal_tracking_control = 2
	detuned_tracking_control = 3
end;

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

function dissipator(v,p,t)
	dephasing_dissipator = -2 * p[DEPHASING_GAMMA] * [v[1], v[2], 0];
	assume_pure_relaxation = true;
	relaxation_coefficient = assume_pure_relaxation ? 1 : get_relaxation_coeff(bath_temp, qbit_freq);
	thermal_dissipator = p[THERMAL_GAMMA] * (-relaxation_coefficient * [v[1]/2, v[2]/2, v[3]-1] - (1-relaxation_coefficient) * [v[1]/2, v[2]/2, v[3]+1]);
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
	detuning_hamiltonian = [0,0,p[DETUNING_FREQ]/2 * 2 * pi];
	# dephasing_hamiltonian(x) = -dephasing_gamma / x[3] * [x[2], -x[1], 0];
	# thermal_hamiltonian(x) = -thermal_gamma / (4 * x[3]) * [x[2], -x[1], 0];
	if p[SIMULATION_TYPE] == detuned_free_decay::SimulationType
		return detuning_hamiltonian;
	elseif p[SIMULATION_TYPE] == ideal_tracking_control::SimulationType
		dephasing_hamiltonian = -p[DEPHASING_GAMMA] / v[3] * [v[2], -v[1], 0];
		thermal_hamiltonian = -p[THERMAL_GAMMA] / (4 * v[3]) * [v[2], -v[1], 0];
		return dephasing_hamiltonian + thermal_hamiltonian;
	elseif p[SIMULATION_TYPE] == detuned_tracking_control::SimulationType
		# Use the ideal case hamiltonian at time t to apply here. The
		# ideal case hamiltonian is a function of the state in ideal case at time t.
		if t > p[IDEAL_TRAJECTORY].t[end]
			return detuning_hamiltonian;
		end
		buff_v = p[IDEAL_TRAJECTORY](t);
		dephasing_hamiltonian = -p[DEPHASING_GAMMA] / buff_v[3] * [buff_v[2], -buff_v[1], 0];
		thermal_hamiltonian = -p[THERMAL_GAMMA] / (4 * buff_v[3]) * [buff_v[2], -buff_v[1], 0];
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
	return 2 * cross(hamiltonian, v) + dissipator(v,p,t)
end

# Methods for finding optimums
using NLopt, ForwardDiff;

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
           grad[1] = prefactor * (vy_dot/sqrt(first(t)) - 2*vy/sqrt(first(t)^3));
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
    if solution.t[crude_argmax] - 3e-6 > 0
        opt.lower_bounds = [solution.t[crude_argmax] - 3e-6]; # lower bound on t
    else
        opt.lower_bounds = [0.0];
    end
    if solution.t[crude_argmax] + 3e-6 > solution.t[end]
        opt.upper_bounds = [solution.t[end]];
    else
        opt.upper_bounds = [solution.t[crude_argmax] + 3e-6]; # upper bound on t
    end
    opt.xtol_rel = 1e-6; # Relative tolerance for t
    opt.max_objective = (x,g) -> max_vyst_objective(x,g,solution,is_positive_detuning);
    (max_vyst,argmax_t,ret) = NLopt.optimize(opt, [solution.t[crude_argmax]]); # best initial guess for t
    
    return (max_vyst,argmax_t[1]);
end

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

# Parameters
trial_run = parsed_args["trial"];
fine_sampling = false;
extra_precision = true;

function logrange(start,stop,len)
	@assert stop > start;
	@assert len > 1;
	stepmul = (stop/start) ^ (1/(len-1));
	return start * (stepmul .^ (0:len-1));
end

## Extract optimal values from df
# filename = "nice_t1_values_sweep.csv";
# df = CSV.read(filename, DataFrame; header=1);
# df.improvement = df.max_vy ./ df.max_vy_r; # Set the improvement column
# # df.improvement = df.max_vyst ./ df.max_vyst_r; # Set the improvement column
# max_filter_df = combine(sdf -> sdf[argmax(sdf.improvement), :], groupby(df, [:t1,:t2,:detune_ratio]));
# max_filter_df[:,[:t1,:t2,:detune_ratio,:vx]]

param_space = Dict();
if trial_run
    param_space["optimiser"] = multiple_shot::Optimisation
	param_space["ideal"] = [[70,50,0.05,0.6]]; # remember to convert to microseconds
    param_space["delta_gamma1"] = [-20,0,20];
    param_space["delta_gamma2"] = [-20,0,20];
else
    # (T1, T2, Detuning * T2, vx value) - remember to convert to microseconds
    param_space["optimiser"] = multiple_shot::Optimisation
    if parsed_args["single"]
        param_space["optimiser"] = single_shot::Optimisation
    end
    if param_space["optimiser"] == multiple_shot::Optimisation
        param_space["ideal"] = [
                        # Remember to choose optimal vx
                        [51,100,0.01,0.809017],
                        [75,100,0.01,0.78043],
                        [100,100,0.01,0.78043],
                        [200,100,0.01,0.81815],
                        [500,100,0.01,0.876307],
                        [1000,100,0.01,0.917755],
                        [51,100,0.1,0.81815],
                        [75,100,0.1,0.790155],
                        [100,100,0.1,0.790155],
                        [200,100,0.1,0.844328],
                        [500,100,0.1,0.883766],
                        [1000,100,0.1,0.929776],
                    ];
    else
        param_space["ideal"] = [
                        [51,100,0.01,0.718126],
                        [75,100,0.01,0.625243],
                        [100,100,0.01,0.575005],
                        [200,100,0.01,0.522499],
                        [500,100,0.01,0.522499],
                        [1000,100,0.01,0.522499],
                        [51,100,0.1,0.728969],
                        [75,100,0.1,0.625243],
                        [100,100,0.1,0.60042],
                        [200,100,0.1,0.575005],
                        [500,100,0.1,0.562083],
                        [1000,100,0.1,0.575005],
                    ];
    end
    # What miscalibration range to work on (in %). 0 represents miscalibrated data.
    param_space["delta_gamma1"] = (-20:2:20);#vcat(collect(-20:0.5:-10.5),collect(10.5:0.5:20));
    param_space["delta_gamma2"] = (-20:2:20); #vcat(collect(-20:0.5:-10.5),collect(10.5:0.5:20));
end

sampling = [];
if fine_sampling
	sampling = 1/sampling_rate;
end

abstol, reltol = 1e-6, 1e-3;
if extra_precision
	abstol, reltol = 1e-8,1e-10;
end

header = "";
header = header * "t1"; # T1
header = header * ",t2"; # T2
header = header * ",gamma1_miscalib_percent"; # T1 miscalibration percentage
header = header * ",gamma2_miscalib_percent"; # T2 miscalibration percentage
header = header * ",detune_ratio"; # Detuning freq * T2
header = header * ",vx"; # Starting state
header = header * ",ideal_breakdown_t"; # Breakdown time or max simulation time
if param_space["optimiser"] == multiple_shot::Optimisation
    header = header * ",vyst"; # Max vy - miscalibrated
    header = header * ",arg_t_vyst"; # Time when reading was taken
    header = header * ",vyst_r"; # Max vy - ramsey
    header = header * ",arg_t_vyst_r"; # Time for max vy - ramsey
else
    header = header * ",vy"; # Max vy - miscalibrated
    header = header * ",arg_t_vy"; # Time when reading was taken
    header = header * ",vy_r"; # Max vy - ramsey
    header = header * ",arg_t_vy_r"; # Time for max vy - ramsey
end
header = header * "\n";
print(header);

for (t1us, t2us, detune_ratio, vx) in param_space["ideal"]
	t2 = t2us * 10^-6;
    # T2 < 2*T1 based on loop condition ideally
    if t2us > 2*t1us
        continue
    end
    t1 = t1us * 10^-6;
    dephasing_gamma = 0.5*((1/t2)-(1/(2*t1)));
    thermal_gamma = 1 / t1;
    detuning_freq = detune_ratio/t2;
    tend = 10*t2;
    # Ramsey interference setup
    simulation_params = (
        detuned_free_decay::SimulationType, # simulation type
        detuning_freq,
        dephasing_gamma,
        thermal_gamma,
        nothing, # ideal solution/trajectory data
    );
    problem = ODEProblem(lindblad, [1,0,0], (0.0, tend), simulation_params);
    # For some reason, there's a problem with sampling rate
    ramsey_solution = solve(problem, alg_hints=[:stiff], saveat=sampling, abstol=abstol, reltol=reltol);
    if param_space["optimiser"] == multiple_shot::Optimisation
        (max_snr_r, argmax_t_snr_r) = get_max_vyst(ramsey_solution, detuning_freq>0);
    else
        (max_snr_r, argmax_t_snr_r) = get_max_vy(ramsey_solution, detuning_freq>0);
    end

    vz = sqrt(1-vx^2);
    # Ideal coherence magnitude preserving control
    v = [vx,0,vz];
    simulation_params = (
        ideal_tracking_control::SimulationType, # simulation type
        detuning_freq,
        dephasing_gamma,
        thermal_gamma,
        nothing, # ideal solution/trajectory data
    );
    problem = ODEProblem(lindblad, v, (0.0, tend), simulation_params);
    ideal_solution = solve(problem, alg_hints=[:stiff], saveat=sampling, abstol=abstol, reltol=reltol);

    # Detuned coherence magnitude preserving control
    simulation_params = (
        detuned_tracking_control::SimulationType, # simulation type
        detuning_freq,
        dephasing_gamma,
        thermal_gamma,
        ideal_solution, # ideal solution/trajectory data
    );
    problem = ODEProblem(lindblad, v, (0.0, tend), simulation_params);
    detuned_solution = solve(problem, alg_hints=[:stiff], saveat=sampling, abstol=abstol, reltol=reltol);
    if param_space["optimiser"] == multiple_shot::Optimisation
        (max_snr, argmax_t_snr) = get_max_vyst(detuned_solution, detuning_freq>0);
    else
        (max_snr, argmax_t_snr) = get_max_vy(detuned_solution, detuning_freq>0);
    end

    # Now change the calibration: T1 and T2 go off
    for gamma1_miscalib_ratio in param_space["delta_gamma1"]
		new_thermal_gamma = (1+gamma1_miscalib_ratio/100) * thermal_gamma;
        for gamma2_miscalib_ratio in param_space["delta_gamma2"]
            new_dephasing_gamma = (1+gamma2_miscalib_ratio/100) * dephasing_gamma;
            # if new_thermal_gamma > 2*new_dephasing_gamma
            #     continue
            # end
            # new_thermal_gamma = 1 / new_t1;
            # new_dephasing_gamma = 0.5*((1/new_t2)-(1/(2*new_t1)));
            
            simulation_params = (
                detuned_tracking_control::SimulationType, # simulation type
                detuning_freq,
                new_dephasing_gamma,
                new_thermal_gamma,
                ideal_solution, # ideal solution/trajectory data
            );
            problem = ODEProblem(lindblad, v, (0.0, tend), simulation_params);
            new_detuned_solution = solve(problem, alg_hints=[:stiff], saveat=sampling, abstol=abstol, reltol=reltol);
            if param_space["optimiser"] == multiple_shot::Optimisation
                new_signal = new_detuned_solution(argmax_t_snr)[2] / sqrt(argmax_t_snr);
            else
                new_signal = new_detuned_solution(argmax_t_snr)[2];
            end
				
            output = "";
            output = output * "$(t1us)"; # T1
            output = output * ",$(t2us)"; # T2
            output = output * ",$(gamma1_miscalib_ratio)"; # T1 miscalibration ratio
            output = output * ",$(gamma2_miscalib_ratio)"; # T2 miscalibration ratio
            output = output * ",$(detune_ratio)"; # Detuning freq * T2
            output = output * ",$(vx)"; # Starting state
            output = output * ",$(ideal_solution.t[end])"; # Breakdown time or max simulation time
            output = output * ",$(new_signal)"; # Max vy - for miscalibration
            output = output * ",$(argmax_t_snr)"; # Time for vy in detuned case, ideally
            output = output * ",$(max_snr_r)"; # Max vy - ramsey
            output = output * ",$(argmax_t_snr_r)"; # Time for max vy - ramsey
            print("$(output)\n");
		end
	end
end













