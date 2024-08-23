using DifferentialEquations;
using LinearAlgebra;
using ArgParse;
using BSON;

s = ArgParseSettings()
@add_arg_table s begin
    "--trial"
        help = "whether this is a trial run"
        action = :store_true
    "--breakdown"
        help = "whether to run simulation over breakdown regions only"
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

param_space = Dict();
if trial_run
    param_space["t2"] = [50] # remember to convert to microseconds
    param_space["t1"] = [[70]] # remember to convert to microseconds
    param_space["detune_ratio"] = [0.44,-0.2,-0.5]
else
    param_space["t2"] = [100] # range(10,200,step=10); # remember to convert to microseconds
    param_space["t1"] = [append!(collect(50:5:100),collect(110:10:200),[500])]
            #[[51,75,100,200,500,1000]]
            #[[x * 100 for x in 0.51:0.1:2]];#[logrange(ceil(x/2),20*x,20) for x in param_space["t2"]]; # remember to convert to microseconds
    param_space["detune_ratio"] = collect(0.5:-0.01:0); 
            # append!(collect(0.01:0.01:0.1), collect(-0.01:-0.01:-0.1), collect(0.1:0.05:0.5), collect(-0.1:-0.05:-0.5))# [0.01,0.1]
            # append!(1 ./ collect(range(10,100,length=10)), -1 ./ collect(range(10,100,length=10)),
            # collect(range(1/10,1/2,length=10)), -1 .* collect(range(1/10,1/2,length=10)))
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
header = header * ",detune_ratio"; # Detuning freq * T2
header = header * ",vx"; # Starting state
header = header * ",ideal_breakdown_t"; # Breakdown time or max simulation time
header = header * ",max_vy"; # Max vy - coherence magnitude
header = header * ",argmax_t_vy"; # Time for max vy - coherence magnitude
header = header * ",max_vy_r"; # Max vy - ramsey
header = header * ",argmax_t_vy_r"; # Time for max vy - ramsey
header = header * ",max_vyst"; # Max vy/sqrt(t) - coherence magnitude
header = header * ",argmax_t_vyst"; # Time for max vy/sqrt(t) - coherence magnitude
header = header * ",max_vyst_r"; # Max vy/sqrt(t) - ramsey
header = header * ",argmax_t_vyst_r"; # Time for max vy/sqrt(t) - ramsey
header = header * ",vy_end"; # vy end - coherence magnitude
header = header * ",vy_end_r"; # vy end - ramsey
header = header * "\n";
print(header);
for detune_ratio in param_space["detune_ratio"]
for (index, t2us) in enumerate(param_space["t2"])
    t2 = t2us * 10^-6;
    for t1us in param_space["t1"][index]
        # T2 < 2*T1 based on loop condition ideally
        if t2us > 2*t1us
            continue
        end
        t1 = t1us * 10^-6;
        dephasing_gamma = 0.5*((1/t2)-(1/(2*t1)));
        thermal_gamma = 1 / t1;
        # vx_maximum for stable regions is supposed to be
        # 0.5*sqrt(2 * thermal_gamma / (4*dephasing_gamma + thermal_gamma)),
        # which is the same as 0.5*sqrt(t2:t1)
        stable_endpoint = 0.5*sqrt(t2us/t1us);
        vx_range = [sin(x*pi) for x in 0.005:0.005:0.49];
        vx_minimum = 0.04; #parsed_args["breakdown"] ? 0.5*sqrt(t2us/t1us) : 0.02;
        vx_maximum = 0.96; #parsed_args["breakdown"] ? 0.96 : 0.5*sqrt(t2us/t1us);
        
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
            max_vyst_r, argmax_t_vyst_r = get_max_vyst(ramsey_solution, detuning_freq>0);
            max_vy_r, argmax_t_vy_r = get_max_vy(ramsey_solution, detuning_freq>0);
            for buff_vx in vx_range
                vx = buff_vx;
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
                
                (max_vyst, argmax_t_vyst) = get_max_vyst(detuned_solution, detuning_freq>0);
                max_vy, argmax_t_vy = get_max_vy(detuned_solution, detuning_freq>0);
                
                # alpha = vx^2*t1/t2-1/4;
                # predicted = -t1*( 0.5*log((0.25+alpha)/((vz-0.5)^2+alpha)) + 1/(2*sqrt(alpha)) * (atan(-0.5/sqrt(alpha))-atan((vz-0.5)/sqrt(alpha))));
                output = "";
                output = output * "$(t1us)"; # T1
                output = output * ",$(t2us)"; # T2
                output = output * ",$(detune_ratio)"; # Detuning freq * T2
                output = output * ",$(vx)"; # Starting state
                output = output * ",$(ideal_solution.t[end])"; # Breakdown time or max simulation time
                output = output * ",$(max_vy)"; # Max vy - coherence magnitude
                output = output * ",$(argmax_t_vy)"; # Time for max vy
                output = output * ",$(max_vy_r)"; # Max vy - ramsey
                output = output * ",$(argmax_t_vy_r)"; # Time for max vy - ramsey
                output = output * ",$(max_vyst)"; # Max vy - coherence magnitude
                output = output * ",$(argmax_t_vyst)"; # Time for max vy
                output = output * ",$(max_vyst_r)"; # Max vy - ramsey
                output = output * ",$(argmax_t_vyst_r)"; # Time for max vy - ramsey
                output = output * ",$(detuned_solution.u[end][2])"; # End vy (CM)
                output = output * ",$(ramsey_solution.u[end][2])"; # End vy (Ramsey)
                print("$(output)\n");
                # filename = "raw_data/$(t1us),$(t2us),$(detune_ratio),$(vx).bson";
                # BSON.@save filename detuned_solution;
            end
        end
    end
end













