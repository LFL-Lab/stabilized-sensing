using DifferentialEquations;
using LinearAlgebra;
using JLD2;

# The setup for the experiment is:
# Pure dephasing + relaxation + interaction with 2nd excited state in tracking control setting
# The ideal solution for tracking control is loaded first. The hamiltonian applied at each time
# step uses the same coefficient as the ideal hamiltonian but with the hamiltonian of the 3
# level system.
#
# Reduced planck's constant is set to 1 for the simulation, so we are working in angular units.
# This means the output is a Hamiltonian, with units in angular frequency. Rabi frequency
# is linear frequency, so divide by 2pi to get in terms of angular frequency.

# Experiment setup
h = 6.62607015 * 1e-34; # Planck's constant - Joule per Hertz
kb = 1.380649 * 1e-23; # Boltzmann's constant - Joule per Kelvin
η = -(210*1e6)*2*pi; # E12-E10 in angular frequency

# 1/t2 = 1/(2t1) + 1/t_phi.
# t_phi - corresponds to dephasing. Equal to 1/gamma
# t1 - corresponds to thermal relaxation.
t1 = 70 * 1e-6;
t2 = 90 * 1e-6;
dephasing_gamma = ((1/t2)-(1/(2*t1)));
thermal_gamma = 1 / t1;

# Arguments
# if length(ARGS) == 0
# 	print("Needs detuning frequency as argument\n");
# 	exit();
# end
# detuning_freq = parse(Float64, ARGS[1]);

# Sampling rate is 2.4 giga samples per sec
sampling_rate = 1000/min(t1,t2);
qbit_freq = 3877496000;
bath_temp = 50 * 10e-3;

# Simulation code
sigma_x = [0 1.0 0; 1.0 0 0; 0 0 0];
sigma_y = [0 -1.0im 0; 1.0im 0 0; 0 0 0];
sigma_z = [1.0 0 0; 0 -1.0 0; 0 0 0];
sigma_i = [1.0 0 0; 0 1.0 0; 0 0 0];

# a = sum_n sqrt(n+1) |n><n+1|
a = Bidiagonal(zeros(Float64,3), [sqrt(i)*(1) for i = 1:2], :U);
h_x = 2*(a + a')/2;
h_y = 2*1im*(-a + a')/2;
h_z = 2*(a'*a); # Extra factor of 2 because energy gap should be same as in sigma_z

@enum SimulationType begin
   ramsey_detuned = 1
   ideal = 2
   detuned = 3
end;

commutator(x, y) = x*y - y*x;

function lindblad(L,rho)
	return L*rho*L'-0.5*(rho*L'*L+L'*L*rho);
end

function dissipator(rho)
	# return zeros(3,3);
	dephasing_dissipator = dephasing_gamma * lindblad(h_z, rho);
	thermal_dissipator = thermal_gamma * lindblad(a, rho);
	# dephasing_dissipator = dephasing_gamma * lindblad(sigma_z, rho);
	# thermal_dissipator = thermal_gamma * lindblad(Bidiagonal(zeros(3),[1.0,0.0],:U), rho);
	return dephasing_dissipator + thermal_dissipator;
end

function target(v)
	return v[1]^2 + v[2]^2;
end

# Returns matrix form of hamiltonian given X, Y, Z coefficients
function vec2ham(v)
	# return v[1]*sigma_x + v[2]*sigma_y + v[3]*sigma_z;
	return v[1]*h_x + v[2]*h_y + v[3]*h_z;
end

# Returns hamiltonian matrix in 3 level system
# Input: 3d vector representing bloch vector in ideal solution
function get_hamiltonian(p,t)
	system_hamiltonian = [0 0 0; 0 0 0; 0 0 η];
	# return sigma_z;
	# return h_x/t2+system_hamiltonian;
	v = p(t);
	dephasing_hamiltonian = -dephasing_gamma / v[3] * vec2ham([v[2], -v[1], 0]);
	thermal_hamiltonian = -thermal_gamma / (4 * v[3]) * vec2ham([v[2], -v[1], 0]);
	return dephasing_hamiltonian + thermal_hamiltonian + system_hamiltonian;
end

function master_equation(rho, p, t)
	hamiltonian = get_hamiltonian(p,t);
	if any(isnan, hamiltonian)
		return [Inf*1.0im Inf Inf;Inf Inf Inf; Inf Inf Inf];
	end
	return -1im * commutator(hamiltonian, rho) + dissipator(rho);
end

JLD2.@load "ideal_solution.jld2" ideal_solution
v = ideal_solution(0);
rho = (v[1] * sigma_x + v[2] * sigma_y + v[3] * sigma_z + sigma_i)/2;
abstol, reltol = 1e-8,1e-6;
tend = 30*t2;

problem = ODEProblem(master_equation, rho, (0.0, tend), (ideal_solution));
@time solution = solve(problem, alg_hints=[:stiff], saveat=t2/10000);

print("Initial state: ", v, "\n");
print("T1, T2: ", t1, ", ", t2, "\n");
print("Simulation time for solution: ", solution.t[end], "\n");
using Plots;
plotly();
time_range = range(0,tend,step=t2/100);
x = [2*real(solution(t)[1,2]) for t in time_range];
y = [2*imag(solution(t)[1,2]) for t in time_range];
z = [2*real(solution(t)[1,1])-1 for t in time_range];
plot(time_range, [x y z], show=true,size=(1500,800), title="T1=$(t1*10^6)us, T2=$(t2*10^6)us", label=["vx (qtrit)" "vy (qtrit)" "vz (qtrit)"]);
JLD2.@load "detuned_solution.jld2" detuned_solution
plot!(detuned_solution, size=(1500,800), show=true, label=["vx detuned (1 kHz)" "vy detuned (1 kHz)" "vz detuned (1 kHz)"]);




