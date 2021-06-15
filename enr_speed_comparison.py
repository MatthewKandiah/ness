import ness_calculator as nc
from time import time

'''
Define model parameters here
'''
epsilon = 0
Delta = 1
g = 0.02

alpha_cold = 0.5
Gamma_cold = 0.04
omega0_cold= 10
cold_bath_temperature = 0.01

alpha_hot = 0.2
Gamma_hot = 0.01
omega0_hot= 20
hot_bath_temperature = 5

'''
Build parameter dictionaries
'''
system_parameter_dict = {}
system_parameter_dict["epsilon"] = epsilon
system_parameter_dict["Delta"] = Delta
system_parameter_dict["interqubit_coupling_strength"] = g

cold_bath_parameter_dict = {}
cold_bath_parameter_dict["alpha"] = alpha_cold
cold_bath_parameter_dict["Gamma"] = Gamma_cold
cold_bath_parameter_dict["omega0"] = omega0_cold

hot_bath_parameter_dict = {}
hot_bath_parameter_dict["alpha"] = alpha_hot
hot_bath_parameter_dict["Gamma"] = Gamma_hot
hot_bath_parameter_dict["omega0"]= omega0_hot

'''
Build spectral densities and baths
'''
cold_bath_spectral_density = nc.spectral_density('underdamped',cold_bath_parameter_dict)
cold_bath = nc.bath(cold_bath_spectral_density, cold_bath_temperature)
hot_bath_spectral_density = nc.spectral_density('underdamped',hot_bath_parameter_dict)
hot_bath = nc.bath(hot_bath_spectral_density, hot_bath_temperature)

# steady state solver options
SOLVER_METHOD = "iterative-gmres"
SOLVER_TOLERANCE = 1e-12

print("Constructing and solving problems for time comparison...")
# calculate supersystem steady state using both "simple" and "max_excitation_number" truncations
max_occupation_number = 3
start_time = time()
liouvillian_simple = nc.build_Liouvillian(max_occupation_number,system_parameter_dict,cold_bath,hot_bath,'simple')
construction_time_liouvillian_simple = time()
steady_state_simple= nc.solve_supersystem_steady_state(liouvillian_simple, SOLVER_METHOD, SOLVER_TOLERANCE)
completion_time_steady_state_simple = time()
liouvillian_enr    = nc.build_Liouvillian(max_occupation_number,system_parameter_dict,cold_bath,hot_bath,'max_excitation_number')
construction_time_liouvillian_enr = time()
steady_state_enr   = nc.solve_supersystem_steady_state(liouvillian_enr, SOLVER_METHOD, SOLVER_TOLERANCE)
completion_time_steady_state_enr = time()

# calculate durations of different processes
construction_duration_liouvillian_simple = construction_time_liouvillian_simple - start_time
construction_duration_liouvillian_enr    = construction_time_liouvillian_enr    - completion_time_steady_state_simple
calculation_duration_steady_state_simple = completion_time_steady_state_simple  - construction_time_liouvillian_simple
calculation_duration_steady_state_enr    = completion_time_steady_state_enr     - construction_time_liouvillian_enr

# report results
print("Results of time comparison:")
print(f"\ttruncation_method = \"simple\":\n\t\tconstruction: \t{construction_duration_liouvillian_simple}\n\t\tcalculation: \t{calculation_duration_steady_state_simple}")
print(f"\ttruncation_method = \"max_excitation_number\":\n\t\tconstruction: \t{construction_duration_liouvillian_enr}\n\t\tcalculation: \t{calculation_duration_steady_state_enr}")
