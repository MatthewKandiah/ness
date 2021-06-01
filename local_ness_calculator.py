import ness_calculator as nc 
from time import time

'''
Save start time so we can check how long calculation took
'''
START_TIME = time()

'''
Filenames defined here
'''
PLOT_FILENAME = "results.png"

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
CONVERGENCE_ATOL = 0.005

# calculate steady state and steady state entanglement of formation
steady_state, rc_dims = nc.calculate_converged_steady_state(CONVERGENCE_ATOL, system_parameter_dict, cold_bath, hot_bath, 'max_excitation_number', SOLVER_METHOD,SOLVER_TOLERANCE, memory_overflow_management = 'strict')
steady_concurrence = nc.calculate_concurrence(steady_state)
steady_entanglement_of_formation = nc.convert_concurrence_to_entanglement_of_formation(steady_concurrence)

# report time taken
CALCULATION_TIME = time() - START_TIME       
nc.debug_message(f"\n\nCalculation of steady state entanglement took {CALCULATION_TIME} seconds.")
nc.debug_message(f"The entanglement of formation was {steady_entanglement_of_formation}.")
nc.debug_message(f"The maximum excitation number used to achieve a convergence tolerance of {CONVERGENCE_ATOL} was n={rc_dims}")