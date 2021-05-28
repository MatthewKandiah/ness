START_TIME = time.time()

'''
Filenames defined here
'''
PLOT_FILENAME = "results.png"

'''
Define model parameters here
'''
system_parameter_dict = {}
system_parameter_dict["epsilon"] = 0 
system_parameter_dict["Delta"] = 1
system_parameter_dict["interqubit_coupling_strength"] = 0.02 

cold_bath_parameter_dict = {}
cold_bath_parameter_dict["alpha"] = 0.5
cold_bath_parameter_dict["Gamma"] = 0.04
cold_bath_parameter_dict["omega0"] = 10
cold_bath_spectral_density = spectral_density('underdamped',cold_bath_parameter_dict)
cold_bath_temperature = 0.01
cold_bath = bath(cold_bath_spectral_density, cold_bath_temperature)

hot_bath_parameter_dict = {}
hot_bath_parameter_dict["alpha"] = 0.2
hot_bath_parameter_dict["Gamma"] = 0.01
hot_bath_parameter_dict["omega0"]= 2
hot_bath_spectral_density = spectral_density('underdamped',hot_bath_parameter_dict)

minimum_hot_bath_temperature = 0.01
maximum_hot_bath_temperature = 10
number_of_plot_points = 2
# hot bath temperatures for plot
temperature_step = (maximum_hot_bath_temperature - minimum_hot_bath_temperature)/number_of_plot_points
hot_bath_temperatures = np.arange(minimum_hot_bath_temperature,maximum_hot_bath_temperature,temperature_step)
beta_hot_array = 1/hot_bath_temperatures

"""
Plot the two bath spectral densities
"""
plot_frequencies = np.arange(0,20.1,0.1)
plot_spectral_density_cold = []
plot_spectral_density_hot = []
for frequency in plot_frequencies:
    plot_spectral_density_cold.append(cold_bath_spectral_density.get_value(frequency))
    plot_spectral_density_hot.append(hot_bath_spectral_density.get_value(frequency))

plt.plot(plot_frequencies,plot_spectral_density_cold)
plt.savefig("spectral_density_cold.png")
plt.clf()

plt.plot(plot_frequencies,plot_spectral_density_hot )
plt.savefig("spectral_density_hot.png" )
plt.clf()

# lists for plot points
concurrence_list = []
entanglement_of_formation_list = []
rc_dims_for_convergence_list = []

# steady state solver options
SOLVER_METHOD = "iterative-gmres"
SOLVER_TOLERANCE = 1e-12
CONVERGENCE_ATOL = 0.001

# calculate steady state entanglement of formation for plot against hot bath temperature
for hot_bath_temperature in hot_bath_temperatures:
    hot_bath = bath(hot_bath_spectral_density, hot_bath_temperature)
    steady_state, rc_dims = calculate_converged_steady_state(CONVERGENCE_ATOL, system_parameter_dict, cold_bath, hot_bath, 'max_excitation_number', SOLVER_METHOD,SOLVER_TOLERANCE, memory_overflow_management = 'strict')
    
    steady_concurrence = calculate_concurrence(steady_state)
    steady_entanglement_of_formation = convert_concurrence_to_entanglement_of_formation(steady_concurrence)
    
    concurrence_list.append(steady_concurrence)
    entanglement_of_formation_list.append(steady_entanglement_of_formation)
    rc_dims_for_convergence_list.append(rc_dims)

        
print(f"\n\nCalculation of plot points took {time.time()-START_TIME} seconds.")

print(f"Saving plot of results to {PLOT_FILENAME}")
plt.plot(hot_bath_temperatures, entanglement_of_formation_list)
plt.savefig(PLOT_FILENAME)
plt.clf()