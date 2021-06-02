# Build a Liouvillian for some set of parameters, and check how sparse it is. 
# Then drop elements of magnitude less than drop_tolerance, and check again for sparsity.
# Repeat for different values of drop_tolerance.

import ness_calculator as nc
from math import prod

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

test_list = [0,1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]
simple_list=[]
enr_list  = []
for i in range(0,len(test_list)):
	'''
	Build Liouvillians, using both simple and max_excitation_number truncation methods
	'''
	drop_tolerance = test_list[i]
	max_occupation_number = 3
	print(f"Constructing Liouvillian {i+1}/{len(test_list)}")
	liouvillian_simple = nc.build_Liouvillian(max_occupation_number,system_parameter_dict,cold_bath,hot_bath,'simple',drop_tolerance=drop_tolerance)
	liouvillian_enr    = nc.build_Liouvillian(max_occupation_number,system_parameter_dict,cold_bath,hot_bath,'max_excitation_number',drop_tolerance=drop_tolerance)

	sparse_matrix_simple = liouvillian_simple.data
	sparse_matrix_enr    = liouvillian_enr.data

	simple_list.append(sparse_matrix_simple)
	enr_list.append(sparse_matrix_enr)

for i in range(0,len(test_list)):
	drop_tolerance = test_list[i]
	sparse_matrix_simple = simple_list[i]
	sparse_matrix_enr    = enr_list[i]
	print(f"drop_tolerance: {drop_tolerance}")
	#print(f"\tsimple truncation:\n\t\texplicitly stored elements:\t{sparse_matrix_simple.nnz}\n\t\tnon-zero elements:\t\t{sparse_matrix_simple.count_nonzero()}\n\t\ttotal elements:\t\t\t{prod(sparse_matrix_simple.shape)}")
	print(f"\tenr truncation:   \n\t\texplicitly stored elements:\t{sparse_matrix_enr.nnz   }\n\t\tnon-zero elements:\t\t{sparse_matrix_enr.count_nonzero()   }\n\t\ttotal elements:\t\t\t{prod(sparse_matrix_enr.shape)   }")
