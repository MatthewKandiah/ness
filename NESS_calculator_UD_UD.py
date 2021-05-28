import numpy as np
import scipy.stats
from qutip import *
import matplotlib.pyplot as plt
import time

# set to true to print extra information to console
DEBUG = True
'''
Choose steadystate solver method (see qutip documentation for valid options)
'''
SOLVER_METHOD = "iterative-gmres"
SOLVER_TOLERANCE = 1e-12

# system operators, expressed in sigma_z eigenbases
sigz_cold = tensor(sigmaz(), qeye(2))
sigz_hot  = tensor(qeye(2), sigmaz())
sigx_cold = tensor(sigmax(), qeye(2))
sigx_hot  = tensor(qeye(2), sigmax())

def debug_message(message_string):
    if DEBUG:
        print(message_string)

class spectral_density:
    allowed_types = ['underdamped']
    
    def __init__(self, type, parameter_dict):
        if type in allowed_types:
            self.type = type 
        else:
            raise ValueError
        if type == 'underdamped':
            self.alpha = parameter_dict[alpha]
            self.Gamma = parameter_dict[Gamma] 
            self.omega0 = parameter_dict[omega0]

    def get_value(frequency):
        if type == 'underdamped':
            value = self.alpha * self.Gamma * self.omega0**2 * frequency / ((self.omega0**2 - frequency**2)**2 + (Gamma*frequency)**2)
        
        return value

    def get_mapped_parameters():
        if type == 'underdamped':
            RC_frequency = self.omega0
            RC_system_coupling_strength = np.sqrt(np.pi * self.alpha * self.omega0 / 2)
            RC_environment_coupling_strength = self.Gamma / (2 * np.pi * self.omega0)

        return RC_frequency, RC_system_coupling_strength, RC_environment_coupling_strength

class bath:
    def __init__(self, spectral_density, temperature):
        self.temperature = temperature
        self.spectral_density = spectral_density
   

# calculate reaction coordinate frame parameters from original representation parameters
# use same notation as accompanying notes
"""
s = coupling strength
Gamma = peak width
omega0 = peak centre
"""
def calculate_RC_frequency(s, Gamma, omega0):
    return omega0
 
def calculate_RC_system_coupling_strength(s, Gamma, omega0):
    return np.sqrt(np.pi * s * omega0 / 2)
    
def calculate_RC_environment_coupling_strength(s, Gamma, omega0):
    return Gamma / (2 * np.pi * omega0)

# define system Hamiltonian
def HamS(epsilon, Delta, interqubit_coupling_strength):
    return (epsilon/2) * (sigz_cold + sigz_hot) + (Delta/2) * (sigx_cold + sigx_hot) + interqubit_coupling_strength*sigz_cold*sigz_hot

# construct super-system Hamiltonian for given maximum reaction coordinate occupation number
# order all tensor products as follows
"""
TENSOR PRODUCT ORDER
cold qubit TENSOR hot qubit TENSOR cold RC TENSOR hot RC
"""
def build_HamSS(max_occupation_number,epsilon, Delta, interqubit_coupling_strength,
                s_cold, Gamma_cold, omega0_cold, s_hot, Gamma_hot, omega0_hot, truncation_method = "simple") :

    # cold RC values 
    RC_frequency_cold = calculate_RC_frequency(s_cold, Gamma_cold, omega0_cold)
    RC_system_coupling_strength_cold = calculate_RC_system_coupling_strength(s_cold, Gamma_cold, omega0_cold)
    RC_environment_coupling_strength_cold = calculate_RC_environment_coupling_strength(s_cold, Gamma_cold, omega0_cold)
    
    # hot RC values
    RC_frequency_hot = calculate_RC_frequency(s_hot, Gamma_hot, omega0_hot)
    RC_system_coupling_strength_hot = calculate_RC_system_coupling_strength(s_hot, Gamma_hot, omega0_hot)
    RC_environment_coupling_strength_hot = calculate_RC_environment_coupling_strength(s_hot, Gamma_hot, omega0_hot)
    
    # dimensions of RC spaces truncated to given max_occupation_number
    rc_dims = max_occupation_number + 1
    
    # construct full supersystem Hamiltonian 
    if truncation_method == "simple":
        # system Hamiltonian
        extended_HamS = tensor(HamS(epsilon, Delta, interqubit_coupling_strength),qeye(rc_dims),qeye(rc_dims))
                
        # cold RC occupation 
        Ham_num_RC_cold = RC_frequency_cold * tensor(qeye(2),qeye(2),num(rc_dims),qeye(rc_dims))
        
        # cold RC - system coupling
        Ham_RC_system_cold = RC_system_coupling_strength_cold * tensor(sigz_cold,create(rc_dims) + destroy(rc_dims),qeye(rc_dims))
        
        # hot RC occupation 
        Ham_num_RC_hot = RC_frequency_hot * tensor(qeye(2),qeye(2),qeye(rc_dims),num(rc_dims))
        
        # hot RC - system coupling
        Ham_RC_system_hot = RC_system_coupling_strength_hot * tensor(sigz_hot, qeye(rc_dims), create(rc_dims) + destroy(rc_dims))
        
    elif truncation_method == "max_excitation_number":
        # interpret max_occupation_number as maximum excitation number for the pair of RCs
        # see comment above build_Liouvillian for more detailed explanation
        
        # load ENR dictionaries (allows us to use the built-in qutip functions for excitation number restricted operators
        dims = [rc_dims] * 2
        excitations = max_occupation_number
        nstates, state2idx, idx2state = enr_state_dictionaries(dims, excitations)
        
        # define ENR destruction operator for cold and hot RCs 
        enr_destroy_cold, enr_destroy_hot = enr_destroy(dims, excitations)
        
        # define ENR creation operator for cold and hot RCs
        enr_create_cold = enr_destroy_cold.dag()
        enr_create_hot  = enr_destroy_hot.dag()
        
        # define ENR number operator for cold and hot RCs
        enr_number_cold = enr_create_cold * enr_destroy_cold
        enr_number_hot  = enr_create_hot  * enr_destroy_hot 
        
        # system Hamiltonian
        extended_HamS = tensor(HamS(epsilon, Delta, interqubit_coupling_strength), enr_identity(dims,excitations))
                
        # cold RC occupation 
        Ham_num_RC_cold = RC_frequency_cold * tensor(qeye(2),qeye(2),enr_number_cold)
        
        # cold RC - system coupling
        Ham_RC_system_cold = RC_system_coupling_strength_cold * tensor(sigz_cold,enr_create_cold + enr_destroy_cold)
        
        # hot RC occupation 
        Ham_num_RC_hot = RC_frequency_hot * tensor(qeye(2),qeye(2),enr_number_hot)
        
        # hot RC - system coupling
        Ham_RC_system_hot = RC_system_coupling_strength_hot * tensor(sigz_hot, enr_create_hot + enr_destroy_hot)  
        
        
    final_HamSS = extended_HamS + Ham_num_RC_cold + Ham_num_RC_hot + Ham_RC_system_cold + Ham_RC_system_hot 
    return final_HamSS
    
# construct Liouvillian operator in vectorised space
# if truncation_method == "simple", then max_occupation_number is interpreted as the largest occupation number of each RC in the truncation.
# i.e. if max_occupation_number = 5 and truncation_method =="simple", each reaction coordinate is truncated to a 6 level system.
# if truncation_method == "max_excitation_number", then max_occupation_number is interpreted as the largest excitation number of the two RCs included in the truncation
# i.e. if max_occupation_number = 3 and truncation_method == "max_excitation_number", then the set of basis states spanning the truncated RC space is
# { Ket(0,0), Ket(0,1), Ket(0,2), Ket(0,3), Ket(1,0), Ket(1,1), Ket(1,2), Ket(2,0), Ket(2,1), Ket(3,0) }
def build_Liouvillian(max_occupation_number,epsilon, Delta, interqubit_coupling_strength,s_cold, Gamma_cold, omega0_cold, s_hot, Gamma_hot, omega0_hot,beta_cold,beta_hot,truncation_method = "simple"):
    
    # build HamSS for given parameters
    HamSS = build_HamSS(max_occupation_number,epsilon, Delta, interqubit_coupling_strength,
                s_cold, Gamma_cold, omega0_cold, s_hot, Gamma_hot, omega0_hot, truncation_method = truncation_method)
                
    # calculate eigensystem of HamSS numerically
    debug_message("\tCalculating supersystem Hamiltonian eigenvectors...")
    # qutip documentation says sparse solver is slower than dense solver, only use sparse if memory is limiting calculation speed
    eigenvalues, eigenstates = HamSS.eigenstates(sparse = True, tol = 1e-20, maxiter = 10000)
    
    debug_message("\t\t...Supersystem eigenvectors successfully calculated!")
    
    # truncated reaction coordinate space dimensions
    rc_dims = max_occupation_number + 1
    
    if truncation_method == "simple":
    
        A_cold = tensor(qeye(2),qeye(2),create(rc_dims) + destroy(rc_dims), qeye(rc_dims))
        A_hot  = tensor(qeye(2),qeye(2),qeye(rc_dims), create(rc_dims) + destroy(rc_dims))
    
    # if using the maximum_excitation_number truncation, then we want to project these operators into the truncated space.
    # take care to make sure that this projection is the same as the projection done in constructing HamSS
    elif truncation_method == "max_excitation_number":
        
        # load ENR dictionaries (allows us to use the built-in qutip functions for excitation number restricted operators
        dims = [rc_dims] * 2
        excitations = max_occupation_number
        nstates, state2idx, idx2state = enr_state_dictionaries(dims, excitations)
        
        # define ENR destruction operator for cold and hot RCs 
        enr_destroy_cold, enr_destroy_hot = enr_destroy(dims, excitations)
        
        # define ENR creation operator for cold and hot RCs
        enr_create_cold = enr_destroy_cold.dag()
        enr_create_hot  = enr_destroy_hot.dag()
        
        A_cold = tensor(qeye(2), qeye(2), enr_create_cold + enr_destroy_cold)
        A_hot  = tensor(qeye(2), qeye(2), enr_create_hot  + enr_destroy_hot )
        
    # calculate rate operators
    debug_message("\tCalculating rate operators chi and Xi...")
    
    chi_operator_cold = 0
    chi_operator_hot  = 0
    Xi_operator_cold = 0
    Xi_operator_hot  = 0
    
    RC_environment_coupling_strength_cold = calculate_RC_environment_coupling_strength(s_cold, Gamma_cold, omega0_cold)
    RC_environment_coupling_strength_hot  = calculate_RC_environment_coupling_strength(s_hot,  Gamma_hot,  omega0_hot )
    
    for j in range(0,len(eigenstates)) :
        for k in range(0,len(eigenstates)) :
            bra_j = eigenstates[j].dag()
            ket_k = eigenstates[k]
            
            Ajkcold = (bra_j * A_cold * ket_k)
            Ajkhot  = (bra_j * A_hot  * ket_k)
            
            # consider cases where eigenvalues are equal or different separately for calculating contributions to chi operators 
            if eigenvalues[j] != eigenvalues[k]:
                chi_operator_cold += (np.pi/2) * (RC_environment_coupling_strength_cold * Ajkcold * (eigenvalues[j] - eigenvalues[k]) / np.tanh(beta_cold * (eigenvalues[j] - eigenvalues[k]) / 2)) * (eigenstates[j] * eigenstates[k].dag())
                
                chi_operator_hot  += (np.pi/2) * (RC_environment_coupling_strength_hot  * Ajkhot  * (eigenvalues[j] - eigenvalues[k]) / np.tanh(beta_hot  * (eigenvalues[j] - eigenvalues[k]) / 2)) * (eigenstates[j] * eigenstates[k].dag())
                
            # if the two eigenvalues are the same, then tanh(0) = 0 so we will get a value error because you aren't allowed to divide by zero. Need to manually account for the fact that x*coth(x) is equal to 1 at x=0.
            # this case is guaranteed to arise when j == k. If HamSS is degenerate, there may other cases too.
            elif eigenvalues[j] == eigenvalues[k]:
                chi_operator_cold += (np.pi/2) * (RC_environment_coupling_strength_cold * Ajkcold) * (2/beta_cold) *(eigenstates[j] * eigenstates[k].dag())
                
                chi_operator_hot += (np.pi/2) * (RC_environment_coupling_strength_hot  * Ajkhot ) * (2/beta_hot ) *(eigenstates[j] * eigenstates[k].dag())
                
            else:
                debug_message("ERROR: looping through eigenvalue pairs in calculation of rate operators, found pair of eigenvalues which are neither equal nor different.")

            # all eigenvalue pairs can be treated the same way when calulating Xi operators
            Xi_operator_cold += (np.pi/2) * (RC_environment_coupling_strength_cold * Ajkcold * (eigenvalues[j] - eigenvalues[k])) * (eigenstates[j] * eigenstates[k].dag())
                
            Xi_operator_hot  += (np.pi/2) * (RC_environment_coupling_strength_hot  * Ajkhot  * (eigenvalues[j] - eigenvalues[k])) * (eigenstates[j] * eigenstates[k].dag())
    debug_message("\t\t...rate operators chi and Xi successfully calculated!")
    
    # construct Liouvillian from master equation
    unitary_liouvillian = -1j * spre(HamSS) + 1j * spost(HamSS)
    cold_liouvillian = -spre(A_cold * chi_operator_cold) + spre(A_cold)*spost(chi_operator_cold) + spre(chi_operator_cold)*spost(A_cold) - spost(chi_operator_cold * A_cold) - spre(A_cold * Xi_operator_cold) - spre(A_cold)*spost(Xi_operator_cold) + spre(Xi_operator_cold)*spost(A_cold) + spost(Xi_operator_cold * A_cold)
    hot_liouvillian  =  - spre(A_hot * chi_operator_hot) + spre(A_hot)*spost(chi_operator_hot) + spre(chi_operator_hot)*spost(A_hot) - spost(chi_operator_hot * A_hot) - spre(A_hot * Xi_operator_hot) - spre(A_hot) * spost(Xi_operator_hot) +spre(Xi_operator_hot) * spost(A_hot) + spost(Xi_operator_hot * A_hot)
    
    final_liouvillian = unitary_liouvillian + cold_liouvillian + hot_liouvillian
    return final_liouvillian


def solve_supersystem_steady_state(liouvillian,sparse = True, solver_method = SOLVER_METHOD, absolute_tolerance = SOLVER_TOLERANCE):
    # see qutip documentation Section 3.6.2 for relevant information on steady state solver
    debug_message("\tCalculating steady state from Liouvillian...")
    rhoSS_steady = steadystate(liouvillian, method=solver_method, sparse=sparse, tol = absolute_tolerance,maxiter = 30000,use_precond = True)
    debug_message("\t\t...Steady state successfully calculated!")
    return rhoSS_steady
    
def solve_system_steady_state(liouvillian, solver_method = SOLVER_METHOD, absolute_tolerance = SOLVER_TOLERANCE):
    rhoSS_steady = solve_supersystem_steady_state(liouvillian, sparse=True, solver_method = solver_method, absolute_tolerance = absolute_tolerance)
    rhoS_steady = rhoSS_steady.ptrace([0,1])
    return rhoS_steady

def find_max_absolute_difference_between_QObjs(QObj1,QObj2):
    # matrices assumed to have the same dimensions
    matrix1 = QObj1.full()
    matrix2 = QObj2.full()
    max_difference = 0
    for i in range(0,len(matrix1)):
        for j in range(0,len(matrix1[0])):
            difference = abs(matrix1[i][j] - matrix2[i][j])
            if difference > max_difference:
                max_difference = difference
                
    return max_difference
 
def find_max_relative_difference_between_QObjs(QObj1, QObj2):
    # matrices assumed to have the same dimensions
    matrix1 = QObj1.full()
    matrix2 = QObj2.full()
    max_relative_difference = 0
    for i in range(0,len(matrix1)):
        for j in range(0,len(matrix1[0])):
            difference = abs(matrix1[i][j] - matrix2[i][j])
            if matrix1[i][j] != 0:
                relative_difference = abs(difference/matrix1[i][j])
            elif matrix2[i][j] != 0:
                relative_difference = abs(difference/matrix2[i][j])
            else:
                # matrix1[i][j] and matrix2[i][j] both zero
                relative_difference = 0
            
            if relative_difference > max_relative_difference:
                max_relative_difference = relative_difference
                
    return max_relative_difference
    
def find_max_absolute_difference_between_populations(rho1, rho2):
    populations1 = rho1.diag()
    populations2 = rho2.diag()
    max_difference = 0
    
    for j in range(0,len(populations1)):
        pop1 = populations1[j]
        pop2 = populations2[j]
        difference = abs(pop1 - pop2)
        if difference > max_difference:
            max_difference = difference
    
    return max_difference
  
# return converged_steady_state, max_occupation_number used to obtain convergence  
# memory_overflow_management set to strict => throw an error if memory overflow prevents calculation being completed.
# memory_overflow_management set to relaxed => return last calculated state as the converged state if memory overflow prevents next calculation. 
def calculate_converged_steady_state(convergence_tolerance, epsilon, Delta, interqubit_coupling_strength, s_cold, Gamma_cold, omega0_cold, s_hot, Gamma_hot, omega0_hot, beta_cold, beta_hot, truncation_method = 'max_excitation_number', solver_method = SOLVER_METHOD, solver_tolerance = SOLVER_TOLERANCE, memory_overflow_management = 'strict'):
    convergence_confirmed = False
    max_occupation_number = 1
    liouvillian = build_Liouvillian(max_occupation_number, epsilon, Delta, interqubit_coupling_strength, s_cold, Gamma_cold, omega0_cold, s_hot, Gamma_hot, omega0_hot, beta_cold, beta_hot)
    previous_steady_state = solve_system_steady_state(liouvillian, solver_method = solver_method, absolute_tolerance = solver_tolerance)
    max_occupation_number += 1
    while not convergence_confirmed : 
        liouvillian = build_Liouvillian(max_occupation_number, epsilon, Delta, interqubit_coupling_strength, s_cold, Gamma_cold, omega0_cold, s_hot, Gamma_hot, omega0_hot, beta_cold, beta_hot)
        
        try:
            next_steady_state = solve_system_steady_state(liouvillian, solver_method = solver_method, absolute_tolerance = solver_tolerance)
        
        except MemoryError:
            debug_message(f"\t\t\t\tMemory overflow prevents calculation for max_occupation_number = {max_occupation_number}")
            if memory_overflow_management == 'relaxed':
                debug_message(f"\t\t\tAccepting last calculated steady state as the converged steady state. WARNING - may give non-physical results.")
                convergence_confirmed = True
                return next_steady_state, max_occupation_number-1
            elif memory_overflow_management == 'strict':
                raise
            else:
                print(f"memory_overflow_management = {memory_overflow_management} not recognised. Must be 'strict' or 'relaxed'.")
                raise
                
        absolute_difference_between_states = find_max_absolute_difference_between_QObjs(previous_steady_state, next_steady_state)
        relative_difference_between_states = find_max_relative_difference_between_QObjs(previous_steady_state, next_steady_state)
        population_difference_between_states = find_max_absolute_difference_between_populations(previous_steady_state, next_steady_state)
        if population_difference_between_states <= convergence_tolerance:
            convergence_confirmed = True
            debug_message(f"\t\t\t\tPopulation difference = {population_difference_between_states}. Convergence confirmed! ")
            return next_steady_state, max_occupation_number
        else:
            previous_steady_state = next_steady_state
            max_occupation_number += 1 
            debug_message(f"\t\t\t\tPopulation difference = {population_difference_between_states}. . Convergence not confirmed, now increasing max_occupation_number to {max_occupation_number}")

def calculate_concurrence(density_operator):
    # check dimensions of density_operator are correct
    if density_operator.dims != [[2,2],[2,2]]:
        print(f"ERROR: can only calculate concurrence for density operator of two qubits\nExpected density_operator.dims == [[2,2],[2,2]], got density_operator.dims == {density_operator.dims}")
        return None
        
    # see qutip section 5.2.2 for documentation about concurrence calculation
    return(concurrence(density_operator))

def binary_entropy(x) :
        return scipy.stats.entropy([x,1-x],base = 2)

def convert_concurrence_to_entanglement_of_formation(concurrence):
    x = (1 + np.sqrt(1-concurrence**2))/2
    return binary_entropy(x)
        
if __name__ == "__main__":

    START_TIME = time.time()

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
    s_cold = 0.5
    Gamma_cold = 0.04
    omega0_cold = 1.1
    s_hot = 0.2
    Gamma_hot = 0.01
    omega0_hot = 0.9
    bath_temperature_cold = 0.01
    beta_cold = 1/bath_temperature_cold

    minimum_hot_bath_temperature = 0.01
    maximum_hot_bath_temperature = 10
    number_of_plot_points = 2
    hot_bath_temperatures = np.arange(minimum_hot_bath_temperature, maximum_hot_bath_temperature, (maximum_hot_bath_temperature-minimum_hot_bath_temperature)/number_of_plot_points)
    hot_bath_betas = 1/hot_bath_temperatures

    convergence_atol = 0.001

    """
    Plot the two bath spectral densities
    """
    plot_frequencies = np.arange(0,20.1,0.1)
    plot_spectral_density_cold = s_cold * Gamma_cold * omega0_cold**2 * plot_frequencies / ((omega0_cold**2 - plot_frequencies**2)**2 + Gamma_cold**2 * plot_frequencies**2 )
    plt.plot(plot_frequencies,plot_spectral_density_cold)
    plt.savefig("spectral_density_cold.png")
    plt.clf()
    plot_spectral_density_hot  = s_hot  * Gamma_hot  * omega0_hot **2 * plot_frequencies / ((omega0_hot **2 - plot_frequencies**2)**2 + Gamma_hot **2 * plot_frequencies**2 )
    plt.plot(plot_frequencies,plot_spectral_density_hot )
    plt.savefig("spectral_density_hot.png" )
    plt.clf()
    """
    Plot the two mapped spectral densities 
    """
    plot_mapped_spectral_density_cold = calculate_RC_environment_coupling_strength(s_cold, Gamma_cold, omega0_cold) * plot_frequencies
    plt.plot(plot_frequencies,plot_mapped_spectral_density_cold)
    plt.savefig("mapped_spectral_density_cold.png")
    plt.clf()
    plot_mapped_spectral_density_hot  = calculate_RC_environment_coupling_strength(s_hot , Gamma_hot , omega0_hot ) * plot_frequencies
    plt.plot(plot_frequencies,plot_mapped_spectral_density_hot )
    plt.savefig("mapped_spectral_density_hot.png" )
    plt.clf()

    temperature_step = (maximum_hot_bath_temperature - minimum_hot_bath_temperature)/number_of_plot_points
    hot_bath_temperatures = np.arange(minimum_hot_bath_temperature,maximum_hot_bath_temperature,temperature_step)
    beta_hot_array = 1/hot_bath_temperatures
    
    # lists for plot points
    concurrence_list = []
    entanglement_of_formation_list = []
    rc_dims_for_convergence_list = []
    
    # calculate steady state entanglement of formation for plot against hot bath temperature
    for beta_hot in hot_bath_betas:
        steady_state, rc_dims = calculate_converged_steady_state(convergence_atol, epsilon, Delta, g, s_cold, Gamma_cold, omega0_cold, s_hot, Gamma_hot, omega0_hot, beta_cold, beta_hot, truncation_method = 'max_excitation_number', solver_method = SOLVER_METHOD, solver_tolerance = SOLVER_TOLERANCE, memory_overflow_management = 'relaxed')
        
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