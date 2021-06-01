import numpy as np
import scipy.stats
from qutip import *
import matplotlib.pyplot as plt

# set to true to print extra information to console
DEBUG = True
'''
Choose steadystate solver method (see qutip documentation for valid options)
'''

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
        if type in self.allowed_types:
            self.type = type 
        else:
            raise ValueError
        if self.type == 'underdamped':
            self.alpha = parameter_dict["alpha"]
            self.Gamma = parameter_dict["Gamma"] 
            self.omega0 = parameter_dict["omega0"]

    def get_value(self,frequency):
        if self.type == 'underdamped':
            value = self.alpha * self.Gamma * self.omega0**2 * frequency / ((self.omega0**2 - frequency**2)**2 + (self.Gamma*frequency)**2)
        return value

    def get_RC_frequency(self):
        if self.type == 'underdamped':
            return self.omega0

    def get_RC_system_coupling_strength(self):
        if self.type == 'underdamped':
            return np.sqrt(np.pi * self.alpha * self.omega0 / 2)

    def get_RC_environment_coupling_strength(self):
        if self.type == 'underdamped':
            return self.Gamma / (2 * np.pi * self.omega0)

    def get_mapped_parameters(self):
        RC_frequency = self.get_RC_frequency()
        RC_system_coupling_strength = self.get_RC_system_coupling_strength()
        RC_environment_coupling_strength = self.get_RC_environment_coupling_strength()
        return RC_frequency, RC_system_coupling_strength, RC_environment_coupling_strength


class bath:
    def __init__(self, spectral_density, temperature):
        self.temperature = temperature
        self.spectral_density = spectral_density
   

# define system Hamiltonian
def HamS(epsilon, Delta, interqubit_coupling_strength):
    return (epsilon/2) * (sigz_cold + sigz_hot) + (Delta/2) * (sigx_cold + sigx_hot) + interqubit_coupling_strength*sigz_cold*sigz_hot

# construct super-system Hamiltonian for given maximum reaction coordinate occupation number
# order all tensor products as follows
"""
TENSOR PRODUCT ORDER
cold qubit TENSOR hot qubit TENSOR cold RC TENSOR hot RC
"""
def build_HamSS(max_occupation_number,system_parameter_dict,
                cold_bath_spectral_density, hot_bath_spectral_density, truncation_method) :

    # extract system parameters
    epsilon = system_parameter_dict["epsilon"]
    Delta = system_parameter_dict["Delta"]
    interqubit_coupling_strength = system_parameter_dict["interqubit_coupling_strength"]

    # cold RC values 
    RC_frequency_cold, RC_system_coupling_strength_cold, RC_environment_coupling_strength_cold = cold_bath_spectral_density.get_mapped_parameters()
    
    # hot RC values
    RC_frequency_hot, RC_system_coupling_strength_hot, RC_environment_coupling_strength_hot = hot_bath_spectral_density.get_mapped_parameters()
    
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
def build_Liouvillian(max_occupation_number,system_parameter_dict,cold_bath,hot_bath,truncation_method):
    
    # build HamSS for given parameters
    HamSS = build_HamSS(max_occupation_number,system_parameter_dict,cold_bath.spectral_density,hot_bath.spectral_density,truncation_method)

    # extract inverse temperatures
    beta_cold = 1/cold_bath.temperature
    beta_hot  = 1/hot_bath.temperature 
                
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
    
    RC_environment_coupling_strength_cold = cold_bath.spectral_density.get_RC_environment_coupling_strength()
    RC_environment_coupling_strength_hot  = hot_bath.spectral_density.get_RC_environment_coupling_strength()
    
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


def solve_supersystem_steady_state(liouvillian, solver_method, absolute_tolerance):
    # see qutip documentation Section 3.6.2 for relevant information on steady state solver
    debug_message("\tCalculating steady state from Liouvillian...")
    rhoSS_steady = steadystate(liouvillian, method=solver_method, sparse=True, tol = absolute_tolerance,maxiter = 30000,use_precond = True)
    debug_message("\t\t...Steady state successfully calculated!")
    return rhoSS_steady
    
def solve_system_steady_state(liouvillian, solver_method, absolute_tolerance):
    rhoSS_steady = solve_supersystem_steady_state(liouvillian, solver_method, absolute_tolerance)
    rhoS_steady = rhoSS_steady.ptrace([0,1])
    print(rhoS_steady.dims)
    return rhoS_steady
   
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
def calculate_converged_steady_state(convergence_tolerance, system_parameter_dict, cold_bath, hot_bath, truncation_method, solver_method, solver_tolerance, memory_overflow_management = 'strict'):
    convergence_confirmed = False
    max_occupation_number = 1
    liouvillian = build_Liouvillian(max_occupation_number, system_parameter_dict, cold_bath, hot_bath, truncation_method)
    previous_steady_state = solve_system_steady_state(liouvillian, solver_method, solver_tolerance)
    max_occupation_number += 1
    while not convergence_confirmed : 
        liouvillian = build_Liouvillian(max_occupation_number, system_parameter_dict, cold_bath, hot_bath, truncation_method)
        
        try:
            next_steady_state = solve_system_steady_state(liouvillian, solver_method, solver_tolerance)
        
        except MemoryError:
            debug_message(f"\t\t\t\tMemory overflow prevents calculation for max_occupation_number = {max_occupation_number}")
            if memory_overflow_management == 'relaxed':
                debug_message(f"\t\t\tAccepting last calculated steady state as the converged steady state. WARNING - may give non-physical results.")
                # reduce max_occupation_number, since most recent attempt failed
                max_occupation_number -= 1
                # treat previous steady state as if it were converged
                converged_steady_state = previous_steady_state
                convergence_confirmed = True
            elif memory_overflow_management == 'strict':
                raise
            else:
                print(f"memory_overflow_management = {memory_overflow_management} not recognised. Must be 'strict' or 'relaxed'.")
                raise
                
        population_difference_between_states = find_max_absolute_difference_between_populations(previous_steady_state, next_steady_state)
        if population_difference_between_states <= convergence_tolerance:
            convergence_confirmed = True
            debug_message(f"\t\t\t\tPopulation difference = {population_difference_between_states}. Convergence confirmed! ")
            converged_steady_state = next_steady_state
        else:
            previous_steady_state = next_steady_state
            max_occupation_number += 1 
            debug_message(f"\t\t\t\tPopulation difference = {population_difference_between_states}. . Convergence not confirmed, now increasing max_occupation_number to {max_occupation_number}")

    return converged_steady_state, max_occupation_number

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