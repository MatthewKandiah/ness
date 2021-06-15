from ness_calculator import *
import unittest
import numpy.testing
from scipy.sparse import csr_matrix

class TestBuildHamSS(unittest.TestCase):
    def setUp(self):
        self.system_parameter_dict = {"epsilon":2, "Delta":3, "interqubit_coupling_strength":4}
        cold_bath_parameter_dict = {"alpha":5, "Gamma":6, "omega0":7}
        hot_bath_parameter_dict = {"alpha":8, "Gamma":9, "omega0":10}
        self.cold_bath_spectral_density = spectral_density('underdamped', cold_bath_parameter_dict)
        self.hot_bath_spectral_density = spectral_density('underdamped', hot_bath_parameter_dict)
        
    def test_dimensions_HamSS(self):
        HamSS0 = build_HamSS(0,self.system_parameter_dict, self.cold_bath_spectral_density,self.hot_bath_spectral_density, "simple")
        self.assertEqual(HamSS0.dims,[[2,2,1,1],[2,2,1,1]])
        self.assertEqual(HamSS0.shape, (4,4))
        self.assertEqual(HamSS0.type, 'oper')
        self.assertTrue(HamSS0.check_herm())
        
        HamSS3 = build_HamSS(3,self.system_parameter_dict, self.cold_bath_spectral_density, self.hot_bath_spectral_density, "simple")
        self.assertEqual(HamSS3.dims,[[2,2,4,4],[2,2,4,4]])
        self.assertEqual(HamSS3.shape, (64,64))
        self.assertEqual(HamSS3.type, 'oper')
        self.assertTrue(HamSS3.check_herm())
        
    def test_convert_to_max_excitation_number_truncation(self):
        hamSS1 = build_HamSS(1,self.system_parameter_dict, self.cold_bath_spectral_density, self.hot_bath_spectral_density,"max_excitation_number")
        hamSS2 = build_HamSS(2,self.system_parameter_dict, self.cold_bath_spectral_density, self.hot_bath_spectral_density,"max_excitation_number")
        hamSS3 = build_HamSS(3,self.system_parameter_dict, self.cold_bath_spectral_density, self.hot_bath_spectral_density,"max_excitation_number")

        self.assertEqual(hamSS1.type, 'oper')
        self.assertEqual(hamSS2.type, 'oper')
        self.assertEqual(hamSS3.type, 'oper')
        
        self.assertEqual(hamSS1.shape, (12,12))
        self.assertEqual(hamSS2.shape, (24,24))
        self.assertEqual(hamSS3.shape, (40,40))
        
        self.assertEqual(hamSS1.dims, [[2,2,2,2],[2,2,2,2]])
        self.assertEqual(hamSS2.dims, [[2,2,3,3],[2,2,3,3]])
        self.assertEqual(hamSS3.dims, [[2,2,4,4],[2,2,4,4]])
        

class TestBuildLiouvillian(unittest.TestCase):
    def setUp(self):
        self.system_parameter_dict = {"epsilon":2, "Delta":3, "interqubit_coupling_strength":4}
        cold_bath_parameter_dict = {"alpha":5, "Gamma":6, "omega0":7}
        hot_bath_parameter_dict = {"alpha":8, "Gamma":9, "omega0":10}
        self.cold_bath_spectral_density = spectral_density('underdamped', cold_bath_parameter_dict)
        self.hot_bath_spectral_density = spectral_density('underdamped', hot_bath_parameter_dict)
        cold_bath_temperature = 0.2
        hot_bath_temperature = 5
        self.cold_bath = bath(self.cold_bath_spectral_density, cold_bath_temperature)
        self.hot_bath = bath(self.hot_bath_spectral_density, hot_bath_temperature)

    def test_dimensions_Liouvillian(self):
        Liouvillian3 = build_Liouvillian(3,self.system_parameter_dict, self.cold_bath, self.hot_bath, "simple")
        self.assertEqual(Liouvillian3.type, 'super')
        self.assertEqual(Liouvillian3.shape, (64**2, 64**2))
        
    def test_steady_state_dimensions(self) :
        Liouvillian3 = build_Liouvillian(3,self.system_parameter_dict, self.cold_bath, self.hot_bath, "simple")
        rhoSS_steady = solve_supersystem_steady_state(Liouvillian3,'direct',1e-10)
        self.assertEqual(rhoSS_steady.type, 'oper')
        self.assertEqual(rhoSS_steady.shape, (64,64))
        self.assertEqual(rhoSS_steady.dims, [[2,2,4,4],[2,2,4,4]])
        
        rhoS_steady = solve_system_steady_state(Liouvillian3,'direct',1e-10)
        self.assertEqual(rhoS_steady.type, 'oper')
        self.assertEqual(rhoS_steady.shape, (4,4))
        self.assertEqual(rhoS_steady.dims, [[2,2],[2,2]])
        
    def test_find_max_differences_between_QObjs(self):
        with self.subTest("Finding maximum population differences"):
            self.assertEqual(find_max_absolute_difference_between_populations(sigmaz(),sigmaz()), 0)
            self.assertEqual(find_max_absolute_difference_between_populations(sigmaz(), qeye(2)), 2)
            self.assertEqual(find_max_absolute_difference_between_populations(sigmaz(),sigmax()), 1)
            self.assertEqual(find_max_absolute_difference_between_populations(sigmax(),sigmay()), 0)
        
    def test_max_excitation_number_truncation_Liouvillian(self):
        Liouvillian1 = build_Liouvillian(1,self.system_parameter_dict, self.cold_bath, self.hot_bath, "max_excitation_number")
        Liouvillian2 = build_Liouvillian(2,self.system_parameter_dict, self.cold_bath, self.hot_bath, "max_excitation_number")
        Liouvillian3 = build_Liouvillian(3,self.system_parameter_dict, self.cold_bath, self.hot_bath, "max_excitation_number")
        
        self.assertEqual(Liouvillian1.type, 'super')
        self.assertEqual(Liouvillian2.type, 'super')
        self.assertEqual(Liouvillian3.type, 'super')
        
        self.assertEqual(Liouvillian1.shape, (12**2, 12**2))
        self.assertEqual(Liouvillian2.shape, (24**2, 24**2))
        self.assertEqual(Liouvillian3.shape, (40**2, 40**2))
        
        self.assertEqual(Liouvillian1.dims, [[[2,2,2,2],[2,2,2,2]],[[2,2,2,2],[2,2,2,2]]])
        self.assertEqual(Liouvillian2.dims, [[[2,2,3,3],[2,2,3,3]],[[2,2,3,3],[2,2,3,3]]])
        self.assertEqual(Liouvillian3.dims, [[[2,2,4,4],[2,2,4,4]],[[2,2,4,4],[2,2,4,4]]])


class TestConcurrenceCalculator(unittest.TestCase):

    def test_concurrence_maximally_entangled_states(self):
        bellstate00 = ket2dm(bell_state(state="00"))
        bellstate01 = ket2dm(bell_state(state="01"))
        bellstate10 = ket2dm(bell_state(state="10"))
        bellstate11 = ket2dm(bell_state(state="11"))
        
        # all calculations are numerical and therefore inexact, so we will accept any solution accurate to an error less than this tolerance
        TESTING_TOLERANCE = 1e-10
        
        concurrence_error00 = abs(calculate_concurrence(bellstate00) - 1)
        concurrence_error01 = abs(calculate_concurrence(bellstate01) - 1)
        concurrence_error10 = abs(calculate_concurrence(bellstate10) - 1)
        concurrence_error11 = abs(calculate_concurrence(bellstate11) - 1)
        
        self.assertLess(concurrence_error00, TESTING_TOLERANCE)
        self.assertLess(concurrence_error01, TESTING_TOLERANCE)
        self.assertLess(concurrence_error10, TESTING_TOLERANCE)
        self.assertLess(concurrence_error11, TESTING_TOLERANCE)
        
    def test_concurrence_unentangled_states(self) :
        # all calculations are numerical and therefore inexact, so we will accept any solution accurate to an error less than this tolerance
        TESTING_TOLERANCE = 1e-10
        
        separable_state = tensor(fock_dm(2,0),fock_dm(2,1))
        
        concurrence_error = abs(calculate_concurrence(separable_state))
        self.assertLess(concurrence_error, TESTING_TOLERANCE)
        
    def test_concurrence_to_entanglement_of_formation(self) :
        self.assertEqual(convert_concurrence_to_entanglement_of_formation(1),1)
        self.assertEqual(convert_concurrence_to_entanglement_of_formation(0),0)
        # all calculations are numerical and therefore inexact, so we will accept any solution accurate to an error less than this tolerance
        TESTING_TOLERANCE = 1e-10
        for i in range(1,10):
            x = i/10
            with self.subTest(x=x):
                binary_entropy_error = abs(binary_entropy(x) - (-x * np.log2(x) - (1-x) * np.log2(1-x)))
                self.assertLess(binary_entropy_error, TESTING_TOLERANCE)


class TestThermalisation(unittest.TestCase) :
    def test_matrix_exponential(self) :
        matrix = (-1j*np.pi/2)*sigmay()
        matrix_exponential = matrix.expm()
        
        expected_answer = -1j * sigmay()
        
        self.assertEqual(matrix_exponential, expected_answer)
        

    # expect supersystem to thermalise to temperature of baths if baths are at equal temperature
    @unittest.skip('Don\'t think this is as good a test as we thought, difficult to say how similar these should actually be. Non-secular master equation => similar, but not the same.')
    def test_baths_at_same_temperature(self) :
        eps = 0
        Del = 1
        g = 0.02
        sC = 0.2
        GamC = 0.4
        om0C = 20
        sH = 0.2
        GamH = 0.4
        om0H = 15
        bath_temperature = 0.5
        inverse_temperature = 1/bath_temperature
        METHOD = "max_excitation_number" 
        
        steady_state, max_excitation_number_for_convergence = calculate_converged_steady_state(0.001, eps,Del,g,sC,GamC,om0C,sH,GamH,om0H,inverse_temperature,inverse_temperature,truncation_method = METHOD,solver_method = 'direct', solver_tolerance = 1e-12)
        hamSS = build_HamSS(max_excitation_number_for_convergence,eps,Del,g,sC,GamC,om0C,sH,GamH,om0H)
        
        gibbs_state = (-inverse_temperature * hamSS).expm()
        partition_function = gibbs_state.norm()
        gibbs_state = gibbs_state/partition_function
       
        with self.subTest("Trace of gibbs_state"):
            print("gibbs_state")
            for i in range(0,gibbs_state.shape[0]) :
                print(gibbs_state.full()[i][i])
            print("\n")
            np.testing.assert_almost_equal(gibbs_state.tr(),1)
            
        with self.subTest("Trace of steady_state"):
            print("steady_state")
            for i in range(0,steady_state.shape[0]) :
                print(steady_state.full()[i][i])
            print("\n")
            np.testing.assert_almost_equal(steady_state.tr(),1)
            
        with self.subTest("Compare partial traces of steady_state and gibbs_state"):
            reduced_gibbs_state = gibbs_state.ptrace([0,1])
            np.testing.assert_allclose(steady_state, reduced_gibbs_state)
                    

    # expect qubit-RC supersystems to approximately thermalise separately to their local bath if interqubit coupling is set to zero
    # plot results to compare to the results in Jake & Ahsan's PRA paper. 
    @unittest.skip('Move from tests to its own script')
    def test_non_interacting_qubits(self) :
        # define parameters
        # choose parameters to match Fig 4A from the PRA paper 
        delta = 200
        eps = 0.5 * delta 
        interqubit_coupling = 0
        system_bath_coupling = 0.5 * delta / np.pi 
        Gamma1 = 10
        Gamma2 = 100
        peak_frequency1 = np.sqrt(Gamma1 * 0.05 * Delta)
        peak_frequency2 = np.sqrt(Gamma2 * 0.05 * Delta)
        METHOD = "max_excitation_number"
        number_of_plot_points = 100
        betaDelta_min = 0.1
        betaDelta_max = 3.0
        betaDelta_array = np.arange(betaDelta_min,betaDelta_max,(betaDelta_max - betaDelta_min)/number_of_plot_points)
        beta_array = betaDelta_array / Delta 
        rho_ee_over_rho_gg_cold_array = np.zeros_like(betaDelta_array)
        rho_ge_cold_array = np.zeros_like(betaDelta_array)
        rho_ee_over_rho_gg_hot_array  = np.zeros_like(betaDelta_array)
        rho_ge_hot_array  = np.zeros_like(betaDelta_array)
        # RC truncation for Gibbs state calculations
        RC_dims = 4
        
        CONVERGENCE_ATOL = 0.017
        SOLVER_TOLERANCE = 1e-10
        
        for j in range(0, len(betaDelta_array)):
            beta = beta_array[j]
            # calculate converged steady state
            steady_state, rc_dims = calculate_converged_steady_state(CONVERGENCE_ATOL, eps, delta, 0, system_bath_coupling, Gamma1, peak_frequency1, system_bath_coupling, Gamma2, peak_frequency2, beta, beta, solver_method = 'direct', memory_overflow_management = 'relaxed', solver_tolerance=SOLVER_TOLERANCE)
            steady_state.tidyup(atol=SOLVER_TOLERANCE)
            steady_state_cold_qubit_and_RC = steady_state.ptrace(0,2)
            steady_state_hot_qubit_and_RC  = steady_state.ptrace(1,3)
            steady_state_cold_qubit = steady_state.ptrace(0)
            steady_state_hot_qubit  = steady_state.ptrace(1)
            
            # update result arrays for plots
            rho_ee_over_rho_gg_cold_array[j] = steady_state_cold_qubit.full()[0][0] / steady_state_cold_qubit.full()[1][1]
            rho_ee_over_rho_gg_hot_array[j]  = steady_state_hot_qubit.full()[0][0]  / steady_state_hot_qubit.full()[1][1]
            rho_ge_cold_array[j] = steady_state_cold_qubit.full()[1][0]
            rho_ge_hot_array[j]  = steady_state_hot_qubit.full()[1][0] 
        
        # plot bath spectral densities to check they are as expected
        frequency_array = np.arange(0,500,1)
        spectral_density_cold_array = system_bath_coupling * Gamma1 * peak_frequency1**2 * frequency_array / ((peak_frequency1**2 - frequency_array**2)**2 + (Gamma1*frequency_array)**2)
        spectral_density_hot_array  = system_bath_coupling * Gamma2 * peak_frequency2**2 * frequency_array / ((peak_frequency2**2 - frequency_array**2)**2 + (Gamma2*frequency_array)**2)
        plt.plot(frequency_array, spectral_density_cold_array)
        plt.savefig("test_plots/cold_bath_spectral_density.png")
        plt.clf()
        plt.plot(frequency_array, spectral_density_hot_array)
        plt.savefig("test_plots/hot_bath_spectral_density.png")
        plt.clf()
        # plots to compare to Jake and Ahsan's results in PRA and/or J Chem Phys
        plt.plot(betaDelta_array, rho_ee_over_rho_gg_cold_array)
        plt.yscale('log')
        plt.savefig("test_plots/cold_population_ratio.png")
        plt.clf()
        plt.plot(betaDelta_array, rho_ee_over_rho_gg_hot_array )
        plt.yscale('linear')
        plt.savefig("test_plots/hot_population_ratio.png" )
        plt.clf()
        plt.plot(betaDelta_array,rho_ge_cold_array)
        plt.savefig("test_plots/cold_coherence.png")
        plt.clf()
        plt.plot(betaDelta_array, rho_ge_hot_array)
        plt.savefig("test_plots/hot_coherence.png")
        plt.clf()

class TestPartialTrace(unittest.TestCase):

    def test_build_basis_set(self):
        basis1 = build_basis_set(1)
        basis3 = build_basis_set(3)
        basis5 = build_basis_set(5)

        expected_basis1 = [basis(1,0)]
        expected_basis3 = [basis(3,0),basis(3,1),basis(3,2)]
        expected_basis5 = [basis(5,0),basis(5,1),basis(5,2),basis(5,3),basis(5,4)]

        for i in range(0,len(basis1)):
            np.testing.assert_array_equal(basis1[i].full(),expected_basis1[i].full())
        for i in range(0,len(basis3)):
            np.testing.assert_array_equal(basis3[i].full(),expected_basis3[i].full())
        for i in range(0,len(basis5)):
            np.testing.assert_array_equal(basis5[i].full(),expected_basis5[i].full())
        for i in range(0,len(expected_basis1)):
            np.testing.assert_array_equal(basis1[i].full(),expected_basis1[i].full())
        for i in range(0,len(expected_basis3)):
            np.testing.assert_array_equal(basis3[i].full(),expected_basis3[i].full())
        for i in range(0,len(expected_basis5)):
            np.testing.assert_array_equal(basis5[i].full(),expected_basis5[i].full())

    def test_ptrace_enr_evironments(self):
        # define states of systems and environments
        system1 = thermal_dm(5,2)
        system2 = thermal_dm(3,0.5)
        environments1 = enr_thermal_dm([3,4],2,1.2)
        environments2 = enr_thermal_dm([3,2],2,1.2)
        # check set up correctly
        self.assertEqual(system1.type, 'oper')
        self.assertEqual(system1.shape, (5,5))
        self.assertEqual(system1.dims, [[5],[5]])
        self.assertEqual(system2.type, 'oper')
        self.assertEqual(system2.shape, (3,3))
        self.assertEqual(system2.dims, [[3],[3]])
        self.assertEqual(environments1.type, 'oper')
        self.assertEqual(environments1.shape, (6,6))
        self.assertEqual(environments1.dims, [[3,4],[3,4]])
        self.assertEqual(environments2.type, 'oper')
        self.assertEqual(environments2.shape, (5,5))
        self.assertEqual(environments2.dims, [[3,2],[3,2]])
        # form composite system states
        composite_system1 = tensor(system1,system2,environments1)
        composite_system2 = tensor(system1,system2,environments2)
        # check dimensions
        self.assertEqual(composite_system1.shape, (90,90))
        self.assertEqual(composite_system1.dims, [[5,3,3,4],[5,3,3,4]])
        self.assertEqual(composite_system2.shape, (75,75)) 
        self.assertEqual(composite_system2.dims, [[5,3,3,2],[5,3,3,2]])
        # peform partial traces and check type and dimensions of output are as expected
        with self.subTest("all modes of higher dimension than max_excitation_number"):
            reduced_state1 = partial_trace_enr_environment(composite_system1,2)
            self.assertEqual(reduced_state1.type, 'oper')
            self.assertEqual(reduced_state1.dims, [[5,3],[5,3]])
            self.assertEqual(reduced_state1.shape, (15,15))
        with self.subTest("one mode of lower dimension than max_excitation_number"):
            reduced_state2 = partial_trace_enr_environment(composite_system2,2)
            self.assertEqual(reduced_state2.type, 'oper')
            self.assertEqual(reduced_state2.dims, [[5,3],[5,3]])
            self.assertEqual(reduced_state2.shape, (15,15))

        # check that partial trace gives correct reduced state for simplest case
        system1 = thermal_dm(5,2)
        system2 = thermal_dm(3,0.5)
        system3 = 0.25 * qeye(4)

        expected_output1 = tensor(system1,system2)
        expected_output2 = tensor(system2,system1)
        expected_output3 = tensor(system1,system3)
        
        composite_system_enr_identity1 = tensor(system1,system2,enr_identity([3,4],3))
        composite_system_enr_identity2 = tensor(system2,system1,enr_identity([4,3],3))
        composite_system_enr_identity3 = tensor(system1, 0.25*qeye(4), enr_identity([2,2],1))
        
        traced_composite_system_enr_identity1 = partial_trace_enr_environment(composite_system_enr_identity1,3)
        traced_composite_system_enr_identity2 = partial_trace_enr_environment(composite_system_enr_identity2,3)
        traced_composite_system_enr_identity3 = partial_trace_enr_environment(composite_system_enr_identity3,1)
        
        with self.subTest("partial tracing enr_identity environment: case 1 - size check"):
            np.testing.assert_array_equal(traced_composite_system_enr_identity1.dims, expected_output1.dims)
            self.assertEqual(traced_composite_system_enr_identity1.full().size,expected_output1.full().size)
        with self.subTest("partial tracing enr_identity environment: case 1 - elementwise check"):
            for n in range(0, len(expected_output1.full())):
                for m in range(0,len(expected_output1.full()[0])):
                    with self.subTest(f"n,m={n,m}"):
                        self.assertEqual(traced_composite_system_enr_identity1.full()[n][m], expected_output1.full()[n][m])
            for n in range(0, len(traced_composite_system_enr_identity1.full())):
                for m in range(0,len(traced_composite_system_enr_identity1.full()[0])):
                    with self.subTest(f"n,m={n,m}"):
                        self.assertEqual(traced_composite_system_enr_identity1.full()[n][m], expected_output1.full()[n][m])

        with self.subTest("partial tracing enr_identity environment: case 2 - size check"):
            np.testing.assert_array_equal(traced_composite_system_enr_identity2.dims, expected_output2.dims)
            self.assertEqual(traced_composite_system_enr_identity2.full().size,expected_output2.full().size)
        with self.subTest("partial tracing enr_identity environment: case 2 - elementwise check"):
            for n in range(0, len(expected_output2.full())):
                for m in range(0,len(expected_output2.full()[0])):
                    with self.subTest(f"n,m={n,m}"):
                        self.assertEqual(traced_composite_system_enr_identity2.full()[n][m], expected_output2.full()[n][m])
            for n in range(0, len(traced_composite_system_enr_identity2.full())):
                for m in range(0,len(traced_composite_system_enr_identity2.full()[0])):
                    with self.subTest(f"n,m={n,m}"):
                        self.assertEqual(traced_composite_system_enr_identity2.full()[n][m], expected_output2.full()[n][m])

        with self.subTest("partial tracing enr_identity environment: case 3 - size check"):
            np.testing.assert_array_equal(traced_composite_system_enr_identity3.dims, expected_output3.dims)
            self.assertEqual(traced_composite_system_enr_identity3.full().size,expected_output3.full().size)
        with self.subTest("partial tracing enr_identity environment: case 3 - elementwise check"):
            for n in range(0, len(expected_output3.full())):
                for m in range(0,len(expected_output3.full()[0])):
                    with self.subTest(f"n,m={n,m}"):
                        self.assertEqual(traced_composite_system_enr_identity3.full()[n][m], expected_output3.full()[n][m])
            for n in range(0, len(traced_composite_system_enr_identity3.full())):
                for m in range(0,len(traced_composite_system_enr_identity3.full()[0])):
                    with self.subTest(f"n,m={n,m}"):
                        self.assertEqual(traced_composite_system_enr_identity3.full()[n][m], expected_output3.full()[n][m])
        
class TestNumpyFunctions(unittest.TestCase):
    # just want to check that array manipulations actually work like I think they do
    def test_take(self):
        drho = [2,3,4]
        sel = [0,1]
        dims_short = np.asarray(drho).take(sel)
        self.assertEqual(dims_short.tolist(), [2,3])

if __name__ == "__main__" :
    unittest.main()