# This is to sidestep a numpy overhead introduced in numpy 1.17:
# https://stackoverflow.com/questions/61983372/is-built-in-method-numpy-core-multiarray-umath-implement-array-function-a-per
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
#----------------------------------
import numpy as np
import numexpr as ne
import scipy.integrate
from scipy.integrate._ivp.base import OdeSolver

# from numba import jit
# import nbkode

from ecoevocrm.type_set import *
from ecoevocrm.resource_set import *
# from ecoevocrm.biochemistry import *
import ecoevocrm.utils as utils

# from ecoevocrm.consumer_resource_dynamics import *


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ConsumerResourceSystem():

    # Define Class constants:
    CONSUMPTION_MODE_FASTEQ = 0
    CONSUMPTION_MODE_LINEAR = 1
    CONSUMPTION_MODE_MONOD  = 2
    INFLOW_MODE_NONE        = 0
    INFLOW_MODE_CONSTANT    = 1

    def __init__(self, 
                 type_set   = None,
                 resource_set  = None,
                 # biochemistry  = None,
                 N_init        = None,
                 R_init        = None,
                 num_types     = None,
                 num_resources = None,
                 sigma         = None,
                 b             = 1,
                 k             = 0,
                 eta           = 1,
                 l             = 0,
                 g             = 1,
                 c             = 0,
                 chi           = None,
                 mu            = 0,
                 J             = None,
                 D             = None,
                 rho           = 0,
                 tau           = 1,
                 omega         = 1,
                 resource_consumption_mode     = 'linear',
                 resource_inflow_mode          = 'constant',
                 threshold_min_abs_abundance   = 1,
                 threshold_min_rel_abundance   = 1e-6,
                 threshold_eq_abundance_change = 1e4,
                 threshold_precise_integrator  = 1e2,
                 seed = None):

        #----------------------------------

        if(seed is not None):
            np.random.seed(seed)
            self.seed = seed

        #----------------------------------
        # Determine the dimensions of the system:
        #----------------------------------
        if(type_set is not None):
            system_num_types     = type_set.num_types
            system_num_resources = type_set.num_traits
        elif(isinstance(sigma, (list, np.ndarray))):
           sigma = np.array(sigma)
           if(sigma.ndim == 2):
                system_num_types     = sigma.shape[0]
                system_num_resources = sigma.shape[1]
        else:
            if(num_types is not None):
                system_num_types = num_types
            elif(isinstance(N_init, (list, np.ndarray))):
                system_num_types = np.array(N_init).ravel().shape[0]
            else:
                utils.error("Error in ConsumerResourceSystem __init__(): Number of types must be specified by providing a) a type set, b) a sigma matrix, c) a num_types value, or d) a list for N_init.")
            #---
            if(resource_set is not None):
                system_num_resources = resource_set.num_resources
            elif(num_resources is not None):
                system_num_resources = num_resources
            elif(isinstance(R_init, (list, np.ndarray))):
                system_num_resources = np.array(R_init).ravel().shape[0]
            else:
                utils.error("Error in ConsumerResourceSystem __init__(): Number of resources must be specified by providing a) a resource set, b) a sigma matrix, c) a num_resources value, or d) a list for R_init.")
            
        #----------------------------------
        # Initialize type set parameters:
        #----------------------------------
        if(type_set is not None):
            if(isinstance(type_set, TypeSet)):
                self.type_set = type_set
            else:
                utils.error(f"Error in ConsumerResourceSystem __init__(): type_set argument expects object of TypeSet type.")
        else:
            self.type_set = TypeSet(num_types=system_num_types, num_traits=system_num_resources, sigma=sigma, b=b, k=k, eta=eta, l=l, g=g, c=c, chi=chi, J=J, mu=mu)
        # Check that the type set dimensions match the system dimensions:
        if(system_num_types != self.type_set.num_types): 
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system types ({system_num_types}) does not match number of type set types ({self.type_set.num_types}).")
        if(system_num_resources != self.type_set.num_traits): 
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system resources ({system_num_resources}) does not match number of type set traits ({self.type_set.num_traits}).")
        
        #----------------------------------
        # Initialize resource set parameters:
        #----------------------------------
        if(resource_set is not None):
            if(isinstance(resource_set, ResourceSet)):
                self.resource_set = resource_set
            else:
                utils.error(f"Error in ConsumerResourceSystem __init__(): resource_set argument expects object of ResourceSet type.")
        else:
            self.resource_set = ResourceSet(num_resources=system_num_resources, rho=rho, tau=tau, omega=omega, D=D)
        # Check that the type set dimensions match the system dimensions:
        if(system_num_resources != self.resource_set.num_resources): 
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system resources ({system_num_resources}) does not match number of resource set resources ({self.resource_set.num_resources}).")

        # #----------------------------------
        # # Initialize biochemistry parameters:
        # #----------------------------------
        # if(biochemistry is not None):
        #     if(isinstance(biochemistry, Biochemistry)):
        #         self.biochemistry = biochemistry
        #     else:
        #         utils.error(f"Error in ConsumerResourceSystem __init__(): biochemistry argument expects object of Biochemistry type.")
        # else:
        #     self.biochemistry = Biochemistry(num_resources=system_num_resources, J=J, D=D)
        # # Check that the type set dimensions match the system dimensions:
        # if(system_num_resources != self.biochemistry.num_resources): 
        #     utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system resources ({system_num_resources}) does not match number of resource set resources ({self.biochemistry.num_resources}).")

        #----------------------------------
        # Initialize system variables:
        #----------------------------------
        if(N_init is None or R_init is None):
            utils.error(f"Error in ConsumerResourceSystem __init__(): Values for N_init and R_init must be provided.")
        self.N_series = utils.reshape(N_init, shape=(system_num_types, 1)).astype(np.float32)
        self.R_series = utils.reshape(R_init, shape=(system_num_resources, 1)).astype(np.float32)

        # self.fitness = np.zeros(self.type_set.num_types)

        #----------------------------------
        # Initialize system time:
        #----------------------------------
        self.t = 0
        self.t_series = np.array([0])

        #----------------------------------
        # Initialize event parameters:
        #----------------------------------
        self.threshold_mutation_propensity = None # is updated in run()
        self.threshold_eq_abundance_change = threshold_eq_abundance_change
        self.threshold_min_abs_abundance   = threshold_min_abs_abundance
        self.threshold_min_rel_abundance   = threshold_min_rel_abundance
        self.threshold_precise_integrator  = threshold_precise_integrator

        #----------------------------------
        # Initialize system options:
        #----------------------------------
        self.resource_consumption_mode = ConsumerResourceSystem.CONSUMPTION_MODE_FASTEQ if resource_consumption_mode=='fast_resource_eq' \
                                          else ConsumerResourceSystem.CONSUMPTION_MODE_LINEAR if resource_consumption_mode=='linear' \
                                          else ConsumerResourceSystem.CONSUMPTION_MODE_MONOD if resource_consumption_mode=='monod' \
                                          else -1

        self.resource_inflow_mode      = ConsumerResourceSystem.INFLOW_MODE_NONE if resource_inflow_mode == 'none' \
                                          else ConsumerResourceSystem.INFLOW_MODE_CONSTANT if resource_inflow_mode == 'constant' \
                                          else -1


        self.extant_mutant_set = self.type_set.generate_mutant_set()

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def run(self, T=10000, dt=None, integration_method='default'):

        t_start   = self.t
        t_elapsed = 0

        numiter = 0

        while(t_elapsed < T):

            #------------------------------
            # Set initial conditions and integration variables:
            #------------------------------
           
            # Set the time interval for this integration epoch:
            t_span = (self.t, self.t+T)

            # Set the time ticks at which to save trajectory values:
            t_eval = np.arange(start=t_span[0], stop=t_span[1]+dt, step=dt) if dt is not None else None,

            # Get the indices and count of extant types (abundance > 0):
            self.extant_type_indices     = self.get_extant_types()
            num_extant_types = len(self.extant_type_indices)
            # print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print("\n\nsigma (master)\n", self.type_set.sigma)
            # print("self.extant_type_indices", self.extant_type_indices, num_extant_types)
            # print("sigma (extant)\n", self.type_set.sigma[self.extant_type_indices])
            


            # Set the initial conditions for this integration epoch:
            N_init = self.N_series[self.extant_type_indices, -1] # only run dynamics for extant types
            R_init = self.R_series[:,-1]
            cumPropMut_init = np.array([0])
            init_cond = np.concatenate([N_init, R_init, cumPropMut_init])

            # Get the params for the dynamics:
            params = self.get_dynamics_params(self.extant_type_indices)

            # Draw a random propensity threshold for triggering the next Gillespie mutation event:
            self.threshold_mutation_propensity = np.random.exponential(1)

            # Reset helper variables for storing latest attributes of mutants (as calc'ed in dynamics):
            # self.mutant_fitnesses      = np.zeros(self.extant_mutant_set.num_types)
            # self.mutation_propensities = np.zeros(self.extant_mutant_set.num_types)
            
            # Set the integration method:
            if(integration_method == 'default'):
                if(num_extant_types <= self.threshold_precise_integrator):
                    _integration_method = 'LSODA' # accurate stiff integrator
                else:
                    _integration_method = 'LSODA' # adaptive stiff/non-stiff integrator
            else:
                _integration_method = integration_method

            #------------------------------
            # Integrate the system dynamics:
            #------------------------------
            sol = scipy.integrate.solve_ivp(self.dynamics, 
                                             y0     = init_cond,
                                             args   = params,
                                             t_span = (self.t, self.t+T),
                                             t_eval = np.arange(start=self.t, stop=self.t+T+dt, step=dt) if dt is not None else None,
                                             events = [self.event_mutation],#, self.event_low_abundance],
                                             method = _integration_method )

            #------------------------------
            # Update the system's trajectories with latest dynamics epoch:
            #------------------------------

            # Update the system time series:
            self.t_series = np.concatenate([self.t_series[:-1], sol.t]) if self.t_series is not None else sol.t
            self.t = self.t_series[-1] + (dt if dt is not None else 0)
            t_elapsed = self.t - t_start

            # Update the system data series:
            N_epoch = np.zeros(shape=(self.type_set.num_types, len(sol.t)))
            N_epoch[self.extant_type_indices] = sol.y[:num_extant_types]

            R_epoch = sol.y[-1-self.resource_set.num_resources:-1]

            self.N_series = np.hstack([self.N_series[:,:-1], N_epoch]) if self.N_series is not None else N_epoch
            self.R_series = np.hstack([self.R_series[:,:-1], R_epoch]) if self.R_series is not None else R_epoch
            # self.N_series = np.hstack([self.N_series[:,:-1], sol.y[:self.type_set.num_types]]) if self.N_series is not None else sol.y[:self.type_set.num_types]
            # self.R_series = np.hstack([self.R_series[:,:-1], sol.y[-1-self.resource_set.num_resources:-1]]) if self.R_series is not None else sol.y[-1-self.resource_set.num_resources:-1]

            extant_type_mutants = self.type_set.get_mutant_indices(self.extant_type_indices)

            # mutant_fitnesses_fullset = np.zeros(self.extant_mutant_set.num_types)
            # mutant_fitnesses_fullset[extant_type_mutants] = self.mutant_fitnesses
            # self.mutant_fitnesses    = mutant_fitnesses_fullset

            # mutation_propensities_fullset = np.zeros(self.extant_mutant_set.num_types)
            # mutation_propensities_fullset[extant_type_mutants] = self.mutation_propensities
            # self.mutation_propensities    = mutation_propensities_fullset

            typeCountStr = f"{num_extant_types}*({len(extant_type_mutants)}|{self.extant_mutant_set.num_types})/{self.type_set.num_types} {N_epoch.shape}"

            #------------------------------
            # Handle events and update the system's states accordingly:
            #------------------------------
            if(sol.status == 1): # An event occurred
                if(len(sol.t_events[0]) > 0):
                    print(f"[ Mutation event occurred at  t={self.t:.4f} {typeCountStr}]\t\r", end="")
                    # print(f"[ Mutation event occurred at  t={self.t:.4f} {typeCountStr}]\t")
                    self.handle_mutation_event()
                    self.handle_type_loss()
                    # exit()
                # if(len(sol.t_events[1]) > 0):
                #     print(f"[ Type loss event occurred at t={self.t:.4f} ]\t\r", end="")
                #     # pass
                #     self.handle_type_loss()
                #     # exit()
            elif(sol.status == 0): # Reached end T successfully
                self.handle_type_loss()
            else: # Error occurred in integration
                utils.error("Error in ConsumerResourceSystem run(): Integration of dynamics using scipy.solve_ivp returned with error status.")
            
            total_epoch_abundance_change = np.sum( self.N_series[:,-1] - self.N_series[:,0] )
            if(total_epoch_abundance_change < self.threshold_eq_abundance_change):
                break

            # numiter += 1
            # if(numiter >= 1):
            #     break


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def resource_demand(N, sigma):
        # TODO: Allow for calculating for single N or time series of Ns
        # print("\n\nresource_demand")
        # print("N\n", N, N.shape)
        # print("sigma\n", sigma, sigma.shape)
        # return np.einsum('ij,ij->j', N[:, np.newaxis], sigma) # equivalent to np.multiply(N_t, sigma).sum(axis=0)
        return np.einsum(('ij,ij->j' if N.ndim == 2 else 'i,ij->j'), N, sigma)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def growth_rate(N, R, sigma, b, k, eta, l, g, energy_costs, omega, resource_consumption_mode, energy_uptake_coeffs=None):
        # TODO: Allow for calculating for single N/R or time series of N/Rs
        energy_uptake_coeffs = omega * (1-l) * sigma * b if energy_uptake_coeffs is None else energy_uptake_coeffs
        # print('>>>', energy_uptake_coeffs.shape)
        #------------------------------
        if(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_FASTEQ):
            resource_demand = ConsumerResourceSystem.resource_demand(N, sigma)
            # print("energy_uptake_coeffs\n", energy_uptake_coeffs, energy_uptake_coeffs.shape)
            # print("(k/(k + resource_demand))\n", (k/(k + resource_demand)), (k/(k + resource_demand)).shape)
            energy_uptake   = np.einsum(('ij,ij->i' if k.ndim == 2 else 'ij,j->i'), energy_uptake_coeffs, k/(k + resource_demand))
            energy_surplus  = energy_uptake - energy_costs
            growth_rate     = g * energy_surplus
        elif(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_LINEAR):
            pass
        elif(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_MONOD):
            pass
        else:
            pass
        #------------------------------
        return growth_rate


    # def growth_rate(self, t=None):
    #     # TODO: allow for passing in single or series of t vals:
    #     return ConsumerResourceSystem.growth_rate(self.N(t), self.R(t), self.type_set.sigma, self.type_set.b, self.type_set.k, self.type_set.eta, self.type_set.l, self.type_set.g, self.type_set.energy_costs, self.resource_set.omega, self.resource_consumption_mode)


    def N(self, t=None, type_index=None, type_id=None):
        # TODO: allow for passing in single or series of t vals:
        # TODO: handle type_index and type_id
        if (t is None):
            return self.N_series[:, -1]


    def R(self, t=None, resource_index=None, resource_id=None):
        # TODO: allow for passing in single or series of t vals:
        # TODO: handle resource_index and resource_id
        if (t is None):
            return self.R_series[:, -1]



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # @jit(nopython=True, fastmath=True, cache=True)
    def dynamics(self, t, variables, 
                    num_types, num_mutants, sigma, b, k, eta, l, g, c, chi, J, mu, energy_costs, energy_uptake_coeffs,
                    # num_mutants, sigma_mut, b_mut, k_mut, eta_mut, l_mut, g_mut, c_mut, chi_mut, J_mut, mu_mut, energy_costs_mut, energy_uptake_coeffs_mut,
                    num_resources, rho, tau, omega, D, 
                    resource_consumption_mode, resource_inflow_mode):

        # N_t = np.zeros((num_types+num_mutants, 1))
        # N_t[:num_types, 0] = variables[:num_types]

        # N_t = variables[:num_types]

        N_t = np.zeros(num_types+num_mutants)
        N_t[:num_types] = variables[:num_types]

        R_t = variables[-1-num_resources:-1]

        # cumulative_mutation_propensity = variables[-1] # value is not needed to calc dCumPropMut

        #------------------------------

        growth_rate = ConsumerResourceSystem.growth_rate(N_t, R_t, sigma, b, k, eta, l, g, energy_costs, omega, resource_consumption_mode, energy_uptake_coeffs)

        # resource_demand = np.einsum('ij,ij->j', N_t, sigma) 
        # energy_surplus  = np.einsum('ij,ij->i', energy_uptake_coeffs, k/(k + resource_demand)) - energy_costs
        # growth_rate     = g * energy_surplus

        
        dNdt = N_t[:num_types] * growth_rate[:num_types] # only calc dNdt for extant (non-mutant types)
        # dNdt = N_t[:num_types, 0] * growth_rate[:num_types] # only calc dNdt for extant (non-mutant types)

        dRdt = np.zeros(num_resources)

        # if(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_FASTEQ):
            
            # # resource_demand = np.sum(np.multiply(N_t, sigma), axis=0)
            # resource_demand = np.einsum('ij,ij->j', N_t, sigma) # equivalent to np.multiply(N_t, sigma).sum(axis=0)
            
            # # energy_surplus  = np.sum(np.multiply(energy_uptake_coeffs, 1/(k + resource_demand)), axis=1) - energy_costs
            # # energy_surplus  = np.einsum('ij,ij->i', energy_uptake_coeffs, 1/(k + resource_demand)) - energy_costs
            # energy_surplus  = np.einsum('ij,ij->i', energy_uptake_coeffs, k/(k + resource_demand)) - energy_costs
            
            # # growth_rate     = np.multiply(g, energy_surplus)
            # growth_rate     = g * energy_surplus

        #     # self.fitness = growth_rate[:num_types]

        #     # dNdt = np.multiply(N_t[:num_types, 0], growth_rate[:num_types]) # only calc dNdt for extant (non-mutant types)
        #     dNdt = N_t[:num_types, 0] * growth_rate[:num_types] # only calc dNdt for extant (non-mutant types)

        #     dRdt = np.zeros(num_resources)

        # elif(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_LINEAR):
        #     pass
        
        # elif(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_MONOD):
        #     pass
        
        # else:
        #     pass

        #------------------------------

        # if(resource_inflow_mode == 0):
        #     resource_inflow = np.zeros((1, num_resources))
        # elif(resource_inflow_mode == 1):
        #     resource_inflow = rho

        # if(resource_inflow_mode == 0): #ConsumerResourceSystem.INFLOW_MODE_NONE):
        #     dRdt = np.zeros((1, num_resources)).ravel()
        # else:
        #     if(D is None):
        #         dRdt = ( resource_inflow - np.multiply(1/tau, R_t) - np.sum(np.multiply(np.expand_dims(N_t, axis=1), resource_consumption), axis=0) ).ravel()
        #     else:
        #         resource_secretion = np.divide(np.dot(np.multiply(omega, np.sum(np.multiply(np.expand_dims(N_t, axis=1), np.multiply(l, resource_consumption)), axis=0)), D.T), omega, where=(omega>0), out=np.zeros_like(omega)).ravel()
        #         dRdt = ( resource_inflow - np.multiply(1/tau, R_t) - np.sum(np.multiply(np.expand_dims(N_t, axis=1), resource_consumption), axis=0) + resource_secretion ).ravel()

        #------------------------------

        # TODO: HERE HERE HERE: Mutant growth rates are being calculated without the presence of the extant types with the following line as is!
        # revert back to calculating growth rates of extant and mutants in one go with params stacked (params that dont vary by type dont need a concat)
        # growth_rate_mut = ConsumerResourceSystem.growth_rate(np.zeros(num_mutants), R_t, sigma_mut, b_mut, k_mut, eta_mut, l_mut, g_mut, energy_costs_mut, omega, resource_consumption_mode, energy_uptake_coeffs_mut)

        # print("growth_rate_mut\n", growth_rate_mut, growth_rate_mut.shape)

        self.mutant_fitnesses      = growth_rate[-num_mutants:]

        # print("self.mutant_fitnesses\n", self.mutant_fitnesses, self.mutant_fitnesses.shape)

        self.mutation_propensities = np.maximum(0, self.mutant_fitnesses * np.repeat(N_t[:num_types] * mu, repeats=num_resources))

        # print("self.mutation_propensities\n", self.mutation_propensities, self.mutation_propensities.shape)
                                                          
        dCumPropMut = np.sum(self.mutation_propensities, keepdims=True)
        
        #------------------------------

        # print("end dynamics")
        # exit()

        return np.concatenate((dNdt, dRdt, dCumPropMut))

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def event_mutation(self, t, variables, *args):
        # cumulative_mutation_propensity = variables[-1]
        return self.threshold_mutation_propensity - variables[-1]
    #----------------------------------
    event_mutation.direction = -1
    event_mutation.terminal  = True

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def handle_mutation_event(self):
        # Pick the mutant that will be established with proababilities proportional to mutants' propensities for establishment:
        mutant_drawprobs = self.mutation_propensities/np.sum(self.mutation_propensities)
        mutant_idx       = np.random.choice(range(self.extant_mutant_set.num_types), p=mutant_drawprobs)
        # print("\tmutant_idx", mutant_idx, self.extant_mutant_set.sigma[mutant_idx])
        # Retrieve the mutant and some of its properties:
        mutant           = self.extant_mutant_set.get_type(mutant_idx)
        mutant_type_id   = self.extant_mutant_set.get_type_id(mutant_idx)
        mutant_fitness   = self.mutant_fitnesses[mutant_idx]
        mutant_abundance = np.maximum(1/mutant_fitness, 1) # forcing abundance of new types to be at least 1, this is a Ryan addition (perhaps controversial)
        # Get the index of the parent of the selected mutant:
        parent_idx       = self.extant_type_indices[ mutant_idx // self.type_set.num_traits ]
        # print("\tparent_idx", parent_idx, self.type_set.sigma[parent_idx])
        #----------------------------------
        if(mutant_type_id in self.type_set.type_ids):
            # print("mutation: pre-existing")
            # This "mutant" is a pre-existing type in the population; get its index:
            preexisting_type_idx = np.where(np.array(self.type_set.type_ids) == mutant_type_id)[0][0]
            # print("\tpreexisting_type_idx", preexisting_type_idx, self.type_set.sigma[preexisting_type_idx])
            # Add abundance equal to the mutant's establishment abundance to the pre-existing type:
            self.set_type_abundance(type_index=preexisting_type_idx, abundance=self.get_type_abundance(preexisting_type_idx)+mutant_abundance)
            # Remove corresonding abundance from the parent type (abundance is moved from parent to mutant):
            self.set_type_abundance(type_index=parent_idx, abundance=self.get_type_abundance(parent_idx)-mutant_abundance)

            if(preexisting_type_idx not in self.extant_type_indices):
                # print("\t> was non-extant, add to mutant set")
                # print("\t......", np.argmax(self.extant_type_indices > preexisting_type_idx), np.argmax(self.extant_type_indices > preexisting_type_idx)*self.type_set.num_traits)
                # print("\textant_mutant_set idx", np.argmax(self.extant_type_indices > preexisting_type_idx)*self.type_set.num_traits)
                self.extant_mutant_set.add_type(mutant.generate_mutant_set(), index=np.argmax(self.extant_type_indices > preexisting_type_idx)*self.type_set.num_traits)


        else:
            # print("mutation: new type, add to mutant set")
            # Add the mutant to the population at an establishment abundance equal to 1/dfitness:
            #  (type set costs, phylogeny, and mutant set get updated as result of add_type())
            self.add_type(mutant, abundance=mutant_abundance, index=parent_idx+1, parent_index=parent_idx)  
            # Remove corresonding abundance from the parent type (abundance is moved from parent to mutant):
            self.set_type_abundance(type_index=parent_idx, abundance=self.get_type_abundance(parent_idx)-mutant_abundance)

            # print("\textant_mutant_set idx", ((mutant_idx//self.type_set.num_traits)+1)*self.type_set.num_traits)
            self.extant_mutant_set.add_type(mutant.generate_mutant_set(), index=((mutant_idx//self.type_set.num_traits)+1)*self.type_set.num_traits)
            

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # self, t, variables, 
    #             0    num_types
    #             1    num_mutants
    #             2    sigma
    #             3    b
    #             4    k
    #             5    eta
    #             6    l
    #             7    g
    #             8    c
    #             9    chi
    #             10    J
    #             11   mu
    #             12    energy_costs,
    #             13    num_resources
    #             14    rho
    #             15    tau
    #             16    omega
    #             17    D
    #             18    energy_uptake_coeffs
    #             19    resource_consumption_mode
    #             20    resource_inflow_mode

    def event_low_abundance(self, t, variables, *args):
        
        num_types    = args[0]
        # sigma        = args[2][:num_types]
        # b            = args[3][:num_types]
        # k            = args[4][:num_types]
        # eta          = args[5][:num_types]
        # l            = args[6][:num_types]
        # g            = args[7][:num_types]
        # energy_costs = args[12][:num_types]
        # omega        = args[16]

        # num_types    = self.type_set.num_types
        # sigma        = self.type_set.sigma
        # b            = self.type_set.b
        # k            = self.type_set.k
        # eta          = self.type_set.eta
        # l            = self.type_set.l
        # g            = self.type_set.g
        # energy_costs = self.type_set.energy_costs
        # omega        = self.resource_set.omega
        # resource_consumption_mode = self.resource_consumption_mode

        # num_types    = args[0]
        # sigma        = args[2][:num_types]
        # b            = args[3][:num_types]
        # k            = args[4][:num_types]
        # eta          = args[5][:num_types]
        # l            = args[6][:num_types]
        # g            = args[7][:num_types]
        # energy_costs = args[12][:num_types]
        # omega        = args[16]
        # resource_consumption_mode = args[19]

        # if(np.any(self.growthrate < 0)):
        #     N_t = variables[:self.type_set.num_types]
        #     #------------------------------
        #     min_abundance_typeidx = np.argmin(N_t[np.nonzero(N_t)])
        #     min_abundance_abs     = N_t[min_abundance_typeidx]
        #     min_abundance_rel     = N_t[min_abundance_typeidx]/np.sum(N_t)
        #     min_abundance_growthrate = self.growthrate[min_abundance_typeidx]

        N_t = variables[:num_types]
        abundance_rel = N_t/np.sum(N_t)
        #------------------------------
        if(np.any((abundance_rel < self.threshold_min_rel_abundance) | (N_t < self.threshold_min_abs_abundance))):
            # print('low abd: whoa')
            return -1

            # R_t = variables[-1-self.resource_set.num_resources:-1]
            # energy_uptake_coeffs  = args[18]
            # #------------------------------
            # min_abundance_typeidx    = np.argmin(N_t[np.nonzero(N_t)])
            # min_abundance_abs        = N_t[min_abundance_typeidx]
            # min_abundance_rel        = N_t[min_abundance_typeidx]/np.sum(N_t)
            # growth_rate              = ConsumerResourceSystem.growth_rate(N_t, R_t, self.type_set.sigma, self.type_set.b, self.type_set.k, self.type_set.eta, self.type_set.l, self.type_set.g, self.type_set.energy_costs, self.resource_set.omega, self.resource_consumption_mode,energy_uptake_coeffs) 
            # min_abundance_growthrate = growth_rate[min_abundance_typeidx]
            # #------------------------------
            # if (min_abundance_growthrate < 0 and (min_abundance_abs < self.threshold_min_abs_abundance or min_abundance_rel < self.threshold_min_rel_abundance)):
            #     return -1
        # print('low abd: no')
        return 1
    #----------------------------------
    # event_low_abundance.direction = -1
    event_low_abundance.terminal  = True

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def handle_type_loss(self):
        # print("handle_type_loss")
        N = self.N_series[:,-1].ravel()
        R = self.R_series[:,-1].ravel()
        growth_rate = ConsumerResourceSystem.growth_rate(N, R, self.type_set.sigma, self.type_set.b, self.type_set.k, self.type_set.eta, self.type_set.l, self.type_set.g, self.type_set.energy_costs, self.resource_set.omega, self.resource_consumption_mode) 
        #----------------------------------
        lost_types = np.where( (N > 0) & (growth_rate < 0) & ((N < self.threshold_min_abs_abundance) | (N/np.sum(N) < self.threshold_min_rel_abundance)) )[0]
        # print(( (N > 0) & (growth_rate < 0) & ((N < self.threshold_min_abs_abundance) | (N/np.sum(N) < self.threshold_min_rel_abundance)) ))
        
        # print("handle_type_loss")

        # print("\t> N:", N)

        self.extant_type_indices = self.get_extant_types()
        # print("\t> extant_type_indices", self.extant_type_indices)

        # print("\t> lost_types", lost_types)
        for i in lost_types:
            self.set_type_abundance(type_index=i, abundance=0.0) # Set the abundance of lost types to 0 (but do not remove from system/type set data)

        

        if(len(lost_types)>0):
            # print("\t> remove", self.type_set.get_mutant_indices(np.where(np.in1d(self.extant_type_indices, lost_types))[0]))
            self.extant_mutant_set.remove_type(self.type_set.get_mutant_indices(np.where(np.in1d(self.extant_type_indices, lost_types))[0]))
            # exit()
        #----------------------------------
        return

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, type_set=None, abundance=0, sigma=None, b=None, k=None, eta=None, l=None, g=None, c=None, chi=None, mu=None, index=None, parent_index=None, parent_id=None):
        abundance      = utils.treat_as_list(abundance)
        new_type_idx   = index if index is not None else self.type_set.num_types # default to adding to end of matrices
        orig_num_types = self.type_set.num_types
        #----------------------------------
        self.type_set.add_type(type_set, sigma, b, k, eta, l, g, c, chi, mu, new_type_idx, parent_index, parent_id)
        num_new_types = self.type_set.num_types - orig_num_types
        #----------------------------------
        self.N_series = np.insert(self.N_series, new_type_idx, np.zeros(shape=(num_new_types, self.N_series.shape[1])), axis=0)
        self.set_type_abundance(type_index=list(range(new_type_idx, new_type_idx+num_new_types)), abundance=abundance)
        #----------------------------------
        # now handled in typeset self.generate_mutant_set()

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_type_abundance(self, abundance, type_index=None, type_id=None, t_index=-1):
        abundance    = utils.treat_as_list(abundance)
        type_indices = [ np.where(self.type_set.type_ids==tid)[0] for tid in utils.treat_as_list(type_id) ] if type_id is not None else utils.treat_as_list(type_index)
        #----------------------------------
        for i, type_idx in enumerate(type_indices):
            self.N_series[type_idx, t_index] = abundance[i]

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_type_abundance(self, type_index=None, type_id=None, t=None, t_index=None):
        type_indices = [ np.where(self.type_set.type_ids == tid)[0] for tid in utils.treat_as_list(type_id) ] if type_id is not None else utils.treat_as_list(type_index) if type_index is not None else list(range(self.type_set.num_types))
        time_indices = [ np.where(self.t_series == t_)[0] for t_ in utils.treat_as_list(t) ] if t is not None else utils.treat_as_list(t_index) if t_index is not None else -1 
        #----------------------------------
        abundances = self.N_series[type_indices, time_indices]
        return abundances if len(type_indices) > 1 else abundances[0]

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_extant_types(self):
        return np.where(self.N_series[:,-1] > 0)[0]

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_extant_type_set(self, type_set=None):
        type_set = self.type_set if type_set is None else type_set
        return type_set.get_type(self.get_extant_types())

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def generate_mutant_set(self, parent_set=None):
    #     parent_set = self.type_set if parent_set is None else parent_set
    #     #----------------------------------
    #     # now in typeset self.extant_mutant_set = self.type_set.generate_mutant_set()
    #     #----------------------------------
    #     # In the case of 'fast resource equilibriation' resource consumption dyanmics,
    #     # growth rates depend on the abundances of types in the population, and thus
    #     # calculating mutant fitnesses requires abundance/parameter information about 
    #     # parent types to be included when calculating mutant growth rates.
    #     # if(self.resource_consumption_mode=='fast_resource_eq'):
    #     #     self.extant_mutant_set.add_type(self.type_set)
    #     # Store the number of actual mutant types in this set so we can later reference values for only mutant types
    #     self.extant_mutant_set.num_types = self.extant_mutant_set.num_types #if self.resource_consumption_mode != 'fast_resource_eq' else self.extant_mutant_set.num_types - parent_set.num_types
    #     #----------------------------------
    #     return self.extant_mutant_set

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, type_indices=None):

        # print("get_dynamics_params()")

        # dynamics() expects args: 
        #     num_types, num_mutants, sigma, b, k, eta, l, g, c, chi, J, mu, energy_costs, energy_uptake_coeffs,
        #     num_resources, rho, tau, omega, D, 
        #     resource_consumption_mode, resource_inflow_mode

        # print("type_indices", type_indices)

        extant_type_params   = self.type_set.get_dynamics_params(type_indices)
        extant_mutant_params = self.extant_mutant_set.get_dynamics_params()
        resource_params      = self.resource_set.get_dynamics_params()

        # print("extant_type_params.sigma", extant_type_params['sigma'].shape)
        # print("extant_mutant_params.sigma", extant_mutant_params['sigma'].shape)

        type_params_wmuts = {
                              'num_types':    extant_type_params['num_types'], 
                              'num_mutants':  extant_mutant_params['num_types'],
                              'sigma':        np.concatenate([extant_type_params['sigma'], extant_mutant_params['sigma']]),
                              'b':            extant_type_params['b'] if extant_type_params['b'].ndim < 2 else np.concatenate([extant_type_params['b'], extant_mutant_params['b']]),
                              'k':            extant_type_params['k'] if extant_type_params['k'].ndim < 2 else np.concatenate([extant_type_params['k'], extant_mutant_params['k']]),
                              'eta':          extant_type_params['eta'] if extant_type_params['eta'].ndim < 2 else np.concatenate([extant_type_params['eta'], extant_mutant_params['eta']]),
                              'g':            extant_type_params['g'] if extant_type_params['g'].ndim < 2 else np.concatenate([extant_type_params['g'], extant_mutant_params['g']]),
                              'c':            extant_type_params['c'] if extant_type_params['c'].ndim < 2 else np.concatenate([extant_type_params['c'], extant_mutant_params['c']]),
                              'l':            extant_type_params['l'] if extant_type_params['l'].ndim < 2 else np.concatenate([extant_type_params['l'], extant_mutant_params['l']]),
                              'chi':          extant_type_params['chi'] if extant_type_params['chi'].ndim < 2 else np.concatenate([extant_type_params['chi'], extant_mutant_params['chi']]),
                              'J':            extant_type_params['J'],
                              'mu':           extant_type_params['mu'] if extant_type_params['mu'].ndim < 2 else np.concatenate([extant_type_params['mu'], extant_mutant_params['mu']]),
                              'energy_costs': np.concatenate([extant_type_params['energy_costs'], extant_mutant_params['energy_costs']]) 
                            }

        energy_uptake_coeffs = resource_params['omega'] * (1-type_params_wmuts['l']) * type_params_wmuts['sigma'] * type_params_wmuts['b']

        # print("energy_uptake_coeffs", energy_uptake_coeffs.shape)

        return (tuple(type_params_wmuts.values()) + (energy_uptake_coeffs,)
                + tuple(resource_params.values())
                + (self.resource_consumption_mode, self.resource_inflow_mode))

        #----------------------------------

        # mutant_indices = self.type_set.get_mutant_indices(type_indices)

        # type_params_wmuts = self.type_set.get_dynamics_params(type_indices, include_mutants=True)
        # resource_params   = self.resource_set.get_dynamics_params()

        # energy_uptake_coeffs = self.resource_set.omega * (1-type_params_wmuts['l']) * type_params_wmuts['sigma'] * type_params_wmuts['b']

        # return (tuple(type_params_wmuts.values()) + (energy_uptake_coeffs,)
        #         + tuple(resource_params.values())
        #         + (self.resource_consumption_mode, self.resource_inflow_mode))

        #----------------------------------
        

        # extant_type_params   = self.type_set.get_dynamics_params(type_indices)
        # extant_mutant_params = self.extant_mutant_set.get_dynamics_params(mutant_indices)
        # resource_params      = self.resource_set.get_dynamics_params()

        # extant_type_params.update({'energy_uptake_coeffs': self.resource_set.omega * (1-extant_type_params['l']) * extant_type_params['sigma'] * extant_type_params['b']})
        # extant_mutant_params.update({'energy_uptake_coeffs': self.resource_set.omega * (1-extant_mutant_params['l']) * extant_mutant_params['sigma'] * extant_mutant_params['b']})

        # return (tuple(extant_type_params.values()) 
        #         + tuple(extant_mutant_params.values()) 
        #         + tuple(resource_params.values())
        #         + (self.resource_consumption_mode, self.resource_inflow_mode))

        # type_params_wmuts = self.type_set.get_dynamics_params(type_indices, include_mutants=True)
        # resource_params   = self.resource_set.get_dynamics_params()

        # sigma_wmuts = type_params_wmuts['sigma']
        # b_wmuts     = type_params_wmuts['b']
        # k_wmuts     = type_params_wmuts['k']
        # l_wmuts     = type_params_wmuts['l']

        # energy_uptake_coeffs = np.multiply(self.resource_set.omega, np.multiply((1-l_wmuts), np.multiply(sigma_wmuts, np.multiply(b_wmuts, k_wmuts))))
        # energy_uptake_coeffs = self.resource_set.omega * (1-l_wmuts) * sigma_wmuts * b_wmuts# * k_wmuts

        # return (tuple(type_params_wmuts.values()) + tuple(resource_params.values())
        #          + (energy_uptake_coeffs,)
        #          + (self.resource_consumption_mode, self.resource_inflow_mode))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def dimensions(self):
        return (self.type_set.num_types, self.resource_set.num_resources)













    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # # @jit(nopython=True)
    # def resource_demand(N, sigma):
    #     return np.sum(np.multiply(N[:, np.newaxis], sigma), axis=0)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # # @jit(nopython=True)
    # def resource_consumption(N, R, sigma, b, k, resource_consumption_mode):
    #     if(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_FASTEQ):
    #         resource_demand = ConsumerResourceSystem.resource_demand(N, sigma)
    #         resource_consumption = np.multiply(sigma, b/(1 + (resource_demand/k)))
    #     elif(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_LINEAR):
    #         resource_consumption = np.multiply(sigma, np.multiply(b, R))
    #     elif(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_MONOD):
    #         resource_consumption = np.multiply(sigma, np.multiply(b, R/(R + k)))
    #     else:
    #         resource_consumption = np.zeros((len(N), len(R)))
    #     #----------------------------------
    #     return resource_consumption

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # def energy_uptake(N, R, sigma, b, k, l, omega):
    #     return np.sum( np.multiply(omega, np.multiply((1-l), ConsumerResourceSystem.resource_consumption(N, R, sigma, b, k))), axis=1)

    # @staticmethod
    # # @jit(nopython=True)
    # def energy_uptake(resource_consumption, l, omega):
    #     return np.sum( np.multiply(omega, np.multiply((1-l), resource_consumption)), axis=1)

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # def energy_surplus(N, R, sigma, b, k, l, omega, energy_costs):
    #     return ConsumerResourceSystem.energy_uptake(N, R, sigma, b, k, l, omega) - energy_costs

    # @staticmethod
    # # @jit(nopython=True)
    # def energy_surplus(resource_consumption, l, omega, energy_costs):
    #     return ConsumerResourceSystem.energy_uptake(resource_consumption, l, omega) - energy_costs

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # def growth_rate(N, R, sigma, b, k, g, l, omega, energy_costs):
    #     return np.multiply(g, ConsumerResourceSystem.energy_surplus(N, R, sigma, b, k, l, omega, energy_costs))

    # @staticmethod
    # # @jit(nopython=True)
    # def growth_rate(resource_consumption, g, l, omega, energy_costs):
    #     return np.multiply(g, ConsumerResourceSystem.energy_surplus(resource_consumption, l, omega, energy_costs))

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # # @jit(nopython=True)
    # def resource_inflow(R, rho, resource_inflow_mode):
    #     if(self.resource_inflow_mode == 'none' or self.resource_inflow_mode == 'zero'):
    #         inflow = np.zeros(len(R))
    #     elif(self.resource_inflow_mode == 'constant'):
    #         inflow = rho
    #     #----------------------------------
    #     return inflow

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # # @jit(nopython=True)
    # def resource_secretion(resource_consumption, N, l, omega, D):
    #     return np.divide(np.dot(np.multiply(omega, np.sum(np.multiply(N[:,np.newaxis], np.multiply(l, resource_consumption)), axis=0)), D.T), omega, where=(omega>0), out=np.zeros_like(omega)).ravel()

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @staticmethod
    # def mutation_propensities(N, R, mu=None, mutant_set=None, parent_set=None, resource_set=None):
    #     mutant_set  = self.extant_mutant_set if mutant_set is None else mutant_set
    #     parent_set  = self.type_set if parent_set is None else parent_set
    #     resource_set = self.resource_set if resource_set is None else resource_set
    #     #----------------------------------
    #     # Compete mutants as single individuals in the context of the current populations' abundances:
    #     mutant_competition_abundances = np.ones(mutant_set.num_mutant_types) if self.resource_consumption_mode != 'fast_resource_eq' else np.concatenate([np.ones(mutant_set.num_mutant_types), N])
    #     mutant_set.fitness = self.growth_rate(self.resource_consumption(N=mutant_competition_abundances, R=R, type_set=mutant_set), type_set=mutant_set)
    #     mutant_fitnesses    = mutant_set.fitness[:mutant_set.num_mutant_types]  
    #     #----------------------------------
    #     mutation_propensities = np.maximum(0, np.multiply(mutant_fitnesses, 
    #                                                       np.multiply(np.repeat(N,              repeats=parent_set.num_traits, axis=0),   # abundance of parent of each mutant 
    #                                                                   np.repeat(parent_set.mu, repeats=parent_set.num_traits, axis=0)))) # mutation rate of parent of each mutant
    #     #----------------------------------
    #     return mutation_propensities











    