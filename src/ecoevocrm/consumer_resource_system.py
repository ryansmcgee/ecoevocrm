# This is to sidestep a numpy overhead introduced in numpy 1.17:
# https://stackoverflow.com/questions/61983372/is-built-in-method-numpy-core-multiarray-umath-implement-array-function-a-per
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
#----------------------------------
import numpy as np
import scipy.integrate
from scipy.integrate._ivp.base import OdeSolver

from ecoevocrm.type_set import *
from ecoevocrm.resource_set import *
import ecoevocrm.utils as utils


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ConsumerResourceSystem():

    # Define Class constants:
    CONSUMPTION_MODE_FASTEQ = 0
    CONSUMPTION_MODE_LINEAR = 1
    CONSUMPTION_MODE_MONOD  = 2
    INFLOW_MODE_NONE        = 0
    INFLOW_MODE_CONSTANT    = 1

    def __init__(self, 
                 type_set      = None,
                 resource_set  = None,
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
                 seed = None ):

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

        #----------------------------------
        # Initialize system variables:
        #----------------------------------
        if(N_init is None or R_init is None):
            utils.error(f"Error in ConsumerResourceSystem __init__(): Values for N_init and R_init must be provided.")
        self._N_series = utils.ExpandableArray(utils.reshape(N_init, shape=(system_num_types, 1)), alloc_shape=(self.resource_set.num_resources*25, 1))
        self._R_series = utils.ExpandableArray(utils.reshape(R_init, shape=(system_num_resources, 1)), alloc_shape=(self.resource_set.num_resources, 1))

        #----------------------------------
        # Initialize system time:
        #----------------------------------
        self._t_series = utils.ExpandableArray([0], alloc_shape=(1, 1))

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

        #----------------------------------
        # Initialize set of mutant types:
        #----------------------------------
        self.mutant_set = self.type_set.generate_mutant_set()


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def get_array(arr):
        return arr.values if isinstance(arr, utils.ExpandableArray) else arr

    @property
    def N_series(self):
        return ConsumerResourceSystem.get_array(self._N_series)

    @property
    def N(self):
        return self.N_series[:, -1]

    @property
    def R_series(self):
        return ConsumerResourceSystem.get_array(self._R_series)

    @property
    def R(self):
        return self.R_series[:, -1]       

    @property
    def t_series(self):
        return ConsumerResourceSystem.get_array(self._t_series).ravel()

    @property
    def t(self):
        return self.t_series[-1]

    @property
    def extant_type_indices(self):
        return np.where(self.N > 0)[0]

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def run(self, T=1e4, dt=None, integration_method='default'):

        t_start   = self.t
        t_elapsed = 0

        self._t_series.expand_alloc((self._t_series.alloc[0], self._t_series.alloc[1]+int(T/dt if dt is not None else T/0.1)))
        self._N_series.expand_alloc((self._N_series.alloc[0], self._N_series.alloc[1]+int(T/dt if dt is not None else T/0.1)))
        self._R_series.expand_alloc((self._R_series.alloc[0], self._R_series.alloc[1]+int(T/dt if dt is not None else T/0.1)))

        while(t_elapsed < T):

            #------------------------------
            # Set initial conditions and integration variables:
            #------------------------------
           
            # Set the time interval for this integration epoch:
            t_span = (self.t, self.t+T)

            # Set the time ticks at which to save trajectory values:
            t_eval = np.arange(start=t_span[0], stop=t_span[1]+dt, step=dt) if dt is not None else None,

            # Get the indices and count of extant types (abundance > 0):
            num_extant_types = len(self.extant_type_indices)

            # Set the initial conditions for this integration epoch:
            N_init = self.N[self.extant_type_indices] 
            R_init = self.R 
            cumPropMut_init = np.array([0])
            init_cond = np.concatenate([N_init, R_init, cumPropMut_init])

            # Get the params for the dynamics:
            params = self.get_dynamics_params(self.extant_type_indices)

            # Draw a random propensity threshold for triggering the next Gillespie mutation event:
            self.threshold_mutation_propensity = np.random.exponential(1)
            
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
                                             events = [self.event_mutation],
                                             method = _integration_method )

            #------------------------------
            # Update the system's trajectories with latest dynamics epoch:
            #------------------------------

            N_epoch = np.zeros(shape=(self._N_series.shape[0], len(sol.t)))
            N_epoch[self.extant_type_indices] = sol.y[:num_extant_types]
            
            R_epoch = sol.y[-1-self.resource_set.num_resources:-1]
            
            self._t_series.add(sol.t, axis=1)
            self._N_series.add(N_epoch, axis=1)
            self._R_series.add(R_epoch, axis=1)
            
            t_elapsed = self.t - t_start

            typeCountStr = f"{num_extant_types}*({self.mutant_set.num_types})/{self.type_set.num_types}"

            #------------------------------
            # Handle events and update the system's states accordingly:
            #------------------------------
            if(sol.status == 1): # An event occurred
                if(len(sol.t_events[0]) > 0):
                    print(f"[ Mutation event occurred at  t={self.t:.4f} {typeCountStr}]\t\r", end="")
                    self.handle_mutation_event()
                    self.handle_type_loss()
            elif(sol.status == 0): # Reached end T successfully
                self.handle_type_loss()
            else: # Error occurred in integration
                utils.error("Error in ConsumerResourceSystem run(): Integration of dynamics using scipy.solve_ivp returned with error status.")

        #------------------------------
        # Finalize data series at end of integration period:
        #------------------------------

        self._t_series.trim()
        self._N_series.trim()
        self._R_series.trim()

        return


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def resource_demand(N, sigma):
        return np.einsum(('ij,ij->j' if N.ndim == 2 else 'i,ij->j'), N, sigma)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def growth_rate(N, R, sigma, b, k, eta, l, g, energy_costs, omega, resource_consumption_mode, energy_uptake_coeffs=None):
        # TODO: Allow for calculating for single N/R or time series of N/Rs
        energy_uptake_coeffs = omega * (1-l) * sigma * b if energy_uptake_coeffs is None else energy_uptake_coeffs
        #------------------------------
        if(resource_consumption_mode == ConsumerResourceSystem.CONSUMPTION_MODE_FASTEQ):
            resource_demand = ConsumerResourceSystem.resource_demand(N, sigma)
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


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def dynamics(self, t, variables, 
                    num_types, num_mutants, sigma, b, k, eta, l, g, c, chi, J, mu, energy_costs, energy_uptake_coeffs,
                    num_resources, rho, tau, omega, D, resource_consumption_mode, resource_inflow_mode):

        N_t = np.zeros(num_types+num_mutants)
        N_t[:num_types] = variables[:num_types]

        R_t = variables[-1-num_resources:-1]

        #------------------------------

        growth_rate = ConsumerResourceSystem.growth_rate(N_t, R_t, sigma, b, k, eta, l, g, energy_costs, omega, resource_consumption_mode, energy_uptake_coeffs)
        
        dNdt = N_t[:num_types] * growth_rate[:num_types] # only calc dNdt for extant (non-mutant) types

        #------------------------------

        dRdt = np.zeros(num_resources)

        #------------------------------

        self.mutant_fitnesses = growth_rate[-num_mutants:]

        self.mutation_propensities = np.maximum(0, self.mutant_fitnesses * np.repeat(N_t[:num_types] * mu, repeats=num_resources))
                                                          
        dCumPropMut = np.sum(self.mutation_propensities, keepdims=True)
        
        #------------------------------

        return np.concatenate((dNdt, dRdt, dCumPropMut))

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def event_mutation(self, t, variables, *args):
        cumulative_mutation_propensity = variables[-1]
        return self.threshold_mutation_propensity - cumulative_mutation_propensity
    #----------------------------------
    event_mutation.direction = -1
    event_mutation.terminal  = True

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def handle_mutation_event(self):
        # Pick the mutant that will be established with proababilities proportional to mutants' propensities for establishment:
        mutant_indices   = self.type_set.get_mutant_indices(self.extant_type_indices)
        mutant_drawprobs = self.mutation_propensities/np.sum(self.mutation_propensities)
        mutant_idx       = np.random.choice(mutant_indices, p=mutant_drawprobs)
        # Retrieve the mutant and some of its properties:
        mutant           = self.mutant_set.get_type(mutant_idx)
        mutant_type_id   = self.mutant_set.get_type_id(mutant_idx)
        mutant_fitness   = self.mutant_fitnesses[np.argmax(mutant_indices == mutant_idx)]
        mutant_abundance = np.maximum(1/mutant_fitness, 1) # forcing abundance of new types to be at least 1, this is a Ryan addition (perhaps controversial)
        # Get the index of the parent of the selected mutant:
        parent_idx       = mutant_idx // self.type_set.num_traits
        #----------------------------------
        if(mutant_type_id in self.type_set.type_ids):
            # This "mutant" is a pre-existing type in the population; get its index:
            preexisting_type_idx = np.where(np.array(self.type_set.type_ids) == mutant_type_id)[0][0]
            # Add abundance equal to the mutant's establishment abundance to the pre-existing type:
            self.set_type_abundance(type_index=preexisting_type_idx, abundance=self.get_type_abundance(preexisting_type_idx)+mutant_abundance)
            # Remove corresonding abundance from the parent type (abundance is moved from parent to mutant):
            self.set_type_abundance(type_index=parent_idx, abundance=max(self.get_type_abundance(parent_idx)-mutant_abundance, 1))
        else:
            # Add the mutant to the population at an establishment abundance equal to 1/dfitness:
            self.add_type(mutant, abundance=mutant_abundance, parent_index=parent_idx)
            # Remove corresonding abundance from the parent type (abundance is moved from parent to mutant):
            self.set_type_abundance(type_index=parent_idx, abundance=max(self.get_type_abundance(parent_idx)-mutant_abundance, 1))
        #----------------------------------
        return
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def handle_type_loss(self):
        N = self.N
        R = self.R
        #----------------------------------
        growth_rate = ConsumerResourceSystem.growth_rate(N, R, self.type_set.sigma, self.type_set.b, self.type_set.k, self.type_set.eta, self.type_set.l, self.type_set.g, self.type_set.energy_costs, self.resource_set.omega, self.resource_consumption_mode) 
        #----------------------------------
        lost_types = np.where( (N < 0) | ((N > 0) & (growth_rate < 0) & ((N < self.threshold_min_abs_abundance) | (N/np.sum(N) < self.threshold_min_rel_abundance))) )[0]
        for i in lost_types:
            self.set_type_abundance(type_index=i, abundance=0.0) # Set the abundance of lost types to 0 (but do not remove from system/type set data)
        #----------------------------------
        return

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, new_type_set=None, abundance=0, parent_index=None, parent_id=None):#, index=None, ):
        abundance      = utils.treat_as_list(abundance)
        #----------------------------------
        self.type_set.add_type(new_type_set, parent_index=parent_index, parent_id=parent_id)
        #----------------------------------
        self._N_series = self._N_series.add(np.zeros(shape=(new_type_set.num_types, self.N_series.shape[1])))
        self.set_type_abundance(type_index=list(range(self.type_set.num_types-new_type_set.num_types, self.type_set.num_types)), abundance=abundance)
        #----------------------------------
        self.mutant_set.add_type(new_type_set.generate_mutant_set())

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_type_abundance(self, abundance, type_index=None, type_id=None, t=None, t_index=None):
        abundance    = utils.treat_as_list(abundance)
        type_indices = [ np.where(self.type_set.type_ids==tid)[0] for tid in utils.treat_as_list(type_id) ] if type_id is not None else utils.treat_as_list(type_index)
        t_idx        = np.argmax(self.t_series >= t) if t is not None else t_index if t_index is not None else -1
        #----------------------------------
        for i, type_idx in enumerate(type_indices):
            self.N_series[type_idx, t_idx] = abundance[i]

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_type_abundance(self, type_index=None, type_id=None, t=None, t_index=None):
        type_indices = [ np.where(self.type_set.type_ids == tid)[0] for tid in utils.treat_as_list(type_id) ] if type_id is not None else utils.treat_as_list(type_index) if type_index is not None else list(range(self.type_set.num_types))
        time_indices = [ np.where(self.t_series == t_)[0] for t_ in utils.treat_as_list(t) ] if t is not None else utils.treat_as_list(t_index) if t_index is not None else -1
        #----------------------------------
        abundances = self.N_series[type_indices, time_indices]
        return abundances if len(type_indices) > 1 else abundances[0]

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, type_indices=None):
        type_params     = self.type_set.get_dynamics_params(type_indices)
        mutant_params   = self.mutant_set.get_dynamics_params(self.type_set.get_mutant_indices(type_indices))
        resource_params = self.resource_set.get_dynamics_params()
        #----------------------------------
        type_params_wmuts = {
                              'num_types':    type_params['num_types'], 
                              'num_mutants':  mutant_params['num_types'],
                              'sigma':        np.concatenate([type_params['sigma'], mutant_params['sigma']]),
                              'b':            type_params['b'] if type_params['b'].ndim < 2 else np.concatenate([type_params['b'], mutant_params['b']]),
                              'k':            type_params['k'] if type_params['k'].ndim < 2 else np.concatenate([type_params['k'], mutant_params['k']]),
                              'eta':          type_params['eta'] if type_params['eta'].ndim < 2 else np.concatenate([type_params['eta'], mutant_params['eta']]),
                              'g':            type_params['g'] if type_params['g'].ndim < 2 else np.concatenate([type_params['g'], mutant_params['g']]),
                              'c':            type_params['c'] if type_params['c'].ndim < 2 else np.concatenate([type_params['c'], mutant_params['c']]),
                              'l':            type_params['l'] if type_params['l'].ndim < 2 else np.concatenate([type_params['l'], mutant_params['l']]),
                              'chi':          type_params['chi'] if type_params['chi'].ndim < 2 else np.concatenate([type_params['chi'], mutant_params['chi']]),
                              'J':            type_params['J'],
                              'mu':           type_params['mu'] if type_params['mu'].ndim < 2 else np.concatenate([type_params['mu'], mutant_params['mu']]),
                              'energy_costs': np.concatenate([type_params['energy_costs'], mutant_params['energy_costs']]) 
                            }
        #----------------------------------
        energy_uptake_coeffs = resource_params['omega'] * (1-type_params_wmuts['l']) * type_params_wmuts['sigma'] * type_params_wmuts['b']
        #----------------------------------
        return (tuple(type_params_wmuts.values()) + (energy_uptake_coeffs,)
                + tuple(resource_params.values())
                + (self.resource_consumption_mode, self.resource_inflow_mode))

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder_types(self, order=None):
        type_order   = np.argsort(self.type_set.lineage_ids) if order is None else order
        mutant_order = self.type_set.get_mutant_indices(type_order)
        #----------------------------------
        self._N_series = self._N_series.reorder(type_order)
        self.type_set.reorder_types(type_order) # don't need to reorder mutant_set because type_set.mutant_indices gets reordered and keeps correct pointers









    