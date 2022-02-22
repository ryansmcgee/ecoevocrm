import numpy as np
import scipy.integrate
from scipy.integrate._ivp.base import OdeSolver

from ecoevocrm.strain_pool import *
from ecoevocrm.resource_set import *
from ecoevocrm.biochemistry import *
import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ConsumerResourceSystem():

    def __init__(self, 
                 strain_pool   = None,
                 resource_set  = None,
                 biochemistry  = None,
                 N_init        = None,
                 R_init        = None,
                 num_types     = None,
                 num_resources = None,
                 sigma         = None,
                 b             = 1,
                 k             = 0,
                 eta           = 1,
                 g             = 1,
                 l             = 0,
                 c             = 0,
                 chi           = None,
                 mu            = 0,
                 J             = None,
                 D             = None,
                 rho           = 0,
                 tau           = 0,
                 omega         = 1,
                 resource_consumption_mode     = 'linear',
                 resource_inflow_mode          = 'constant',
                 threshold_min_abs_abundance   = 1,
                 threshold_min_rel_abundance   = 1e-6,
                 threshold_eq_abundance_change = 1e4,
                 threshold_precise_integrator  = 1e2,
                 seed = None):

        #----------------------------------------

        if(seed is not None):
            np.random.seed(seed)
            self.seed = seed

        #----------------------------------------
        # Determine the dimensions of the system:
        #----------------------------------------
        if(strain_pool is not None):
            system_num_types     = strain_pool.num_types
            system_num_resources = strain_pool.num_traits
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
                utils.error("Error in ConsumerResourceSystem __init__(): Number of types must be specified by providing a) a strain pool, b) a sigma matrix, c) a num_types value, or d) a list for N_init.")
            #---
            if(resource_set is not None):
                system_num_resources = resource_set.num_resources
            elif(num_resources is not None):
                system_num_resources = num_resources
            elif(isinstance(R_init, (list, np.ndarray))):
                system_num_resources = np.array(R_init).ravel().shape[0]
            else:
                utils.error("Error in ConsumerResourceSystem __init__(): Number of resources must be specified by providing a) a resource set, b) a sigma matrix, c) a num_resources value, or d) a list for R_init.")
            
        #----------------------------------------
        # Initialize strain pool parameters:
        #----------------------------------------
        if(strain_pool is not None):
            if(isinstance(strain_pool, StrainPool)):
                self.strain_pool = strain_pool
            else:
                utils.error(f"Error in ConsumerResourceSystem __init__(): strain_pool argument expects object of StrainPool type.")
        else:
            self.strain_pool = StrainPool(num_types=system_num_types, num_traits=system_num_resources, sigma=sigma, b=b, k=k, eta=eta, l=l, g=g, c=c, chi=chi, mu=mu)
        # Check that the strain pool dimensions match the system dimensions:
        if(system_num_types != self.strain_pool.num_types): 
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system types ({system_num_types}) does not match number of strain pool types ({self.strain_pool.num_types}).")
        if(system_num_resources != self.strain_pool.num_traits): 
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system resources ({system_num_resources}) does not match number of strain pool traits ({self.strain_pool.num_traits}).")

        #----------------------------------------
        # Initialize resource set parameters:
        #----------------------------------------
        if(resource_set is not None):
            if(isinstance(resource_set, ResourceSet)):
                self.resource_set = resource_set
            else:
                utils.error(f"Error in ConsumerResourceSystem __init__(): resource_set argument expects object of ResourceSet type.")
        else:
            self.resource_set = ResourceSet(num_resources=system_num_resources, rho=rho, tau=tau, omega=omega)
        # Check that the strain pool dimensions match the system dimensions:
        if(system_num_resources != self.resource_set.num_resources): 
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system resources ({system_num_resources}) does not match number of resource set resources ({self.resource_set.num_resources}).")

        #----------------------------------------
        # Initialize biochemistry parameters:
        #----------------------------------------
        if(biochemistry is not None):
            if(isinstance(biochemistry, Biochemistry)):
                self.biochemistry = biochemistry
            else:
                utils.error(f"Error in ConsumerResourceSystem __init__(): biochemistry argument expects object of Biochemistry type.")
        else:
            self.biochemistry = Biochemistry(num_resources=system_num_resources, J=J, D=D)
        # Check that the strain pool dimensions match the system dimensions:
        if(system_num_resources != self.biochemistry.num_resources): 
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system resources ({system_num_resources}) does not match number of resource set resources ({self.biochemistry.num_resources}).")

        #----------------------------------------
        # Initialize system variables:
        #----------------------------------------
        if(N_init is None or R_init is None):
            utils.error(f"Error in ConsumerResourceSystem __init__(): Values for N_init and R_init must be provided.")
        self.N_series = utils.reshape(N_init, shape=(system_num_types, 1))
        self.R_series = utils.reshape(R_init, shape=(system_num_resources, 1))

        #----------------------------------------
        # Initialize system time:
        #----------------------------------------
        self.t = 0
        self.t_series = np.array([0])

        #----------------------------------------
        # Initialize event parameters:
        #----------------------------------------
        self.threshold_mutation_propensity = None # is updated in run()
        self.threshold_eq_abundance_change = threshold_eq_abundance_change
        self.threshold_min_abs_abundance   = threshold_min_abs_abundance
        self.threshold_min_rel_abundance   = threshold_min_rel_abundance
        self.threshold_precise_integrator  = threshold_precise_integrator

        #----------------------------------------
        # Initialize system options:
        #----------------------------------------
        self.resource_consumption_mode = resource_consumption_mode
        self.resource_inflow_mode      = resource_inflow_mode

        #----------------------------------------
        # Update set of mutants for current strain pool:
        #----------------------------------------
        self.generate_mutant_pool()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def run(self, T, output_dt=None, integration_method='default'):

        t_start   = self.t
        t_elapsed = 0

        while(t_elapsed < T):

            self.threshold_mutation_propensity = np.random.exponential(1)
           
            # Set the time interval for this integration epoch:
            t_span = (self.t, T)

            # Set the initial conditions for this integration epoch:
            N_init = self.N_series[:,-1] #if self.N_series is not None else self.N_init
            R_init = self.R_series[:,-1] #if self.R_series is not None else self.R_init
            cumPropMut_init = np.array([0])
            init_cond = np.concatenate([N_init, R_init, cumPropMut_init])

            # Set the integration method:
            if(integration_method == 'default'):
                num_extant_types = np.count_nonzero(self.N_series[:,-1])
                if(num_extant_types < self.threshold_precise_integrator):
                    _integration_method = 'Radau' # accurate stiff integrator
                else:
                    _integration_method = 'LSODA' # adaptive stiff/non-stiff integrator
            else:
                _integration_method = integration_method

            # Integrate the system dynamics:
            sol = scipy.integrate.solve_ivp( self.dynamics, 
                                             y0     = init_cond,
                                             t_span = t_span,
                                             t_eval = np.arange(start=t_span[0], stop=t_span[1], step=output_dt) if output_dt is not None else None,
                                             events = [self.event_mutation, self.event_type_loss],
                                             method = _integration_method )

            # Update the system time series:
            self.t_series = np.concatenate([self.t_series[:-1], sol.t]) if self.t_series is not None else sol.t
            self.t = self.t_series[-1] + (output_dt if output_dt is not None else 0)
            t_elapsed = self.t - t_start
            # print(f"t={self.t} t_elapsed={t_elapsed}")

            # Update the system data series:
            self.N_series = np.hstack([self.N_series[:,:-1], sol.y[:self.strain_pool.num_types]]) if self.N_series is not None else sol.y[:self.strain_pool.num_types]
            self.R_series = np.hstack([self.R_series[:,:-1], sol.y[-1-self.resource_set.num_resources:-1]]) if self.R_series is not None else sol.y[-1-self.resource_set.num_resources:-1]

            # print()
            # print(sol.t_events)            
            # print("r<0:", np.where((self.N_series[:,-1] > 0) & (self.growth_rates < 0)[0]))
            # print("N<1:", np.where((self.N_series[:,-1] > 0) & (self.N_series[:,-1] < 1)[0]))
            # print("p<~:", np.where((self.N_series[:,-1] > 0) & (self.N_series[:,-1]/np.sum(self.N_series[:,-1]) < self.threshold_min_rel_abundance))[0])

            if(sol.status == 1): # An event occurred
                if(len(sol.t_events[0]) > 0):
                    print(f"[ Mutation event occurred at t={self.t} ]\t\r", end="")
                    self.handle_mutation_event()
                    self.handle_type_loss()
                if(len(sol.t_events[1]) > 0):
                    print(f"[ Type loss event occurred at t={self.t} ]\t\r", end="")
                    self.handle_type_loss()
            elif(sol.status == 0): # Reached end T successfully
                self.handle_type_loss()
            else: # Error occurred in integration
                utils.error("Error in ConsumerResourceSystem run(): Integration of dynamics using scipy.solve_ivp returned with error status.")
            
            total_epoch_abundance_change = np.sum( self.N_series[:,-1] - self.N_series[:,0] )
            if(total_epoch_abundance_change < self.threshold_eq_abundance_change):
                break

            # break


    def dynamics(self, t, variables):
        if(len(variables) != (self.strain_pool.num_types + self.resource_set.num_resources + 1)):
            utils.error(f"Error in dynamics(): length of variables array must equal num_types + num_resources + 1 (cum. prop. mutation) = {self.strain_pool.num_types + self.resource_set.num_resources + 1}, but {len(variables)} were given.")
        #------------------------------
        _N_t = variables[:self.strain_pool.num_types]
        _R_t = variables[-1-self.resource_set.num_resources:-1]
        _h_t = self.resource_consumption(_N_t, _R_t)
        #------------------------------
        self.growth_rates = self.growth_rate(_h_t)
        #------------------------------
        dNdt = np.multiply(_N_t, self.growth_rates)
        #------------------------------
        if(self.resource_consumption_mode == 'fast_resource_eq'):
            dRdt = np.zeros_like(_R_t)
        else:
            dRdt = self.resource_inflow(_R_t) - np.multiply(1/self.resource_set.tau, _R_t) - np.sum(np.multiply(_N_t[:,np.newaxis], _h_t), axis=0) + self.resource_secretion(_N_t, _h_t)
            dRdt = dRdt.ravel()
        #------------------------------
        self.latest_total_abundance_change = np.sum(dNdt)
        #------------------------------
        cumulative_mutation_propensity = variables[-1]
        # print("cumprop", cumulative_mutation_propensity)
        dCumPropMut = np.array([np.sum(self.mutation_propensities(_N_t, _R_t))])
        #------------------------------
        return np.concatenate((dNdt, dRdt, dCumPropMut))


    def resource_consumption(self, N, R, sigma=None, b=None, k=None, strain_pool=None, resource_set=None):
        strain_pool  = self.strain_pool if strain_pool is None else strain_pool
        resource_set = self.resource_set if resource_set is None else resource_set
        sigma = strain_pool.sigma if sigma is None else sigma
        b     = strain_pool.b if b is None else b
        k     = strain_pool.k if k is None else k
        #----------------------------------
        if(self.resource_consumption_mode == 'linear'):
            resource_consumption = np.multiply(sigma, np.multiply(b, R))
        elif(self.resource_consumption_mode == 'monod'):
            resource_consumption = np.multiply(sigma, np.multiply(b, R/(R + k)))
        elif(self.resource_consumption_mode == 'fast_resource_eq'):
            T = np.sum(np.multiply(N[:,np.newaxis], sigma), axis=0)
            resource_consumption = np.multiply(sigma, b/(1 + (T/k)))
        else:
            resource_consumption = np.zeros((self.strain_pool.num_types, self.resource_set.num_resources))
        #----------------------------------
        return resource_consumption


    def energy_uptake(self, resource_consumption, l=None, omega=None, strain_pool=None, resource_set=None):
        strain_pool  = self.strain_pool if strain_pool is None else strain_pool
        resource_set = self.resource_set if resource_set is None else resource_set
        l     = strain_pool.l if l is None else l
        omega = resource_set.omega if omega is None else omega
        #----------------------------------
        return np.sum( np.multiply(omega, np.multiply((1-l), resource_consumption)), axis=1)


    def energy_surplus(self, resource_consumption, strain_pool=None):
        strain_pool = self.strain_pool if strain_pool is None else strain_pool
        return self.energy_uptake(resource_consumption, strain_pool=strain_pool) - strain_pool.energy_costs


    def growth_rate(self, resource_consumption, strain_pool=None):
        strain_pool = self.strain_pool if strain_pool is None else strain_pool
        return np.multiply(strain_pool.g, self.energy_surplus(resource_consumption, strain_pool=strain_pool))


    def resource_inflow(self, R, rho=None, resource_set=None):
        resource_set = self.resource_set if resource_set is None else resource_set
        rho = resource_set.rho if rho is None else rho
        #----------------------------------
        if(self.resource_inflow_mode == 'none' or self.resource_inflow_mode == 'zero'):
            inflow = np.zeros_like(R)
        elif(self.resource_inflow_mode == 'constant'):
            inflow = rho
        #----------------------------------
        return inflow


    def resource_secretion(self, resource_consumption, N, l=None, omega=None, D=None, strain_pool=None, resource_set=None, biochemistry=None):
        strain_pool  = self.strain_pool if strain_pool is None else strain_pool
        resource_set = self.resource_set if resource_set is None else resource_set
        biochemistry = self.biochemistry if biochemistry is None else biochemistry
        l     = strain_pool.l if l is None else l
        omega = resource_set.omega if omega is None else omega
        D     = biochemistry.D if D is None else D
        #----------------------------------
        if(D is not None):
            secretion = np.divide(np.dot(np.multiply(omega, np.sum(np.multiply(N[:,np.newaxis], np.multiply(l, resource_consumption)), axis=0)), D.T), omega, where=(omega>0), out=np.zeros_like(omega)).ravel()
        else:
            secretion = np.zeros(self.resource_set.num_resources)
        #----------------------------------
        return secretion


    def mutation_propensities(self, N, R, mu=None, mutant_pool=None, parent_pool=None, resource_set=None):
        mutant_pool  = self.mutant_pool if mutant_pool is None else mutant_pool
        parent_pool  = self.strain_pool if parent_pool is None else parent_pool
        resource_set = self.resource_set if resource_set is None else resource_set
        #----------------------------------

        # >>> This is where I am:
        #     If tikhonov consumption mode, then mutant pool needs to include parent types. 
        #         Stick parent matrix rows at the bottom of the mutant pool matrices
        #         Pass concat(mutant abds, parent abds) to N for resource_consumption call in mutation_propensities just below
        #         Only keep the first [:num mutant types] propensities that are returned from growth_rate() call (cutting off propensities calculated incidentally for parents due to their inclusion in the mutant_pool)
        #         Then the draw probs and everything should normalize fine since the parents were cut out of propensities list
        #     For other consumption modes, dont both with any of the above


        # print(N)
        # print(N.shape)
        # print(np.ones(mutant_pool.num_types))
        # print(np.ones(mutant_pool.num_types).shape)
        # print(np.concatenate([np.ones(mutant_pool.num_types), N]))
        # print(np.concatenate([np.ones(mutant_pool.num_types), N]).shape)
        # exit()

        # Compete mutants as single individuals in the context of the current populations' abundances:
        mutant_competition_abundances = np.ones(mutant_pool.num_mutant_types) if self.resource_consumption_mode != 'fast_resource_eq' else np.concatenate([np.ones(mutant_pool.num_mutant_types), N])

        # mutant_pool.fitness = self.growth_rate(self.resource_consumption(N=np.ones(mutant_pool.num_types), R=R, strain_pool=mutant_pool), strain_pool=mutant_pool)
        mutant_pool.fitness = self.growth_rate(self.resource_consumption(N=mutant_competition_abundances, R=R, strain_pool=mutant_pool), strain_pool=mutant_pool)
        mutant_fitnesses    = mutant_pool.fitness[:mutant_pool.num_mutant_types]  
        # print("mutant_pool.fitness", mutant_pool.fitness, mutant_pool.fitness.shape)
        # print("mutant_fitnesses", mutant_fitnesses, mutant_fitnesses.shape)
        # exit()
        #----------------------------------
        mutation_propensities = np.maximum(0, np.multiply(mutant_fitnesses, 
                                                          np.multiply(np.repeat(N,              repeats=parent_pool.num_traits, axis=0),   # abundance of parent of each mutant 
                                                                      np.repeat(parent_pool.mu, repeats=parent_pool.num_traits, axis=0)))) # mutation rate of parent of each mutant
        #----------------------------------
        return mutation_propensities


    def event_mutation(self, t, variables):
        # print("check event - mutation @ t =", t) 
        cumulative_mutation_propensity = variables[-1]
        # print(t, "EVENT - mutation:", 
        #     self.threshold_mutation_propensity, "-",
        #     cumulative_mutation_propensity, ":",  
        #     "return", self.threshold_mutation_propensity - cumulative_mutation_propensity)
        # print(f"check event - mutation @ t =  {t}, return {self.threshold_mutation_propensity - cumulative_mutation_propensity}")
        return self.threshold_mutation_propensity - cumulative_mutation_propensity
    #----------------------------------
    event_mutation.terminal  = True
    event_mutation.direction = -1


    def handle_mutation_event(self):
        # print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
        _N_t_event = self.N_series[:,-1].ravel()
        _R_t_event = self.R_series[:,-1].ravel()
        # Pick the mutant that will be established with proababilities proportional to mutants' propensities for establishment:
        mutation_propensities = self.mutation_propensities(_N_t_event, _R_t_event)
        # print("\nmutation_propensities", mutation_propensities, mutation_propensities.shape)
        mutant_drawprobs      = mutation_propensities/np.sum(mutation_propensities)
        # print("mutant_drawprobs", mutant_drawprobs, mutant_drawprobs.shape)
        mutant_idx            = np.random.choice(range(self.mutant_pool.num_mutant_types), p=mutant_drawprobs)
        # print("mutant_idx", mutant_idx)
        # Retrieve the mutant and some of its properties:
        mutant           = self.mutant_pool.get_type(mutant_idx)
        mutant_type_id   = self.mutant_pool.type_ids[mutant_idx]
        mutant_fitness   = self.mutant_pool.fitness[mutant_idx]
        mutant_abundance = np.maximum(1/mutant_fitness, 1) # forcing abundance of new types to be at least 1, this is a Ryan addition (perhaps controversial)
        # Get the index of the parent of the selected mutant:
        parent_idx     = mutant_idx // self.strain_pool.num_traits
        
        # for m in range(self.mutant_pool.sigma.shape[0]):
        #     p = m // self.strain_pool.num_traits
        #     print(p, self.strain_pool.sigma[p], '|', m, self.mutant_pool.sigma[m], self.mutant_pool.fitness[m], mutant_drawprobs[m])
        # print('pick vvvvv')
        # print("mutant_idx", mutant_idx, self.mutant_pool.sigma[mutant_idx])
        # print("parent_idx", parent_idx, self.strain_pool.sigma[parent_idx])
        # print("parent_lin", self.strain_pool.lineage_ids[parent_idx])
        # print('mutant', mutant_type_id, mutant_fitness)
        #----------------------------------
        if(mutant_type_id in self.strain_pool.type_ids):
            # print("[[[[[ pre-existing ]]]]]")
            # This "mutant" is a pre-existing type in the population; get its index:
            preexisting_type_idx = np.where(np.array(self.strain_pool.type_ids) == mutant_type_id)[0][0]
            # Add abundance equal to the mutant's establishment abundance to the pre-existing type:
            self.set_type_abundance(type_index=preexisting_type_idx, abundance=self.get_type_abundance(preexisting_type_idx)+mutant_abundance)
            # Remove corresonding abundance from the parent type (abundance is moved from parent to mutant):
            self.set_type_abundance(type_index=parent_idx, abundance=self.get_type_abundance(parent_idx)-mutant_abundance)
        else:
            # Add the mutant to the population at an establishment abundance equal to 1/dfitness:
            # (strain pool costs, phylogeny, and mutant pool get updated as result of add_type())
            self.add_type(mutant, abundance=mutant_abundance, index=parent_idx+1, parent_index=parent_idx)  
            # Remove corresonding abundance from the parent type (abundance is moved from parent to mutant):
            self.set_type_abundance(type_index=parent_idx, abundance=self.get_type_abundance(parent_idx)-mutant_abundance)
        # print(self.strain_pool.lineage_ids)
        # print("mutant_pool size", self.mutant_pool.num_types)





    











    def event_type_loss(self, t, variables):
        # print("check event - type loss @ t =", t) 
        _N_t = variables[:self.strain_pool.num_types]
        _R_t = variables[-1-self.resource_set.num_resources:-1]
        _h_t = self.resource_consumption(_N_t, _R_t)
        #------------------------------
        # print(_N_t)
        min_abundance_typeidx = np.argmin(_N_t[np.nonzero(_N_t)])
        min_abundance_abs     = _N_t[min_abundance_typeidx]
        min_abundance_rel     = _N_t[min_abundance_typeidx]/np.sum(_N_t)
        min_abundance_growthrate = self.growth_rate(_h_t)[min_abundance_typeidx]
        #------------------------------
        # print(f'min abundance: idx {min_abundance_typeidx}, abs {min_abundance_abs}, rel {min_abundance_rel} :: growth_rate {min_abundance_growthrate}')
        # print(f"check event - type loss @ t =  {t}, return {-1 if ((min_abundance_abs < self.threshold_min_abs_abundance) or (min_abundance_rel < self.threshold_min_rel_abundance and min_abundance_growthrate < 0)) else 1}")
        return -1 if ((min_abundance_abs < self.threshold_min_abs_abundance) or (min_abundance_rel < self.threshold_min_rel_abundance and min_abundance_growthrate < 0)) else 1



    #----------------------------------
    event_type_loss.terminal  = True


    def handle_type_loss(self):
        # print('handle_type_loss', self.t)
        _N_t_event = self.N_series[:,-1].ravel()
        _R_t_event = self.R_series[:,-1].ravel()
        _h_t_event = self.resource_consumption(_N_t_event, _R_t_event)

        # print(_N_t_event)
        # print(_N_t_event/np.sum(_N_t_event))

        # print(np.where(_N_t_event < self.threshold_min_abs_abundance))
        # print(np.where(_N_t_event/np.sum(_N_t_event) < self.threshold_min_rel_abundance))
        # print(np.where(self.growth_rate(_h_t_event) < 0))
        # print( np.where( ((_N_t_event < self.threshold_min_abs_abundance) 
                        # | (_N_t_event/np.sum(_N_t_event) < self.threshold_min_rel_abundance)) 
                        # & (self.growth_rate(_h_t_event) < 0) ) )

        gg = self.growth_rate(_h_t_event)

        lost_types = np.where( (_N_t_event > 0)
                                & ( 
                                    (_N_t_event < self.threshold_min_abs_abundance)
                                    | ((_N_t_event/np.sum(_N_t_event) < self.threshold_min_rel_abundance) & (gg < 0))
                                ) )[0]
        # print(lost_types)
        for i in lost_types:
            # Set the abundance of lost types to 0 (but do not remove from system/strain pool data):
            # print("lost", i, self.strain_pool.type_ids[i], gg[i])
            self.set_type_abundance(type_index=i, abundance=0.0)

        # print(self.N_series)

        # exit()
        return









    # def event_equilibrium(self, t, variables):
    #     print("check event - equilibrium @ t =", t) 
    #     # print(t, "EVENT - event_equilibrium:", 
    #         # self.latest_total_abundance_change, 
    #         # self.threshold_eq_abundance_change, 
    #         # "return", self.latest_total_abundance_change - self.threshold_eq_abundance_change)
    #     return self.latest_total_abundance_change - self.threshold_eq_abundance_change
    # #----------------------------------
    # event_equilibrium.terminal  = True






    def add_type(self, strain_pool=None, abundance=0, sigma=None, b=None, k=None, eta=None, l=None, g=None, c=None, chi=None, mu=None, index=None, parent_index=None, parent_id=None):
        abundance      = utils.treat_as_list(abundance)
        new_type_idx   = index if index is not None else self.strain_pool.num_types # default to adding to end of matrices
        orig_num_types = self.strain_pool.num_types
        #----------------------------------
        self.strain_pool.add_type(strain_pool, sigma, b, k, eta, l, g, c, chi, mu, new_type_idx, parent_index, parent_id)
        num_new_types = self.strain_pool.num_types - orig_num_types
        #----------------------------------
        self.N_series = np.insert(self.N_series, new_type_idx, np.zeros(shape=(num_new_types, self.N_series.shape[1])), axis=0)
        self.set_type_abundance(type_index=list(range(new_type_idx, new_type_idx+num_new_types)), abundance=abundance)
        #----------------------------------
        self.generate_mutant_pool()


    def remove_type(self, type_id=None, t_index=-1, remove_data_series=False):
        # type_indices = [ np.where(self.strain_pool.type_ids==tid)[0] for tid in utils.treat_as_list(type_id) ] if type_id is not None else utils.treat_as_list(type_index)
        # #----------------------------------
        # self.set_type_abundance(type_index=list(range(new_type_idx, new_type_idx+num_new_types)), abundance=abundance)
        # TODO (also TODO for StrainPool class)
        return


    def set_type_abundance(self, abundance, type_index=None, type_id=None, t_index=-1):
        abundance    = utils.treat_as_list(abundance)
        type_indices = [ np.where(self.strain_pool.type_ids==tid)[0] for tid in utils.treat_as_list(type_id) ] if type_id is not None else utils.treat_as_list(type_index)
        #----------------------------------
        for i, type_idx in enumerate(type_indices):
            self.N_series[type_idx, t_index] = abundance[i]


    def get_type_abundance(self, type_index=None, type_id=None, t_index=-1):
        type_indices = [ np.where(self.strain_pool.type_ids==tid)[0] for tid in utils.treat_as_list(type_id) ] if type_id is not None else utils.treat_as_list(type_index)
        #----------------------------------
        abundances = []
        for i, type_idx in enumerate(type_indices):
            abundances.append(self.N_series[type_idx, t_index])
        return abundances if len(type_indices) > 1 else abundances[0]


    def get_extant_types(self):
        return np.where(self.N_series[:,-1] > 0)[0]


    def get_extant_strain_pool(self, strain_pool=None):
        strain_pool = self.strain_pool if strain_pool is None else strain_pool
        return strain_pool.get_type(self.get_extant_types())


    def generate_mutant_pool(self, parent_pool=None):
        parent_pool = self.strain_pool if parent_pool is None else parent_pool
        #----------------------------------
        self.mutant_pool = self.strain_pool.generate_mutant_pool()
        #----------------------------------
        # In the case of 'fast resource equilibriation' resource consumption dyanmics,
        # growth rates depend on the abundances of types in the population, and thus
        # calculating mutant fitnesses requires abundance/parameter information about 
        # parent strains to be included when calculating mutant growth rates.
        if(self.resource_consumption_mode=='fast_resource_eq'):
            self.mutant_pool.add_type(self.strain_pool)
        # Store the number of actual mutant types in this pool so we can later reference values for only mutant types
        self.mutant_pool.num_mutant_types = self.mutant_pool.num_types if self.resource_consumption_mode != 'fast_resource_eq' else self.mutant_pool.num_types - parent_pool.num_types
        #----------------------------------
        return self.mutant_pool


    def dimensions(self):
        return (self.strain_pool.num_types, self.resource_set.num_resources)










