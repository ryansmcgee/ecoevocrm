# This is to sidestep a numpy overhead introduced in numpy 1.17:
# https://stackoverflux.com/questions/61983372/is-built-in-method-numpy-core-multiarray-umath-implement-array-function-a-per
import os

os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
#----------------------------------
import scipy.integrate

from ecoevocrm.type_set import *
from ecoevocrm.resource_set import *
import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ConsumerResourceSystem():

    # Define Class constants:
    RESOURCE_DYNAMICS_FASTEQ = 0
    RESOURCE_DYNAMICS_EXPLICIT = 1
    RESOURCE_CROSSFEEDING_NONE = 0
    RESOURCE_CROSSFEEDING_HOMOTYPES = 1
    RESOURCE_CROSSFEEDING_HETEROTYPES = 2

    def __init__(self,
                 type_set                    = None,
                 resource_set                = None,
                 N_init                      = None,
                 R_init                      = None,
                 num_types                   = None,
                 num_resources               = None,
                 # TypeSet params:
                 traits                      = None,  # previously sigma
                 consumption_rate            = 1,     # previously beta
                 carrying_capacity           = 1e9,   # previously kappa
                 growth_factor               = 1,     # previously gamma
                 energy_passthru             = 0,     # previously lamda
                 cost_baseline               = 0,     # previously xi
                 cost_trait                  = 0,     # previously chi
                 cost_interaction            = None,  # previously J
                 cost_landscape              = None,
                 mutation_rate               = 1e-9,  # previously mu
                 segregation_rate            = 0,
                 transfer_rate_donor         = 0,
                 transfer_rate_recip         = 0,
                 mutant_attributes           = None,
                 segregant_attributes        = None,
                 transconjugant_attributes   = None,
                 segregation_linkage         = None,
                 transfer_linkage            = None,
                 # ResourceSet params:
                 influx_rate                 = 1,     # previously rho
                 decay_rate                  = 1,     # previously tau
                 energy_content              = 1,     # previously omega
                 cross_production            = None,  # previously D
                 # Simulation options:
                 resource_dynamics_mode      = 'explicit',
                 threshold_min_abs_abundance = 1,
                 convergent_lineages         = False,
                 seed                        = None):

        #----------------------------------

        if(seed is not None):
            np.random.seed(seed)
            self.seed = seed

        #----------------------------------
        # Determine the dimensions of the system:
        #----------------------------------
        system_num_types     = None
        system_num_resources = None
        if(type_set is not None):
            system_num_types = type_set.num_types
            system_num_resources = type_set.num_traits
        elif(isinstance(traits, (list, np.ndarray))):
            traits = np.array(traits)
            if(traits.ndim == 2):
                system_num_types = traits.shape[0]
                system_num_resources = traits.shape[1]
        else:
            if(num_types is not None):
                system_num_types = num_types
            elif(isinstance(N_init, (list, np.ndarray))):
                system_num_types = np.array(N_init).ravel().shape[0]
            else:
                utils.error("Error in ConsumerResourceSystem __init__(): Number of types must be specified by providing a) a type set, b) a traits matrix, c) a num_types value, or d) a list for N_init.")
            # ---
            if(resource_set is not None):
                system_num_resources = resource_set.num_resources
            elif(num_resources is not None):
                system_num_resources = num_resources
            elif(isinstance(R_init, (list, np.ndarray))):
                system_num_resources = np.array(R_init).ravel().shape[0]
            else:
                utils.error("Error in ConsumerResourceSystem __init__(): Number of resources must be specified by providing a) a resource set, b) a traits matrix, c) a num_resources value, or d) a list for R_init.")

        #----------------------------------
        # Initialize type set parameters:
        #----------------------------------
        if(type_set is not None):
            if(isinstance(type_set, TypeSet)):
                self.type_set = type_set
            else:
                utils.error(f"Error in ConsumerResourceSystem __init__(): type_set argument expects object of TypeSet type.")
        else:
            self.type_set = TypeSet(num_types=system_num_types, num_traits=system_num_resources, traits=traits,
                                    consumption_rate=consumption_rate, carrying_capacity=carrying_capacity, energy_passthru=energy_passthru, growth_factor=growth_factor,
                                    cost_baseline=cost_baseline, cost_trait=cost_trait, cost_interaction=cost_interaction, cost_landscape=cost_landscape,
                                    mutation_rate=mutation_rate, segregation_rate=segregation_rate, transfer_rate_donor=transfer_rate_donor, transfer_rate_recip=transfer_rate_recip,
                                    mutant_attributes=mutant_attributes, segregant_attributes=segregant_attributes, transconjugant_attributes=transconjugant_attributes,
                                    segregation_linkage=segregation_linkage, transfer_linkage=transfer_linkage)
        # Check that the type set dimensions match the system dimensions:
        if(system_num_types != self.type_set.num_types):
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system types ({system_num_types}) does not match number of type set types ({self.type_set.num_types}).")
        if(system_num_resources != self.type_set.num_traits):
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system resources ({system_num_resources}) does not match number of type set traits ({self.type_set.num_traits}).")

        # Reference the TypeSet's lineageIDs property to induce assignment of a non-None lineage_id attribute 
        # (waiting to induce assignment of lineage ids until after a large sim can cause a RecursionError):
        self.type_set.lineageIDs

        #----------------------------------
        # Initialize resource set parameters:
        #----------------------------------
        if(resource_set is not None):
            if(isinstance(resource_set, ResourceSet)):
                self.resource_set = resource_set
            else:
                utils.error(f"Error in ConsumerResourceSystem __init__(): resource_set argument expects object of ResourceSet type.")
        else:
            self.resource_set = ResourceSet(num_resources=system_num_resources, influx_rate=influx_rate, decay_rate=decay_rate, energy_content=energy_content, cross_production=cross_production)
        # Check that the type set dimensions match the system dimensions:
        if(system_num_resources != self.resource_set.num_resources):
            utils.error(f"Error in ConsumerResourceSystem __init__(): Number of system resources ({system_num_resources}) does not match number of resource set resources ({self.resource_set.num_resources}).")

        #----------------------------------
        # Initialize system variables:
        #----------------------------------
        if(N_init is None or R_init is None):
            utils.error(f"Error in ConsumerResourceSystem __init__(): Values for N_init and R_init must be provided.")
        self._N_series = utils.ExpandableArray(utils.reshape(N_init, shape=(system_num_types, 1)), alloc_shape=(max(self.resource_set.num_resources * 25, system_num_types), 1))
        self._R_series = utils.ExpandableArray(utils.reshape(R_init, shape=(system_num_resources, 1)), alloc_shape=(self.resource_set.num_resources, 1))

        #----------------------------------
        # Initialize system time:
        #----------------------------------
        self._t_series     = utils.ExpandableArray([0], alloc_shape=(1, 1))

        #----------------------------------
        # Initialize simulation parameters:
        #----------------------------------
        self.threshold_event_propensity  = None  # is updated in run()  # formerly threshold_mutation_propensity
        self.threshold_min_abs_abundance = threshold_min_abs_abundance
        # self.check_event_low_abundance   = check_event_low_abundance
        self.convergent_lineages         = convergent_lineages

        self.resource_dynamics_mode = ConsumerResourceSystem.RESOURCE_DYNAMICS_FASTEQ if resource_dynamics_mode == 'fasteq' \
            else ConsumerResourceSystem.RESOURCE_DYNAMICS_EXPLICIT if resource_dynamics_mode == 'explicit' \
            else -1

        self.resource_crossfeeding_mode = ConsumerResourceSystem.RESOURCE_CROSSFEEDING_NONE if np.all(
            self.resource_set.cross_production == 0) or np.all(self.type_set.energy_passthru == 0) \
            else ConsumerResourceSystem.RESOURCE_CROSSFEEDING_HOMOTYPES if self.type_set.energy_passthru.ndim == 1 \
            else ConsumerResourceSystem.RESOURCE_CROSSFEEDING_HETEROTYPES if self.type_set.energy_passthru.ndim == 2 \
            else -1

        #----------------------------------
        # Initialize sets of newly generated types:
        #----------------------------------
        self.mutant_set         = self.type_set.generate_mutant_set()
        self.segregant_set      = self.type_set.generate_segregant_set()
        self.transconjugant_set = self.type_set.generate_transconjugant_set()

        self.mutation_propensities    = None
        self.segregation_propensities = None
        self.transfer_propensities    = None

        #""""""""""""""""""""""""""""""""""

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

    @property
    def extant_type_set(self):
        return self.get_extant_type_set()

    @property
    def abundance(self):
        return self.N

    @property
    def rel_abundance(self):
        return self.N / np.sum(self.N)

    @property
    def biomass(self):
        return np.sum(self.N)

    @property
    def fitness(self):
        return self.growth_rate(self.N, self.R, self.t, self.type_set.traits, self.type_set.consumption_rate, self.type_set.carrying_capacity, self.type_set.energy_passthru, self.type_set.growth_factor, self.type_set.energy_costs,
                                self.resource_set.influx_rate, self.resource_set.decay_rate, self.resource_set.energy_content, self.resource_dynamics_mode, self.resource_set.resource_influx_mode)

    @property
    def num_types(self):
        return self.type_set.num_types

    @property
    def num_resources(self):
        return self.resource_set.num_resources

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def run(self, T, dt=None, reorder_types_by_phylogeny=True, integration_method='default', max_time_step=np.inf):

        t_start   = self.t
        t_end     = self.t + T
        t_elapsed = 0

        self._t_series.expand_alloc((self._t_series.alloc[0], self._t_series.alloc[1] + int(T / dt if dt is not None else 10000)))
        self._N_series.expand_alloc((self._N_series.alloc[0], self._N_series.alloc[1] + int(T / dt if dt is not None else 10000)))
        self._R_series.expand_alloc((self._R_series.alloc[0], self._R_series.alloc[1] + int(T / dt if dt is not None else 10000)))

        while (t_elapsed < T):

            #------------------------------
            # Set initial conditions and integration variables:
            #------------------------------

            # Set the time interval for this integration epoch:
            t_span = (self.t, t_end)

            # Set the time ticks at which to save trajectory values:
            t_eval = np.arange(start=t_span[0], stop=t_span[1]+dt, step=dt) if dt is not None else None

            # Get the indices and count of extant types (abundance > 0):
            self._active_type_indices = self.extant_type_indices
            num_extant_types = len(self._active_type_indices)

            # Set the initial conditions for this integration epoch:
            N_init = self.N[self._active_type_indices]
            R_init = self.R
            cumPropMut_init = np.array([0])
            init_cond = np.concatenate([N_init, R_init, cumPropMut_init])

            # Get the params for the dynamics:
            params = self.get_dynamics_params(type_index=self._active_type_indices)

            # Draw a random propensity threshold for triggering the next Gillespie event:
            self.threshold_event_propensity = np.random.exponential(1)

            # Set the integration method:
            _integration_method = 'LSODA' if(integration_method == 'default') else integration_method

            # Define the set of events that may trigger:
            events = []
            if(np.any(self.type_set.mutation_rate > 0) or np.any(self.type_set.segregation_rate > 0) or np.any(self.type_set.transfer_rate_donor > 0)):
                events.append(self.event_lineage_establishment)
            # if(self.check_event_low_abundance):
            #     events.append(self.event_low_abundance)

            #------------------------------
            # Integrate the system dynamics:
            #------------------------------

            sol = scipy.integrate.solve_ivp(self.dynamics,
                                            y0=init_cond,
                                            args=params,
                                            t_span=t_span,
                                            t_eval=t_eval,
                                            events=events,
                                            method=_integration_method,
                                            max_step=max_time_step)

            #------------------------------
            # Update the system's trajectories with latest dynamics epoch:
            #------------------------------

            N_epoch = np.zeros(shape=(self._N_series.shape[0], len(sol.t)))
            N_epoch[self._active_type_indices] = sol.y[:num_extant_types]
            N_epoch[N_epoch < self.threshold_min_abs_abundance] = 0  # clip abundances below threshold_min_abs_abundance to 0

            R_epoch = sol.y[-1 - self.resource_set.num_resources:-1]

            self._t_series.add(sol.t[1:], axis=1)
            self._N_series.add(N_epoch[:, 1:], axis=1)
            self._R_series.add(R_epoch[:, 1:], axis=1)

            t_elapsed = self.t - t_start

            #------------------------------
            # Handle events and update the system's states accordingly:
            #------------------------------
            if(sol.status == 1):  #-> An event occurred
                if(len(sol.t_events[0]) > 0):
                    self.handle_lineage_establishment()
            elif(sol.status == 0):  #-> Reached end T successfully
                pass
            else:  #-> Error occurred in integration
                utils.error("Error in ConsumerResourceSystem run(): Integration of dynamics using scipy.solve_ivp returned with error status.")

        #------------------------------
        # Finalize data series at end of integration period:
        #------------------------------

        self._t_series.trim()
        self._N_series.trim()
        self._R_series.trim()

        if(reorder_types_by_phylogeny):
            self.reorder_types()

        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def dynamics(self, t, variables,
                 num_types, traits, consumption_rate, carrying_capacity, energy_passthru, growth_factor, energy_costs,
                 num_mutants, mutant_rates, mutant_parent_indices,
                 num_segregants, segregant_rates, segregant_parent_indices,
                 num_transconjugants, transconjugant_rates, transconjugant_donor_indices, transconjugant_recip_indices,
                 num_resources, influx_rate, decay_rate, energy_content, cross_production_energy,
                 uptake_coeffs, consumption_coeffs, resource_dynamics_mode, resource_influx_mode, resource_crossfeeding_mode):

        N_t = np.zeros(num_types + num_mutants + num_segregants + num_transconjugants)
        N_t[:num_types] = variables[:num_types]
        R_t = variables[-1 - num_resources:-1]

        #------------------------------

        # TODO: Need to add segregation to dNdt

        growth_rate = ConsumerResourceSystem.growth_rate(N_t, R_t, t, traits, consumption_rate, carrying_capacity, energy_passthru, growth_factor, energy_costs,
                                                         influx_rate, decay_rate, energy_content, resource_dynamics_mode, resource_influx_mode, uptake_coeffs, consumption_coeffs)

        dNdt = N_t[:num_types] * growth_rate[:num_types]  # only compute dNdt for extant types

        dRdt = ConsumerResourceSystem.resource_change(N_t, R_t, t, traits, consumption_rate, carrying_capacity, energy_passthru,
                                                      influx_rate, decay_rate, cross_production_energy, resource_dynamics_mode, resource_influx_mode, resource_crossfeeding_mode,
                                                      consumption_coeffs)

        #------------------------------

        mean_fitness     = np.mean(growth_rate[:num_types])
        dTotalPropensity = 0

        if(num_mutants > 0):
            self.mutant_fitnesses = growth_rate[num_types : num_types+num_mutants]
            self.mutant_selcoeffs = self.mutant_fitnesses - mean_fitness
            self.mutant_selcoeffs[self.mutant_selcoeffs < 0] = 0
            # - - - -
            self.mutation_propensities = N_t[mutant_parent_indices] * mutant_rates * self.mutant_fitnesses * self.mutant_selcoeffs # TODO: Should the final term corresponding to prob of establishment be (mutant fitness - mean fitness of pop) or (mutant_fitness - fitness of parent?)
            self.mutation_propensities[self.mutation_propensities < 0] = 0  # negative propensities due to negative growthrate or selcoeff are zeroed out
            # - - - -
            dTotalPropensity += np.sum(self.mutation_propensities)

        if(num_segregants > 0):
            self.segregant_fitnesses = growth_rate[num_types+num_mutants : num_types+num_mutants+num_segregants]
            self.segregant_selcoeffs = self.segregant_fitnesses - mean_fitness
            self.segregant_selcoeffs[self.segregant_selcoeffs < 0] = 0
            # - - - -
            self.segregation_propensities = N_t[segregant_parent_indices] * segregant_rates * self.segregant_fitnesses * self.segregant_selcoeffs
            self.segregation_propensities[self.segregation_propensities < 0] = 0  # negative propensities due to negative growthrate or selcoeff are zeroed out
            # - - - -
            dTotalPropensity += np.sum(self.segregation_propensities)

        if(num_transconjugants > 0):
            self.transconjugant_fitnesses = growth_rate[num_types+num_mutants+num_segregants : num_types+num_mutants+num_segregants+num_transconjugants]
            self.transconjugant_selcoeffs = self.transconjugant_fitnesses - mean_fitness
            self.transconjugant_selcoeffs[self.transconjugant_selcoeffs < 0] = 0
            # - - - -
            self.transfer_propensities = N_t[transconjugant_donor_indices] * N_t[transconjugant_recip_indices] * transconjugant_rates * self.transconjugant_fitnesses * self.transconjugant_selcoeffs
            self.transfer_propensities[self.transfer_propensities < 0] = 0  # negative propensities due to negative growthrate or selcoeff are zeroed out
            # - - - -
            dTotalPropensity += np.sum(self.transfer_propensities)

        dTotalPropensity = [dTotalPropensity]

        #------------------------------

        return np.concatenate((dNdt, dRdt, dTotalPropensity))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    # TODO: Allow for calculating for single N/R or time series of N/Rs
    def growth_rate(N, R, t, traits, consumption_rate, carrying_capacity, energy_passthru, growth_factor, energy_costs,
                    influx_rate, decay_rate, energy_content, resource_dynamics_mode, resource_influx_mode,
                    uptake_coeffs=None, consumption_coeffs=None):
        #------------------------------
        energy_uptake = ConsumerResourceSystem.energy_uptake(N, R, t, traits, consumption_rate, carrying_capacity, energy_passthru,
                                                             influx_rate, decay_rate, energy_content, resource_dynamics_mode, resource_influx_mode,
                                                             uptake_coeffs, consumption_coeffs)
        energy_surplus = energy_uptake - energy_costs
        growth_rate    = growth_factor * energy_surplus
        #------------------------------
        return growth_rate

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    # TODO: Allow for calculating for single N/R or time series of N/Rs
    def energy_uptake(N, R, t, traits, consumption_rate, carrying_capacity, energy_passthru,
                      influx_rate, decay_rate, energy_content, resource_dynamics_mode, resource_influx_mode,
                      uptake_coeffs=None, consumption_coeffs=None):
        #------------------------------
        consumption_rates_bytrait = None
        if(uptake_coeffs is None):
            consumption_rates_bytrait = np.einsum('ij,ij->ij', traits, consumption_rate) if consumption_rate.ndim == 2 else np.einsum('ij,j->ij', traits, consumption_rate)
            uptake_coeffs = consumption_rates_bytrait
            if(resource_dynamics_mode != ConsumerResourceSystem.RESOURCE_DYNAMICS_FASTEQ):
                if(np.any(energy_passthru != 0)):
                    uptake_coeffs = uptake_coeffs * (1 - energy_passthru)
                if(np.any(energy_content != 1)):
                    uptake_coeffs = uptake_coeffs * energy_content
        #------------------------------
        energy_uptake = None
        if(resource_dynamics_mode == ConsumerResourceSystem.RESOURCE_DYNAMICS_FASTEQ):
            if(consumption_rates_bytrait is None):
                consumption_rates_bytrait = np.einsum('ij,ij->ij', traits, consumption_rate) if consumption_rate.ndim == 2 else np.einsum('ij,j->ij', traits, consumption_rate)
            consumption_coeffs = consumption_rates_bytrait / carrying_capacity if consumption_coeffs is None else consumption_coeffs
            _influx_rate    = influx_rate(t).ravel() if resource_influx_mode == ResourceSet.RESOURCE_INFLUX_TEMPORAL else influx_rate
            resource_uptake = _influx_rate / (decay_rate + np.einsum('ij,i->j', consumption_coeffs, N))
            energy_uptake = np.einsum('ij,j->i', uptake_coeffs, resource_uptake)
        elif(resource_dynamics_mode == ConsumerResourceSystem.RESOURCE_DYNAMICS_EXPLICIT):
            energy_uptake = np.einsum('ij,j->i', uptake_coeffs, R)
        #------------------------------
        return energy_uptake

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def resource_change(N, R, t, traits, consumption_rate, carrying_capacity, energy_passthru,
                        influx_rate, decay_rate, cross_production_energy, resource_dynamics_mode, resource_influx_mode, resource_crossfeeding_mode,
                        consumption_coeffs=None):
        if(consumption_coeffs is None):
            consumption_rates_bytrait = np.einsum('ij,ij->ij', traits, consumption_rate) if consumption_rate.ndim == 2 else np.einsum('ij,j->ij', traits, consumption_rate)
            consumption_coeffs = consumption_rates_bytrait / carrying_capacity
        _influx_rate = influx_rate(t).ravel() if resource_influx_mode == ResourceSet.RESOURCE_INFLUX_TEMPORAL else influx_rate
        #------------------------------
        dRdt = None
        if(resource_dynamics_mode == ConsumerResourceSystem.RESOURCE_DYNAMICS_FASTEQ):
            dRdt = np.zeros(len(R))
        elif(resource_dynamics_mode == ConsumerResourceSystem.RESOURCE_DYNAMICS_EXPLICIT):
            if(resource_crossfeeding_mode == ConsumerResourceSystem.RESOURCE_CROSSFEEDING_NONE):
                resource_consumption_rate = np.einsum('ij,j->j', np.einsum('ij,i->ij', consumption_coeffs, N), R)
                dRdt = _influx_rate - decay_rate * R - resource_consumption_rate
            elif(resource_crossfeeding_mode == ConsumerResourceSystem.RESOURCE_CROSSFEEDING_HOMOTYPES):
                resource_consumption_rate = np.einsum('ij,j->j', np.einsum('ij,i->ij', consumption_coeffs, N), R)  # shape = (num_resources,)
                resource_leak_rate = energy_passthru * resource_consumption_rate
                resource_conversion_rate = np.einsum('ij,j->i', cross_production_energy, resource_leak_rate)
                dRdt = _influx_rate - decay_rate * R - resource_consumption_rate + resource_conversion_rate
            elif(resource_crossfeeding_mode == ConsumerResourceSystem.RESOURCE_CROSSFEEDING_HETEROTYPES):
                resource_consumption_terms = np.einsum('ij,j->ij', np.einsum('ij,i->ij', consumption_coeffs, N), R)  # gives shape = (num_types, num_resources)
                resource_consumption_rate = np.sum(resource_consumption_terms, axis=0)  # gives shape = (num_resources,)
                resource_leak_rate = np.einsum('ij,ij->j', energy_passthru, resource_consumption_terms)  # gives shape = (num_resources,)
                resource_conversion_rate = np.einsum('ij,j->i', cross_production_energy, resource_leak_rate)
                dRdt = _influx_rate - decay_rate * R - resource_consumption_rate + resource_conversion_rate
        #------------------------------
        return dRdt

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, type_index=None):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.num_types)
        #----------------------------------
        type_params = self.type_set.get_dynamics_params(type_idx)
        del type_params['creation_rate']
        del type_params['parent_indices']
        del type_params['donor_indices']
        del type_params['recip_indices']
        #----------------------------------
        if(self.mutant_set.num_types > 0):
            mutant_params = self.mutant_set.get_dynamics_params(self.type_set.get_mutant_indices(type_idx))
            type_params['traits']                = np.concatenate([type_params['traits'], mutant_params['traits']])
            type_params['consumption_rate']      = utils.SystemParameter.combine(type_params['consumption_rate'], mutant_params['consumption_rate']) # <--
            type_params['carrying_capacity']     = utils.SystemParameter.combine(type_params['carrying_capacity'], mutant_params['carrying_capacity']) # <--
            type_params['energy_passthru']       = utils.SystemParameter.combine(type_params['energy_passthru'], mutant_params['energy_passthru']) # <--
            type_params['growth_factor']         = utils.SystemParameter.combine(type_params['growth_factor'], mutant_params['growth_factor']) # <--
            type_params['energy_costs']          = np.concatenate([type_params['energy_costs'], mutant_params['energy_costs']])
            type_params['num_mutants']           = mutant_params['num_types']
            type_params['mutant_rates']          = mutant_params['creation_rate']
            type_params['mutant_parent_indices'] = [utils.find_first(pidx, type_idx) for pidx in mutant_params['parent_indices']] # old: = mutant_params['parent_indices']
        else:
            type_params['num_mutants']           = 0
            type_params['mutant_rates']          = 0
            type_params['mutant_parent_indices'] = 0
        #----------------------------------
        if(self.segregant_set.num_types > 0):
            segregant_params = self.segregant_set.get_dynamics_params(self.type_set.get_segregant_indices(type_idx))
            type_params['traits']                   = np.concatenate([type_params['traits'], segregant_params['traits']])
            type_params['consumption_rate']         = utils.SystemParameter.combine(type_params['consumption_rate'], segregant_params['consumption_rate']) # <--
            type_params['carrying_capacity']        = utils.SystemParameter.combine(type_params['carrying_capacity'], segregant_params['carrying_capacity']) # <--
            type_params['energy_passthru']          = utils.SystemParameter.combine(type_params['energy_passthru'], segregant_params['energy_passthru']) # <--
            type_params['growth_factor']            = utils.SystemParameter.combine(type_params['growth_factor'], segregant_params['growth_factor']) # <--
            type_params['energy_costs']             = np.concatenate([type_params['energy_costs'], segregant_params['energy_costs']])
            type_params['num_segregants']           = segregant_params['num_types']
            type_params['segregant_rates']          = segregant_params['creation_rate']
            type_params['segregant_parent_indices'] = [utils.find_first(pidx, type_idx) for pidx in segregant_params['parent_indices']] # old: = segregant_params['parent_indices']
        else:
            type_params['num_segregants']           = 0
            type_params['segregant_rates']          = 0
            type_params['segregant_parent_indices'] = 0
        #----------------------------------
        if(self.transconjugant_set.num_types > 0):
            transconjugant_params = self.transconjugant_set.get_dynamics_params(self.type_set.get_transconjugant_indices(type_idx))
            type_params['traits']                       = np.concatenate([type_params['traits'], transconjugant_params['traits']])
            type_params['consumption_rate']             = utils.SystemParameter.combine(type_params['consumption_rate'], transconjugant_params['consumption_rate'])
            type_params['carrying_capacity']            = utils.SystemParameter.combine(type_params['carrying_capacity'], transconjugant_params['carrying_capacity'])
            type_params['energy_passthru']              = utils.SystemParameter.combine(type_params['energy_passthru'], transconjugant_params['energy_passthru'])
            type_params['growth_factor']                = utils.SystemParameter.combine(type_params['growth_factor'], transconjugant_params['growth_factor'])
            type_params['energy_costs']                 = np.concatenate([type_params['energy_costs'], transconjugant_params['energy_costs']])
            type_params['num_transconjugants']          = transconjugant_params['num_types']
            type_params['transconjugant_rates']         = transconjugant_params['creation_rate']
            type_params['transconjugant_donor_indices'] = [utils.find_first(didx, type_idx) for didx in transconjugant_params['donor_indices']]
            type_params['transconjugant_recip_indices'] = [utils.find_first(ridx, type_idx) for ridx in transconjugant_params['recip_indices']]
        else:
            type_params['num_transconjugants']          = 0
            type_params['transconjugant_rates']         = 0
            type_params['transconjugant_donor_indices'] = 0
            type_params['transconjugant_recip_indices'] = 0
        #----------------------------------
        type_params['consumption_rate']  = type_params['consumption_rate'].values()
        type_params['carrying_capacity'] = type_params['carrying_capacity'].values()
        type_params['energy_passthru']   = type_params['energy_passthru'].values()
        type_params['growth_factor']     = type_params['growth_factor'].values()
        #----------------------------------
        resource_params = self.resource_set.get_dynamics_params()
        #----------------------------------
        consumption_rates_bytrait = np.einsum('ij,ij->ij', type_params['traits'], type_params['consumption_rate']) if type_params['consumption_rate'].ndim == 2 else np.einsum('ij,j->ij', type_params['traits'], type_params['consumption_rate'])
        # ------------------
        uptake_coeffs = consumption_rates_bytrait
        if(self.resource_dynamics_mode != ConsumerResourceSystem.RESOURCE_DYNAMICS_FASTEQ):
            if(np.any(type_params['energy_passthru'] != 0)):
                uptake_coeffs = uptake_coeffs * (1 - type_params['energy_passthru'])
            if(np.any(resource_params['energy_content'] != 1)):
                uptake_coeffs = uptake_coeffs * resource_params['energy_content']
        # ------------------
        consumption_coeffs = consumption_rates_bytrait / type_params['carrying_capacity']
        #----------------------------------
        return (  tuple(type_params.values())
                + tuple(resource_params.values())
                + (uptake_coeffs, consumption_coeffs)
                + (self.resource_dynamics_mode, self.resource_set.resource_influx_mode, self.resource_crossfeeding_mode) )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def event_lineage_establishment(self, t, variables, *args):
        cumulative_mutation_propensity = variables[-1]
        return self.threshold_event_propensity - cumulative_mutation_propensity
    #----------------------------------
    event_lineage_establishment.direction = -1
    event_lineage_establishment.terminal = True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def handle_lineage_establishment(self):
        total_mutation_propensity    = np.sum(self.mutation_propensities) if self.mutation_propensities is not None else 0
        total_segregation_propensity = np.sum(self.segregation_propensities) if self.segregation_propensities is not None else 0
        total_transfer_propensity    = np.sum(self.transfer_propensities) if self.transfer_propensities is not None else 0
        total_establishment_propensity = total_mutation_propensity + total_segregation_propensity + total_transfer_propensity
        #----------------------------------
        r = np.random.uniform(0, total_establishment_propensity)
        if(r < total_mutation_propensity and total_mutation_propensity > 0):
            self.handle_mutant_establishment()
        elif(r < total_mutation_propensity + total_segregation_propensity and total_segregation_propensity > 0):
            self.handle_segregant_establishment()
        elif(total_transfer_propensity > 0):
            self.handle_transconjugant_establishment()
        else:
            pass

    def handle_mutant_establishment(self):
        # Pick the mutant that will be established with proababilities proportional to mutants' propensities for establishment:
        mutant_indices = self.type_set.get_mutant_indices(self._active_type_indices)
        mutant_drawrates = self.mutation_propensities / np.sum(self.mutation_propensities)
        mutant_idx = np.random.choice(mutant_indices, p=mutant_drawrates)
        # Retrieve the mutant and some of its properties:
        mutant = self.mutant_set.get_type(mutant_idx)
        mutant_type_id = mutant.typeIDs[0]
        mutant_fitness = self.mutant_fitnesses[np.argmax(mutant_indices == mutant_idx)]
        mutant_abundance = np.maximum(1 / mutant_fitness, 1)
        #----------------------------------
        # Get the index of the parent of the selected mutant:
        parent_idx = mutant.parent_indices[0]
        #----------------------------------
        if(self.convergent_lineages and mutant_type_id in self.type_set.typeIDs):
            # This "mutant" is a pre-existing type in the population; get its index:
            preexisting_type_idx = np.where(np.array(self.type_set.typeIDs) == mutant_type_id)[0][0]
            # Add abundance equal to the mutant's establishment abundance to the pre-existing type:
            self.set_type_abundance(type_index=preexisting_type_idx, abundance=self.get_type_abundance(preexisting_type_idx) + mutant_abundance)
            # Remove corresonding abundance from the parent type (abundance is moved from parent to mutant):
            self.set_type_abundance(type_index=parent_idx, abundance=max(self.get_type_abundance(parent_idx) - mutant_abundance, 1))
        else:
            # Add the mutant to the population at an establishment abundance equal to 1/dfitness:
            self.add_type(mutant, abundance=mutant_abundance)#, parent_index=parent_idx)
            # Remove corresonding abundance from the parent type (abundance is moved from parent to mutant):
            self.set_type_abundance(type_index=parent_idx, abundance=max(self.get_type_abundance(parent_idx) - mutant_abundance, 1))
        #----------------------------------
        print(f"[ Mutation event occurred at t={self.t:.2f} ]\t{self.type_set.trait_keys[parent_idx]} --> {mutant.trait_keys[0]}\n", end="")
        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def handle_segregant_establishment(self):
        # Pick the segregant that will be established with proababilities proportional to segregants' propensities for establishment:
        segregant_indices   = self.type_set.get_segregant_indices(self._active_type_indices)
        segregant_drawrates = self.segregation_propensities / np.sum(self.segregation_propensities)
        segregant_idx       = np.random.choice(segregant_indices, p=segregant_drawrates)
        # Retrieve the segregant and some of its properties:
        segregant           = self.segregant_set.get_type(segregant_idx)
        segregant_type_id   = segregant.typeIDs[0]
        segregant_fitness   = self.segregant_fitnesses[np.argmax(segregant_indices == segregant_idx)]
        segregant_abundance = np.maximum(1 / segregant_fitness, 1)
        #----------------------------------
        # Get the index of the parent of the selected segregant:
        parent_idx = segregant.parent_indices[0]
        #----------------------------------
        if(self.convergent_lineages and segregant_type_id in self.type_set.typeIDs):
            # This "segregant" is a pre-existing type in the population; get its index:
            preexisting_type_idx = np.where(np.array(self.type_set.typeIDs) == segregant_type_id)[0][0]
            # Add abundance equal to the segregant's establishment abundance to the pre-existing type:
            self.set_type_abundance(type_index=preexisting_type_idx, abundance=self.get_type_abundance(preexisting_type_idx) + segregant_abundance)
            # Remove corresonding abundance from the parent type (abundance is moved from parent to segregant):
            self.set_type_abundance(type_index=parent_idx, abundance=max(self.get_type_abundance(parent_idx) - segregant_abundance, 1))
        else:
            # Add the segregant to the population at an establishment abundance equal to 1/dfitness:
            self.add_type(segregant, abundance=segregant_abundance)#, parent_index=parent_idx)
            # Remove corresonding abundance from the parent type (abundance is moved from parent to segregant):
            self.set_type_abundance(type_index=parent_idx, abundance=max(self.get_type_abundance(parent_idx) - segregant_abundance, 1))
        #----------------------------------
        print(f"[ Segregation event occurred at t={self.t:.2f} ]\t{self.type_set.trait_keys[parent_idx]} --> {segregant.trait_keys[0]}\n", end="")
        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def handle_transconjugant_establishment(self):
        # Pick the transconjugant that will be established with proababilities proportional to transconjugants' propensities for establishment:
        transconjugant_indices   = self.type_set.get_transconjugant_indices(self._active_type_indices)
        transconjugant_drawrates = self.transfer_propensities / np.sum(self.transfer_propensities)
        transconjugant_idx       = np.random.choice(transconjugant_indices, p=transconjugant_drawrates)
        # Retrieve the transconjugant and some of its properties:
        transconjugant           = self.transconjugant_set.get_type(transconjugant_idx)
        transconjugant_type_id   = transconjugant.typeIDs[0]
        transconjugant_fitness   = self.transconjugant_fitnesses[np.argmax(transconjugant_indices == transconjugant_idx)]
        transconjugant_abundance = np.maximum(1 / transconjugant_fitness, 1)
        #----------------------------------
        # Get the index of the donor and recipient types associated with the selected transconjugant:
        donor_idx = transconjugant.donor_indices[0]
        recip_idx = transconjugant.recip_indices[0]
        # recip
        #----------------------------------
        if(self.convergent_lineages and transconjugant_type_id in self.type_set.typeIDs):
            # This "transconjugant" is a pre-existing type in the population; get its index:
            preexisting_type_idx = np.where(np.array(self.type_set.typeIDs) == transconjugant_type_id)[0][0]
            # Add abundance equal to the transconjugant's establishment abundance to the pre-existing type:
            self.set_type_abundance(type_index=preexisting_type_idx, abundance=self.get_type_abundance(preexisting_type_idx) + transconjugant_abundance)
            # Remove corresonding abundance from the recipient type (abundance is moved from recipient to transconjugant):
            self.set_type_abundance(type_index=recip_idx, abundance=max(self.get_type_abundance(recip_idx) - transconjugant_abundance, 1))
        else:
            # Add the transconjugant to the population at an establishment abundance equal to 1/dfitness:
            self.add_type(transconjugant, abundance=transconjugant_abundance)#, recip_index=recip_idx)
            # Remove corresonding abundance from the recipient type (abundance is moved from recipient to transconjugant):
            self.set_type_abundance(type_index=recip_idx, abundance=max(self.get_type_abundance(recip_idx) - transconjugant_abundance, 1))
        #----------------------------------
        print(f"[ Transfer event occurred at t={self.t:.2f} ]\t{self.type_set.trait_keys[donor_idx]}+{self.type_set.trait_keys[recip_idx]} --> {transconjugant.trait_keys[0]}\n", end="")
        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def event_low_abundance(self, t, variables, *args):
    #     num_types = args[0]
    #     N_t = variables[:num_types]
    #     #------------------------------
    #     abundances_abs = N_t[N_t > 0]
    #     return -1 if np.any(
    #         abundances_abs < self.threshold_min_abs_abundance) else 1  # this line sometimes leads to "ValueError: f(a) and f(b) must have different signs"
    #     # return (np.min(abundances_abs - self.threshold_min_abs_abundance)) # this line -also- sometimes leads to "ValueError: f(a) and f(b) must have different signs"
    # #------------------------------
    # event_low_abundance.direction = -1
    # event_low_abundance.terminal = True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def handle_low_abundance(self):
    #     N = self.N
    #     #----------------------------------
    #     lost_types = np.where((N != 0) & (N <= self.threshold_min_abs_abundance))[0]
    #     for i in lost_types:
    #         self.set_type_abundance(type_index=i, abundance=0.0)  # Set the abundance of lost types to 0 (but do not remove from system/type set data)
    #     #----------------------------------
    #     return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, new_type_set=None, abundance=0):#, parent_index=None, parent_id=None):  # , index=None, ):
        abundance = utils.treat_as_list(abundance)
        #----------------------------------
        # preexisting_typeIDs = self.type_set.typeIDs
        #----------------------------------
        new_type_indices = self.type_set.add_type(new_type_set)#, parent_index=parent_index, parent_id=parent_id)
        #----------------------------------
        self._N_series = self._N_series.add(np.zeros(shape=(new_type_set.num_types, self.N_series.shape[1])))
        self.set_type_abundance(type_index=list(range(self.type_set.num_types - new_type_set.num_types, self.type_set.num_types)), abundance=abundance)
        #----------------------------------
        for new_type_idx in new_type_indices:
            new_mutants        = self.type_set.generate_mutant_set(new_type_idx, update_mutant_indices=False)
            new_mutant_indices = self.mutant_set.add_type(new_mutants)
            self.type_set.mutant_indices[new_type_idx] = new_mutant_indices
            # - - - -
            new_segregants        = self.type_set.generate_segregant_set(new_type_idx, update_segregant_indices=False)
            new_segregant_indices = self.segregant_set.add_type(new_segregants)
            self.type_set.segregant_indices[new_type_idx] = new_segregant_indices
            # - - - -
            new_transconjugants_newtypeasdonor = self.type_set.generate_transconjugant_set(donor_index=new_type_idx, update_transconjugant_indices=False)
            new_transconjugants_newtypeasrecip = self.type_set.generate_transconjugant_set(recip_index=new_type_idx, update_transconjugant_indices=False)
            new_transconjugant_indices_bydonor = self.transconjugant_set.add_type(new_transconjugants_newtypeasdonor)
            new_transconjugant_indices_byrecip = self.transconjugant_set.add_type(new_transconjugants_newtypeasrecip)
            self.type_set.transconjugant_indices_bydonor[new_type_idx] = new_transconjugant_indices_bydonor
            self.type_set.transconjugant_indices_byrecip[new_type_idx] = new_transconjugant_indices_byrecip

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_fitness(self, t=None, t_index=None, N=None, R=None, return_mutants=False):
        t_idx = np.argmax(self.t_series >= t) if t is not None else t_index if t_index is not None else -1
        #----------------------------------
        if(return_mutants):
            _N = np.zeros(self.type_set.num_types + self.mutant_set.num_types)
            _N[:self.type_set.num_types] = self.N_series[:, t_idx] if N is None else N[:self.type_set.num_types]
            _R = self.R_series[:, t_idx] if R is None else R
            params = self.get_dynamics_params(type_index=t_idx)
            # --------------------------
            # These params indices should be right, but could be off
            return self.growth_rate(N=_N, R=_R, t=self.t_series[t_idx], traits=params[2], consumption_rate=params[3], carrying_capacity=params[4], energy_passthru=params[6], growth_factor=params[7], energy_costs=params[12],
                                    influx_rate=params[14], decay_rate=params[15], energy_content=params[16], resource_dynamics_mode=params[24], resource_influx_mode=params[25])
        else:
            _N = self.N_series[:, t_idx] if N is None else N[:self.type_set.num_types]
            _R = self.R_series[:, t_idx] if R is None else R
            # --------------------------
            return self.growth_rate(N=_N, R=_R, t=self.t_series[t_idx], traits=self.type_set.traits, consumption_rate=self.type_set.consumption_rate, carrying_capacity=self.type_set.carrying_capacity, energy_passthru=self.type_set.energy_passthru, growth_factor=self.type_set.growth_factor, energy_costs=self.type_set.energy_costs,
                                    influx_rate=self.resource_set.influx_rate, decay_rate=self.resource_set.decay_rate, energy_content=self.resource_set.energy_content, resource_dynamics_mode=self.resource_dynamics_mode, resource_influx_mode=self.resource_set.resource_influx_mode)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_growth_rate(self, t=None, t_index=None, N=None, R=None, return_mutants=False):
        return self.get_fitness(t, t_index, N, R, return_mutants)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_energy_uptake(self, t=None, t_index=None, N=None, R=None, return_mutants=False):
        t_idx = np.argmax(self.t_series >= t) if t is not None else t_index if t_index is not None else -1
        #----------------------------------
        if(return_mutants):
            _N = np.zeros(self.type_set.num_types + self.mutant_set.num_types)
            _N[:self.type_set.num_types] = self.N_series[:, t_idx] if N is None else N[:self.type_set.num_types]
            _R = self.R_series[:, t_idx] if R is None else R
            params = self.get_dynamics_params(type_index=t_idx)
            # --------------------------
            return self.energy_uptake(N=_N, R=_R, t=self.t_series[t_idx], traits=params[2], consumption_rate=params[3], carrying_capacity=params[4], energy_passthru=params[6],
                                      influx_rate=params[14], decay_rate=params[15], energy_content=params[16], resource_dynamics_mode=params[24], resource_influx_mode=params[25])
        else:
            _N = self.N_series[:, t_idx] if N is None else N[:self.type_set.num_types]
            _R = self.R_series[:, t_idx] if R is None else R
            # --------------------------
            return self.energy_uptake(N=_N, R=_R, t=self.t_series[t_idx], traits=self.type_set.traits, consumption_rate=self.type_set.consumption_rate, carrying_capacity=self.type_set.carrying_capacity, energy_passthru=self.type_set.energy_passthru,
                                      influx_rate=self.resource_set.influx_rate, decay_rate=self.resource_set.decay_rate, energy_content=self.resource_set.energy_content, resource_dynamics_mode=self.resource_dynamics_mode, resource_influx_mode=self.resource_set.resource_influx_mode)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_type_abundance(self, abundance, type_index=None, type_id=None, t=None, t_index=None):
        abundance = utils.treat_as_list(abundance)
        type_indices = [np.where(self.type_set.typeIDs == tid)[0] for tid in
                        utils.treat_as_list(type_id)] if type_id is not None else utils.treat_as_list(type_index)
        t_idx = np.argmax(self.t_series >= t) if t is not None else t_index if t_index is not None else -1
        #----------------------------------
        for i, type_idx in enumerate(type_indices):
            self.N_series[type_idx, t_idx] = abundance[i]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_type_abundance(self, type_index=None, type_id=None, t=None, t_index=None):
        type_indices = [np.where(np.array(self.type_set.typeIDs) == tid)[0] for tid in
                        utils.treat_as_list(type_id)] if type_id is not None else utils.treat_as_list(
            type_index) if type_index is not None else list(range(self.type_set.num_types))
        time_indices = [np.argmax(self.t_series >= t_) for t_ in
                        utils.treat_as_list(t)] if t is not None else utils.treat_as_list(
            t_index) if t_index is not None else -1
        #----------------------------------
        abundances = self.N_series[type_indices, :][:, time_indices]
        return abundances if len(type_indices) > 1 else abundances[0]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder_types(self, order=None):
        type_order   = np.argsort(self.type_set.lineageIDs) if order is None else order
        # mutant_order = self.type_set.get_mutant_indices(type_order)
        #----------------------------------
        self._N_series = self._N_series.reorder(type_order)
        self.type_set.reorder_types(type_order)  # don't need to reorder mutant_set because type_set.mutant_indices gets reordered and keeps correct pointers

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_extant_type_set(self, type_set=None, t=None, t_index=None):
        t_idx = np.argmax(self.t_series >= t) if t is not None else t_index if t_index is not None else -1
        type_set = self.type_set if type_set is None else type_set
        #----------------------------------
        if(t_idx == -1):
            return type_set.get_type(self.extant_type_indices)
        else:
            _extant_type_indices = np.where(self.N_series[:, t_idx] > 0)[0]
            return type_set.get_type(_extant_type_indices)

    def get_extant_type_indices(self, t=None, t_index=None):
        t_idx = np.argmax(self.t_series >= t) if t is not None else t_index if t_index is not None else -1
        #----------------------------------
        return np.where(self.N_series[:, t_idx] > 0)[0]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def combine(self, added_system, merge_on_type_id=True):
        # This implementation assumes that the 'self' system that is combined 'into' keeps its thresholds and other metadata attributes.
        # TODO: Properly combine resource sets (right now the 'self' system resources are kept as is)
        #----------------------------------
        # TODO: At the moment, it seems that there is no good way to reconcile parent indices and phylogenies/lineage ids from multiple systems,
        # so these are currently being reset in the combined system.
        self.type_set._parent_indices = [None for i in range(self.num_types)]
        self.type_set._lineageIDs = None
        #----------------------------------
        for comb_type_idx in range(added_system.type_set.num_types):
            # Retrieve the added type and some of its properties:
            comb_type = added_system.type_set.get_type(comb_type_idx)
            comb_type_id = added_system.type_set.get_type_id(comb_type_idx)
            comb_type_abundance = added_system.N[comb_type_idx]
            #----------------------------------
            if(merge_on_type_id and comb_type_id in self.type_set.typeIDs):
                # The added type is a pre-existing type in the current population; get its index:
                preexisting_type_idx = np.where(np.array(self.type_set.typeIDs) == comb_type_id)[0][0]
                # Add abundance equal to the added types abundance:
                self.set_type_abundance(type_index=preexisting_type_idx,
                                        abundance=self.get_type_abundance(preexisting_type_idx) + comb_type_abundance)
            else:
                # The added type is not present in the current population:
                # Add the new type to the population at its abundance in the added_system:
                self.add_type(comb_type, abundance=comb_type_abundance)#, parent_index=None)  # note that parent_index is None here under assumption that parent indices need to be reset in combined systems
            #----------------------------------
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def perturb(self, param, dist, args, mode='multiplicative_proportional', element_wise=True):
        params = utils.treat_as_list(param)
        #----------------------------------
        for param in params:
            if(param == 'consumption_rate'):
                perturb_vals = utils.get_perturbations(self.type_set.consumption_rate, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.type_set.consumption_rate = (self.type_set.consumption_rate * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.type_set.consumption_rate * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.type_set.consumption_rate + perturb_vals) if mode == 'additive' else self.type_set.consumption_rate
            elif(param == 'carrying_capacity'):
                perturb_vals = utils.get_perturbations(self.type_set.carrying_capacity, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.type_set.carrying_capacity = (self.type_set.carrying_capacity * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.type_set.carrying_capacity * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.type_set.carrying_capacity + perturb_vals) if mode == 'additive' else self.type_set.carrying_capacity
            elif(param == 'energy_passthru'):
                perturb_vals = utils.get_perturbations(self.type_set.energy_passthru, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.type_set.energy_passthru = (self.type_set.energy_passthru * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.type_set.energy_passthru * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.type_set.energy_passthru + perturb_vals) if mode == 'additive' else self.type_set.energy_passthru
            elif(param == 'growth_factor'):
                perturb_vals = utils.get_perturbations(self.type_set.growth_factor, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.type_set.growth_factor = (self.type_set.growth_factor * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.type_set.growth_factor * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.type_set.growth_factor + perturb_vals) if mode == 'additive' else self.type_set.growth_factor
            elif(param == 'cost_baseline'):
                perturb_vals = utils.get_perturbations(self.type_set.cost_baseline, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.type_set.cost_baseline = (self.type_set.cost_baseline * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.type_set.cost_baseline * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.type_set.cost_baseline + perturb_vals) if mode == 'additive' else self.type_set.cost_baseline
            elif(param == 'cost_trait'):
                perturb_vals = utils.get_perturbations(self.type_set.cost_trait, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.type_set.cost_trait = (self.type_set.cost_trait * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.type_set.cost_trait * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.type_set.cost_trait + perturb_vals) if mode == 'additive' else self.type_set.cost_trait
            elif(param == 'mutation_rate'):
                perturb_vals = utils.get_perturbations(self.type_set.mutation_rate, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.type_set.mutation_rate = (self.type_set.mutation_rate * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.type_set.mutation_rate * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.type_set.mutation_rate + perturb_vals) if mode == 'additive' else self.type_set.mutation_rate
            elif(param == 'influx_rate'):
                perturb_vals = utils.get_perturbations(self.resource_set.influx_rate, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.resource_set.influx_rate = (self.resource_set.influx_rate * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.resource_set.influx_rate * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.resource_set.influx_rate + perturb_vals) if mode == 'additive' else self.resource_set.influx_rate
            elif(param == 'decay_rate'):
                perturb_vals = utils.get_perturbations(self.resource_set.decay_rate, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.resource_set.decay_rate = (self.resource_set.decay_rate * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.resource_set.decay_rate * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.resource_set.decay_rate + perturb_vals) if mode == 'additive' else self.resource_set.decay_rate
            elif(param == 'energy_content'):
                perturb_vals = utils.get_perturbations(self.resource_set.energy_content, dist=dist, args=args, mode=mode, element_wise=element_wise)
                self.resource_set.energy_content = (self.resource_set.energy_content * (1 + np.maximum(perturb_vals, -1))) if mode == 'multiplicative_proportional' else (self.resource_set.energy_content * np.maximum(perturb_vals, 0)) if mode == 'multiplicative' else (self.resource_set.energy_content + perturb_vals) if mode == 'additive' else self.resource_set.energy_content
        #----------------------------------
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_most_fit_types(self, rank_cutoff=None, fitness_cutoff=None, t=None, t_index=None):
        rank_cutoff = 1 if rank_cutoff is None else rank_cutoff
        fitness_cutoff = np.min(self.fitness) if fitness_cutoff is None else fitness_cutoff
        t_idx = np.argmax(self.t_series >= t) if t is not None else t_index if t_index is not None else -1
        #----------------------------------
        return self.type_set.get_type(self.get_fitness(t_index=t_idx)[self.get_fitness(t_index=t_idx) >= fitness_cutoff].argsort()[::-1][:rank_cutoff])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_lowest_cost_types(self, rank_cutoff=None, cost_cutoff=None):
        rank_cutoff = 1 if rank_cutoff is None else rank_cutoff
        cost_cutoff = np.max(self.type_set.energy_costs) if cost_cutoff is None else cost_cutoff
        #----------------------------------
        return self.type_set.get_type(self.type_set.energy_costs[self.type_set.energy_costs <= cost_cutoff].argsort()[:rank_cutoff])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def resource_demand(N, traits):
        return np.einsum(('ij,ij->j' if N.ndim == 2 else 'i,ij->j'), N, traits)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_resource_demand(self, type_index=None, type_id=None, t=None, t_index=None, trait_subset=None, relative_demand=False):
        type_indices = [np.where(np.array(self.type_set.typeIDs) == tid)[0] for tid in utils.treat_as_list(type_id)] if type_id is not None else utils.treat_as_list(type_index) if type_index is not None else list(range(self.type_set.num_types))
        time_indices = [np.argmax(self.t_series >= t_) for t_ in utils.treat_as_list(t)] if t is not None else utils.treat_as_list(t_index) if t_index is not None else -1
        trait_subset = np.array(range(self.type_set.num_traits) if trait_subset is None else trait_subset)
        #----------------------------------
        abundances = self.get_type_abundance(type_index=type_indices, t_index=time_indices)
        #----------------------------------
        resource_demand = np.einsum(('ij,jk->ik' if abundances.ndim == 2 else 'i,ij->j'), abundances.T, self.type_set.traits[type_indices, :][:, trait_subset])
        if(relative_demand):
            resource_demand /= resource_demand.sum()
        #----------------------------------
        return (resource_demand.T)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_biomass(self, type_index=None, type_id=None, t=None, t_index=None, trait_subset=None):
        type_indices = [np.where(np.array(self.type_set.typeIDs) == tid)[0] for tid in utils.treat_as_list(type_id)] if type_id is not None else utils.treat_as_list(type_index) if type_index is not None else list(range(self.type_set.num_types))
        time_indices = [np.argmax(self.t_series >= t_) for t_ in utils.treat_as_list(t)] if t is not None else utils.treat_as_list(t_index) if t_index is not None else -1
        trait_subset = np.array(range(self.type_set.num_traits) if trait_subset is None else trait_subset)
        #----------------------------------
        resource_demand = self.get_resource_demand(type_index=type_indices, t_index=time_indices, trait_subset=trait_subset)
        #----------------------------------
        biomass = resource_demand.sum(axis=0)
        #----------------------------------
        return (biomass)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_num_extant_types(self, type_index=None, type_id=None, t=None, t_index=None):
        type_indices = [np.where(np.array(self.type_set.typeIDs) == tid)[0] for tid in utils.treat_as_list(type_id)] if type_id is not None else utils.treat_as_list(type_index) if type_index is not None else list(range(self.type_set.num_types))
        time_indices = [np.argmax(self.t_series >= t_) for t_ in utils.treat_as_list(t)] if t is not None else utils.treat_as_list(t_index) if t_index is not None else -1
        #----------------------------------
        abundances = self.get_type_abundance(type_index=type_indices, t_index=time_indices)
        #----------------------------------
        num_extant_types = np.count_nonzero(abundances, axis=0)
        #----------------------------------
        return num_extant_types

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_num_extant_phenotypes(self, type_index=None, type_id=None, t=None, t_index=None, trait_subset=None):
        type_indices = [np.where(np.array(self.type_set.typeIDs) == tid)[0] for tid in utils.treat_as_list(type_id)] if type_id is not None else utils.treat_as_list(type_index) if type_index is not None else list(range(self.type_set.num_types))
        time_indices = [np.argmax(self.t_series >= t_) for t_ in utils.treat_as_list(t)] if t is not None else utils.treat_as_list(t_index) if t_index is not None else -1
        trait_subset = np.array(range(self.type_set.num_traits) if trait_subset is None else trait_subset)
        #----------------------------------
        numphenos_by_time = []
        for tidx in time_indices:
            extant_type_indices = self.get_extant_type_indices(t_index=tidx)
            extant_type_indices_of_interest = list(set(extant_type_indices).intersection(set(type_indices)))
            traits_subset = self.type_set.traits[extant_type_indices_of_interest, :][:, trait_subset]
            num_phenotypes = np.unique(traits_subset, axis=0).shape[0]
            numphenos_by_time.append(num_phenotypes)
        #----------------------------------
        return np.array(numphenos_by_time)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_num_traits_per_type(self, type_index=None, type_id=None, t=None, t_index=None, trait_subset=None, summary_stat=None):
        type_indices = [np.where(np.array(self.type_set.typeIDs) == tid)[0] for tid in utils.treat_as_list(type_id)] if type_id is not None else utils.treat_as_list(type_index) if type_index is not None else list(range(self.type_set.num_types))
        time_indices = [np.argmax(self.t_series >= t_) for t_ in utils.treat_as_list(t)] if t is not None else utils.treat_as_list(t_index) if t_index is not None else -1
        trait_subset = np.array(range(self.type_set.num_traits) if trait_subset is None else trait_subset)
        #----------------------------------
        if(isinstance(time_indices, (list, np.ndarray))):
            numtraits_by_time = []
            for tidx in time_indices:
                extant_type_indices = self.get_extant_type_indices(t_index=tidx)
                extant_type_indices_of_interest = list(set(extant_type_indices).intersection(set(type_indices)))
                traits_subset = self.type_set.traits[extant_type_indices_of_interest, :][:, trait_subset]
                num_traits = np.count_nonzero(traits_subset, axis=1)
                if(summary_stat == 'mean' or summary_stat == 'average'):
                    numtraits_by_time.append(np.mean(num_traits))
                elif(summary_stat == 'median'):
                    numtraits_by_time.append(np.median(num_traits))
                elif(summary_stat == 'min'):
                    numtraits_by_time.append(np.min(num_traits))
                elif(summary_stat == 'max'):
                    numtraits_by_time.append(np.max(num_traits))
                elif(summary_stat == 'std' or summary_stat == 'stdev'):
                    numtraits_by_time.append(np.std(num_traits))
                else:
                    numtraits_by_time.append(num_traits)
            return np.array(numtraits_by_time)
        else:
            extant_type_indices = self.get_extant_type_indices(t_index=time_indices)
            extant_type_indices_of_interest = list(set(extant_type_indices).intersection(set(type_indices)))
            traits_subset = self.type_set.traits[extant_type_indices_of_interest, :][:, trait_subset]
            num_traits = np.count_nonzero(traits_subset, axis=1)
            if(summary_stat == 'mean' or summary_stat == 'average'):
                return np.mean(num_traits)
            elif(summary_stat == 'median'):
                return np.median(num_traits)
            elif(summary_stat == 'min'):
                return np.min(num_traits)
            elif(summary_stat == 'max'):
                return np.max(num_traits)
            elif(summary_stat == 'std' or summary_stat == 'stdev'):
                return np.std(num_traits)
            else:
                return num_traits

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def get_optima(self, traits_sample, t=None, t_index=None, N_eval=None, R_eval=None): #include_
    #     t_idx = np.argmax(self.t_series >= t) if t is not None else t_index if t_index is not None else -1
    #     #----------------------------------
    #
    #     eval_system = copy.deepcopy(self)
    #     eval_system.add_type(new_type_set=traits_sample, abundance=0)
    #
    #     # print(self.traits, self.traits.shape)
    #     # print(self.N, self.N.shape)
    #     # print(self.energy_cost, self.energy_cost.shape)
