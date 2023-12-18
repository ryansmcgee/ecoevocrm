import numpy as np

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TypeSet():

    def __init__(self, num_types            = None,
                       num_traits           = None,
                       traits               = None,
                       consumption_rate     = 1,
                       carrying_capacity    = 1e10,
                       growth_factor        = 1,
                       energy_passthru      = 0,
                       cost_baseline        = 0,
                       cost_trait           = None,
                       cost_interaction     = None,
                       mutation_rate        = 1e-10,
                       creation_rates       = None,
                       lineageIDs           = None,
                       parent_indices       = None,
                       binarize_trait_costs = True,
                       binarize_interaction_costs = True ):

        #----------------------------------
        # Determine the number of types and traits,
        # and initialize traits matrix:
        #----------------------------------
        if(isinstance(traits, (list, np.ndarray))):
            traits = np.array(traits)
            if(traits.ndim == 2):
                num_types  = traits.shape[0]
                num_traits = traits.shape[1]
            elif(traits.ndim == 1):
                num_types  = 1
                num_traits = len(traits)
        elif(num_types is not None and num_traits is not None):
            num_types  = num_types
            num_traits = num_traits
        else:
            utils.error("Error in TypeSet __init__(): Number of types and traits must be specified by providing a) a traits matrix, or b) both num_types and num_traits values.")
        #----------------------------------
        
        self.num_traits = num_traits

        # /!\ Non-binary traitss is deprecated as of 2023-09-01
        # self.normalize_phenotypes = normalize_phenotypes
        # if(self.normalize_phenotypes):
        #     norm_denom = np.atleast_2d(traits).sum(axis=1, keepdims=1)
        #     norm_denom[norm_denom == 0] = 1
        #     traits = traits/norm_denom

        self._traits = utils.ExpandableArray(utils.reshape(traits, shape=(num_types, num_traits)), dtype='int')

        #----------------------------------
        # Initialize parameter vectors/matrices:
        #----------------------------------

        # print("Start!")
        # print("_consumption_rate ...")
        self._consumption_rate   = self.preprocess_params(consumption_rate,  has_trait_dim=True)
        # print("_carrying_capacity ...")
        self._carrying_capacity  = self.preprocess_params(carrying_capacity, has_trait_dim=True)
        # print("_energy_passthru ...")
        self._energy_passthru  = self.preprocess_params(energy_passthru, has_trait_dim=True)
        # print("_growth_factor ...")
        self._growth_factor  = self.preprocess_params(growth_factor, has_trait_dim=False)
        # print("_cost_baseline ...")
        self._cost_baseline     = self.preprocess_params(cost_baseline,    has_trait_dim=False) #, force_expandable_array=(mean_cost_baseline_mut > 0))
        # print("_cost_trait ...")
        self._cost_trait    = self.preprocess_params(cost_trait,   has_trait_dim=True) if cost_trait is not None else None
        # print("_cost_interaction ...")
        self._cost_interaction      = utils.reshape(cost_interaction, shape=(self.num_traits, self.num_traits)) if cost_interaction is not None else None
        # print("_mutation_rate ...")
        self._mutation_rate     = self.preprocess_params(mutation_rate,    has_trait_dim=True)
        # print("_creation_rates ...")
        self._creation_rates = self.preprocess_params(creation_rates, has_trait_dim=False) if creation_rates is not None else None

        # self._mean_cost_baseline_mut = mean_cost_baseline_mut
        # self.__mean_cost_baseline_mut = self.preprocess_params(__mean_cost_baseline_mut, has_trait_dim=False)

        #----------------------------------
        # Initialize other type properties/metadata:
        #----------------------------------

        self._energy_costs = None 

        self._typeIDs   = None

        # self._parentIDs = parentIDs

        # self._mutantIDs = None

        # self._typeID_indices = {typeID: i for i, typeID in enumerate(self.typeIDs)}

        self._parent_indices = utils.treat_as_list(parent_indices) if parent_indices is not None else [None for i in range(self.num_types)]  
        self._mutant_indices =  None   

        self._lineageIDs = lineageIDs
        
        self.phylogeny = {}

        self.binarize_trait_costs = binarize_trait_costs
        self.binarize_interaction_costs   = binarize_interaction_costs
                
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def get_array(arr):
        return arr.values if isinstance(arr, utils.ExpandableArray) else arr

    @property
    def num_types(self):
        return self.traits.shape[0]
    
    @property
    def traits(self):
        return TypeSet.get_array(self._traits)

    @property
    def consumption_rate(self):
        return TypeSet.get_array(self._consumption_rate)

    @consumption_rate.setter
    def consumption_rate(self, vals):
        self._consumption_rate = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def carrying_capacity(self):
        return TypeSet.get_array(self._carrying_capacity)

    @carrying_capacity.setter
    def carrying_capacity(self, vals):
        self._carrying_capacity = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def energy_passthru(self):
        return TypeSet.get_array(self._energy_passthru)

    @energy_passthru.setter
    def energy_passthru(self, vals):
        self._energy_passthru = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def growth_factor(self):
        return TypeSet.get_array(self._growth_factor)

    @growth_factor.setter
    def growth_factor(self, vals):
        self._growth_factor = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def cost_baseline(self):
        return TypeSet.get_array(self._cost_baseline)

    @cost_baseline.setter
    def cost_baseline(self, vals):
        self._cost_baseline = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def cost_trait(self):
        return TypeSet.get_array(self._cost_trait)

    @cost_trait.setter
    def cost_trait(self, vals):
        self._cost_trait = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def mutation_rate(self):
        return TypeSet.get_array(self._mutation_rate)

    @mutation_rate.setter
    def mutation_rate(self, vals):
        self._mutation_rate = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def creation_rates(self):
        return self._creation_rates

    @creation_rates.setter
    def creation_rates(self, vals):
        self._creation_rates = self.preprocess_params(vals, has_trait_dim=False)

    # @property
    # def _mean_cost_baseline_mut(self):
    #     return self.__mean_cost_baseline_mut

    # @_mean_cost_baseline_mut.setter
    # def _mean_cost_baseline_mut(self, vals):
    #     self.__mean_cost_baseline_mut = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def cost_interaction(self):
        return TypeSet.get_array(self._cost_interaction)

    @property
    def energy_costs(self):
        if(self._energy_costs is None):
            costs = 0 + (self.cost_baseline.ravel() if self.cost_baseline.ndim == 2 else self.cost_baseline)
            costs += self.cost_trait_cost_terms
            costs += self.cost_interaction_cost_terms
            if(np.any(costs < 0)):
                raise ValueError('Negative energy_costs encountered for one or more types.')
            self._energy_costs = utils.ExpandableArray(costs)
        return TypeSet.get_array(self._energy_costs).ravel()

    @property
    def cost_baseline_cost_terms(self):
        return (self.cost_baseline.ravel() if self.cost_baseline.ndim == 2 else self.cost_baseline)

    @property
    def cost_trait_cost_terms(self):
        _traits = self.traits if not self.binarize_trait_costs else (self.traits > 0).astype(int)
        return np.sum(_traits * self.cost_trait, axis=1) if self._cost_trait is not None else 0
    
    @property
    def cost_interaction_cost_terms(self):
        _traits = self.traits if not self.binarize_interaction_costs else (self.traits > 0).astype(int)
        return -1 * np.sum(_traits * np.dot(_traits, self.cost_interaction), axis=1) if self._cost_interaction is not None else 0
    
    @property
    def typeIDs(self):
        if(self._typeIDs is None):
            self._typeIDs = np.array(self.assign_type_ids())
        return self._typeIDs

    # @property
    # def mutantIDs(self):
    #     return self._mutantIDs

    @property
    def parent_indices(self):
        return np.array(self._parent_indices)

    @property
    def mutant_indices(self):
        # if(self._mutant_indices is None):
        #     self._mutant_indices = utils.ExpandableArray(np.arange(0, self.num_types*self.num_traits).reshape(self.num_types, self.num_traits), dtype='int')
        # return TypeSet.get_array(self._mutant_indices)
        return self._mutant_indices

    @property
    def lineageIDs(self):
        if(self._lineageIDs is None):
            self.phylogeny = {}
            lineageIDs = []
            for i in range(self.num_types):
                new_lineage_id = self.add_type_to_phylogeny(i)
                lineageIDs.append(new_lineage_id)
            self._lineageIDs = lineageIDs
        return self._lineageIDs

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def preprocess_params(self, vals, has_trait_dim, force_expandable_array=False, dtype='float64'):
        # print(f"preprocess_params({vals})")
        arr = np.array(vals, dtype=dtype) if(isinstance(vals, (list, np.ndarray))) else np.array([vals], dtype=dtype)
        #----------------------------------
        if(has_trait_dim):
            if(arr.ndim == 1):
                if(len(arr) == 1):
                    return np.repeat(arr, self.num_traits)
                elif(len(arr) == self.num_traits):
                    return arr # as is
                elif(len(arr) == self.num_types):
                    if(np.all(arr == arr[0])): # all elements equal
                        return np.full(shape=(self.num_traits,), fill_value=arr[0])
                    else:
                        return utils.ExpandableArray( np.tile(arr.reshape(self.num_types, 1), (1, self.num_traits)) )
            elif(arr.ndim == 2):
                if(arr.shape[0] == self.num_types and arr.shape[1] == self.num_traits):
                    return utils.ExpandableArray( arr ) # as is
        #----------------------------------
        else:
            if(arr.ndim == 1):
                if(len(arr) == 1 and not force_expandable_array):
                    return arr[0] # single val as scalar
                elif(len(arr) == self.num_types):
                    if(np.all(arr == arr[0]) and not force_expandable_array): # all elements equal
                        return arr[0] # single val as scalar
                    else:
                        return utils.ExpandableArray( arr.reshape(self.num_types, 1) )
            elif(arr.ndim == 2):
                if(arr.shape[0] == self.num_types and arr.shape[1] == 1):
                    return utils.ExpandableArray( arr ) # as is
        #----------------------------------
        # If none of the above conditions met (hasn't returned by now):
        if(has_trait_dim):
            utils.error(f"Error in TypeSet.preprocess_params(): input with shape {arr.shape} does not correspond to the number of types ({self.num_types}) and/or traits ({self.num_traits}).")
        else:
            utils.error(f"Error in TypeSet.preprocess_params(): input with shape {arr.shape} does not correspond to the number of types ({self.num_types}) (has_trait_dim is {has_trait_dim}).")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def generate_mutant_phenotypes(self, traits=None):
    #     traits = self.traits if traits is None else traits
    #     traits = (traits != 0).astype(float)
    #     #----------------------------------
    #     mutations = np.tile(np.identity(traits.shape[1]), reps=(traits.shape[0], 1))
    #     traits_mut = 1 * np.logical_xor( np.repeat(traits, repeats=traits.shape[1], axis=0), mutations )
    #     #----------------------------------
    #     # /!\ Non-binary traitss is deprecated as of 2023-09-01
    #     # if(self.normalize_phenotypes):
    #     #     norm_denom = traits_mut.sum(axis=1, keepdims=1)
    #     #     norm_denom[norm_denom == 0] = 1
    #     #     traits_mut = traits_mut/norm_denom
    #     #----------------------------------
    #     return traits_mut


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def generate_mutant_set(self, update_mutantIDs=True): #
    #     traits_mut = self.generate_mutant_phenotypes()
    #     #---------------------------------- 
    #     consumption_rate_mut  = np.repeat(self.consumption_rate,  repeats=traits_mut.shape[0], axis=0) if self.consumption_rate.ndim == 2  else self.consumption_rate
    #     carrying_capacity_mut = np.repeat(self.carrying_capacity, repeats=traits_mut.shape[0], axis=0) if self.carrying_capacity.ndim == 2 else self.carrying_capacity
    #     energy_passthru_mut = np.repeat(self.energy_passthru, repeats=traits_mut.shape[0], axis=0) if self.energy_passthru.ndim == 2 else self.energy_passthru
    #     growth_factor_mut = np.repeat(self.growth_factor, repeats=traits_mut.shape[0], axis=0) if self.growth_factor.ndim == 2 else self.growth_factor
    #     cost_baseline_mut    = np.repeat(self.cost_baseline,    repeats=traits_mut.shape[0], axis=0) if self.cost_baseline.ndim == 2    else self.cost_baseline
    #     cost_trait_mut   = np.repeat(self.cost_trait,   repeats=traits_mut.shape[0], axis=0) if self.cost_trait.ndim == 2   else self.cost_trait
    #     mutation_rate_mut    = np.repeat(self.mutation_rate,    repeats=traits_mut.shape[0], axis=0) if self.mutation_rate.ndim == 2    else self.mutation_rate
    #     #----------------------------------
    #     # if(self._mean_cost_baseline_mut > 0):
    #     #     cost_baseline_mut = self.cost_baseline.ravel() - np.random.exponential(scale=self._mean_cost_baseline_mut, size=traits_mut.shape[0])
    #     # else:
    #     #     cost_baseline_mut = np.repeat(self.cost_baseline, repeats=traits_mut.shape[0], axis=0) if self.cost_baseline.ndim == 2 else self.cost_baseline
    #     #----------------------------------
    #     mutant_set = TypeSet(traits=traits_mut, consumption_rate=consumption_rate_mut, carrying_capacity=carrying_capacity_mut, energy_passthru=energy_passthru_mut, growth_factor=growth_factor_mut, cost_baseline=cost_baseline_mut, cost_trait=cost_trait_mut, cost_interaction=self.cost_interaction, mutation_rate=mutation_rate_mut, # mean_cost_baseline_mut=self._mean_cost_baseline_mut,
    #                          # normalize_phenotypes=self.normalize_phenotypes, 
    #                          binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
    #     #----------------------------------
    #     if(update_mutantIDs):
    #         self._mutantIDs = mutant_set.typeIDs.reshape((self.num_types, self.num_traits))
    #     #----------------------------------
    #     return mutant_set

    def generate_mutant_set(self, type_index=None, update_mutant_indices=True): #
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.traits.shape[0])
        #----------------------------------
        traits_mut            = []
        consumption_rate_mut             = [] if self.consumption_rate.ndim == 2 else self.consumption_rate
        carrying_capacity_mut            = [] if self.carrying_capacity.ndim == 2 else self.carrying_capacity
        energy_passthru_mut            = [] if self.energy_passthru.ndim == 2 else self.energy_passthru
        growth_factor_mut            = [] if self.growth_factor.ndim == 2 else self.growth_factor
        cost_baseline_mut               = [] if self.cost_baseline.ndim == 2 else self.cost_baseline
        cost_trait_mut              = [] if self.cost_trait.ndim == 2 else self.cost_trait
        mutation_rate_mut               = [] if self.mutation_rate.ndim == 2 else self.mutation_rate
        parent_indices_mut   = []
        creation_rates_mut = []
        mutant_indices       = []
        #----------------------------------
        for p, parent_idx in enumerate(type_idx):
            mutation_rate_p = self.mutation_rate[parent_idx] if self.mutation_rate.ndim == 2 else self.mutation_rate
            mutant_indices.append([])
            if(np.any(mutation_rate_p > 0)):
                for i in (np.where(mutation_rate_p > 0)[0] if mutation_rate_p.ndim == 1 else range(self.traits.shape[1])):
                    traits_mut.append(self.traits[parent_idx] ^ [0 if j!=i else 1 for j in range(self.traits.shape[1])])
                    # - - - - -
                    if(self.consumption_rate.ndim == 2):    consumption_rate_mut.append(self.consumption_rate[parent_idx])
                    if(self.carrying_capacity.ndim == 2):   carrying_capacity_mut.append(self.carrying_capacity[parent_idx])
                    if(self.energy_passthru.ndim == 2):   energy_passthru_mut.append(self.energy_passthru[parent_idx])
                    if(self.growth_factor.ndim == 2):   growth_factor_mut.append(self.growth_factor[parent_idx])
                    if(self.cost_baseline.ndim == 2):      cost_baseline_mut.append(self.cost_baseline[parent_idx])
                    if(self.cost_trait.ndim == 2):     cost_trait_mut.append(self.cost_trait[parent_idx])
                    if(self.mutation_rate.ndim == 2):      mutation_rate_mut.append(self.mutation_rate[parent_idx])
                    # - - - - -
                    creation_rates_mut.append(mutation_rate_p[i] if mutation_rate_p.ndim == 1 else mutation_rate_p)
                    # - - - - -
                    parent_indices_mut.append(parent_idx)
                    # - - - - -
                    mutant_indices[p].append(len(traits_mut)-1)
        #----------------------------------
        mutant_set = TypeSet(traits=traits_mut, consumption_rate=consumption_rate_mut, carrying_capacity=carrying_capacity_mut, energy_passthru=energy_passthru_mut, growth_factor=growth_factor_mut, cost_baseline=cost_baseline_mut, cost_trait=cost_trait_mut, cost_interaction=self.cost_interaction, mutation_rate=mutation_rate_mut,
                             creation_rates=creation_rates_mut, parent_indices=parent_indices_mut,
                             binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
        #----------------------------------
        if(update_mutant_indices):
            self._mutant_indices = mutant_indices
        #----------------------------------
        return mutant_set


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, type_set=None, traits=None, consumption_rate=None, carrying_capacity=None, energy_passthru=None, growth_factor=None, cost_baseline=None, cost_trait=None, mutation_rate=None, creation_rates=None, parent_index=None, parent_id=None, ref_type_idx=None): # type_index=None, mean_cost_baseline_mut=None,
        parent_idx   = np.where(self.typeIDs==parent_id)[0] if parent_id is not None else parent_index
        ref_type_idx = ref_type_idx if ref_type_idx is not None else parent_idx if parent_idx is not None else 0
        #----------------------------------
        added_type_set = None
        if(type_set is not None):
            if(isinstance(type_set, TypeSet)):
                added_type_set = type_set
            else:
                utils.error(f"Error in TypeSet add_type(): type_set argument expects object of TypeSet type.")
        else:
            added_type_set = TypeSet(traits=traits if traits is not None else self.traits[ref_type_idx],
                                         consumption_rate=consumption_rate if consumption_rate is not None else self.consumption_rate[ref_type_idx],
                                         carrying_capacity=carrying_capacity if carrying_capacity is not None else self.carrying_capacity[ref_type_idx],
                                         energy_passthru=energy_passthru if energy_passthru is not None else self.energy_passthru[ref_type_idx],
                                         growth_factor=growth_factor if growth_factor is not None else self.growth_factor[ref_type_idx],
                                         cost_baseline=cost_baseline if cost_baseline is not None else self.cost_baseline[ref_type_idx],
                                         cost_trait=cost_trait if cost_trait is not None else self.cost_trait[ref_type_idx],
                                         mutation_rate=mutation_rate if mutation_rate is not None else self.mutation_rate[ref_type_idx],
                                         # mean_cost_baseline_mut=mean_cost_baseline_mut if mean_cost_baseline_mut is not None else self._mean_cost_baseline_mut
                                         creation_rates=creation_rates if creation_rates is not None else self.creation_rates[ref_type_idx])
        # Check that the type set dimensions match the system dimensions:
        if(self.num_traits != added_type_set.num_traits): 
            utils.error(f"Error in TypeSet add_type(): The number of traits for added types ({added_type_set.num_traits}) does not match the number of type set traits ({self.num_traits}).")
        #----------------------------------
        added_type_indices = list(range(self.num_types, self.num_types+added_type_set.num_types))
        #----------------------------------
        self._traits = self._traits.add(added_type_set.traits)
        self._consumption_rate  = self._consumption_rate.add(added_type_set.consumption_rate)   if isinstance(self._consumption_rate,  utils.ExpandableArray) else self._consumption_rate
        self._carrying_capacity = self._carrying_capacity.add(added_type_set.carrying_capacity) if isinstance(self._carrying_capacity, utils.ExpandableArray) else self._carrying_capacity
        self._energy_passthru = self._energy_passthru.add(added_type_set.energy_passthru) if isinstance(self._energy_passthru, utils.ExpandableArray) else self._energy_passthru
        self._growth_factor = self._growth_factor.add(added_type_set.growth_factor) if isinstance(self._growth_factor, utils.ExpandableArray) else self._growth_factor
        self._cost_baseline    = self._cost_baseline.add(added_type_set.cost_baseline)       if isinstance(self._cost_baseline,    utils.ExpandableArray) else self._cost_baseline
        self._cost_trait   = self._cost_trait.add(added_type_set.cost_trait)     if isinstance(self._cost_trait,   utils.ExpandableArray) else self._cost_trait
        self._mutation_rate    = self._mutation_rate.add(added_type_set.mutation_rate)       if isinstance(self._mutation_rate,    utils.ExpandableArray) else self._mutation_rate
        self._creation_rates = self._creation_rates.add(added_type_set.creation_rates) if isinstance(self._creation_rates, utils.ExpandableArray) else self._creation_rates
        #----------------------------------
        # print()
        # print("self._parent_indices", self._parent_indices, type(self._parent_indices), "added_type_set.parent_indices", added_type_set.parent_indices, type(added_type_set.parent_indices))
        self._parent_indices = [pidx for idxlist in [self._parent_indices, added_type_set.parent_indices] for pidx in idxlist] 
        # print("* self._parent_indices", self._parent_indices)
        #----------------------------------
        # print("self._mutant_indices", self._mutant_indices, "added_type_set.mutant_indices", added_type_set.mutant_indices)
        if(self._mutant_indices is not None):
            if(added_type_set.mutant_indices is None):
                self._mutant_indices = [mindices for indiceslist in [self._mutant_indices, [[] for mut in range(added_type_set.num_types)]] for mindices in indiceslist] 
            else:
                self._mutant_indices = [mindices for indiceslist in [self._mutant_indices, added_type_set.mutant_indices] for mindices in indiceslist] 
        # print("* self._mutant_indices", self._mutant_indices)
        #----------------------------------
        if(self._energy_costs is not None):
            self._energy_costs.add(added_type_set.energy_costs, axis=1)
        #----------------------------------
        if(self._typeIDs is not None):
            # self._typeIDs.extend(added_type_set.typeIDs)
            self._typeIDs = [tid for idlist in [self._typeIDs, added_type_set.typeIDs] for tid in idlist] 
        #----------------------------------
        if(self._lineageIDs is not None):
            for i in range((self.num_types-1), (self.num_types-1)+added_type_set.num_types):
                added_lineage_id = self.add_type_to_phylogeny(i)
                self._lineageIDs.append(added_lineage_id)
        #----------------------------------
        return added_type_indices


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type_to_phylogeny(self, type_index=None, type_id=None, parent_id=None):
        type_idx   = np.where(np.in1d(self.typeIDs, utils.treat_as_list(type_id))) if type_id is not None else type_index
        parent_idx = np.where(self.typeIDs==parent_id)[0] if parent_id is not None else self.parent_indices[type_idx]
        #----------------------------------
        if(parent_idx is None or np.isnan(parent_idx)):
            new_lineage_id = str( len(self.phylogeny.keys())+1 )
            self.phylogeny.update({ new_lineage_id: {} })
        else:
            parent_lineage_id = self.lineageIDs[parent_idx.astype(int)]
            if('.' in parent_lineage_id):
                parent_lineage_id_parts = parent_lineage_id.split('.')
                lineageSubtree = self.phylogeny
                for l in range(1, len(parent_lineage_id_parts)+1):
                    lineageSubtree = lineageSubtree['.'.join(parent_lineage_id_parts[:l])]
            else:
                lineageSubtree = self.phylogeny[parent_lineage_id]
            new_lineage_id = parent_lineage_id +'.'+ str(len(lineageSubtree.keys())+1)
            lineageSubtree[new_lineage_id] = {}
        #----------------------------------
        return new_lineage_id


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_type(self, type_index=None):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else None
        if(type_idx is None):
            utils.error(f"Error in TypeSet get_type(): A type index or type id must be given.")
        #----------------------------------
        return TypeSet(traits  = self.traits[type_idx],
                        consumption_rate  = self.consumption_rate[type_idx]  if self.consumption_rate.ndim == 2   else self.consumption_rate,
                        carrying_capacity = self.carrying_capacity[type_idx] if self.carrying_capacity.ndim == 2  else self.carrying_capacity,
                        energy_passthru = self.energy_passthru[type_idx] if self.energy_passthru.ndim == 2  else self.energy_passthru,
                        growth_factor = self.growth_factor[type_idx] if self.growth_factor.ndim == 2  else self.growth_factor,
                        cost_baseline    = self.cost_baseline[type_idx]    if self.cost_baseline.ndim == 2     else self.cost_baseline,
                        cost_trait   = self.cost_trait[type_idx]   if self.cost_trait.ndim == 2    else self.cost_trait,
                        mutation_rate    = self.mutation_rate[type_idx]    if self.mutation_rate.ndim == 2     else self.mutation_rate,
                        cost_interaction     = self.cost_interaction,
                        creation_rates=self.creation_rates[type_idx] if self.creation_rates is not None and self.creation_rates.ndim == 2 else self.creation_rates,
                        parent_indices=self.parent_indices[type_idx],
                        # mean_cost_baseline_mut = self._mean_cost_baseline_mut,
                        # normalize_phenotypes           = self.normalize_phenotypes,
                        binarize_trait_costs = self.binarize_trait_costs,
                        binarize_interaction_costs   = self.binarize_interaction_costs )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|||

    def assign_type_ids(self, type_index=None, traits=None):
        # print("assignID >>", "type_index=", type_index, "traits=", traits)
        traits = traits if traits is not None else self.traits
        # type_idx = utils.treat_as_list(type_index) if type_index is not None else range(traits.shape[0])
        # Convert binary traits arrays to integer IDs
        typeIDs = []
        for u in range(len(traits)):
            intID = 0
            for bit in traits[u].ravel():
                intID = (intID << 1) | bit
            typeIDs.append(intID)
        return typeIDs #if len(typeIDs) > 1 else typeIDs[0]


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|||

    # new but bad
    # def get_indices(self, type_id=None, type_index=None, traits=None):
    #     traits = traits if traits is not None else self.traits
    #     return utils.treat_as_list(type_index) if type_index is not None else [self._typeID_indices[tid] for tid in utils.treat_as_list(type_id)] if type_id is not None else np.arange(0, len(traits), 1)
    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|||

    #/!\ deprecated
    # def get_mutant_indices(self, index):
        # type_idx = utils.treat_as_list(index)
        # return self.mutant_indices[type_idx].ravel()

    # new but bad
    # def get_mutant_ids(self, type_id=None, type_index=None):
    #     type_idx = self.get_indices(type_id=type_id, type_index=type_index)
    #     return [m for mutIDs_u in ([self._mutantIDs[u] for u in type_idx]) for m in mutIDs_u]
    def get_mutant_indices(self, type_index=None):
        type_idx = utils.treat_as_list(type_index)
        return [m for mutIndices_u in ([self._mutant_indices[u] for u in type_idx]) for m in mutIndices_u]


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # /!\  deprecated
    # def get_dynamics_params(self, index=None, type_id=None, include_mutants=False):
    #     type_idx = np.where(np.in1d(self.typeIDs, utils.treat_as_list(type_id))) if type_id is not None else index
    #     if(type_idx is None):
    #         type_idx = np.arange(0, self.num_types, 1)
    #     #----------------------------------
    #     return {'num_types':    len(type_idx),
    #             'traits':        self.traits[type_idx],
    #             'consumption_rate':         self.consumption_rate  if self.consumption_rate.ndim < 2  else self.consumption_rate[type_idx],
    #             'carrying_capacity':        self.carrying_capacity if self.carrying_capacity.ndim < 2 else self.carrying_capacity[type_idx],
    #             'energy_passthru':        self.energy_passthru if self.energy_passthru.ndim < 2 else self.energy_passthru[type_idx],
    #             'growth_factor':        self.growth_factor if self.growth_factor.ndim < 2 else self.growth_factor[type_idx],
    #             'cost_baseline':           self.cost_baseline    if self.cost_baseline.ndim < 2    else self.cost_baseline[type_idx],
    #             'cost_trait':          self.cost_trait   if self.cost_trait.ndim < 2   else self.cost_trait[type_idx],
    #             'cost_interaction':            self.cost_interaction,
    #             'mutation_rate':           self.mutation_rate    if self.mutation_rate.ndim < 2    else self.mutation_rate[type_idx],
    #             'energy_costs': self.energy_costs[type_idx]}

    # new but bad
    # def get_dynamics_params(self, type_id=None, type_index=None):
    #     type_idx = self.get_indices(type_id=type_id, type_index=type_index)
    #     print("get_dynamics_params @@@", type_id, type_index, "-> type_idx", type_idx)
    #     #----------------------------------
    #     return {'num_types':    len(type_idx),
    #             'traits':        self.traits[type_idx],
    #             'consumption_rate':         self.consumption_rate  if self.consumption_rate.ndim < 2  else self.consumption_rate[type_idx],
    #             'carrying_capacity':        self.carrying_capacity if self.carrying_capacity.ndim < 2 else self.carrying_capacity[type_idx],
    #             'energy_passthru':        self.energy_passthru if self.energy_passthru.ndim < 2 else self.energy_passthru[type_idx],
    #             'growth_factor':        self.growth_factor if self.growth_factor.ndim < 2 else self.growth_factor[type_idx],
    #             'cost_baseline':           self.cost_baseline    if self.cost_baseline.ndim < 2    else self.cost_baseline[type_idx],
    #             'cost_trait':          self.cost_trait   if self.cost_trait.ndim < 2   else self.cost_trait[type_idx],
    #             'cost_interaction':            self.cost_interaction,
    #             # 'mutation_rate':           self.mutation_rate    if self.mutation_rate.ndim < 2    else self.mutation_rate[type_idx],
    #             'creation_rates': (self.creation_rates if self.creation_rates.ndim < 2 else self.creation_rates[type_idx]) if self.creation_rates is not None else None,
    #             'energy_costs': self.energy_costs[type_idx]}

    def get_dynamics_params(self, type_index=None):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.num_types)
        # print("get_dynamics_params @@@", type_index, "-> type_idx", type_idx)
        #----------------------------------
        return {'num_types':    len(type_idx),
                'traits':        self.traits[type_idx],
                'consumption_rate':         self.consumption_rate  if self.consumption_rate.ndim < 2  else self.consumption_rate[type_idx],
                'carrying_capacity':        self.carrying_capacity if self.carrying_capacity.ndim < 2 else self.carrying_capacity[type_idx],
                'energy_passthru':        self.energy_passthru if self.energy_passthru.ndim < 2 else self.energy_passthru[type_idx],
                'growth_factor':        self.growth_factor if self.growth_factor.ndim < 2 else self.growth_factor[type_idx],
                'cost_baseline':           self.cost_baseline    if self.cost_baseline.ndim < 2    else self.cost_baseline[type_idx],
                'cost_trait':          self.cost_trait   if self.cost_trait.ndim < 2   else self.cost_trait[type_idx],
                'cost_interaction':            self.cost_interaction,
                'energy_costs': self.energy_costs[type_idx],
                # 'mu':           self.mu    if self.mu.ndim < 2    else self.mutation_rate[type_idx],
                'creation_rates': (self.creation_rates if self.creation_rates.ndim < 2 else self.creation_rates[type_idx]) if self.creation_rates is not None else None,
                'parent_indices': self.parent_indices[type_idx]
                }


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder_types(self, order=None):
        type_order   = np.argsort(self.lineageIDs) if order is None else order
        if(len(type_order) < self.num_types):
            utils.error("Error in TypeSet.reorder_types(): The ordering provided has fewer indices than types.")
        #----------------------------------
        self._traits = self._traits.reorder(type_order)
        self._consumption_rate  = self._consumption_rate.reorder(type_order)  if isinstance(self._consumption_rate,  utils.ExpandableArray) else self._consumption_rate
        self._carrying_capacity = self._carrying_capacity.reorder(type_order) if isinstance(self._carrying_capacity, utils.ExpandableArray) else self._carrying_capacity
        self._energy_passthru = self._energy_passthru.reorder(type_order) if isinstance(self._energy_passthru, utils.ExpandableArray) else self._energy_passthru
        self._growth_factor = self._growth_factor.reorder(type_order) if isinstance(self._growth_factor, utils.ExpandableArray) else self._growth_factor
        self._cost_baseline    = self._cost_baseline.reorder(type_order)    if isinstance(self._cost_baseline,    utils.ExpandableArray) else self._cost_baseline
        self._cost_trait   = self._cost_trait.reorder(type_order)   if isinstance(self._cost_trait,   utils.ExpandableArray) else self._cost_trait
        self._mutation_rate    = self._mutation_rate.reorder(type_order)    if isinstance(self._mutation_rate,    utils.ExpandableArray) else self._mutation_rate
        self._creation_rates = self._creation_rates.reorder(type_order) if isinstance(self._creation_rates, utils.ExpandableArray) else self._creation_rates
        self._energy_costs   = None # reset to recalculate upon next reference
        self._typeIDs       = np.array(self._typeIDs)[type_order].tolist() if self._typeIDs is not None else None
        self._lineageIDs    = np.array(self._lineageIDs)[type_order].tolist() if self._lineageIDs is not None else None
        self._mutant_indices = np.array(self._mutant_indices)[type_order].tolist() if self._mutant_indices is not None else None
        #----------------------------------
        # Parent indices require special handling because simply reordering the parent indices list makes the index pointers point to incorrect places relative to the reordered lists
        _parent_indices_tempreorder = np.array(self._parent_indices)[type_order].tolist()
        self._parent_indices = [np.where(type_order == pidx)[0][0] if pidx is not None else None for pidx in _parent_indices_tempreorder]
        #----------------------------------
        return


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_lineage_depths(self):
        lineage_depths = [len(lin_id.split('.')) for lin_id in self.lineageIDs]
        return lineage_depths

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_num_mutations(self):
        return np.array(self.get_lineage_depths()) - 1


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_phenotype_strings(self):
        return [ ''.join(['1' if traits_vector[i] != 0 else '0' for i in range(len(traits_vector))]) for traits_vector in self.traits ]




