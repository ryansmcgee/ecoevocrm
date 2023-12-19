import numpy as np

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TypeSet():

    def __init__(self, num_types            = None,
                       num_traits           = None,
                       traits               = None,
                       consumption_rate     = 1,
                       carrying_capacity    = 1e9,
                       growth_factor        = 1,
                       energy_passthru      = 0,
                       cost_baseline        = 0,
                       cost_trait           = 0,
                       cost_interaction     = None,
                       cost_landscape       = None,
                       mutation_prob        = 1e-9,
                       segregation_prob     = 0,
                       transfer_donor_rate  = 0,
                       transfer_recip_rate  = 0,
                       mutant_prob          = None,
                       segregant_prob       = None,
                       transconjugant_rate  = None,
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
                num_traits = len(traits)
                num_types  = 1 if num_traits > 0 else 0
        elif(num_types is not None and num_traits is not None):
            num_types  = num_types
            num_traits = num_traits
        else:
            utils.error("Error in TypeSet __init__(): Number of types and traits must be specified by providing a) a traits matrix, or b) both num_types and num_traits values.")
        #----------------------------------
        self.num_traits = num_traits
        self._traits = utils.ExpandableArray(utils.reshape(traits, shape=(num_types, num_traits)), dtype='int')

        #----------------------------------
        # Initialize parameter vectors/matrices:
        #----------------------------------
        self._consumption_rate    = self.preprocess_params(consumption_rate,  has_trait_dim=True)
        self._carrying_capacity   = self.preprocess_params(carrying_capacity, has_trait_dim=True)
        self._energy_passthru     = self.preprocess_params(energy_passthru,   has_trait_dim=True)
        self._growth_factor       = self.preprocess_params(growth_factor,     has_trait_dim=False)
        self._cost_baseline       = self.preprocess_params(cost_baseline,     has_trait_dim=False)
        self._cost_trait          = self.preprocess_params(cost_trait,        has_trait_dim=True) if cost_trait is not None else None
        self._cost_interaction    = utils.reshape(cost_interaction, shape=(self.num_traits, self.num_traits)) if cost_interaction is not None else None
        self._cost_landscape      = cost_landscape

        self._mutation_prob       = self.preprocess_params(mutation_prob,       has_trait_dim=True)
        self._segregation_prob    = self.preprocess_params(segregation_prob,    has_trait_dim=True)
        self._transfer_donor_rate = self.preprocess_params(transfer_donor_rate, has_trait_dim=True)
        self._transfer_recip_rate = self.preprocess_params(transfer_recip_rate, has_trait_dim=True)

        self._mutant_prob         = self.preprocess_params(mutant_prob, has_trait_dim=False) if mutant_prob is not None else None
        self._segregant_prob      = self.preprocess_params(segregant_prob, has_trait_dim=False) if segregant_prob is not None else None
        self._transconjugant_rate = self.preprocess_params(transconjugant_rate, has_trait_dim=False) if transconjugant_rate is not None else None

        #----------------------------------
        # Initialize other type properties/metadata:
        #----------------------------------
        self._typeIDs                = None
        self._parent_indices         = utils.treat_as_list(parent_indices) if parent_indices is not None else [None for i in range(self.num_types)]
        self._mutant_indices         = None
        self._segregant_indices      = None
        self._transconjugant_indices = None

        self._lineageIDs = lineageIDs
        self.phylogeny = {}

        self._energy_costs = None
        self.binarize_trait_costs = binarize_trait_costs
        self.binarize_interaction_costs = binarize_interaction_costs
    
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
    def cost_interaction(self):
        return TypeSet.get_array(self._cost_interaction)

    @property
    def cost_landscape(self):
        return TypeSet.get_array(self._cost_landscape)

    @property
    def energy_costs(self):
        if(self._energy_costs is None):
            costs = 0
            costs += self.cost_baseline_bytype
            costs += self.cost_trait_bytype
            costs += self.cost_interaction_bytype
            costs += self.cost_landscape_bytype
            if(np.any(costs < 0)):
                raise ValueError('Negative energy_costs encountered for one or more types.')
            self._energy_costs = utils.ExpandableArray(costs)
        return TypeSet.get_array(self._energy_costs).ravel()

    @property
    def cost_baseline_bytype(self):
        return (self.cost_baseline.ravel() if self.cost_baseline.ndim == 2 else self.cost_baseline)

    @property
    def cost_trait_bytype(self):
        _traits = self.traits if not self.binarize_trait_costs else (self.traits > 0).astype(int)
        return np.sum(_traits * self.cost_trait, axis=1) if self._cost_trait is not None else 0
    
    @property
    def cost_interaction_bytype(self):
        _traits = self.traits if not self.binarize_interaction_costs else (self.traits > 0).astype(int)
        return -1 * np.sum(_traits * np.dot(_traits, self.cost_interaction), axis=1) if self._cost_interaction is not None else 0

    @property
    def cost_landscape_bytype(self):
        return [self._cost_landscape[k] for k in self.type_keys]

    @property
    def mutation_prob(self):
        return TypeSet.get_array(self._mutation_prob)

    @mutation_prob.setter
    def mutation_prob(self, vals):
        self._mutation_prob = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def mutant_prob(self):
        return self._mutant_prob

    @mutant_prob.setter
    def mutant_prob(self, vals):
        self._mutant_prob = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def segregation_prob(self):
        return TypeSet.get_array(self._segregation_prob)

    @segregation_prob.setter
    def segregation_prob(self, vals):
        self._segregation_prob = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def segregant_prob(self):
        return self._segregant_prob

    @segregant_prob.setter
    def segregant_prob(self, vals):
        self._segregant_prob = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def transfer_donor_rate(self):
        return TypeSet.get_array(self._transfer_donor_rate)

    @transfer_donor_rate.setter
    def transfer_donor_rate(self, vals):
        self._transfer_donor_rate = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def transfer_recip_rate(self):
        return TypeSet.get_array(self._transfer_recip_rate)

    @transfer_recip_rate.setter
    def transfer_recip_rate(self, vals):
        self._transfer_recip_rate = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def transconjugant_rate(self):
        return self._transconjugant_rate

    @transconjugant_rate.setter
    def transconjugant_rate(self, vals):
        self._transconjugant_rate = self.preprocess_params(vals, has_trait_dim=False)
    
    @property
    def typeIDs(self):
        if(self._typeIDs is None):
            self._typeIDs = np.array(self.assign_type_ids())
        return self._typeIDs

    @property
    def type_keys(self):
        return [''.join(str(a) for a in traits_u) for traits_u in (self.traits != 0).astype(int)]

    @property
    def parent_indices(self):
        return np.array(self._parent_indices)

    @property
    def mutant_indices(self):
        return self._mutant_indices

    @property
    def segregant_indices(self):
        return self._segregant_indices

    @property
    def transconjugant_indices(self):
        return self._transconjugant_indices

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
                elif(self.num_types == 0 and self.num_traits == 0):
                    return np.array([])
            elif(arr.ndim == 2):
                if(arr.shape[0] == self.num_types and arr.shape[1] == self.num_traits):
                    return utils.ExpandableArray( arr ) # as is
        #----------------------------------
        else:
            if(arr.ndim == 1):
                if(len(arr) == 1 and not force_expandable_array):
                    return arr[0] # single val as scalar
                elif(len(arr) == self.num_types):
                    if(self.num_types == 0):
                        return np.array([])
                    elif(np.all(arr == arr[0]) and not force_expandable_array): # all elements equal
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

    def generate_mutant_set(self, type_index=None, update_mutant_indices=True):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.traits.shape[0])
        #----------------------------------
        traits_mut              = []
        consumption_rate_mut    = [] if self.consumption_rate.ndim == 2 else self.consumption_rate
        carrying_capacity_mut   = [] if self.carrying_capacity.ndim == 2 else self.carrying_capacity
        energy_passthru_mut     = [] if self.energy_passthru.ndim == 2 else self.energy_passthru
        growth_factor_mut       = [] if self.growth_factor.ndim == 2 else self.growth_factor
        cost_baseline_mut       = [] if self.cost_baseline.ndim == 2 else self.cost_baseline
        cost_trait_mut          = [] if self.cost_trait.ndim == 2 else self.cost_trait
        mutation_prob_mut       = [] if self.mutation_prob.ndim == 2 else self.mutation_prob
        segregation_prob_mut    = [] if self.segregation_prob.ndim == 2 else self.segregant_prob
        transfer_donor_rate_mut = [] if self.transfer_donor_rate.ndim == 2 else self.transfer_donor_rate
        transfer_recip_rate_mut = [] if self.transfer_recip_rate.ndim == 2 else self.transfer_recip_rate
        parent_indices_mut      = []
        mutant_prob_mut         = []
        mutant_indices          = []
        #----------------------------------
        for p, parent_idx in enumerate(type_idx):
            mutation_prob_p = self.mutation_prob[parent_idx] if self.mutation_prob.ndim == 2 else self.mutation_prob
            mutant_indices.append([])
            if(np.any(mutation_prob_p > 0)):
                for i in (np.where(mutation_prob_p > 0)[0] if mutation_prob_p.ndim == 1 else range(self.traits.shape[1])):
                    traits_mut.append(self.traits[parent_idx] ^ [0 if j!=i else 1 for j in range(self.traits.shape[1])])
                    # - - - - -
                    if(self.consumption_rate.ndim == 2):    consumption_rate_mut.append(self.consumption_rate[parent_idx])
                    if(self.carrying_capacity.ndim == 2):   carrying_capacity_mut.append(self.carrying_capacity[parent_idx])
                    if(self.energy_passthru.ndim == 2):     energy_passthru_mut.append(self.energy_passthru[parent_idx])
                    if(self.growth_factor.ndim == 2):       growth_factor_mut.append(self.growth_factor[parent_idx])
                    if(self.cost_baseline.ndim == 2):       cost_baseline_mut.append(self.cost_baseline[parent_idx])
                    if(self.cost_trait.ndim == 2):          cost_trait_mut.append(self.cost_trait[parent_idx])
                    if(self.mutation_prob.ndim == 2):       mutation_prob_mut.append(self.mutation_prob[parent_idx])
                    if(self.segregation_prob.ndim == 2):    segregation_prob_mut.append(self.segregation_prob[parent_idx])
                    if(self.transfer_donor_rate.ndim == 2): transfer_donor_rate_mut.append(self.transfer_donor_rate[parent_idx])
                    if(self.transfer_recip_rate.ndim == 2): transfer_recip_rate_mut.append(self.transfer_recip_rate[parent_idx])
                    # - - - - -
                    mutant_prob_mut.append(mutation_prob_p[i] if mutation_prob_p.ndim == 1 else mutation_prob_p)
                    # - - - - -
                    parent_indices_mut.append(parent_idx)
                    # - - - - -
                    mutant_indices[p].append(len(traits_mut)-1)
        #----------------------------------
        mutant_set = TypeSet(traits=traits_mut, consumption_rate=consumption_rate_mut, carrying_capacity=carrying_capacity_mut, energy_passthru=energy_passthru_mut, growth_factor=growth_factor_mut,
                             cost_baseline=cost_baseline_mut, cost_trait=cost_trait_mut, cost_interaction=self.cost_interaction, cost_landscape=self.cost_landscape,
                             mutation_prob=mutation_prob_mut, segregation_prob=segregation_prob_mut, transfer_donor_rate=transfer_donor_rate_mut, transfer_recip_rate=transfer_recip_rate_mut,
                             mutant_prob=mutant_prob_mut, parent_indices=parent_indices_mut, binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
        #----------------------------------
        if(update_mutant_indices):
            self._mutant_indices = mutant_indices
        #----------------------------------
        return mutant_set


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, type_set=None, traits=None, consumption_rate=None, carrying_capacity=None, energy_passthru=None, growth_factor=None, cost_baseline=None, cost_trait=None,
                 mutation_prob=None, mutant_prob=None, segregation_prob=None, segregant_prob=None, transfer_donor_rate=None, transfer_recip_rate=None, transconjugant_rate=None,
                 parent_index=None, parent_id=None, ref_type_idx=None):
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
                                         mutation_prob=mutation_prob if mutation_prob is not None else self.mutation_prob[ref_type_idx],
                                         mutant_prob=mutant_prob if mutant_prob is not None else self.mutant_prob[ref_type_idx],
                                         segregation_prob=segregation_prob if segregation_prob is not None else self.segregation_prob[ref_type_idx],
                                         segregant_prob=segregant_prob if segregant_prob is not None else self.segregant_prob[ref_type_idx],
                                         transfer_donor_rate=transfer_donor_rate if transfer_donor_rate is not None else self.transfer_donor_rate[ref_type_idx],
                                         transfer_recip_rate=transfer_recip_rate if transfer_recip_rate is not None else self.transfer_recip_rate[ref_type_idx],
                                         transconjugant_rate=transconjugant_rate if transconjugant_rate is not None else self.transconjugant_rate[ref_type_idx],
                                     )
        # Check that the type set dimensions match the system dimensions:
        if(self.num_traits != added_type_set.num_traits): 
            utils.error(f"Error in TypeSet add_type(): The number of traits for added types ({added_type_set.num_traits}) does not match the number of type set traits ({self.num_traits}).")
        #----------------------------------
        added_type_indices = list(range(self.num_types, self.num_types+added_type_set.num_types))
        #----------------------------------
        self._traits = self._traits.add(added_type_set.traits)
        self._consumption_rate    = self._consumption_rate.add(added_type_set.consumption_rate)       if isinstance(self._consumption_rate, utils.ExpandableArray) else self._consumption_rate
        self._carrying_capacity   = self._carrying_capacity.add(added_type_set.carrying_capacity)     if isinstance(self._carrying_capacity, utils.ExpandableArray) else self._carrying_capacity
        self._energy_passthru     = self._energy_passthru.add(added_type_set.energy_passthru)         if isinstance(self._energy_passthru, utils.ExpandableArray) else self._energy_passthru
        self._growth_factor       = self._growth_factor.add(added_type_set.growth_factor)             if isinstance(self._growth_factor, utils.ExpandableArray) else self._growth_factor
        self._cost_baseline       = self._cost_baseline.add(added_type_set.cost_baseline)             if isinstance(self._cost_baseline, utils.ExpandableArray) else self._cost_baseline
        self._cost_trait          = self._cost_trait.add(added_type_set.cost_trait)                   if isinstance(self._cost_trait, utils.ExpandableArray) else self._cost_trait
        self._mutation_prob       = self._mutation_prob.add(added_type_set.mutation_prob)             if isinstance(self._mutation_prob, utils.ExpandableArray) else self._mutation_prob
        self._mutant_prob         = self._mutant_prob.add(added_type_set.mutant_prob)                 if isinstance(self._mutant_prob, utils.ExpandableArray) else self._mutant_prob
        self._segregation_prob    = self._segregation_prob.add(added_type_set.segregation_prob)       if isinstance(self._segregation_prob, utils.ExpandableArray) else self._segregation_prob
        self._segregant_prob      = self._segregant_prob.add(added_type_set.segregant_prob)           if isinstance(self._segregant_prob, utils.ExpandableArray) else self._segregant_prob
        self._transfer_donor_rate = self._transfer_donor_rate.add(added_type_set.transfer_donor_rate) if isinstance(self._transfer_donor_rate, utils.ExpandableArray) else self._transfer_donor_rate
        self._transfer_recip_rate = self._transfer_recip_rate.add(added_type_set.transfer_recip_rate) if isinstance(self._transfer_recip_rate, utils.ExpandableArray) else self._transfer_recip_rate
        self._transconjugant_rate = self._transconjugant_rate.add(added_type_set.transconjugant_rate) if isinstance(self._transconjugant_rate, utils.ExpandableArray) else self._transconjugant_rate
        #----------------------------------
        self._parent_indices = [pidx for idxlist in [self._parent_indices, added_type_set.parent_indices] for pidx in idxlist]
        #----------------------------------
        if(self._mutant_indices is not None):
            if(added_type_set.mutant_indices is None):
                self._mutant_indices = [mindices for indiceslist in [self._mutant_indices, [[] for addedtype in range(added_type_set.num_types)]] for mindices in indiceslist]
            else:
                self._mutant_indices = [mindices for indiceslist in [self._mutant_indices, added_type_set.mutant_indices] for mindices in indiceslist]
        #--------
        if(self._segregant_indices is not None):
            if(added_type_set.segregant_indices is None):
                self._segregant_indices = [sindices for indiceslist in [self._segregant_indices, [[] for addedtype in range(added_type_set.num_types)]] for sindices in indiceslist]
            else:
                self._segregant_indices = [sindices for indiceslist in [self._segregant_indices, added_type_set.segregant_indices] for sindices in indiceslist]
        #--------
        if(self._transconjugant_indices is not None):
            if(added_type_set.transconjugant_indices is None):
                self._transconjugant_indices = [tindices for indiceslist in [self._transconjugant_indices, [[] for addedtype in range(added_type_set.num_types)]] for tindices in indiceslist]
            else:
                self._transconjugant_indices = [tindices for indiceslist in [self._transconjugant_indices, added_type_set.transconjugant_indices] for tindices in indiceslist]
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
        return TypeSet(traits = self.traits[type_idx],
                        consumption_rate    = self.consumption_rate[type_idx]    if self.consumption_rate.ndim == 2    else self.consumption_rate,
                        carrying_capacity   = self.carrying_capacity[type_idx]   if self.carrying_capacity.ndim == 2   else self.carrying_capacity,
                        energy_passthru     = self.energy_passthru[type_idx]     if self.energy_passthru.ndim == 2     else self.energy_passthru,
                        growth_factor       = self.growth_factor[type_idx]       if self.growth_factor.ndim == 2       else self.growth_factor,
                        cost_baseline       = self.cost_baseline[type_idx]       if self.cost_baseline.ndim == 2       else self.cost_baseline,
                        cost_trait          = self.cost_trait[type_idx]          if self.cost_trait.ndim == 2          else self.cost_trait,
                        cost_interaction    = self.cost_interaction,
                        cost_landscape      = self.cost_landscape,
                        mutation_prob       = self.mutation_prob[type_idx]       if self.mutation_prob.ndim == 2       else self.mutation_prob,
                        mutant_prob         = self.mutant_prob[type_idx]         if self.mutant_prob is not None and self.mutant_prob.ndim == 2 else self.mutant_prob,
                        segregation_prob    = self.segregation_prob[type_idx]    if self.segregation_prob.ndim == 2    else self.segregation_prob,
                        segregant_prob      = self.segregant_prob[type_idx]      if self.segregant_prob is not None and self.segregant_prob.ndim == 2 else self.segregant_prob,
                        transfer_donor_rate = self.transfer_donor_rate[type_idx] if self.transfer_donor_rate.ndim == 2 else self.transfer_donor_rate,
                        transfer_recip_rate = self.transfer_recip_rate[type_idx] if self.transfer_recip_rate.ndim == 2 else self.transfer_recip_rate,
                        transconjugant_rate = self.transconjugant_rate[type_idx] if self.transconjugant_rate is not None and self.transconjugant_rate.ndim == 2 else self.transconjugant_rate,
                        parent_indices      = self.parent_indices[type_idx],
                        binarize_trait_costs = self.binarize_trait_costs,
                        binarize_interaction_costs = self.binarize_interaction_costs )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|||

    def assign_type_ids(self, traits=None):
        # TODO: Implement option to assign typeIDs using parameter values for future support of param val evolution
        traits = traits if traits is not None else self.traits
        # Convert binary traits arrays to integer IDs
        typeIDs = []
        for u in range(len(traits)):
            intID = 0
            for bit in traits[u].ravel():
                intID = (intID << 1) | bit
            typeIDs.append(intID)
        return typeIDs

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_mutant_indices(self, type_index=None):
        type_idx = utils.treat_as_list(type_index)
        return [m for mutIndices_u in ([self._mutant_indices[u] for u in type_idx]) for m in mutIndices_u]

    def get_segregant_indices(self, type_index=None):
        type_idx = utils.treat_as_list(type_index)
        return [s for segIndices_u in ([self._segregant_indices[u] for u in type_idx]) for s in segIndices_u]

    def get_transconjugant_indices(self, type_index=None):
        type_idx = utils.treat_as_list(type_index)
        return [x for tranIndices_u in ([self._transconjugant_indices[u] for u in type_idx]) for x in tranIndices_u]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, type_index=None):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.num_types)
        # print("get_dynamics_params @@@", type_index, "-> type_idx", type_idx)
        #----------------------------------
        return { 'num_types':            len(type_idx),
                 'traits':               self.traits[type_idx],
                 'consumption_rate':     self.consumption_rate  if self.consumption_rate.ndim < 2  else self.consumption_rate[type_idx],
                 'carrying_capacity':    self.carrying_capacity if self.carrying_capacity.ndim < 2 else self.carrying_capacity[type_idx],
                 'energy_passthru':      self.energy_passthru if self.energy_passthru.ndim < 2 else self.energy_passthru[type_idx],
                 'growth_factor':        self.growth_factor if self.growth_factor.ndim < 2 else self.growth_factor[type_idx],
                 'energy_costs':         self.energy_costs[type_idx],
                 'mutant_prob':          (self.mutant_prob if self.mutant_prob.ndim < 2 else self.mutant_prob[type_idx]) if self.mutant_prob is not None else None,
                 'segregant_prob':       (self.segregant_prob if self.segregant_prob.ndim < 2 else self.segregant_prob[type_idx]) if self.segregant_prob is not None else None,
                 'transconjugant_rate':  (self.transconjugant_rate if self.transconjugant_rate.ndim < 2 else self.transconjugant_rate[type_idx]) if self.transconjugant_rate is not None else None,
                 'parent_indices':       self.parent_indices[type_idx] }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder_types(self, order=None):
        type_order   = np.argsort(self.lineageIDs) if order is None else order
        if(len(type_order) < self.num_types):
            utils.error("Error in TypeSet.reorder_types(): The ordering provided has fewer indices than types.")
        #----------------------------------
        self._traits                 = self._traits.reorder(type_order)
        self._consumption_rate       = self._consumption_rate.reorder(type_order)     if isinstance(self._consumption_rate, utils.ExpandableArray)    else self._consumption_rate
        self._carrying_capacity      = self._carrying_capacity.reorder(type_order)    if isinstance(self._carrying_capacity, utils.ExpandableArray)   else self._carrying_capacity
        self._energy_passthru        = self._energy_passthru.reorder(type_order)      if isinstance(self._energy_passthru, utils.ExpandableArray)     else self._energy_passthru
        self._growth_factor          = self._growth_factor.reorder(type_order)        if isinstance(self._growth_factor, utils.ExpandableArray)       else self._growth_factor
        self._cost_baseline          = self._cost_baseline.reorder(type_order)        if isinstance(self._cost_baseline, utils.ExpandableArray)       else self._cost_baseline
        self._cost_trait             = self._cost_trait.reorder(type_order)           if isinstance(self._cost_trait, utils.ExpandableArray)          else self._cost_trait
        self._mutation_prob          = self._mutation_prob.reorder(type_order)        if isinstance(self._mutation_prob, utils.ExpandableArray)       else self._mutation_prob
        self._mutant_prob            = self._mutant_prob.reorder(type_order)          if isinstance(self._mutant_prob, utils.ExpandableArray)         else self._mutant_prob
        self._segregation_prob       = self._segregation_prob.reorder(type_order)     if isinstance(self._segregation_prob, utils.ExpandableArray)    else self._segregation_prob
        self._segregant_prob         = self._segregant_prob.reorder(type_order)       if isinstance(self._segregant_prob, utils.ExpandableArray)      else self._segregant_prob
        self._transfer_donor_rate    = self._transfer_donor_rate.reorder(type_order)  if isinstance(self._transfer_donor_rate, utils.ExpandableArray) else self._transfer_donor_rate
        self._transfer_recip_rate    = self._transfer_recip_rate.reorder(type_order)  if isinstance(self._transfer_recip_rate, utils.ExpandableArray) else self._transfer_recip_rate
        self._transconjugant_rate    = self._transconjugant_rate.reorder(type_order)  if isinstance(self._transconjugant_rate, utils.ExpandableArray) else self._transconjugant_rate
        self._energy_costs           = None # reset to recalculate upon next reference
        self._typeIDs                = np.array(self._typeIDs)[type_order].tolist() if self._typeIDs is not None else None
        self._lineageIDs             = np.array(self._lineageIDs)[type_order].tolist() if self._lineageIDs is not None else None
        self._mutant_indices         = [self._mutant_indices[i] for i in type_order] if self._mutant_indices is not None else None # Rows may have unequal lengths, so keep as list of lists (not 2d array)
        self._segregant_indices      = [self._segregant_indices[i] for i in type_order] if self._segregant_indices is not None else None # Rows may have unequal lengths, so keep as list of lists (not 2d array)
        self._transconjugant_indices = [self._transconjugant_indices[i] for i in type_order] if self._transconjugant_indices is not None else None # Rows may have unequal lengths, so keep as list of lists (not 2d array)
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
