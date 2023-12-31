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
                       mutation_rate        = 1e-9,
                       segregation_rate     = 0,
                       transfer_donor_rate  = 0,
                       transfer_recip_rate  = 0,

                       cost_interaction     = None,
                       cost_landscape       = None,

                       creation_rate        = None,
                       segregation_linkage  = None,
                       transfer_linkage     = None,
                       parent_indices       = None,
                       lineageIDs           = None,
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
        self._consumption_rate    = consumption_rate if isinstance(consumption_rate, utils.SystemParameter) else utils.SystemParameter(values=consumption_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True)
        # self._consumption_rate  = self.preprocess_params(consumption_rate,  has_trait_dim=True)
        self._carrying_capacity   = carrying_capacity if isinstance(carrying_capacity, utils.SystemParameter) else utils.SystemParameter(values=carrying_capacity, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True)
        # self._carrying_capacity = self.preprocess_params(carrying_capacity, has_trait_dim=True)
        self._energy_passthru     = energy_passthru if isinstance(energy_passthru, utils.SystemParameter) else utils.SystemParameter(values=energy_passthru, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True)
        # self._energy_passthru   = self.preprocess_params(energy_passthru,   has_trait_dim=True)
        self._growth_factor     = growth_factor if isinstance(growth_factor, utils.SystemParameter) else utils.SystemParameter(values=growth_factor, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=False)
        # self._growth_factor       = self.preprocess_params(growth_factor,     has_trait_dim=False)
        self._cost_baseline     = cost_baseline if isinstance(cost_baseline, utils.SystemParameter) else utils.SystemParameter(values=cost_baseline, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=False)
        # self._cost_baseline       = self.preprocess_params(cost_baseline,     has_trait_dim=False)
        self._cost_trait        = cost_trait if (cost_trait is None or isinstance(cost_trait, utils.SystemParameter)) else utils.SystemParameter(values=cost_trait, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=False)
        # self._cost_trait          = self.preprocess_params(cost_trait,        has_trait_dim=True) if cost_trait is not None else None
        self._cost_interaction    = utils.reshape(cost_interaction, shape=(self.num_traits, self.num_traits)) if cost_interaction is not None else None
        self._cost_landscape      = cost_landscape
        self._mutation_rate       = mutation_rate if isinstance(mutation_rate, utils.SystemParameter) else utils.SystemParameter(values=mutation_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True)
        self._segregation_rate    = segregation_rate if isinstance(segregation_rate, utils.SystemParameter) else utils.SystemParameter(values=segregation_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True)
        self._transfer_donor_rate = transfer_donor_rate if isinstance(transfer_donor_rate, utils.SystemParameter) else utils.SystemParameter(values=transfer_donor_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True)
        self._transfer_recip_rate = transfer_recip_rate if isinstance(transfer_recip_rate, utils.SystemParameter) else utils.SystemParameter(values=transfer_recip_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True)
        # self._mutation_rate       = self.preprocess_params(mutation_rate,       has_trait_dim=True)
        # self._segregation_rate    = self.preprocess_params(segregation_rate,    has_trait_dim=True)
        # self._transfer_donor_rate = self.preprocess_params(transfer_donor_rate, has_trait_dim=True)
        # self._transfer_recip_rate = self.preprocess_params(transfer_recip_rate, has_trait_dim=True)
        self._segregation_linkage = segregation_linkage
        self._transfer_linkage    = transfer_linkage

        self._creation_rate       = utils.treat_as_list(creation_rate) if creation_rate is not None else None # [None for i in range(self.num_types)]
        # = self.preprocess_params(creation_rate, has_trait_dim=False) if creation_rate is not None else None


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
        return self._consumption_rate.values()
        # return TypeSet.get_array(self._consumption_rate)

    # @consumption_rate.setter
    # def consumption_rate(self, vals):
    #     self._consumption_rate = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def carrying_capacity(self):
        return self._consumption_rate.values()
        # return TypeSet.get_array(self._carrying_capacity)

    # @carrying_capacity.setter
    # def carrying_capacity(self, vals):
    #     self._carrying_capacity = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def energy_passthru(self):
        return self._energy_passthru.values()
        # return TypeSet.get_array(self._energy_passthru)

    # @energy_passthru.setter
    # def energy_passthru(self, vals):
    #     self._energy_passthru = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def growth_factor(self):
        return self._growth_factor.values()
        # return TypeSet.get_array(self._growth_factor)

    # @growth_factor.setter
    # def growth_factor(self, vals):
    #     self._growth_factor = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def cost_baseline(self):
        return self._cost_baseline.values()
        # return TypeSet.get_array(self._cost_baseline)

    # @cost_baseline.setter
    # def cost_baseline(self, vals):
    #     self._cost_baseline = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def cost_trait(self):
        return self._cost_trait.values()
        # return TypeSet.get_array(self._cost_trait)

    # @cost_trait.setter
    # def cost_trait(self, vals):
    #     self._cost_trait = self.preprocess_params(vals, has_trait_dim=True)

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
        return (self.cost_baseline.ravel() if self.cost_baseline.ndim == 2 else self.cost_baseline)  # TODO can this be simplified?
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
    def mutation_rate(self):
        return self._mutation_rate.values()
        # return TypeSet.get_array(self._mutation_rate)

    # @mutation_rate.setter
    # def mutation_rate(self, vals):
    #     self._mutation_rate = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def creation_rate(self):
        # return TypeSet.get_array(self._creation_rate).ravel() if self._creation_rate is not None else None
        return np.array(self._creation_rate) if self._creation_rate is not None else None

    @property
    def segregation_rate(self):
        return self._segregation_rate.values()
        # return TypeSet.get_array(self._segregation_rate)

    # @segregation_rate.setter
    # def segregation_rate(self, vals):
    #     self._segregation_rate = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def transfer_donor_rate(self):
        return self._transfer_donor_rate.values()
        # return TypeSet.get_array(self._transfer_donor_rate)

    # @transfer_donor_rate.setter
    # def transfer_donor_rate(self, vals):
    #     self._transfer_donor_rate = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def transfer_recip_rate(self):
        return self._transfer_recip_rate.values()
        # return TypeSet.get_array(self._transfer_recip_rate)

    # @transfer_recip_rate.setter
    # def transfer_recip_rate(self, vals):
    #     self._transfer_recip_rate = self.preprocess_params(vals, has_trait_dim=True)
    
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
                    # elif(np.all(arr == arr[0]) and not force_expandable_array): # all elements equal
                    #     return arr[0] # single val as scalar
                    else:
                        return utils.ExpandableArray( arr.reshape(self.num_types, 1) )
            elif(arr.ndim == 2):
                if(arr.shape[0] == self.num_types and arr.shape[1] == 1):
                    return utils.ExpandableArray( arr ) # as is
        #----------------------------------
        # If none of the above conditions met (hasn't returned by now):
        if(has_trait_dim):
            print(arr)
            utils.error(f"Error in TypeSet.preprocess_params(): input with shape {arr.shape} does not correspond to the number of types ({self.num_types}) and/or traits ({self.num_traits}).")
        else:
            print(arr)
            utils.error(f"Error in TypeSet.preprocess_params(): input with shape {arr.shape} does not correspond to the number of types ({self.num_types}) (has_trait_dim is {has_trait_dim}).")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_mutant_set(self, type_index=None, update_mutant_indices=True):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.traits.shape[0])
        #----------------------------------
        traits_mut              = []
        consumption_rate_mut    = None
        # consumption_rate_mut  = [] if self.consumption_rate.ndim == 2 else self.consumption_rate
        carrying_capacity_mut   = None
        # carrying_capacity_mut = [] if self.carrying_capacity.ndim == 2 else self.carrying_capacity
        energy_passthru_mut     = None
        # energy_passthru_mut   = [] if self.energy_passthru.ndim == 2 else self.energy_passthru
        growth_factor_mut     = None
        # growth_factor_mut       = [] if self.growth_factor.ndim == 2 else self.growth_factor
        cost_baseline_mut     = None
        # cost_baseline_mut       = [] if self.cost_baseline.ndim == 2 else self.cost_baseline
        cost_trait_mut     = None
        # cost_trait_mut          = [] if self.cost_trait.ndim == 2 else self.cost_trait
        mutation_rate_mut       = None
        segregation_rate_mut    = None
        transfer_donor_rate_mut = None
        transfer_recip_rate_mut = None
        # mutation_rate_mut       = [] if self.mutation_rate.ndim == 2 else self.mutation_rate
        # segregation_rate_mut    = [] if self.segregation_rate.ndim == 2 else self.segregation_rate
        # transfer_donor_rate_mut = [] if self.transfer_donor_rate.ndim == 2 else self.transfer_donor_rate
        # transfer_recip_rate_mut = [] if self.transfer_recip_rate.ndim == 2 else self.transfer_recip_rate
        parent_indices_mut      = []
        creation_rate_mut       = []
        mutant_indices          = []
        #----------------------------------
        for p, parent_idx in enumerate(type_idx):
            mutation_rate_p = self._mutation_rate.get_type(parent_idx, values_only=True)
            # mutation_rate_p = self.mutation_rate[parent_idx] if self.mutation_rate.ndim == 2 else self.mutation_rate
            mutant_indices.append([])
            if(np.any(mutation_rate_p > 0)):
                for i in (np.where(mutation_rate_p > 0)[0] if mutation_rate_p.ndim == 1 else range(self.traits.shape[1])):
                    traits_mut.append(self.traits[parent_idx] ^ [0 if j!=i else 1 for j in range(self.traits.shape[1])])
                    # - - - - -
                    consumption_rate_mut = utils.SystemParameter.combine(consumption_rate_mut, self._consumption_rate.get_type(parent_idx))
                    # if(self.consumption_rate.ndim == 2):    consumption_rate_mut.append(self.consumption_rate[parent_idx])
                    carrying_capacity_mut = utils.SystemParameter.combine(carrying_capacity_mut, self._carrying_capacity.get_type(parent_idx))
                    # if(self.carrying_capacity.ndim == 2):   carrying_capacity_mut.append(self.carrying_capacity[parent_idx])
                    energy_passthru_mut = utils.SystemParameter.combine(energy_passthru_mut, self._energy_passthru.get_type(parent_idx))
                    # if(self.energy_passthru.ndim == 2):     energy_passthru_mut.append(self.energy_passthru[parent_idx])
                    growth_factor_mut = utils.SystemParameter.combine(growth_factor_mut, self._growth_factor.get_type(parent_idx))
                    # if(self.growth_factor.ndim == 2):       growth_factor_mut.append(self.growth_factor[parent_idx])
                    cost_baseline_mut = utils.SystemParameter.combine(cost_baseline_mut, self._cost_baseline.get_type(parent_idx))
                    # if(self.cost_baseline.ndim == 2):       cost_baseline_mut.append(self.cost_baseline[parent_idx])
                    cost_trait_mut = utils.SystemParameter.combine(cost_trait_mut, self._cost_trait.get_type(parent_idx))
                    # if(self.cost_trait.ndim == 2):          cost_trait_mut.append(self.cost_trait[parent_idx])
                    mutation_rate_mut       = utils.SystemParameter.combine(mutation_rate_mut, self._mutation_rate.get_type(parent_idx))
                    segregation_rate_mut    = utils.SystemParameter.combine(segregation_rate_mut, self._segregation_rate.get_type(parent_idx))
                    transfer_donor_rate_mut = utils.SystemParameter.combine(transfer_donor_rate_mut, self._transfer_donor_rate.get_type(parent_idx))
                    transfer_recip_rate_mut = utils.SystemParameter.combine(transfer_recip_rate_mut, self._transfer_recip_rate.get_type(parent_idx))
                    # if(self.mutation_rate.ndim == 2):       mutation_rate_mut.append(self.mutation_rate[parent_idx])
                    # if(self.segregation_rate.ndim == 2):    segregation_rate_mut.append(self.segregation_rate[parent_idx])
                    # if(self.transfer_donor_rate.ndim == 2): transfer_donor_rate_mut.append(self.transfer_donor_rate[parent_idx])
                    # if(self.transfer_recip_rate.ndim == 2): transfer_recip_rate_mut.append(self.transfer_recip_rate[parent_idx])
                    # - - - - -
                    creation_rate_mut.append(mutation_rate_p[i] if mutation_rate_p.ndim == 1 else mutation_rate_p)
                    # - - - - -
                    parent_indices_mut.append(parent_idx)
                    # - - - - -
                    mutant_indices[p].append(len(traits_mut)-1)
        #----------------------------------
        mutant_set = TypeSet(traits=traits_mut, consumption_rate=consumption_rate_mut, carrying_capacity=carrying_capacity_mut, energy_passthru=energy_passthru_mut, growth_factor=growth_factor_mut,
                             cost_baseline=cost_baseline_mut, cost_trait=cost_trait_mut, cost_interaction=self.cost_interaction, cost_landscape=self.cost_landscape,
                             mutation_rate=mutation_rate_mut, segregation_rate=segregation_rate_mut, transfer_donor_rate=transfer_donor_rate_mut, transfer_recip_rate=transfer_recip_rate_mut,
                             creation_rate=creation_rate_mut, parent_indices=parent_indices_mut, binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
        #----------------------------------
        if(update_mutant_indices):
            self._mutant_indices = mutant_indices
        #----------------------------------
        return mutant_set

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_segregant_set(self, type_index=None, update_segregant_indices=True):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.traits.shape[0])
        #----------------------------------
        traits_seg              = []
        consumption_rate_seg    = None
        # consumption_rate_seg    = [] if self.consumption_rate.ndim == 2 else self.consumption_rate
        carrying_capacity_seg    = None
        # carrying_capacity_seg   = [] if self.carrying_capacity.ndim == 2 else self.carrying_capacity
        energy_passthru_seg    = None
        # energy_passthru_seg     = [] if self.energy_passthru.ndim == 2 else self.energy_passthru
        growth_factor_seg    = None
        # growth_factor_seg       = [] if self.growth_factor.ndim == 2 else self.growth_factor
        cost_baseline_seg    = None
        # cost_baseline_seg       = [] if self.cost_baseline.ndim == 2 else self.cost_baseline
        cost_trait_seg    = None
        # cost_trait_seg          = [] if self.cost_trait.ndim == 2 else self.cost_trait
        mutation_rate_seg       = None
        segregation_rate_seg    = None
        transfer_donor_rate_seg = None
        transfer_recip_rate_seg = None
        # mutation_rate_seg       = [] if self.mutation_rate.ndim == 2 else self.mutation_rate
        # segregation_rate_seg    = [] if self.segregation_rate.ndim == 2 else self.segregation_rate
        # transfer_donor_rate_seg = [] if self.transfer_donor_rate.ndim == 2 else self.transfer_donor_rate
        # transfer_recip_rate_seg = [] if self.transfer_recip_rate.ndim == 2 else self.transfer_recip_rate
        parent_indices_seg      = []
        creation_rate_seg       = []
        segregant_indices       = []
        #----------------------------------
        for p, parent_idx in enumerate(type_idx):
            segregation_rate_p = self._segregation_rate.get_type(parent_idx, values_only=True)
            # segregation_rate_p = self.segregation_rate[parent_idx] if self.segregation_rate.ndim == 2 else self.segregation_rate
            segregant_indices.append([])
            if(np.any(segregation_rate_p > 0)):
                for i in (np.where(segregation_rate_p > 0)[0] if segregation_rate_p.ndim == 1 else range(self.traits.shape[1])):
                    if(self.traits[parent_idx][i] != 0):
                        _traits_seg = self.traits[parent_idx].copy()
                        _traits_seg[i] = 0
                        if(self._segregation_linkage is not None and i in self._segregation_linkage):
                            _traits_seg[self._segregation_linkage[i]] = 0
                        traits_seg.append(_traits_seg)
                        # - - - - -
                        consumption_rate_seg = utils.SystemParameter.combine(consumption_rate_seg, self._consumption_rate.get_type(parent_idx))
                        # if(self.consumption_rate.ndim == 2):    consumption_rate_seg.append(self.consumption_rate[parent_idx])
                        carrying_capacity_seg = utils.SystemParameter.combine(carrying_capacity_seg, self._carrying_capacity.get_type(parent_idx))
                        # if(self.carrying_capacity.ndim == 2):   carrying_capacity_seg.append(self.carrying_capacity[parent_idx])
                        energy_passthru_seg = utils.SystemParameter.combine(energy_passthru_seg, self._energy_passthru.get_type(parent_idx))
                        # if(self.energy_passthru.ndim == 2):     energy_passthru_seg.append(self.energy_passthru[parent_idx])
                        growth_factor_seg = utils.SystemParameter.combine(growth_factor_seg, self._growth_factor.get_type(parent_idx))
                        # if(self.growth_factor.ndim == 2):       growth_factor_seg.append(self.growth_factor[parent_idx])
                        cost_baseline_seg = utils.SystemParameter.combine(cost_baseline_seg, self._cost_baseline.get_type(parent_idx))
                        # if(self.cost_baseline.ndim == 2):       cost_baseline_seg.append(self.cost_baseline[parent_idx])
                        cost_trait_seg = utils.SystemParameter.combine(cost_trait_seg, self._cost_trait.get_type(parent_idx))
                        # if(self.cost_trait.ndim == 2):          cost_trait_seg.append(self.cost_trait[parent_idx])
                        mutation_rate_seg = utils.SystemParameter.combine(mutation_rate_seg, self._mutation_rate.get_type(parent_idx))
                        segregation_rate_seg = utils.SystemParameter.combine(segregation_rate_seg, self._segregation_rate.get_type(parent_idx))
                        transfer_donor_rate_seg = utils.SystemParameter.combine(transfer_donor_rate_seg, self._transfer_donor_rate.get_type(parent_idx))
                        transfer_recip_rate_seg = utils.SystemParameter.combine(transfer_recip_rate_seg, self._transfer_recip_rate.get_type(parent_idx))
                        # if(self.mutation_rate.ndim == 2):       mutation_rate_seg.append(self.mutation_rate[parent_idx])
                        # if(self.segregation_rate.ndim == 2):    segregation_rate_seg.append(self.segregation_rate[parent_idx])
                        # if(self.transfer_donor_rate.ndim == 2): transfer_donor_rate_seg.append(self.transfer_donor_rate[parent_idx])
                        # if(self.transfer_recip_rate.ndim == 2): transfer_recip_rate_seg.append(self.transfer_recip_rate[parent_idx])
                        # - - - - -
                        creation_rate_seg.append(segregation_rate_p[i] if segregation_rate_p.ndim == 1 else segregation_rate_p)
                        # - - - - -
                        parent_indices_seg.append(parent_idx)
                        # - - - - -
                        segregant_indices[p].append(len(traits_seg)-1)
        #----------------------------------
        segregant_set = TypeSet(traits=traits_seg, consumption_rate=consumption_rate_seg, carrying_capacity=carrying_capacity_seg, energy_passthru=energy_passthru_seg, growth_factor=growth_factor_seg,
                                 cost_baseline=cost_baseline_seg, cost_trait=cost_trait_seg, cost_interaction=self.cost_interaction, cost_landscape=self.cost_landscape,
                                 mutation_rate=mutation_rate_seg, segregation_rate=segregation_rate_seg, transfer_donor_rate=transfer_donor_rate_seg, transfer_recip_rate=transfer_recip_rate_seg,
                                 creation_rate=creation_rate_seg, parent_indices=parent_indices_seg, binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
        #----------------------------------
        if(update_segregant_indices):
            self._segregant_indices = segregant_indices
        #----------------------------------
        return segregant_set

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, type_set=None, traits=None, consumption_rate=None, carrying_capacity=None, energy_passthru=None, growth_factor=None, cost_baseline=None, cost_trait=None,
                 mutation_rate=None, segregation_rate=None, transfer_donor_rate=None, transfer_recip_rate=None, creation_rate=None,
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
            # TODO: Why do we create this TypeSet? Is it just to preprocess all of the passed-in parameter values? If so, we can skip making this TypeSet and just handle SystemParameter objs for the param values.
            added_type_set = TypeSet(traits=traits if traits is not None else self.traits[ref_type_idx],
                                         consumption_rate=consumption_rate if consumption_rate is not None else self._consumption_rate.get_type(ref_type_idx), # TODO I'm not sure this is the right thing to do, should we even be creating this added TypeSet?
                                         # consumption_rate=consumption_rate if consumption_rate is not None else self.consumption_rate[ref_type_idx],
                                         carrying_capacity=carrying_capacity if carrying_capacity is not None else self._carrying_capacity.get_type(ref_type_idx),
                                         # carrying_capacity=carrying_capacity if carrying_capacity is not None else self.carrying_capacity[ref_type_idx],
                                         energy_passthru=energy_passthru if energy_passthru is not None else self._energy_passthru.get_type(ref_type_idx),
                                         # energy_passthru=energy_passthru if energy_passthru is not None else self.energy_passthru[ref_type_idx],
                                         growth_factor=growth_factor if growth_factor is not None else self._growth_factor.get_type(ref_type_idx),
                                         # growth_factor=growth_factor if growth_factor is not None else self.growth_factor[ref_type_idx],
                                         cost_baseline=cost_baseline if cost_baseline is not None else self._cost_baseline.get_type(ref_type_idx),
                                         # cost_baseline=cost_baseline if cost_baseline is not None else self.cost_baseline[ref_type_idx],
                                         cost_trait=cost_trait if cost_trait is not None else self._cost_trait.get_type(ref_type_idx),
                                         # cost_trait=cost_trait if cost_trait is not None else self.cost_trait[ref_type_idx],
                                         mutation_rate=mutation_rate if mutation_rate is not None else self._mutation_rate.get_type(ref_type_idx),
                                         segregation_rate=segregation_rate if segregation_rate is not None else self._segregation_rate.get_type(ref_type_idx),
                                         transfer_donor_rate=transfer_donor_rate if transfer_donor_rate is not None else self._transfer_donor_rate.get_type(ref_type_idx),
                                         transfer_recip_rate=transfer_recip_rate if transfer_recip_rate is not None else self._transfer_recip_rate.get_type(ref_type_idx),
                                         # mutation_rate=mutation_rate if mutation_rate is not None else self.mutation_rate[ref_type_idx],
                                         # segregation_rate=segregation_rate if segregation_rate is not None else self.segregation_rate[ref_type_idx],
                                         # transfer_donor_rate=transfer_donor_rate if transfer_donor_rate is not None else self.transfer_donor_rate[ref_type_idx],
                                         # transfer_recip_rate=transfer_recip_rate if transfer_recip_rate is not None else self.transfer_recip_rate[ref_type_idx],
                                         creation_rate=creation_rate if creation_rate is not None else self.creation_rate[ref_type_idx],
                                     )
        # Check that the type set dimensions match the system dimensions:
        if(self.num_traits != added_type_set.num_traits): 
            utils.error(f"Error in TypeSet add_type(): The number of traits for added types ({added_type_set.num_traits}) does not match the number of type set traits ({self.num_traits}).")
        #----------------------------------
        added_type_indices = list(range(self.num_types, self.num_types+added_type_set.num_types))
        #----------------------------------
        self._traits = self._traits.add(added_type_set.traits)
        self._consumption_rate    = utils.SystemParameter.combine(self._consumption_rate, added_type_set._consumption_rate) # TODO: Should the 2nd arg - the added param set - be passed in as a SystemParameter or as a numpy array? Currently treating as a SystemParameter
        # self._consumption_rate    = self._consumption_rate.add(added_type_set.consumption_rate)       if isinstance(self._consumption_rate, utils.ExpandableArray) else self._consumption_rate
        self._carrying_capacity   = utils.SystemParameter.combine(self._carrying_capacity, added_type_set._carrying_capacity)
        # self._carrying_capacity   = self._carrying_capacity.add(added_type_set.carrying_capacity)     if isinstance(self._carrying_capacity, utils.ExpandableArray) else self._carrying_capacity
        self._energy_passthru   = utils.SystemParameter.combine(self._energy_passthru, added_type_set._energy_passthru)
        # self._energy_passthru     = self._energy_passthru.add(added_type_set.energy_passthru)         if isinstance(self._energy_passthru, utils.ExpandableArray) else self._energy_passthru
        self._growth_factor   = utils.SystemParameter.combine(self._growth_factor, added_type_set._growth_factor)
        # self._growth_factor       = self._growth_factor.add(added_type_set.growth_factor)             if isinstance(self._growth_factor, utils.ExpandableArray) else self._growth_factor
        self._cost_baseline   = utils.SystemParameter.combine(self._cost_baseline, added_type_set._cost_baseline)
        # self._cost_baseline       = self._cost_baseline.add(added_type_set.cost_baseline)             if isinstance(self._cost_baseline, utils.ExpandableArray) else self._cost_baseline
        self._cost_trait   = utils.SystemParameter.combine(self._cost_trait, added_type_set._cost_trait)
        # self._cost_trait          = self._cost_trait.add(added_type_set.cost_trait)                   if isinstance(self._cost_trait, utils.ExpandableArray) else self._cost_trait
        self._mutation_rate    = utils.SystemParameter.combine(self._mutation_rate, added_type_set._mutation_rate)
        self._segregation_rate    = utils.SystemParameter.combine(self._segregation_rate, added_type_set._segregation_rate)
        self._transfer_donor_rate    = utils.SystemParameter.combine(self._transfer_donor_rate, added_type_set._transfer_donor_rate)
        self._transfer_recip_rate    = utils.SystemParameter.combine(self._transfer_recip_rate, added_type_set._transfer_recip_rate)
        # self._mutation_rate       = self._mutation_rate.add(added_type_set.mutation_rate)             if isinstance(self._mutation_rate, utils.ExpandableArray) else self._mutation_rate
        # self._segregation_rate    = self._segregation_rate.add(added_type_set.segregation_rate)       if isinstance(self._segregation_rate, utils.ExpandableArray) else self._segregation_rate
        # self._transfer_donor_rate = self._transfer_donor_rate.add(added_type_set.transfer_donor_rate) if isinstance(self._transfer_donor_rate, utils.ExpandableArray) else self._transfer_donor_rate
        # self._transfer_recip_rate = self._transfer_recip_rate.add(added_type_set.transfer_recip_rate) if isinstance(self._transfer_recip_rate, utils.ExpandableArray) else self._transfer_recip_rate
        # self._creation_rate       = self._creation_rate.add(added_type_set.creation_rate)             if isinstance(self._creation_rate, utils.ExpandableArray) else self._creation_rate
        #----------------------------------
        self._creation_rate = [rate for ratelist in [self._creation_rate, added_type_set._creation_rate] for rate in ratelist] if self._creation_rate is not None else None
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
                        consumption_rate    = self._consumption_rate.get_type(type_idx),
                        # consumption_rate    = self.consumption_rate[type_idx]    if self.consumption_rate.ndim == 2    else self.consumption_rate,
                        carrying_capacity   = self._carrying_capacity.get_type(type_idx),
                        # carrying_capacity   = self.carrying_capacity[type_idx]   if self.carrying_capacity.ndim == 2   else self.carrying_capacity,
                        energy_passthru   = self._energy_passthru.get_type(type_idx),
                        # energy_passthru     = self.energy_passthru[type_idx]     if self.energy_passthru.ndim == 2     else self.energy_passthru,
                        growth_factor   = self._growth_factor.get_type(type_idx),
                        # growth_factor       = self.growth_factor[type_idx]       if self.growth_factor.ndim == 2       else self.growth_factor,
                        cost_baseline   = self._cost_baseline.get_type(type_idx),
                        # cost_baseline       = self.cost_baseline[type_idx]       if self.cost_baseline.ndim == 2       else self.cost_baseline,
                        cost_trait   = self._cost_trait.get_type(type_idx),
                        # cost_trait          = self.cost_trait[type_idx]          if self.cost_trait.ndim == 2          else self.cost_trait,
                        cost_interaction    = self.cost_interaction,
                        cost_landscape      = self.cost_landscape,
                        mutation_rate       = self._mutation_rate.get_type(type_idx),
                        segregation_rate    = self._segregation_rate.get_type(type_idx),
                        transfer_donor_rate = self._transfer_donor_rate.get_type(type_idx),
                        transfer_recip_rate = self._transfer_recip_rate.get_type(type_idx),
                        # mutation_rate       = self.mutation_rate[type_idx]       if self.mutation_rate.ndim == 2       else self.mutation_rate,
                        # segregation_rate    = self.segregation_rate[type_idx]    if self.segregation_rate.ndim == 2    else self.segregation_rate,
                        # transfer_donor_rate = self.transfer_donor_rate[type_idx] if self.transfer_donor_rate.ndim == 2 else self.transfer_donor_rate,
                        # transfer_recip_rate = self.transfer_recip_rate[type_idx] if self.transfer_recip_rate.ndim == 2 else self.transfer_recip_rate,
                        creation_rate       = self.creation_rate[type_idx]       if self.creation_rate is not None else None,
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

    def get_dynamics_params(self, type_index=None, values_only=False):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.num_types)
        # print("get_dynamics_params @@@", type_index, "-> type_idx", type_idx)
        #----------------------------------
        return { 'num_types':            len(type_idx),
                 'traits':               self.traits[type_idx],
                 'consumption_rate':     self._consumption_rate.get_type(type_idx, values_only),
                 # 'consumption_rate':     self.consumption_rate  if self.consumption_rate.ndim < 2  else self.consumption_rate[type_idx],
                 'carrying_capacity':    self._carrying_capacity.get_type(type_idx, values_only),
                 # 'carrying_capacity':    self.carrying_capacity if self.carrying_capacity.ndim < 2 else self.carrying_capacity[type_idx],
                 'energy_passthru':    self._energy_passthru.get_type(type_idx, values_only),
                 # 'energy_passthru':      self.energy_passthru if self.energy_passthru.ndim < 2 else self.energy_passthru[type_idx],
                 'growth_factor':    self._growth_factor.get_type(type_idx, values_only),
                 # 'growth_factor':        self.growth_factor if self.growth_factor.ndim < 2 else self.growth_factor[type_idx],
                 'energy_costs':         self.energy_costs[type_idx],
                 'creation_rate':        self.creation_rate[type_idx] if self.creation_rate is not None else None,
                 'parent_indices':       self.parent_indices[type_idx] }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder_types(self, order=None):
        type_order   = np.argsort(self.lineageIDs) if order is None else order
        if(len(type_order) < self.num_types):
            utils.error("Error in TypeSet.reorder_types(): The ordering provided has fewer indices than types.")
        #----------------------------------
        self._traits                 = self._traits.reorder(type_order)
        # TODO: The below line hasn't been updated to use SystemParameter yet. Probably need a reorder() inside SystemParameter.
        # TODO: I think this is the last thing to update to use SystemParameter for consumption_rate (although none of the updates have been tested yet).
        self._consumption_rate.reorder(type_order)
        # self._consumption_rate       = self._consumption_rate.reorder(type_order)     if isinstance(self._consumption_rate, utils.ExpandableArray)    else self._consumption_rate
        self._carrying_capacity.reorder(type_order)
        # self._carrying_capacity      = self._carrying_capacity.reorder(type_order)    if isinstance(self._carrying_capacity, utils.ExpandableArray)   else self._carrying_capacity
        self._energy_passthru.reorder(type_order)
        # self._energy_passthru        = self._energy_passthru.reorder(type_order)      if isinstance(self._energy_passthru, utils.ExpandableArray)     else self._energy_passthru
        self._growth_factor.reorder(type_order)
        # self._growth_factor          = self._growth_factor.reorder(type_order)        if isinstance(self._growth_factor, utils.ExpandableArray)       else self._growth_factor
        self._cost_baseline.reorder(type_order)
        # self._cost_baseline          = self._cost_baseline.reorder(type_order)        if isinstance(self._cost_baseline, utils.ExpandableArray)       else self._cost_baseline
        self._cost_trait.reorder(type_order)
        # self._cost_trait             = self._cost_trait.reorder(type_order)           if isinstance(self._cost_trait, utils.ExpandableArray)          else self._cost_trait
        self._mutation_rate.reorder(type_order)
        self._segregation_rate.reorder(type_order)
        self._transfer_donor_rate.reorder(type_order)
        self._transfer_recip_rate.reorder(type_order)
        # self._mutation_rate          = self._mutation_rate.reorder(type_order)        if isinstance(self._mutation_rate, utils.ExpandableArray)       else self._mutation_rate
        # self._segregation_rate       = self._segregation_rate.reorder(type_order)     if isinstance(self._segregation_rate, utils.ExpandableArray)    else self._segregation_rate
        # self._transfer_donor_rate    = self._transfer_donor_rate.reorder(type_order)  if isinstance(self._transfer_donor_rate, utils.ExpandableArray) else self._transfer_donor_rate
        # self._transfer_recip_rate    = self._transfer_recip_rate.reorder(type_order)  if isinstance(self._transfer_recip_rate, utils.ExpandableArray) else self._transfer_recip_rate
        # self._creation_rate          = self._creation_rate.reorder(type_order)        if isinstance(self._creation_rate, utils.ExpandableArray)       else self._creation_rate
        self._creation_rate          = np.array(self._creation_rate)[type_order].tolist() if self._creation_rate is not None else None
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
