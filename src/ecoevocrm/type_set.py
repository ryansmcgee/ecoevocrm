import numpy as np
import re

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TypeSet():

    def __init__(self, num_types                  = None,
                       num_traits                 = None,

                       traits                     = None,
                       consumption_rate           = 1,
                       carrying_capacity          = 1e9,
                       growth_factor              = 1,
                       energy_passthru            = 0,
                       cost_baseline              = 0,
                       cost_trait                 = 0,
                       mutation_rate              = 1e-9,
                       segregation_rate           = 0,
                       transfer_donor_rate        = 0,
                       transfer_recip_rate        = 0,

                       cost_interaction           = None,
                       cost_landscape             = None,

                       creation_rate              = None,
                       mutant_attributes          = None,
                       segregant_attributes       = None,
                       transconjugant_attributes  = None,
                       segregation_linkage        = None,
                       transfer_linkage           = None,
                       parent_indices             = None,
                       lineageIDs                 = None,
                       binarize_trait_costs       = True,
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
        self._params = {
            'consumption_rate':    consumption_rate if isinstance(consumption_rate, utils.SystemParameter) else utils.SystemParameter(values=consumption_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'carrying_capacity':   carrying_capacity if isinstance(carrying_capacity, utils.SystemParameter) else utils.SystemParameter(values=carrying_capacity, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'energy_passthru':     energy_passthru if isinstance(energy_passthru, utils.SystemParameter) else utils.SystemParameter(values=energy_passthru, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'growth_factor':       growth_factor if isinstance(growth_factor, utils.SystemParameter) else utils.SystemParameter(values=growth_factor, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=False),
            'cost_baseline':       cost_baseline if isinstance(cost_baseline, utils.SystemParameter) else utils.SystemParameter(values=cost_baseline, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=False),
            'cost_trait':          cost_trait if (cost_trait is None or isinstance(cost_trait, utils.SystemParameter)) else utils.SystemParameter(values=cost_trait, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=False),
            'mutation_rate':       mutation_rate if isinstance(mutation_rate, utils.SystemParameter) else utils.SystemParameter(values=mutation_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'segregation_rate':    segregation_rate if isinstance(segregation_rate, utils.SystemParameter) else utils.SystemParameter(values=segregation_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'transfer_donor_rate': transfer_donor_rate if isinstance(transfer_donor_rate, utils.SystemParameter) else utils.SystemParameter(values=transfer_donor_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'transfer_recip_rate': transfer_recip_rate if isinstance(transfer_recip_rate, utils.SystemParameter) else utils.SystemParameter(values=transfer_recip_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
        }

        self._cost_interaction         = utils.reshape(cost_interaction, shape=(self.num_traits, self.num_traits)) if cost_interaction is not None else None
        self._cost_landscape           = cost_landscape

        self._creation_rate            = utils.treat_as_list(creation_rate) if creation_rate is not None else None # [None for i in range(self.num_types)]

        self.mutant_attributes         = mutant_attributes
        self.segregant_attributes      = segregant_attributes
        self.transconjugant_attributes = transconjugant_attributes
        self._segregation_linkage      = segregation_linkage
        self._transfer_linkage         = transfer_linkage


        #----------------------------------
        # Initialize other type properties/metadata:
        #----------------------------------
        self._typeIDs                = None
        self._parent_indices         = utils.treat_as_list(parent_indices) if parent_indices is not None else [None for i in range(self.num_types)]
        self._mutant_indices         = None
        self._segregant_indices      = None
        self._transconjugant_indices = None

        self._lineageIDs = lineageIDs
        self.phylogeny   = {}

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
        return self._params['consumption_rate'].values()

    @property
    def carrying_capacity(self):
        return self._params['carrying_capacity'].values()

    @property
    def energy_passthru(self):
        return self._params['energy_passthru'].values()

    @property
    def growth_factor(self):
        return self._params['growth_factor'].values()

    @property
    def cost_baseline(self):
        return self._params['cost_baseline'].values()

    @property
    def cost_trait(self):
        return self._params['cost_trait'].values()

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
        return self._params['cost_baseline'].values()

    @property
    def cost_trait_bytype(self):
        _traits = self.traits if not self.binarize_trait_costs else (self.traits > 0).astype(int)
        return np.sum(_traits * self.cost_trait, axis=1) if self._params['cost_trait'] is not None else 0
    
    @property
    def cost_interaction_bytype(self):
        _traits = self.traits if not self.binarize_interaction_costs else (self.traits > 0).astype(int)
        return -1 * np.sum(_traits * np.dot(_traits, self.cost_interaction), axis=1) if self._cost_interaction is not None else 0

    @property
    def cost_landscape_bytype(self):
        return [self._cost_landscape[k] for k in self.trait_keys]

    @property
    def mutation_rate(self):
        return self._params['mutation_rate'].values()

    @property
    def segregation_rate(self):
        return self._params['segregation_rate'].values()

    @property
    def transfer_donor_rate(self):
        return self._params['transfer_donor_rate'].values()

    @property
    def transfer_recip_rate(self):
        return self._params['transfer_recip_rate'].values()

    @property
    def creation_rate(self):
        return np.array(self._creation_rate) if self._creation_rate is not None else None

    @property
    def typeIDs(self):
        if(self._typeIDs is None):
            self._typeIDs = np.array(self.assign_type_ids())
        return self._typeIDs

    # @property
    # def trait_keys(self):
    #     return [''.join(str(a) for a in traits_u) for traits_u in (self.traits != 0).astype(int)]

    def get_trait_key(self, traits):
        _traits = np.array(traits)
        if(traits.ndim == 2):
            trait_keys = [''.join(str(a) for a in traits_u) for traits_u in (traits != 0).astype(int)]
            return trait_keys
        else:
            trait_keys = ''.join(str(a) for a in traits)
            return trait_keys

    @property
    def trait_keys(self):
        return self.get_trait_key(self.traits)

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

    def generate_mutant_set(self, type_index=None, update_mutant_indices=True):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.traits.shape[0])
        #----------------------------------
        traits_mut         = []
        params_mut         = {param: None for param in self._params.keys()}
        parent_indices_mut = []
        creation_rate_mut  = []
        mutant_indices     = []
        #----------------------------------
        for p, parent_idx in enumerate(type_idx):
            mutation_rate_p = self._params['mutation_rate'].get_type(parent_idx, values_only=True)
            mutant_indices.append([])
            if(np.any(mutation_rate_p > 0)):
                for i in (np.where(mutation_rate_p > 0)[0] if mutation_rate_p.ndim == 1 else range(self.traits.shape[1])):
                    # traits_mut.append(self.traits[parent_idx] ^ [0 if j!=i else 1 for j in range(self.traits.shape[1])])
                    # # - - - - -
                    # for param, param_vals in params_mut.items():
                    #     params_mut[param] = utils.SystemParameter.combine(params_mut[param], self._params[param].get_type(parent_idx))
                    # - - - - -
                    # mutant_attributes_p = {}
                    # if(self.mutant_attributes is not None):
                    #     for key in self.mutant_attributes.keys():
                    #         if(re.match(key, self.trait_keys[parent_idx])):
                    #             mutant_attributes_p = self.mutant_attributes[key]
                    #             break
                    # # - - - - -
                    # if('traits' in mutant_attributes_p):
                    #     _traits_mut = mutant_attributes_p['traits']
                    # else:
                    #     _traits_mut = self.traits[parent_idx] ^ [0 if j!=i else 1 for j in range(self.traits.shape[1])]
                    # traits_mut.append(_traits_mut)
                    # # - - - - -
                    # for param, param_vals in params_mut.items():
                    #     if(param in mutant_attributes_p):
                    #         params_mut[param] = utils.SystemParameter.combine(params_mut[param], utils.SystemParameter(mutant_attributes_p[param], num_types=1, num_traits=self._params[param].num_traits, force_type_dim=self._params[param].force_type_dim, force_trait_dim=self._params[param].force_trait_dim))
                    #     else:
                    #         params_mut[param] = utils.SystemParameter.combine(params_mut[param], self._params[param].get_type(parent_idx))
                    # - - - - -
                    # Generate the (default) mutant trait profile:
                    _traits_mut = self.traits[parent_idx] ^ [0 if j!=i else 1 for j in range(self.traits.shape[1])]
                    # - - - - -
                    # Check if attributes have been manually specified for this mutant trait profile...
                    mutant_attributes_mut = {}
                    if(self.mutant_attributes is not None):
                        for key in self.mutant_attributes.keys():
                            if(re.match(key, self.get_trait_key(_traits_mut))):
                                mutant_attributes_mut = self.mutant_attributes[key]
                                break
                    # ...overriding the default mutant trait profile, if applicable:
                    if('traits' in mutant_attributes_mut):
                        _traits_mut = mutant_attributes_mut['traits']
                    # Append the mutant trait profile to the list of mutant traits:
                    traits_mut.append(_traits_mut)
                    # - - - - -
                    for param, param_vals in params_mut.items():
                        if(param in mutant_attributes_mut):
                            params_mut[param] = utils.SystemParameter.combine(params_mut[param], utils.SystemParameter(mutant_attributes_mut[param], num_types=1, num_traits=self._params[param].num_traits, force_type_dim=self._params[param].force_type_dim, force_trait_dim=self._params[param].force_trait_dim))
                        else:
                            params_mut[param] = utils.SystemParameter.combine(params_mut[param], self._params[param].get_type(parent_idx))
                    # - - - - -
                    creation_rate_mut.append(mutation_rate_p[i] if mutation_rate_p.ndim == 1 else mutation_rate_p)
                    # - - - - -
                    parent_indices_mut.append(parent_idx)
                    # - - - - -
                    mutant_indices[p].append(len(traits_mut)-1)
        #----------------------------------
        mutant_set = TypeSet(traits=traits_mut, consumption_rate=params_mut['consumption_rate'], carrying_capacity=params_mut['carrying_capacity'], energy_passthru=params_mut['energy_passthru'], growth_factor=params_mut['growth_factor'],
                             cost_baseline=params_mut['cost_baseline'], cost_trait=params_mut['cost_trait'], cost_interaction=self.cost_interaction, cost_landscape=self.cost_landscape,
                             mutation_rate=params_mut['mutation_rate'], segregation_rate=params_mut['segregation_rate'], transfer_donor_rate=params_mut['transfer_donor_rate'], transfer_recip_rate=params_mut['transfer_recip_rate'],
                             mutant_attributes=self.mutant_attributes, segregant_attributes=self.segregant_attributes, transconjugant_attributes=self.transconjugant_attributes,
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
        traits_seg         = []
        params_seg         = {param: None for param in self._params.keys()}
        parent_indices_seg = []
        creation_rate_seg  = []
        segregant_indices  = []
        #----------------------------------
        for p, parent_idx in enumerate(type_idx):
            segregation_rate_p = self._params['segregation_rate'].get_type(parent_idx, values_only=True)
            segregant_indices.append([])
            if(np.any(segregation_rate_p > 0)):
                for i in (np.where(segregation_rate_p > 0)[0] if segregation_rate_p.ndim == 1 else range(self.traits.shape[1])):
                    if(self.traits[parent_idx][i] != 0):
                        # - - - - -
                        # segregant_attributes_p = {}
                        # if(self.segregant_attributes is not None):
                        #     for key in self.segregant_attributes.keys():
                        #         if(re.match(key, self.trait_keys[parent_idx])):
                        #             segregant_attributes_p = self.segregant_attributes[key]
                        #             break
                        # # - - - - -
                        # if('traits' in segregant_attributes_p):
                        #     _traits_seg = segregant_attributes_p['traits']
                        # else:
                        #     _traits_seg = self.traits[parent_idx].copy()
                        #     _traits_seg[i] = 0
                        #     if(self._segregation_linkage is not None and i in self._segregation_linkage):
                        #         _traits_seg[self._segregation_linkage[i]] = 0
                        # traits_seg.append(_traits_seg)
                        # # - - - - -
                        # for param, param_vals in params_seg.items():
                        #     if(param in segregant_attributes_p):
                        #         params_seg[param] = utils.SystemParameter.combine(params_seg[param], utils.SystemParameter(segregant_attributes_p[param], num_types=1, num_traits=self._params[param].num_traits, force_type_dim=self._params[param].force_type_dim, force_trait_dim=self._params[param].force_trait_dim))
                        #     else:
                        #         params_seg[param] = utils.SystemParameter.combine(params_seg[param], self._params[param].get_type(parent_idx))
                        # - - - - -
                        # Generate the (default) segregant trait profile:
                        _traits_seg = self.traits[parent_idx].copy()
                        _traits_seg[i] = 0
                        if(self._segregation_linkage is not None and i in self._segregation_linkage):
                            _traits_seg[self._segregation_linkage[i]] = 0
                        # - - - - -
                        # Check if attributes have been manually specified for this segregant trait profile...
                        segregant_attributes_seg = {}
                        if(self.segregant_attributes is not None):
                            for key in self.segregant_attributes.keys():
                                if(re.match(key, self.get_trait_key(_traits_seg))):
                                    segregant_attributes_seg = self.segregant_attributes[key]
                                    break
                        # ...overriding the default segregant trait profile, if applicable:
                        if('traits' in segregant_attributes_seg):
                            _traits_seg = segregant_attributes_seg['traits']
                        # Append the segregant trait profile to the list of segregant traits:
                        traits_seg.append(_traits_seg)
                        # - - - - -
                        for param, param_vals in params_seg.items():
                            if(param in segregant_attributes_seg):
                                params_seg[param] = utils.SystemParameter.combine(params_seg[param], utils.SystemParameter(segregant_attributes_seg[param], num_types=1, num_traits=self._params[param].num_traits, force_type_dim=self._params[param].force_type_dim, force_trait_dim=self._params[param].force_trait_dim))
                            else:
                                params_seg[param] = utils.SystemParameter.combine(params_seg[param], self._params[param].get_type(parent_idx))
                        # - - - - -
                        creation_rate_seg.append(segregation_rate_p[i] if segregation_rate_p.ndim == 1 else segregation_rate_p)
                        # - - - - -
                        parent_indices_seg.append(parent_idx)
                        # - - - - -
                        segregant_indices[p].append(len(traits_seg)-1)
        #----------------------------------
        segregant_set = TypeSet(traits=traits_seg, consumption_rate=params_seg['consumption_rate'], carrying_capacity=params_seg['carrying_capacity'], energy_passthru=params_seg['energy_passthru'], growth_factor=params_seg['growth_factor'],
                                cost_baseline=params_seg['cost_baseline'], cost_trait=params_seg['cost_trait'], cost_interaction=self.cost_interaction, cost_landscape=self.cost_landscape,
                                mutation_rate=params_seg['mutation_rate'], segregation_rate=params_seg['segregation_rate'], transfer_donor_rate=params_seg['transfer_donor_rate'], transfer_recip_rate=params_seg['transfer_recip_rate'],
                                mutant_attributes=self.mutant_attributes, segregant_attributes=self.segregant_attributes, transconjugant_attributes=self.transconjugant_attributes,
                                creation_rate=creation_rate_seg, parent_indices=parent_indices_seg, binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
        #----------------------------------
        if(update_segregant_indices):
            self._segregant_indices = segregant_indices
        #----------------------------------
        return segregant_set

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, added_type_set):
        if(not isinstance(added_type_set, TypeSet)):
            utils.error(f"Error in TypeSet add_type(): type_set argument expects object of TypeSet type.")
        #----------------------------------
        if(added_type_set.num_types == 0):
            added_type_indices = []
            return added_type_indices
        #----------------------------------
        # Check that the type set dimensions match the system dimensions:
        if(self.num_traits != added_type_set.num_traits): 
            utils.error(f"Error in TypeSet add_type(): The number of traits for added types ({added_type_set.num_traits}) does not match the number of type set traits ({self.num_traits}).")
        #----------------------------------
        added_type_indices = list(range(self.num_types, self.num_types+added_type_set.num_types))
        #----------------------------------
        self._traits = self._traits.add(added_type_set.traits)
        for param, param_vals in self._params.items():
            self._params[param] = utils.SystemParameter.combine(self._params[param], added_type_set._params[param])
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
                        consumption_rate     = self._params['consumption_rate'].get_type(type_idx),
                        carrying_capacity    = self._params['carrying_capacity'].get_type(type_idx),
                        energy_passthru      = self._params['energy_passthru'].get_type(type_idx),
                        growth_factor        = self._params['growth_factor'].get_type(type_idx),
                        cost_baseline        = self._params['cost_baseline'].get_type(type_idx),
                        cost_trait           = self._params['cost_trait'].get_type(type_idx),
                        cost_interaction     = self.cost_interaction,
                        cost_landscape       = self.cost_landscape,
                        mutation_rate        = self._params['mutation_rate'].get_type(type_idx),
                        segregation_rate     = self._params['segregation_rate'].get_type(type_idx),
                        transfer_donor_rate  = self._params['transfer_donor_rate'].get_type(type_idx),
                        transfer_recip_rate  = self._params['transfer_recip_rate'].get_type(type_idx),
                        creation_rate        = self.creation_rate[type_idx] if self.creation_rate is not None else None,
                        mutant_attributes    = self.mutant_attributes,
                        segregant_attributes = self.segregant_attributes,
                        transconjugant_attributes = self.transconjugant_attributes,
                        parent_indices       = self.parent_indices[type_idx],
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
        return { 'num_types':         len(type_idx),
                 'traits':            self.traits[type_idx],
                 'consumption_rate':  self._params['consumption_rate'].get_type(type_idx, values_only),
                 'carrying_capacity': self._params['carrying_capacity'].get_type(type_idx, values_only),
                 'energy_passthru':   self._params['energy_passthru'].get_type(type_idx, values_only),
                 'growth_factor':     self._params['growth_factor'].get_type(type_idx, values_only),
                 'energy_costs':      self.energy_costs[type_idx],
                 'creation_rate':     self.creation_rate[type_idx] if self.creation_rate is not None else None,
                 'parent_indices':    self.parent_indices[type_idx] }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder_types(self, order=None):
        type_order   = np.argsort(self.lineageIDs) if order is None else order
        if(len(type_order) < self.num_types):
            utils.error("Error in TypeSet.reorder_types(): The ordering provided has fewer indices than types.")
        #----------------------------------
        self._traits                 = self._traits.reorder(type_order)
        for param in self._params.keys():
            self._params[param].reorder(type_order)
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
