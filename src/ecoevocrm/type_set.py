import numpy as np
import re

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TypeSet():

    def __init__(self, num_types                  = None,
                       num_traits                 = None,

                       traits                     = None,   # A      | previously sigma
                       consumption_rate           = 1,      # pi     | previously beta
                       carrying_capacity          = 1e9,    # k      | previously kappa
                       growth_factor              = 1,      # w      | previously gamma
                       energy_passthru            = 0,      # p      | previously lamda
                       cost_baseline              = 0,      # xi     | previously xi
                       cost_pertrait              = 0,      # theta  | previously chi
                       cost_perpair               = 0,
                       mutation_rate              = 1e-9,   # m      | previously mu
                       segregation_rate           = 0,      # l
                       transfer_rate_donor        = 0,      # beta
                       transfer_rate_recip        = 0,      # alpha
                       cost_interaction           = None,   # J      | previously J
                       cost_landscape             = None,   # lambda
                       creation_rate              = None,
                       mutant_overrides           = None,
                       segregant_overrides        = None,
                       transconjugant_overrides   = None,
                       segregation_linkage        = None,
                       transfer_linkage           = None,
                       mutation_parent_indices    = None,
                       segregation_parent_indices = None,
                       transfer_donor_indices     = None,
                       transfer_recip_indices     = None,
                       lineageIDs                 = None,
                       lineageID_traits           = None,
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
            'cost_pertrait':       cost_pertrait if (cost_pertrait is None or isinstance(cost_pertrait, utils.SystemParameter)) else utils.SystemParameter(values=cost_pertrait, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=False),
            'cost_perpair':        cost_perpair if isinstance(cost_perpair, utils.SystemParameter) else utils.SystemParameter(values=cost_perpair, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=False),
            'mutation_rate':       mutation_rate if isinstance(mutation_rate, utils.SystemParameter) else utils.SystemParameter(values=mutation_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'segregation_rate':    segregation_rate if isinstance(segregation_rate, utils.SystemParameter) else utils.SystemParameter(values=segregation_rate, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'transfer_rate_donor': transfer_rate_donor if isinstance(transfer_rate_donor, utils.SystemParameter) else utils.SystemParameter(values=transfer_rate_donor, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
            'transfer_rate_recip': transfer_rate_recip if isinstance(transfer_rate_recip, utils.SystemParameter) else utils.SystemParameter(values=transfer_rate_recip, num_types=self.num_types, num_traits=self.num_traits, force_type_dim=False, force_trait_dim=True),
        }

        self._cost_interaction         = utils.reshape(cost_interaction, shape=(self.num_traits, self.num_traits)) if cost_interaction is not None else None
        self._cost_landscape           = cost_landscape

        self._creation_rate            = utils.treat_as_list(creation_rate) if creation_rate is not None else None

        self.mutant_overrides          = mutant_overrides
        self.segregant_overrides       = segregant_overrides
        self.transconjugant_overrides  = transconjugant_overrides

        self._segregation_linkage      = segregation_linkage
        self._transfer_linkage         = transfer_linkage

        #----------------------------------
        # Initialize other type properties/metadata:
        #----------------------------------
        self._typeIDs                        = None
        self._mutation_parent_indices        = utils.treat_as_list(mutation_parent_indices) if mutation_parent_indices is not None else [None for i in range(self.num_types)]
        self._segregation_parent_indices     = utils.treat_as_list(segregation_parent_indices) if segregation_parent_indices is not None else [None for i in range(self.num_types)]
        self._transfer_donor_indices         = utils.treat_as_list(transfer_donor_indices) if transfer_donor_indices is not None else [None for i in range(self.num_types)]
        self._transfer_recip_indices         = utils.treat_as_list(transfer_recip_indices) if transfer_recip_indices is not None else [None for i in range(self.num_types)]
        self._mutant_indices                 = None
        self._segregant_indices              = None
        self._transconjugant_indices_bydonor = None
        self._transconjugant_indices_byrecip = None

        self.phylogeny   = {}
        self._lineageID_traits = lineageID_traits
        if(lineageIDs is not None):
            self._lineageIDs = []
            for i in range(self.num_types):
                self._lineageIDs.append( self.add_type_to_phylogeny(type_index=i, lineage_id=lineageIDs[i]) )
        else:
            self._lineageIDs = None

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
    def cost_pertrait(self):
        return self._params['cost_pertrait'].values()

    @property
    def cost_perpair(self):
        return self._params['cost_perpair'].values()

    @property
    def cost_interaction(self):
        return TypeSet.get_array(self._cost_interaction)

    @property
    def cost_landscape(self):
        return TypeSet.get_array(self._cost_landscape)

    def update_cost_landscape(self, new_landscape_dict):
        self._cost_landscape.update(new_landscape_dict)

    @property
    def energy_costs(self):
        if(self._energy_costs is None):
            costs = 0
            costs += self.cost_baseline_bytype
            costs += self.cost_pertrait_bytype
            costs += self.cost_perpair_bytype
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
    def cost_pertrait_bytype(self):
        _traits = self.traits if not self.binarize_trait_costs else (self.traits > 0).astype(int)
        return np.sum(_traits * self.cost_pertrait, axis=1) if self._params['cost_pertrait'] is not None else 0

    @property
    def cost_perpair_bytype(self):
        # Use binarized traits if requested (mirrors cost_trait_bytype behavior)
        _traits = self.traits if not self.binarize_trait_costs else (self.traits > 0).astype(int)
        n = _traits.sum(axis=1)
        return 0.5 * self.cost_perpair * n * (n - 1)   # α * C(n,2)
    
    @property
    def cost_interaction_bytype(self):
        _traits = self.traits if not self.binarize_interaction_costs else (self.traits > 0).astype(int)
        return -1 * np.sum(_traits * np.dot(_traits, self.cost_interaction), axis=1) if self._cost_interaction is not None else 0

    @property
    def cost_landscape_bytype(self):
        if self._cost_landscape is not None:
            if not hasattr(self, "landscape_patterns"):
                self.landscape_patterns = []
                for trait_pattern, landscape_value in self._cost_landscape.items():
                    bitmask = 0
                    target_bits = 0
                    for character in trait_pattern:
                        bitmask = (bitmask << 1) | (0 if character == '*' else 1)
                        target_bits = (target_bits << 1) | (1 if character == '1' else 0)
                    pattern_length = len(trait_pattern)
                    self.landscape_patterns.append((bitmask, target_bits, float(landscape_value), pattern_length))

            l = []
            for trait_key in self.trait_keys:
                trait_length = len(trait_key)
                trait_int = None
                matched_value = 0.0
                for bitmask, target_bits, value, pattern_length in self.landscape_patterns:
                    if pattern_length != trait_length:
                        continue
                    if trait_int is None:
                        trait_int = int(trait_key, 2)
                    if (trait_int & bitmask) == target_bits:
                        matched_value = value
                        # print("-->", bitmask, "matched value = value", matched_value, "=", value)
                        break
                l.append(matched_value)
#             print("return", l)
            return l
        else:
#             print("return", 0)
            return 0

    @property
    def mutation_rate(self):
        return self._params['mutation_rate'].values()

    @property
    def segregation_rate(self):
        return self._params['segregation_rate'].values()

    @property
    def transfer_rate_donor(self):
        return self._params['transfer_rate_donor'].values()

    @property
    def transfer_rate_recip(self):
        return self._params['transfer_rate_recip'].values()

    @property
    def creation_rate(self):
        return np.array(self._creation_rate) if self._creation_rate is not None else None

    @property
    def typeIDs(self):
        if(self._typeIDs is None):
            self._typeIDs = np.array(self.assign_type_ids())
        return self._typeIDs

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
    def mutation_parent_indices(self):
        return np.array(self._mutation_parent_indices)

    @property
    def segregation_parent_indices(self):
        return np.array(self._segregation_parent_indices)

    @property
    def transfer_donor_indices(self):
        return np.array(self._transfer_donor_indices)

    @property
    def transfer_recip_indices(self):
        return np.array(self._transfer_recip_indices)

    @property
    def mutant_indices(self):
        return self._mutant_indices

    @property
    def segregant_indices(self):
        return self._segregant_indices

    @property
    def transconjugant_indices_bydonor(self):
        return self._transconjugant_indices_bydonor

    @property
    def transconjugant_indices_byrecip(self):
        return self._transconjugant_indices_byrecip

    @property
    def lineageIDs(self):
        if(self._lineageIDs is None):
            self.phylogeny = {}
            lineageIDs = []
            for i in range(self.num_types):
                new_lineage_id = self.add_type_to_phylogeny(type_index=i)
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
            mutation_rate_par = self._params['mutation_rate'].get_type(parent_idx, values_only=True)
            mutant_indices.append([])
            for i in (np.where(mutation_rate_par > 0)[0] if mutation_rate_par.ndim == 1 else range(self.traits.shape[1])):
                # - - - - -
                # Generate the (default) mutant trait profile:
                _traits_mut = self.traits[parent_idx] ^ [0 if j!=i else 1 for j in range(self.traits.shape[1])]
                # - - - - -
                # Check if attributes have been manually specified for this mutant trait profile...
                mutant_overrides_mut = {}
                if(self.mutant_overrides is not None):
                    for key in self.mutant_overrides.keys():
                        if(re.match(key, self.get_trait_key(_traits_mut))):
                            mutant_overrides_mut = self.mutant_overrides[key]
                            break
                # Override the default mutant trait profile, if applicable:
                if('traits' in mutant_overrides_mut):
                    _traits_mut = mutant_overrides_mut['traits']
                # Append the mutant trait profile to the list of mutant traits:
                traits_mut.append(_traits_mut)
                # - - - - -
                for param, param_vals in params_mut.items():
                    if(param in mutant_overrides_mut):
                        params_mut[param] = utils.SystemParameter.combine(params_mut[param], utils.SystemParameter(mutant_overrides_mut[param], num_types=1, num_traits=self._params[param].num_traits, force_type_dim=self._params[param].force_type_dim, force_trait_dim=self._params[param].force_trait_dim))
                    else:
                        params_mut[param] = utils.SystemParameter.combine(params_mut[param], self._params[param].get_type(parent_idx))
                # - - - - -
                creation_rate_mut.append(mutation_rate_par[i] if mutation_rate_par.ndim == 1 else mutation_rate_par)
                # - - - - -
                parent_indices_mut.append(parent_idx)
                # - - - - -
                mutant_indices[p].append(len(traits_mut)-1)
        #----------------------------------
        mutant_set = TypeSet(traits=traits_mut, consumption_rate=params_mut['consumption_rate'], carrying_capacity=params_mut['carrying_capacity'], energy_passthru=params_mut['energy_passthru'], growth_factor=params_mut['growth_factor'],
                             cost_baseline=params_mut['cost_baseline'], cost_pertrait=params_mut['cost_pertrait'], cost_perpair=params_mut['cost_perpair'], cost_interaction=self.cost_interaction, cost_landscape=self.cost_landscape,
                             mutation_rate=params_mut['mutation_rate'], segregation_rate=params_mut['segregation_rate'], transfer_rate_donor=params_mut['transfer_rate_donor'], transfer_rate_recip=params_mut['transfer_rate_recip'],
                             mutant_overrides=self.mutant_overrides, segregant_overrides=self.segregant_overrides, transconjugant_overrides=self.transconjugant_overrides,
                             creation_rate=creation_rate_mut, mutation_parent_indices=parent_indices_mut, binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
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
            segregation_rate_par = self._params['segregation_rate'].get_type(parent_idx, values_only=True)
            segregant_indices.append([])
            for i in (np.where(segregation_rate_par > 0)[0] if segregation_rate_par.ndim == 1 else range(self.traits.shape[1])):
                if(self.traits[parent_idx][i] != 0):
                    # - - - - -
                    # Generate the (default) segregant trait profile:
                    _traits_seg = self.traits[parent_idx].copy()
                    _traits_seg[i] = 0
                    if(self._segregation_linkage is not None and i in self._segregation_linkage):
                        _traits_seg[self._segregation_linkage[i]] = 0
                    # - - - - -
                    # Check if attributes have been manually specified for this segregant trait profile...
                    segregant_overrides_seg = {}
                    if(self.segregant_overrides is not None):
                        for key in self.segregant_overrides.keys():
                            if(re.match(key, self.get_trait_key(_traits_seg))):
                                segregant_overrides_seg = self.segregant_overrides[key]
                                break
                    # Override the default segregant trait profile, if applicable:
                    if('traits' in segregant_overrides_seg):
                        _traits_seg = segregant_overrides_seg['traits']
                    # Append the segregant trait profile to the list of segregant traits:
                    traits_seg.append(_traits_seg)
                    # - - - - -
                    for param, param_vals in params_seg.items():
                        if(param in segregant_overrides_seg):
                            params_seg[param] = utils.SystemParameter.combine(params_seg[param], utils.SystemParameter(segregant_overrides_seg[param], num_types=1, num_traits=self._params[param].num_traits, force_type_dim=self._params[param].force_type_dim, force_trait_dim=self._params[param].force_trait_dim))
                        else:
                            params_seg[param] = utils.SystemParameter.combine(params_seg[param], self._params[param].get_type(parent_idx))
                    # - - - - -
                    creation_rate_seg.append(segregation_rate_par[i] if segregation_rate_par.ndim == 1 else segregation_rate_par)
                    # - - - - -
                    parent_indices_seg.append(parent_idx)
                    # - - - - -
                    segregant_indices[p].append(len(traits_seg)-1)
        #----------------------------------
        segregant_set = TypeSet(traits=traits_seg, consumption_rate=params_seg['consumption_rate'], carrying_capacity=params_seg['carrying_capacity'], energy_passthru=params_seg['energy_passthru'], growth_factor=params_seg['growth_factor'],
                                cost_baseline=params_seg['cost_baseline'], cost_pertrait=params_seg['cost_pertrait'], cost_perpair=params_seg['cost_perpair'], cost_interaction=self.cost_interaction, cost_landscape=self.cost_landscape,
                                mutation_rate=params_seg['mutation_rate'], segregation_rate=params_seg['segregation_rate'], transfer_rate_donor=params_seg['transfer_rate_donor'], transfer_rate_recip=params_seg['transfer_rate_recip'],
                                mutant_overrides=self.mutant_overrides, segregant_overrides=self.segregant_overrides, transconjugant_overrides=self.transconjugant_overrides,
                                creation_rate=creation_rate_seg, segregation_parent_indices=parent_indices_seg, binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
        #----------------------------------
        if(update_segregant_indices):
            self._segregant_indices = segregant_indices
        #----------------------------------
        return segregant_set

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_transconjugant_set(self, donor_index=None, recip_index=None, update_transconjugant_indices=True):
        donor_indices = utils.treat_as_list(donor_index) if donor_index is not None else range(self.traits.shape[0])
        recip_indices = utils.treat_as_list(recip_index) if recip_index is not None else range(self.traits.shape[0])
        #----------------------------------
        traits_xconj           = []
        params_xconj           = {param: None for param in self._params.keys()}
        creation_rate_xconj    = []
        donor_indices_xconj    = []
        recip_indices_xconj    = []
        transconjugant_indices_bydonor = [[] for d in range(len(donor_indices))]
        transconjugant_indices_byrecip = [[] for r in range(len(recip_indices))]
        #----------------------------------
        # Iterate over potential donor types:
        for d, donor_idx in enumerate(donor_indices):
            transfer_rate_donor = self._params['transfer_rate_donor'].get_type(donor_idx, values_only=True)
            for i in (np.where(transfer_rate_donor > 0)[0] if transfer_rate_donor.ndim == 1 else range(self.traits.shape[1])):
                if(self.traits[donor_idx][i] != 0):
                    # Iterate over potential recipient types:
                    for r, recip_idx in enumerate(recip_indices):
                        if(recip_idx == donor_idx):
                            continue
                        transfer_rate_recip   = self._params['transfer_rate_recip'].get_type(recip_idx, values_only=True)
                        transfer_rate_recip_i = transfer_rate_recip[i] if transfer_rate_recip.ndim == 1 else transfer_rate_recip
                        if(self.traits[recip_idx][i] == 0 and transfer_rate_recip_i > 0):
                            # - - - - -
                            transferred_traits = [i]
                            if(self._transfer_linkage is not None and i in self._transfer_linkage):
                                transferred_traits = np.concatenate([transferred_traits, self._transfer_linkage[i]])
                            # - - - - -
                            # Generate the (default) transconjugant trait profile:
                            _traits_xconj = self.traits[recip_idx].copy()
                            _traits_xconj[transferred_traits] = self.traits[donor_idx][transferred_traits]
                            # - - - - -
                            # Check if attributes have been manually specified for this transconjugant trait profile...
                            transconjugant_overrides_xconj = {}
                            if(self.transconjugant_overrides is not None):
                                for key in self.transconjugant_overrides.keys():
                                    if(re.match(key, self.get_trait_key(_traits_xconj))):
                                        transconjugant_overrides_xconj = self.transconjugant_overrides[key]
                                        break
                            # Override the default transconjugant trait profile if applicable:
                            if('traits' in transconjugant_overrides_xconj):
                                override_traits = transconjugant_overrides_xconj['traits']['traits'] if 'traits' in transconjugant_overrides_xconj['traits'] else range(self.traits.shape[1])
                                _traits_xconj[override_traits] = transconjugant_overrides_xconj['traits']['values']
                            # Append the transconjugant trait profile to the list of transconjugant traits:
                            traits_xconj.append(_traits_xconj)
                            # - - - - -
                            for param, param_vals in params_xconj.items():
                                xconj_values = self._params[param].values(type=recip_idx)
                                donor_values = self._params[param].values(type=donor_idx, trait=transferred_traits)
                                if(xconj_values.ndim == 1):
                                    xconj_values[transferred_traits] = donor_values if donor_values.ndim <= 1 else donor_values.ravel()
                                elif(xconj_values.ndim == 2):
                                    xconj_values = xconj_values.ravel()
                                    xconj_values[transferred_traits] = donor_values if donor_values.ndim <= 1 else donor_values.ravel()
                                params_xconj[param] = utils.SystemParameter.combine(params_xconj[param], utils.SystemParameter(xconj_values, num_types=1, num_traits=self._params[param].num_traits, force_type_dim=self._params[param].force_type_dim, force_trait_dim=self._params[param].force_trait_dim))
                                pass
                            # - - - - -
                            transfer_rate_donor_i = transfer_rate_donor[i] if transfer_rate_donor.ndim == 1 else transfer_rate_donor
                            creation_rate_xconj.append(transfer_rate_donor_i * transfer_rate_recip_i)
                            # - - - - -
                            donor_indices_xconj.append(donor_idx)
                            recip_indices_xconj.append(recip_idx) # TODO This needs to be donor_indices and recip_indices
                            # - - - - -
                            transconjugant_indices_bydonor[d].append(len(traits_xconj)-1)
                            transconjugant_indices_byrecip[r].append(len(traits_xconj)-1)
        #----------------------------------
        transconjugant_set = TypeSet(traits=traits_xconj, consumption_rate=params_xconj['consumption_rate'], carrying_capacity=params_xconj['carrying_capacity'], energy_passthru=params_xconj['energy_passthru'], growth_factor=params_xconj['growth_factor'],
                                     cost_baseline=params_xconj['cost_baseline'], cost_pertrait=params_xconj['cost_pertrait'], cost_perpair=params_xconj['cost_perpair'], cost_interaction=self.cost_interaction, cost_landscape=self.cost_landscape,
                                     mutation_rate=params_xconj['mutation_rate'], segregation_rate=params_xconj['segregation_rate'], transfer_rate_donor=params_xconj['transfer_rate_donor'], transfer_rate_recip=params_xconj['transfer_rate_recip'],
                                     mutant_overrides=self.mutant_overrides, segregant_overrides=self.segregant_overrides, transconjugant_overrides=self.transconjugant_overrides,
                                     creation_rate=creation_rate_xconj, transfer_donor_indices=donor_indices_xconj, transfer_recip_indices=recip_indices_xconj, binarize_trait_costs=self.binarize_trait_costs, binarize_interaction_costs=self.binarize_interaction_costs)
        #----------------------------------
        if(update_transconjugant_indices):
            self._transconjugant_indices_bydonor = transconjugant_indices_bydonor
            self._transconjugant_indices_byrecip = transconjugant_indices_byrecip
        #----------------------------------
        return transconjugant_set

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, added_type_set, keep_lineage_ids=False):
        if(not isinstance(added_type_set, TypeSet)):
            utils.error(f"Error in TypeSet add_type(): type_set argument expects object of TypeSet type.")
        #----------------------------------
        if(added_type_set.num_types == 0):
            added_type_indices = []
            return added_type_indices
        #----------------------------------
        # Check that the type set dimensions match the system dimensions:
        if(self.num_traits != added_type_set.num_traits and self.num_traits != 0):
            utils.error(f"Error in TypeSet add_type(): The number of traits for added types ({added_type_set.num_traits}) does not match the number of type set traits ({self.num_traits}).")
        #----------------------------------
        added_type_indices = list(range(self.num_types, self.num_types+added_type_set.num_types))
        #----------------------------------
        if(self._traits is None or self._traits.shape[0]==0 and self.traits.shape[1]==0):
            self._traits = added_type_set._traits
        else:
            self._traits = self._traits.add(added_type_set.traits)
        #----------------------------------
        for param, param_vals in self._params.items():
            # _ = self._params[param]
            # __ = added_type_set._params[param]
            # if(_.shape is None):
            #     pass
            # if(__.shape is None):
            #     pass
            self._params[param] = utils.SystemParameter.combine(self._params[param], added_type_set._params[param])
        #----------------------------------
        self._creation_rate  = [rate for ratelist in [self._creation_rate, added_type_set._creation_rate] for rate in ratelist] if self._creation_rate is not None else None
        #----------------------------------
        self._mutation_parent_indices    = [pidx for idxlist in [self._mutation_parent_indices, added_type_set.mutation_parent_indices] for pidx in idxlist]
        self._segregation_parent_indices = [pidx for idxlist in [self._segregation_parent_indices, added_type_set.segregation_parent_indices] for pidx in idxlist]
        self._transfer_donor_indices     = [didx for idxlist in [self._transfer_donor_indices, added_type_set.transfer_donor_indices] for didx in idxlist]
        self._transfer_recip_indices     = [ridx for idxlist in [self._transfer_recip_indices, added_type_set.transfer_recip_indices] for ridx in idxlist]
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
        if(self._transconjugant_indices_bydonor is not None):
            if(added_type_set.transconjugant_indices_bydonor is None):
                self._transconjugant_indices_bydonor = [tindices for indiceslist in [self._transconjugant_indices_bydonor, [[] for addedtype in range(added_type_set.num_types)]] for tindices in indiceslist]
            else:
                self._transconjugant_indices_bydonor = [tindices for indiceslist in [self._transconjugant_indices_bydonor, added_type_set.transconjugant_indices_bydonor] for tindices in indiceslist]
        #--------
        if(self._transconjugant_indices_byrecip is not None):
            if(added_type_set.transconjugant_indices_byrecip is None):
                self._transconjugant_indices_byrecip = [tindices for indiceslist in [self._transconjugant_indices_byrecip, [[] for addedtype in range(added_type_set.num_types)]] for tindices in indiceslist]
            else:
                self._transconjugant_indices_byrecip = [tindices for indiceslist in [self._transconjugant_indices_byrecip, added_type_set.transconjugant_indices_byrecip] for tindices in indiceslist]
        #----------------------------------
        if(self._energy_costs is not None):
            self._energy_costs.add(added_type_set.energy_costs, axis=1)
        #----------------------------------
        if(self._typeIDs is not None):
            self._typeIDs = [tid for idlist in [self._typeIDs, added_type_set.typeIDs] for tid in idlist]
        #----------------------------------
        if(self._lineageIDs is not None):
            # for _i, i in enumerate(range((self.num_types-1), (self.num_types-1)+added_type_set.num_types)):
            for _i, i in enumerate(range(self.num_types-added_type_set.num_types, self.num_types)):
                added_lineage_id = self.add_type_to_phylogeny(type_index=i, lineage_id=(added_type_set.lineageIDs[_i] if keep_lineage_ids else None))
                self._lineageIDs.append(added_lineage_id)
        #----------------------------------
        return added_type_indices

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type_to_phylogeny(self, type_index=None, lineage_id=None):
        if(self.mutation_parent_indices[type_index] is not None):
            progenitor_index  = self.mutation_parent_indices[type_index]
            progeny_id_prefix = 'm'
            progeny_id_suffix = ''
        elif(self.segregation_parent_indices[type_index] is not None):
            progenitor_index  = self.segregation_parent_indices[type_index]
            progeny_id_prefix = 's'
            progeny_id_suffix = ''
        elif(self.transfer_recip_indices[type_index] is not None):
            progenitor_index  = self.transfer_recip_indices[type_index]
            donor_index       = self.transfer_donor_indices[type_index]
            progeny_id_prefix = 't'
            progeny_id_suffix = f"(d{self.lineageIDs[donor_index].replace('.', '-')})"
        else:
            progenitor_index  = None
            progeny_id_prefix = None
            progeny_id_suffix = None
        #----------------------------------
        lineage_id_traits     = ''
        if(self._lineageID_traits is not None):
            trait_key = self.trait_keys[type_index]
            _ = ''.join([trait_key[i] for i in self._lineageID_traits])
            lineage_id_traits = f"[{''.join([trait_key[i] for i in self._lineageID_traits])}]"
        #----------------------------------
        if(progenitor_index is None or np.isnan(progenitor_index)):
            new_lineage_id = f"{str( len(self.phylogeny.keys())+1 )}{lineage_id_traits}" if lineage_id is None else f"{lineage_id}{lineage_id_traits}"
            self.phylogeny.update({ new_lineage_id: {} })
        else:
            progenitor_lineage_id = self.lineageIDs[progenitor_index.astype(int)]
            if('.' in progenitor_lineage_id):
                progenitor_lineage_id_parts = progenitor_lineage_id.split('.')
                lineageSubtree = self.phylogeny
                for l in range(1, len(progenitor_lineage_id_parts)+1):
                    lineageSubtree = lineageSubtree['.'.join(progenitor_lineage_id_parts[:l])]
            else:
                lineageSubtree = self.phylogeny[progenitor_lineage_id]
            # new_lineage_id = progenitor_lineage_id +'.'+ str(len(lineageSubtree.keys())+1) if lineage_id is None else lineage_id
            new_lineage_id = f"{progenitor_lineage_id}.{progeny_id_prefix}{str(len(lineageSubtree.keys())+1)}{progeny_id_suffix}{lineage_id_traits}" if lineage_id is None else lineage_id
            lineageSubtree[new_lineage_id] = {}
        #----------------------------------
        return new_lineage_id

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_type(self, type_index=None, keep_progenitor_indices=True, keep_lineage_ids=False):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else None
        if(type_idx is None):
            utils.error(f"Error in TypeSet get_type(): A type index or type id must be given.")
        #----------------------------------
        return TypeSet(traits = self.traits[type_idx],
                        consumption_rate           = self._params['consumption_rate'].get_type(type_idx),
                        carrying_capacity          = self._params['carrying_capacity'].get_type(type_idx),
                        energy_passthru            = self._params['energy_passthru'].get_type(type_idx),
                        growth_factor              = self._params['growth_factor'].get_type(type_idx),
                        cost_baseline              = self._params['cost_baseline'].get_type(type_idx),
                        cost_pertrait              = self._params['cost_pertrait'].get_type(type_idx),
                        cost_perpair               = self._params['cost_perpair'].get_type(type_idx),
                        cost_interaction           = self.cost_interaction,
                        cost_landscape             = self.cost_landscape,
                        mutation_rate              = self._params['mutation_rate'].get_type(type_idx),
                        segregation_rate           = self._params['segregation_rate'].get_type(type_idx),
                        transfer_rate_donor        = self._params['transfer_rate_donor'].get_type(type_idx),
                        transfer_rate_recip        = self._params['transfer_rate_recip'].get_type(type_idx),
                        creation_rate              = self.creation_rate[type_idx] if self.creation_rate is not None else None,
                        mutant_overrides           = self.mutant_overrides,
                        segregant_overrides        = self.segregant_overrides,
                        transconjugant_overrides   = self.transconjugant_overrides,
                        mutation_parent_indices    = self.mutation_parent_indices[type_idx] if keep_progenitor_indices else None,
                        segregation_parent_indices = self.segregation_parent_indices[type_idx] if keep_progenitor_indices else None,
                        transfer_donor_indices     = self.transfer_donor_indices[type_idx] if keep_progenitor_indices else None,
                        transfer_recip_indices     = self.transfer_recip_indices[type_idx] if keep_progenitor_indices else None,
                        lineageIDs                 = [self.lineageIDs[i] for i in type_idx] if keep_lineage_ids else None,
                        binarize_trait_costs       = self.binarize_trait_costs,
                        binarize_interaction_costs = self.binarize_interaction_costs)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        transconjugant_indices_bydonor      = [x for xconjIndices_u in ([self._transconjugant_indices_bydonor[u] for u in type_idx]) for x in xconjIndices_u]
        transconjugant_indices_byrecip      = [x for xconjIndices_u in ([self._transconjugant_indices_byrecip[u] for u in type_idx]) for x in xconjIndices_u]
        transconjugant_indices_intersection = [x for x in transconjugant_indices_bydonor if x in frozenset(transconjugant_indices_byrecip)]
        return transconjugant_indices_intersection

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_progenitor_indices(self, type_index=None, return_progenitor_class=False):
        type_idx = utils.treat_as_list(type_index)
        progenitor_indices = [None for u in type_idx]
        progenitor_classes = [None for u in type_idx]
        for u, idx in enumerate(type_idx):
            if(self.mutation_parent_indices[idx] is not None):
                progenitor_indices[u] = self.mutation_parent_indices[idx]
                progenitor_classes[u] = 'mutation'
            elif(self.segregation_parent_indices[idx] is not None):
                progenitor_indices[u] = self.segregation_parent_indices[idx]
                progenitor_classes[u] = 'segregation'
            elif(self.transfer_recip_indices[idx] is not None):
                progenitor_indices[u] = self.transfer_recip_indices[idx]
                progenitor_classes[u] = 'transfer'
            else:
                pass  # leave progenitor_indices[u] as None
        return progenitor_indices if not return_progenitor_class else (progenitor_indices, progenitor_classes)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, type_index=None, values_only=False):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.num_types)
        #----------------------------------
        return { 'num_types':                  len(type_idx),
                 'traits':                     self.traits[type_idx],
                 'consumption_rate':           self._params['consumption_rate'].get_type(type_idx, values_only),
                 'carrying_capacity':          self._params['carrying_capacity'].get_type(type_idx, values_only),
                 'energy_passthru':            self._params['energy_passthru'].get_type(type_idx, values_only),
                 'growth_factor':              self._params['growth_factor'].get_type(type_idx, values_only),
                 'energy_costs':               self.energy_costs[type_idx],
                 'creation_rate':              self.creation_rate[type_idx] if self.creation_rate is not None else None,
                 'mutation_parent_indices':    self.mutation_parent_indices[type_idx],
                 'segregation_parent_indices': self.segregation_parent_indices[type_idx],
                 'transfer_donor_indices':     self.transfer_donor_indices[type_idx],
                 'transfer_recip_indices':     self.transfer_recip_indices[type_idx] }

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
        self._transconjugant_indices_bydonor = [self._transconjugant_indices_bydonor[i] for i in type_order] if self._transconjugant_indices_bydonor is not None else None # Rows may have unequal lengths, so keep as list of lists (not 2d array)
        self._transconjugant_indices_byrecip = [self._transconjugant_indices_byrecip[i] for i in type_order] if self._transconjugant_indices_byrecip is not None else None # Rows may have unequal lengths, so keep as list of lists (not 2d array)
        #----------------------------------
        # Parent indices require special handling because simply reordering the parent indices list makes the index pointers point to incorrect places relative to the reordered lists
        _mutation_parent_indices_tempreorder = np.array(self._mutation_parent_indices)[type_order].tolist()
        self._mutation_parent_indices = [np.where(type_order == pidx)[0][0] if pidx is not None else None for pidx in _mutation_parent_indices_tempreorder]
        _segregation_parent_indices_tempreorder = np.array(self._segregation_parent_indices)[type_order].tolist()
        self._segregation_parent_indices = [np.where(type_order == pidx)[0][0] if pidx is not None else None for pidx in _segregation_parent_indices_tempreorder]
        _transfer_donor_indices_tempreorder = np.array(self._transfer_donor_indices)[type_order].tolist()
        self._transfer_donor_indices  = [np.where(type_order == didx)[0][0] if didx is not None else None for didx in _transfer_donor_indices_tempreorder]
        _transfer_recip_indices_tempreorder = np.array(self._transfer_recip_indices)[type_order].tolist()
        self._transfer_recip_indices  = [np.where(type_order == ridx)[0][0] if ridx is not None else None for ridx in _transfer_recip_indices_tempreorder]
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
