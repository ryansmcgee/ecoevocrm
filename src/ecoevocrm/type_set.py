import numpy as np

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TypeSet():

    def __init__(self, num_types   = None,
                       num_traits  = None,
                       sigma       = None,
                       beta        = 1,
                       kappa       = 1e10,
                       eta         = 1,
                       lamda       = 0,
                       growthfactor       = 1,
                       xi          = 0,
                       chi         = None,
                       J           = None,
                       mu          = 1e-10, 
                       generation_rates = None,
                       lineageIDs  = None,
                       parent_indices = None,
                       normalize_phenotypes = False,
                       binarize_traits_chi_cost_terms = False,
                       binarize_traits_J_cost_terms = False
                    ):

        #----------------------------------
        # Determine the number of types and traits,
        # and initialize sigma matrix:
        #----------------------------------
        if(isinstance(sigma, (list, np.ndarray))):
            sigma = np.array(sigma)
            if(sigma.ndim == 2):
                num_types  = sigma.shape[0]
                num_traits = sigma.shape[1]
            elif(sigma.ndim == 1):
                num_types  = 1
                num_traits = len(sigma)
        elif(num_types is not None and num_traits is not None):
            num_types  = num_types
            num_traits = num_traits
        else:
            utils.error("Error in TypeSet __init__(): Number of types and traits must be specified by providing a) a sigma matrix, or b) both num_types and num_traits values.")
        #----------------------------------
        
        self.num_traits = num_traits

        # /!\ Non-binary sigmas is deprecated as of 2023-09-01
        # self.normalize_phenotypes = normalize_phenotypes
        # if(self.normalize_phenotypes):
        #     norm_denom = np.atleast_2d(sigma).sum(axis=1, keepdims=1)
        #     norm_denom[norm_denom == 0] = 1
        #     sigma = sigma/norm_denom

        self._sigma = utils.ExpandableArray(utils.reshape(sigma, shape=(num_types, num_traits)), dtype='int')

        #----------------------------------
        # Initialize parameter vectors/matrices:
        #----------------------------------

        # print("Start!")
        # print("_beta ...")
        self._beta   = self.preprocess_params(beta,  has_trait_dim=True)
        # print("_kappa ...")
        self._kappa  = self.preprocess_params(kappa, has_trait_dim=True)
        # print("_eta ...")
        self._eta    = self.preprocess_params(eta,   has_trait_dim=True)
        # print("_lamda ...")
        self._lamda  = self.preprocess_params(lamda, has_trait_dim=True)
        # print("_growthfactor ...")
        self._growthfactor  = self.preprocess_params(growthfactor, has_trait_dim=False)
        # print("_xi ...")
        self._xi     = self.preprocess_params(xi,    has_trait_dim=False) #, force_expandable_array=(mean_xi_mut > 0))
        # print("_chi ...")
        self._chi    = self.preprocess_params(chi,   has_trait_dim=True) if chi is not None else None
        # print("_J ...")
        self._J      = utils.reshape(J, shape=(self.num_traits, self.num_traits)) if J is not None else None
        # print("_mu ...")
        self._mu     = self.preprocess_params(mu,    has_trait_dim=True)
        # print("_generation_rates ...")
        self._generation_rates = self.preprocess_params(generation_rates, has_trait_dim=False) if generation_rates is not None else None

        # self._mean_xi_mut = mean_xi_mut
        # self.__mean_xi_mut = self.preprocess_params(__mean_xi_mut, has_trait_dim=False)

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

        self.binarize_traits_chi_cost_terms = binarize_traits_chi_cost_terms
        self.binarize_traits_J_cost_terms   = binarize_traits_J_cost_terms
                
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def get_array(arr):
        return arr.values if isinstance(arr, utils.ExpandableArray) else arr

    @property
    def num_types(self):
        return self.sigma.shape[0]
    
    @property
    def sigma(self):
        return TypeSet.get_array(self._sigma)

    @property
    def beta(self):
        return TypeSet.get_array(self._beta)

    @beta.setter
    def beta(self, vals):
        self._beta = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def kappa(self):
        return TypeSet.get_array(self._kappa)

    @kappa.setter
    def kappa(self, vals):
        self._kappa = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def eta(self):
        return TypeSet.get_array(self._eta)

    @eta.setter
    def eta(self, vals):
        self._eta = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def lamda(self):
        return TypeSet.get_array(self._lamda)

    @lamda.setter
    def lamda(self, vals):
        self._lamda = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def growthfactor(self):
        return TypeSet.get_array(self._growthfactor)

    @growthfactor.setter
    def growthfactor(self, vals):
        self._growthfactor = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def xi(self):
        return TypeSet.get_array(self._xi)

    @xi.setter
    def xi(self, vals):
        self._xi = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def chi(self):
        return TypeSet.get_array(self._chi)

    @chi.setter
    def chi(self, vals):
        self._chi = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def mu(self):
        return TypeSet.get_array(self._mu)

    @mu.setter
    def mu(self, vals):
        self._mu = self.preprocess_params(vals, has_trait_dim=True)

    @property
    def generation_rates(self):
        return self._generation_rates

    @generation_rates.setter
    def generation_rates(self, vals):
        self._generation_rates = self.preprocess_params(vals, has_trait_dim=False)

    # @property
    # def _mean_xi_mut(self):
    #     return self.__mean_xi_mut

    # @_mean_xi_mut.setter
    # def _mean_xi_mut(self, vals):
    #     self.__mean_xi_mut = self.preprocess_params(vals, has_trait_dim=False)

    @property
    def J(self):
        return TypeSet.get_array(self._J)

    @property
    def energy_costs(self):
        if(self._energy_costs is None):
            costs = 0 + (self.xi.ravel() if self.xi.ndim == 2 else self.xi)
            costs += self.chi_cost_terms
            costs += self.J_cost_terms
            if(np.any(costs < 0)):
                raise ValueError('Negative energy_costs encountered for one or more types.')
            self._energy_costs = utils.ExpandableArray(costs)
        return TypeSet.get_array(self._energy_costs).ravel()

    @property
    def xi_cost_terms(self):
        return (self.xi.ravel() if self.xi.ndim == 2 else self.xi)

    @property
    def chi_cost_terms(self):
        _sigma = self.sigma if not self.binarize_traits_chi_cost_terms else (self.sigma > 0).astype(int)
        return np.sum(_sigma * self.chi, axis=1) if self._chi is not None else 0
    
    @property
    def J_cost_terms(self):
        _sigma = self.sigma if not self.binarize_traits_J_cost_terms else (self.sigma > 0).astype(int)
        return -1 * np.sum(_sigma * np.dot(_sigma, self.J), axis=1) if self._J is not None else 0
    
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

    # def generate_mutant_phenotypes(self, sigma=None):
    #     sigma = self.sigma if sigma is None else sigma
    #     sigma = (sigma != 0).astype(float)
    #     #----------------------------------
    #     mutations = np.tile(np.identity(sigma.shape[1]), reps=(sigma.shape[0], 1))
    #     sigma_mut = 1 * np.logical_xor( np.repeat(sigma, repeats=sigma.shape[1], axis=0), mutations )
    #     #----------------------------------
    #     # /!\ Non-binary sigmas is deprecated as of 2023-09-01
    #     # if(self.normalize_phenotypes):
    #     #     norm_denom = sigma_mut.sum(axis=1, keepdims=1)
    #     #     norm_denom[norm_denom == 0] = 1
    #     #     sigma_mut = sigma_mut/norm_denom
    #     #----------------------------------
    #     return sigma_mut


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def generate_mutant_set(self, update_mutantIDs=True): #
    #     sigma_mut = self.generate_mutant_phenotypes()
    #     #---------------------------------- 
    #     beta_mut  = np.repeat(self.beta,  repeats=sigma_mut.shape[0], axis=0) if self.beta.ndim == 2  else self.beta
    #     kappa_mut = np.repeat(self.kappa, repeats=sigma_mut.shape[0], axis=0) if self.kappa.ndim == 2 else self.kappa
    #     eta_mut   = np.repeat(self.eta,   repeats=sigma_mut.shape[0], axis=0) if self.eta.ndim == 2   else self.eta
    #     lamda_mut = np.repeat(self.lamda, repeats=sigma_mut.shape[0], axis=0) if self.lamda.ndim == 2 else self.lamda
    #     growthfactor_mut = np.repeat(self.growthfactor, repeats=sigma_mut.shape[0], axis=0) if self.growthfactor.ndim == 2 else self.growthfactor
    #     xi_mut    = np.repeat(self.xi,    repeats=sigma_mut.shape[0], axis=0) if self.xi.ndim == 2    else self.xi
    #     chi_mut   = np.repeat(self.chi,   repeats=sigma_mut.shape[0], axis=0) if self.chi.ndim == 2   else self.chi
    #     mu_mut    = np.repeat(self.mu,    repeats=sigma_mut.shape[0], axis=0) if self.mu.ndim == 2    else self.mu
    #     #----------------------------------
    #     # if(self._mean_xi_mut > 0):
    #     #     xi_mut = self.xi.ravel() - np.random.exponential(scale=self._mean_xi_mut, size=sigma_mut.shape[0])
    #     # else:
    #     #     xi_mut = np.repeat(self.xi, repeats=sigma_mut.shape[0], axis=0) if self.xi.ndim == 2 else self.xi
    #     #----------------------------------
    #     mutant_set = TypeSet(sigma=sigma_mut, beta=beta_mut, kappa=kappa_mut, eta=eta_mut, lamda=lamda_mut, growthfactor=growthfactor_mut, xi=xi_mut, chi=chi_mut, J=self.J, mu=mu_mut, # mean_xi_mut=self._mean_xi_mut,
    #                          # normalize_phenotypes=self.normalize_phenotypes, 
    #                          binarize_traits_chi_cost_terms=self.binarize_traits_chi_cost_terms, binarize_traits_J_cost_terms=self.binarize_traits_J_cost_terms)
    #     #----------------------------------
    #     if(update_mutantIDs):
    #         self._mutantIDs = mutant_set.typeIDs.reshape((self.num_types, self.num_traits))
    #     #----------------------------------
    #     return mutant_set

    def generate_mutant_set(self, type_index=None, update_mutant_indices=True): #
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.sigma.shape[0])
        #----------------------------------
        sigma_mut            = []
        beta_mut             = [] if self.beta.ndim == 2 else self.beta
        kappa_mut            = [] if self.kappa.ndim == 2 else self.kappa
        eta_mut              = [] if self.eta.ndim == 2 else self.eta
        lamda_mut            = [] if self.lamda.ndim == 2 else self.lamda
        growthfactor_mut            = [] if self.growthfactor.ndim == 2 else self.growthfactor
        xi_mut               = [] if self.xi.ndim == 2 else self.xi
        chi_mut              = [] if self.chi.ndim == 2 else self.chi
        mu_mut               = [] if self.mu.ndim == 2 else self.mu
        parent_indices_mut   = []
        generation_rates_mut = []
        mutant_indices       = []
        #----------------------------------
        for p, parent_idx in enumerate(type_idx):
            mu_p = self.mu[parent_idx] if self.mu.ndim == 2 else self.mu
            mutant_indices.append([])
            if(np.any(mu_p > 0)):
                for i in (np.where(mu_p > 0)[0] if mu_p.ndim == 1 else range(self.sigma.shape[1])):
                    sigma_mut.append(self.sigma[parent_idx] ^ [0 if j!=i else 1 for j in range(self.sigma.shape[1])])
                    # - - - - -
                    if(self.beta.ndim == 2):    beta_mut.append(self.beta[parent_idx])
                    if(self.kappa.ndim == 2):   kappa_mut.append(self.kappa[parent_idx])
                    if(self.eta.ndim == 2):     eta_mut.append(  self.eta[parent_idx])
                    if(self.lamda.ndim == 2):   lamda_mut.append(self.lamda[parent_idx])
                    if(self.growthfactor.ndim == 2):   growthfactor_mut.append(self.growthfactor[parent_idx])
                    if(self.xi.ndim == 2):      xi_mut.append(self.xi[parent_idx])
                    if(self.chi.ndim == 2):     chi_mut.append(self.chi[parent_idx])
                    if(self.mu.ndim == 2):      mu_mut.append(self.mu[parent_idx])
                    # - - - - -
                    generation_rates_mut.append(mu_p[i] if mu_p.ndim == 1 else mu_p)
                    # - - - - -
                    parent_indices_mut.append(parent_idx)
                    # - - - - -
                    mutant_indices[p].append(len(sigma_mut)-1)
        #----------------------------------
        mutant_set = TypeSet(sigma=sigma_mut, beta=beta_mut, kappa=kappa_mut, eta=eta_mut, lamda=lamda_mut, growthfactor=growthfactor_mut, xi=xi_mut, chi=chi_mut, J=self.J, mu=mu_mut, 
                             generation_rates=generation_rates_mut, parent_indices=parent_indices_mut,
                             binarize_traits_chi_cost_terms=self.binarize_traits_chi_cost_terms, binarize_traits_J_cost_terms=self.binarize_traits_J_cost_terms)
        #----------------------------------
        if(update_mutant_indices):
            self._mutant_indices = mutant_indices
        #----------------------------------
        return mutant_set


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, type_set=None, sigma=None, beta=None, kappa=None, eta=None, lamda=None, growthfactor=None, xi=None, chi=None, mu=None, generation_rates=None, parent_index=None, parent_id=None, ref_type_idx=None): # type_index=None, mean_xi_mut=None, 
        parent_idx   = np.where(self.typeIDs==parent_id)[0] if parent_id is not None else parent_index
        ref_type_idx = ref_type_idx if ref_type_idx is not None else parent_idx if parent_idx is not None else 0
        #----------------------------------
        if(type_set is not None):
            if(isinstance(type_set, TypeSet)):
                added_type_set = type_set
            else:
                utils.error(f"Error in TypeSet add_type(): type_set argument expects object of TypeSet type.")
        else:
            added_type_set = TypeSet(sigma=sigma if sigma is not None else self.sigma[ref_type_idx], 
                                         beta=beta if beta is not None else self.beta[ref_type_idx],  
                                         kappa=kappa if kappa is not None else self.kappa[ref_type_idx],  
                                         eta=eta if eta is not None else self.eta[ref_type_idx],  
                                         lamda=lamda if lamda is not None else self.lamda[ref_type_idx],  
                                         growthfactor=growthfactor if growthfactor is not None else self.growthfactor[ref_type_idx],  
                                         xi=xi if xi is not None else self.xi[ref_type_idx],  
                                         chi=chi if chi is not None else self.chi[ref_type_idx], 
                                         mu=mu if mu is not None else self.mu[ref_type_idx],
                                         # mean_xi_mut=mean_xi_mut if mean_xi_mut is not None else self._mean_xi_mut
                                         generation_rates=generation_rates if generation_rates is not None else self.generation_rates[ref_type_idx]
                                         )
        # Check that the type set dimensions match the system dimensions:
        if(self.num_traits != added_type_set.num_traits): 
            utils.error(f"Error in TypeSet add_type(): The number of traits for added types ({added_type_set.num_traits}) does not match the number of type set traits ({self.num_traits}).")
        #----------------------------------
        added_type_indices = list(range(self.num_types, self.num_types+added_type_set.num_types))
        #----------------------------------
        self._sigma = self._sigma.add(added_type_set.sigma)
        self._beta  = self._beta.add(added_type_set.beta)   if isinstance(self._beta,  utils.ExpandableArray) else self._beta
        self._kappa = self._kappa.add(added_type_set.kappa) if isinstance(self._kappa, utils.ExpandableArray) else self._kappa
        self._eta   = self._eta.add(added_type_set.eta)     if isinstance(self._eta,   utils.ExpandableArray) else self._eta
        self._lamda = self._lamda.add(added_type_set.lamda) if isinstance(self._lamda, utils.ExpandableArray) else self._lamda
        self._growthfactor = self._growthfactor.add(added_type_set.growthfactor) if isinstance(self._growthfactor, utils.ExpandableArray) else self._growthfactor
        self._xi    = self._xi.add(added_type_set.xi)       if isinstance(self._xi,    utils.ExpandableArray) else self._xi
        self._chi   = self._chi.add(added_type_set.chi)     if isinstance(self._chi,   utils.ExpandableArray) else self._chi
        self._mu    = self._mu.add(added_type_set.mu)       if isinstance(self._mu,    utils.ExpandableArray) else self._mu
        self._generation_rates = self._generation_rates.add(added_type_set.generation_rates) if isinstance(self._generation_rates, utils.ExpandableArray) else self._generation_rates
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

    def add_type_to_phylogeny(self, type_index=None, type_id=None, parent_index=None, parent_id=None):
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
        return TypeSet(sigma  = self.sigma[type_idx], 
                        beta  = self.beta[type_idx]  if self.beta.ndim == 2   else self.beta, 
                        kappa = self.kappa[type_idx] if self.kappa.ndim == 2  else self.kappa, 
                        eta   = self.eta[type_idx]   if self.eta.ndim == 2    else self.eta, 
                        lamda = self.lamda[type_idx] if self.lamda.ndim == 2  else self.lamda, 
                        growthfactor = self.growthfactor[type_idx] if self.growthfactor.ndim == 2  else self.growthfactor, 
                        xi    = self.xi[type_idx]    if self.xi.ndim == 2     else self.xi, 
                        chi   = self.chi[type_idx]   if self.chi.ndim == 2    else self.chi, 
                        mu    = self.mu[type_idx]    if self.mu.ndim == 2     else self.mu,
                        J     = self.J,
                        generation_rates=self.generation_rates[type_idx] if self.generation_rates is not None and self.generation_rates.ndim == 2 else self.generation_rates, 
                        parent_indices=self.parent_indices[type_idx],
                        # mean_xi_mut = self._mean_xi_mut,
                        # normalize_phenotypes           = self.normalize_phenotypes,
                        binarize_traits_chi_cost_terms = self.binarize_traits_chi_cost_terms,
                        binarize_traits_J_cost_terms   = self.binarize_traits_J_cost_terms ) 


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|||

    def assign_type_ids(self, type_index=None, sigma=None):
        # print("assignID >>", "type_index=", type_index, "sigma=", sigma)
        sigma = sigma if sigma is not None else self.sigma
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(sigma.shape[0])
        # Convert binary sigma arrays to integer IDs
        typeIDs = []
        for u in range(len(sigma)):
            intID = 0
            for bit in sigma[u].ravel():
                intID = (intID << 1) | bit
            typeIDs.append(intID)
        return typeIDs #if len(typeIDs) > 1 else typeIDs[0]


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|||

    # new but bad
    # def get_indices(self, type_id=None, type_index=None, sigma=None):
    #     sigma = sigma if sigma is not None else self.sigma
    #     return utils.treat_as_list(type_index) if type_index is not None else [self._typeID_indices[tid] for tid in utils.treat_as_list(type_id)] if type_id is not None else np.arange(0, len(sigma), 1)
    

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
    #             'sigma':        self.sigma[type_idx],
    #             'beta':         self.beta  if self.beta.ndim < 2  else self.beta[type_idx],
    #             'kappa':        self.kappa if self.kappa.ndim < 2 else self.kappa[type_idx],
    #             'eta':          self.eta   if self.eta.ndim < 2   else self.eta[type_idx],
    #             'lamda':        self.lamda if self.lamda.ndim < 2 else self.lamda[type_idx],
    #             'growthfactor':        self.growthfactor if self.growthfactor.ndim < 2 else self.growthfactor[type_idx],
    #             'xi':           self.xi    if self.xi.ndim < 2    else self.xi[type_idx],
    #             'chi':          self.chi   if self.chi.ndim < 2   else self.chi[type_idx],
    #             'J':            self.J,
    #             'mu':           self.mu    if self.mu.ndim < 2    else self.mu[type_idx],
    #             'energy_costs': self.energy_costs[type_idx]}

    # new but bad
    # def get_dynamics_params(self, type_id=None, type_index=None):
    #     type_idx = self.get_indices(type_id=type_id, type_index=type_index)
    #     print("get_dynamics_params @@@", type_id, type_index, "-> type_idx", type_idx)
    #     #----------------------------------
    #     return {'num_types':    len(type_idx),
    #             'sigma':        self.sigma[type_idx],
    #             'beta':         self.beta  if self.beta.ndim < 2  else self.beta[type_idx],
    #             'kappa':        self.kappa if self.kappa.ndim < 2 else self.kappa[type_idx],
    #             'eta':          self.eta   if self.eta.ndim < 2   else self.eta[type_idx],
    #             'lamda':        self.lamda if self.lamda.ndim < 2 else self.lamda[type_idx],
    #             'growthfactor':        self.growthfactor if self.growthfactor.ndim < 2 else self.growthfactor[type_idx],
    #             'xi':           self.xi    if self.xi.ndim < 2    else self.xi[type_idx],
    #             'chi':          self.chi   if self.chi.ndim < 2   else self.chi[type_idx],
    #             'J':            self.J,
    #             # 'mu':           self.mu    if self.mu.ndim < 2    else self.mu[type_idx],
    #             'generation_rates': (self.generation_rates if self.generation_rates.ndim < 2 else self.generation_rates[type_idx]) if self.generation_rates is not None else None,
    #             'energy_costs': self.energy_costs[type_idx]}

    def get_dynamics_params(self, type_index=None):
        type_idx = utils.treat_as_list(type_index) if type_index is not None else range(self.num_types)
        # print("get_dynamics_params @@@", type_index, "-> type_idx", type_idx)
        #----------------------------------
        return {'num_types':    len(type_idx),
                'sigma':        self.sigma[type_idx],
                'beta':         self.beta  if self.beta.ndim < 2  else self.beta[type_idx],
                'kappa':        self.kappa if self.kappa.ndim < 2 else self.kappa[type_idx],
                'eta':          self.eta   if self.eta.ndim < 2   else self.eta[type_idx],
                'lamda':        self.lamda if self.lamda.ndim < 2 else self.lamda[type_idx],
                'growthfactor':        self.growthfactor if self.growthfactor.ndim < 2 else self.growthfactor[type_idx],
                'xi':           self.xi    if self.xi.ndim < 2    else self.xi[type_idx],
                'chi':          self.chi   if self.chi.ndim < 2   else self.chi[type_idx],
                'J':            self.J,
                'energy_costs': self.energy_costs[type_idx],
                # 'mu':           self.mu    if self.mu.ndim < 2    else self.mu[type_idx],
                'generation_rates': (self.generation_rates if self.generation_rates.ndim < 2 else self.generation_rates[type_idx]) if self.generation_rates is not None else None,
                'parent_indices': self.parent_indices[type_idx]
                }


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder_types(self, order=None):
        type_order   = np.argsort(self.lineageIDs) if order is None else order
        if(len(type_order) < self.num_types):
            utils.error("Error in TypeSet.reorder_types(): The ordering provided has fewer indices than types.")
        #----------------------------------
        self._sigma = self._sigma.reorder(type_order)
        self._beta  = self._beta.reorder(type_order)  if isinstance(self._beta,  utils.ExpandableArray) else self._beta
        self._kappa = self._kappa.reorder(type_order) if isinstance(self._kappa, utils.ExpandableArray) else self._kappa
        self._eta   = self._eta.reorder(type_order)   if isinstance(self._eta,   utils.ExpandableArray) else self._eta
        self._lamda = self._lamda.reorder(type_order) if isinstance(self._lamda, utils.ExpandableArray) else self._lamda
        self._growthfactor = self._growthfactor.reorder(type_order) if isinstance(self._growthfactor, utils.ExpandableArray) else self._growthfactor
        self._xi    = self._xi.reorder(type_order)    if isinstance(self._xi,    utils.ExpandableArray) else self._xi
        self._chi   = self._chi.reorder(type_order)   if isinstance(self._chi,   utils.ExpandableArray) else self._chi
        self._mu    = self._mu.reorder(type_order)    if isinstance(self._mu,    utils.ExpandableArray) else self._mu
        self._generation_rates = self._generation_rates.reorder(type_order) if isinstance(self._generation_rates, utils.ExpandableArray) else self._generation_rates
        self._energy_costs   = None # reset to recalculate upon next reference
        self._typeIDs       = np.array(self._typeIDs)[type_order].tolist() if self._typeIDs is not None else None
        self._lineageIDs    = np.array(self._lineageIDs)[type_order].tolist() if self._lineageIDs is not None else None
        self._mutant_indices = np.array(self._mutant_indices)[type_order].tolist() if self._mutant_indices is not None else None
        #----------------------------------
        # Parent indices require special handling because simply reordering the parent indices list makes the index pointers point to incorrect places relative to the reordered lists
        _parent_indices_tempreorder = np.array(self._parent_indices)[type_order].tolist()
        self._parent_indices = [np.where(type_order == pidx)[0][0] if pidx != None else None for pidx in _parent_indices_tempreorder]
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
        return [ ''.join(['1' if sigma_vector[i] != 0 else '0' for i in range(len(sigma_vector))]) for sigma_vector in self.sigma ]






