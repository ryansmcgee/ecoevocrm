import numpy as np

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TypeSet():

    def __init__(self, num_types      = None,
                       num_traits     = None,
                       sigma          = None,
                       b              = 1,
                       k              = 0,
                       eta            = 1,
                       l              = 0,
                       g              = 1,
                       xi             = 0,
                       chi            = None,
                       J              = None,
                       mu             = 0,
                       lineage_ids    = None ):

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

        self._sigma = utils.ExpandableArray(utils.reshape(sigma, shape=(num_types, num_traits)))

        #----------------------------------
        # Initialize parameter vectors/matrices:
        #----------------------------------

        self._b     = self.preprocess_params(b,   has_trait_dim=True)
        self._k     = self.preprocess_params(k,   has_trait_dim=True)
        self._eta   = self.preprocess_params(eta, has_trait_dim=True)
        self._l     = self.preprocess_params(l,   has_trait_dim=True)
        self._g     = self.preprocess_params(g,   has_trait_dim=False)
        self._xi    = self.preprocess_params(xi,  has_trait_dim=False)
        self._chi   = self.preprocess_params(chi, has_trait_dim=True)
        self._mu    = self.preprocess_params(mu,  has_trait_dim=False)

        self._J     = utils.reshape(J, shape=(self.num_traits, self.num_traits)) if J is not None else None

        #----------------------------------
        # Initialize other type properties/metadata:
        #----------------------------------

        self._energy_costs = None 

        self._type_ids = None

        self._parent_indices = [None for i in range(self.num_types)]

        self._mutant_indices = None

        self._lineage_ids = lineage_ids if lineage_ids is not None else None
        
        self.phylogeny = {}
                
    
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
    def b(self):
        return TypeSet.get_array(self._b)

    @property
    def k(self):
        return TypeSet.get_array(self._k)

    @property
    def eta(self):
        return TypeSet.get_array(self._eta)

    @property
    def l(self):
        return TypeSet.get_array(self._l)

    @property
    def g(self):
        return TypeSet.get_array(self._g)

    @property
    def xi(self):
        return TypeSet.get_array(self._xi)

    @property
    def chi(self):
        return TypeSet.get_array(self._chi)

    @property
    def mu(self):
        return TypeSet.get_array(self._mu)

    @property
    def J(self):
        return TypeSet.get_array(self._J)

    @property
    def energy_costs(self):
        if(self._energy_costs is None):
            costs = 0 + (self.xi.ravel() if self.xi.ndim == 2 else self.xi)
            if(self.chi is not None):
                costs += np.sum(self.sigma * self.chi, axis=1)
            if(self.J is not None):
                costs += -1 * np.sum(self.sigma * np.dot(self.sigma, self.J), axis=1)
            self._energy_costs = utils.ExpandableArray(costs)
        return TypeSet.get_array(self._energy_costs).ravel()

    @property
    def type_ids(self):
        if(self._type_ids is None):
            self._type_ids = [self.get_type_id(i) for i in range(self.num_types)]
        return self._type_ids

    @property
    def parent_indices(self):
        return self._parent_indices

    @property
    def mutant_indices(self):
        if(self._mutant_indices is None):
            self._mutant_indices = utils.ExpandableArray(np.arange(0, self.num_types*self.num_traits).reshape(self.num_types, self.num_traits), dtype='int')
        return TypeSet.get_array(self._mutant_indices)

    @property
    def lineage_ids(self):
        if(self._lineage_ids is None):
            self.phylogeny = {}
            lineage_ids = []
            for i in range(self.num_types):
                new_lineage_id = self.add_type_to_phylogeny(i)
                lineage_ids.append(new_lineage_id)
            self._lineage_ids = lineage_ids
        return self._lineage_ids

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def preprocess_params(self, vals, has_trait_dim, dtype='float64'):
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
                if(len(arr) == 1):
                    return arr[0] # single val as scalar
                elif(len(arr) == self.num_types):
                    if(np.all(arr == arr[0])): # all elements equal
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

    def generate_mutant_phenotypes(self, sigma=None):
        sigma = self.sigma if sigma is None else sigma
        #----------------------------------
        mutations = np.tile(np.identity(sigma.shape[1]), reps=(sigma.shape[0], 1))
        sigma_mut = 1 * np.logical_xor( np.repeat(sigma, repeats=sigma.shape[1], axis=0), mutations )
        #----------------------------------
        return sigma_mut


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_mutant_set(self):
        sigma_mut = self.generate_mutant_phenotypes()
        #---------------------------------- 
        b_mut     = np.repeat(self.b, repeats=self.sigma.shape[1], axis=0)   if self.b.ndim == 2   else self.b
        k_mut     = np.repeat(self.k, repeats=self.sigma.shape[1], axis=0)   if self.k.ndim == 2   else self.k
        eta_mut   = np.repeat(self.eta, repeats=self.sigma.shape[1], axis=0) if self.eta.ndim == 2 else self.eta
        l_mut     = np.repeat(self.l, repeats=self.sigma.shape[1], axis=0)   if self.l.ndim == 2   else self.l
        g_mut     = np.repeat(self.g, repeats=self.sigma.shape[1], axis=0)   if self.g.ndim == 2   else self.g
        xi_mut    = np.repeat(self.xi, repeats=self.sigma.shape[1], axis=0)  if self.xi.ndim == 2  else self.xi
        chi_mut   = np.repeat(self.chi, repeats=self.sigma.shape[1], axis=0) if self.chi.ndim == 2 else self.chi
        mu_mut    = np.repeat(self.mu, repeats=self.sigma.shape[1], axis=0)  if self.mu.ndim == 2  else self.mu
        #----------------------------------
        mutant_set = TypeSet(sigma=sigma_mut, b=b_mut, k=k_mut, eta=eta_mut, l=l_mut, g=g_mut, xi=xi_mut, chi=chi_mut, J=self.J, mu=mu_mut)
        #----------------------------------
        return mutant_set


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type(self, type_set=None, sigma=None, b=None, k=None, eta=None, l=None, g=None, xi=None, chi=None, mu=None, parent_index=None, parent_id=None, ref_type_idx=None): # index=None, 
        parent_idx   = np.where(self.type_ids==parent_id)[0] if parent_id is not None else parent_index
        ref_type_idx = ref_type_idx if ref_type_idx is not None else parent_idx if parent_idx is not None else 0
        #----------------------------------
        if(type_set is not None):
            if(isinstance(type_set, TypeSet)):
                new_type_set = type_set
            else:
                utils.error(f"Error in TypeSet add_type(): type_set argument expects object of TypeSet type.")
        else:
            new_type_set = TypeSet(sigma=sigma if sigma is not None else self.sigma[ref_type_idx], 
                                         b=b if b is not None else self.b[ref_type_idx],  
                                         k=k if k is not None else self.k[ref_type_idx],  
                                         eta=eta if eta is not None else self.eta[ref_type_idx],  
                                         l=l if l is not None else self.l[ref_type_idx],  
                                         g=g if g is not None else self.g[ref_type_idx],  
                                         xi=xi if xi is not None else self.xi[ref_type_idx],  
                                         chi=chi if chi is not None else self.chi[ref_type_idx],  
                                         mu=mu if mu is not None else self.mu[ref_type_idx]
                                         )
        # Check that the type set dimensions match the system dimensions:
        if(self.num_traits != new_type_set.num_traits): 
            utils.error(f"Error in TypeSet add_type(): The number of traits for added types ({new_type_set.num_traits}) does not match the number of type set traits ({self.num_traits}).")
        #----------------------------------
        self._sigma = self._sigma.add(new_type_set.sigma)
        self._b     = self._b.add(new_type_set.b)     if isinstance(self._b,   utils.ExpandableArray) else self._b
        self._k     = self._k.add(new_type_set.k)     if isinstance(self._k,   utils.ExpandableArray) else self._k
        self._eta   = self._eta.add(new_type_set.eta) if isinstance(self._eta, utils.ExpandableArray) else self._eta
        self._l     = self._l.add(new_type_set.l)     if isinstance(self._l,   utils.ExpandableArray) else self._l
        self._g     = self._g.add(new_type_set.g)     if isinstance(self._g,   utils.ExpandableArray) else self._g
        self._xi    = self._xi.add(new_type_set.xi)   if isinstance(self._xi,   utils.ExpandableArray) else self._xi
        self._chi   = self._chi.add(new_type_set.chi) if isinstance(self._chi, utils.ExpandableArray) else self._chi
        self._mu    = self._mu.add(new_type_set.mu)   if isinstance(self._mu,  utils.ExpandableArray) else self._mu
        #----------------------------------
        self._parent_indices.append(parent_index) # TODO: this does not work with lists of parent indexes
        #----------------------------------
        if(self._mutant_indices is not None):
            self._mutant_indices.add(np.arange((self.num_types-1)*self.num_traits, (self.num_types)*self.num_traits))
        #----------------------------------
        if(self._energy_costs is not None):
            self._energy_costs.add(new_type_set.energy_costs)
        #----------------------------------
        if(self._type_ids is not None):
            self._type_ids.extend(new_type_set.type_ids)
        #----------------------------------
        if(self._lineage_ids is not None):
            for i in range((self.num_types-1), (self.num_types-1)+new_type_set.num_types):
                new_lineage_id = self.add_type_to_phylogeny(i)
                self._lineage_ids.append(new_lineage_id)
        #----------------------------------
        return


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type_to_phylogeny(self, index=None, type_id=None, parent_index=None, parent_id=None):
        type_idx   = np.where(np.in1d(self.type_ids, utils.treat_as_list(type_id))) if type_id is not None else index
        parent_idx = np.where(self.type_ids==parent_id)[0] if parent_id is not None else self.parent_indices[type_idx]
        #----------------------------------
        if(parent_idx is None or np.isnan(parent_idx)):
            new_lineage_id = str( len(self.phylogeny.keys())+1 )
            self.phylogeny.update({ new_lineage_id: {} })
        else:
            parent_lineage_id = self.lineage_ids[parent_idx.astype(int)]
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

    def get_type(self, index=None, type_id=None):
        type_idx = np.where(np.in1d(self.type_ids, utils.treat_as_list(type_id))) if type_id is not None else index
        if(type_idx is None):
            utils.error(f"Error in TypeSet get_type(): A type index or type id must be given.")
        _num_types = 1 if type_idx.ndim == 0 else len(type_idx)
        #----------------------------------
        return TypeSet(sigma = self.sigma[type_idx], 
                        b    = self.b[type_idx]   if self.b.ndim == 2   else self.b, 
                        k    = self.k[type_idx]   if self.k.ndim == 2   else self.k, 
                        eta  = self.eta[type_idx] if self.eta.ndim == 2 else self.eta, 
                        l    = self.l[type_idx]   if self.l.ndim == 2   else self.l, 
                        g    = self.g[type_idx]   if self.g.ndim == 2   else self.g, 
                        xi   = self.xi[type_idx]  if self.xi.ndim == 2  else self.xi, 
                        chi  = self.chi[type_idx] if self.chi.ndim == 2 else self.chi, 
                        mu   = self.mu[type_idx]  if self.mu.ndim == 2  else self.mu,
                        J    = self.J ) 


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_type_id(self, index):
        return hash(tuple( self.sigma[index].tolist() ))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_mutant_indices(self, index):
        type_idx = utils.treat_as_list(index)
        return self.mutant_indices[type_idx].ravel()


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, index=None, type_id=None, include_mutants=False):
        type_idx = np.where(np.in1d(self.type_ids, utils.treat_as_list(type_id))) if type_id is not None else index
        if(type_idx is None):
            type_idx = np.arange(0, self.num_types, 1)
        #----------------------------------
        return {'num_types':    len(type_idx),
                'sigma':        self.sigma[type_idx],
                'b':            self.b   if self.b.ndim < 2   else self.b[type_idx],
                'k':            self.k   if self.k.ndim < 2   else self.k[type_idx],
                'eta':          self.eta if self.eta.ndim < 2 else self.eta[type_idx],
                'l':            self.l   if self.l.ndim < 2   else self.l[type_idx],
                'g':            self.g   if self.g.ndim < 2   else self.g[type_idx],
                'xi':           self.xi  if self.xi.ndim < 2  else self.xi[type_idx],
                'chi':          self.chi if self.chi.ndim < 2 else self.chi[type_idx],
                'J':            self.J,
                'mu':           self.mu  if self.mu.ndim < 2  else self.mu[type_idx],
                'energy_costs': self.energy_costs[type_idx]}


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder_types(self, order=None):
        type_order   = np.argsort(self.lineage_ids) if order is None else order
        if(len(type_order) < self.num_types):
            utils.error("Error in TypeSet.reorder_types(): The ordering provided has fewer indices than types.")
        #----------------------------------
        self._sigma = self._sigma.reorder(type_order)
        self._b     = self._b.reorder(type_order)   if isinstance(self._b,   utils.ExpandableArray)  else self._b
        self._k     = self._k.reorder(type_order)   if isinstance(self._k,   utils.ExpandableArray)  else self._k
        self._eta   = self._eta.reorder(type_order) if isinstance(self._eta, utils.ExpandableArray)  else self._eta
        self._l     = self._l.reorder(type_order)   if isinstance(self._l,   utils.ExpandableArray)  else self._l
        self._g     = self._g.reorder(type_order)   if isinstance(self._g,   utils.ExpandableArray)  else self._g
        self._xi    = self._xi.reorder(type_order)  if isinstance(self._xi,   utils.ExpandableArray) else self._xi
        self._chi   = self._chi.reorder(type_order) if isinstance(self._chi, utils.ExpandableArray)  else self._chi
        self._mu    = self._mu.reorder(type_order)  if isinstance(self._mu,  utils.ExpandableArray)  else self._mu
        self._energy_costs   = None # reset to recalculate upon next reference
        self._parent_indices = np.array(self.parent_indices)[type_order].tolist()
        self._type_ids       = np.array(self._type_ids)[type_order].tolist()
        self._lineage_ids    = np.array(self._lineage_ids)[type_order].tolist()
        self._mutant_indices = self._mutant_indices = self._mutant_indices.reorder(type_order) if self._mutant_indices is not None else None
        #----------------------------------
        return






