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
                       c              = 0,
                       chi            = None,
                       J              = None,
                       mu             = 0,
                       lineage_ids    = None,
                       has_type_ids   = True,
                       has_phylogeny  = True,
                       has_mutant_set = True ):

        #----------------------------------
        # Determine the number of types and traits,
        # and initialize sigma matrix:
        #----------------------------------
        if(isinstance(sigma, (list, np.ndarray))):
            sigma = np.array(sigma)
            if(sigma.ndim == 2):
                self.num_types  = sigma.shape[0]
                self.num_traits = sigma.shape[1]
            elif(sigma.ndim == 1):
                self.num_types  = 1
                self.num_traits = len(sigma)
        elif(num_types is not None and num_traits is not None):
            self.num_types  = num_types
            self.num_traits = num_traits
        else:
            utils.error("Error in TypeSet __init__(): Number of types and traits must be specified by providing a) a sigma matrix, or b) both num_types and num_traits values.")
        #----------------------------------
        self.sigma = utils.reshape(sigma, shape=(self.num_types, self.num_traits))


        #----------------------------------
        # Initialize parameter vectors/matrices:
        #----------------------------------

        # print("b->", b.shape)
        self.b   = self.preprocess_params(b,   has_trait_dim=True)
        # print("b\n", self.b, self.b.shape)
        self.k   = self.preprocess_params(k,   has_trait_dim=True)
        # print("k\n", self.k, self.k.shape)
        self.eta = self.preprocess_params(eta, has_trait_dim=True)
        # print("eta\n", self.eta, self.eta.shape)
        self.l   = self.preprocess_params(l,   has_trait_dim=True)
        # print("l\n", self.l, self.l.shape)
        self.g   = self.preprocess_params(g,   has_trait_dim=False)
        # print("g\n", self.g, self.g.shape)
        # print("c->", c.shape)
        self.c   = self.preprocess_params(c,   has_trait_dim=False)
        # print("c\n", self.c, self.c.shape)
        self.chi = self.preprocess_params(chi, has_trait_dim=True)
        # print("chi\n", self.chi, self.chi.shape)
        self.mu  = self.preprocess_params(mu,  has_trait_dim=False)
        # print("mu\n", self.mu, self.mu.shape)

        # self.b     = utils.reshape(b,     shape=(self.num_types, self.num_traits))
        # self.k     = utils.reshape(k,     shape=(self.num_types, self.num_traits))
        # self.eta   = utils.reshape(eta,   shape=(self.num_types, self.num_traits))
        # self.l     = utils.reshape(l,     shape=(self.num_types, self.num_traits))
        # self.g     = utils.reshape(g,     shape=(self.num_types, 1)).ravel()
        # self.c     = utils.reshape(c,     shape=(self.num_types, 1)).ravel()
        # self.chi   = utils.reshape(chi,   shape=(self.num_types, self.num_traits)) if chi is not None else None
        self.J     = utils.reshape(J,     shape=(self.num_traits, self.num_traits)) if J is not None else None
        # self.mu    = utils.reshape(mu,    shape=(self.num_types, 1)).ravel()

        # Calculate initial (biochem independent) phenotypic costs:
        self.update_phenotypic_costs()

        # Assign unique ids to each type (determined by hash of all param values):
        self.type_ids = np.array([self.get_type_id(i) for i in range(self.num_types)]) if has_type_ids else None

        self.lineage_ids    = np.array([None for i in range(self.num_types)] if lineage_ids is None else lineage_ids)
        self.parent_indices = np.array([None for i in range(self.num_types)])
        self.phylogeny      = {}
        if(has_phylogeny):
            for i in range(self.num_types):
                new_lineage_id = self.add_type_to_phylogeny(i)
                self.lineage_ids[i] = new_lineage_id

        if(has_mutant_set):
            self.generate_mutant_set() 
        else:
            self.mutant_set = None

        # print("done __init__")
        # exit()
                
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def preprocess_params(self, vals, has_trait_dim, dtype='float64'):
        arr = np.array(vals, dtype=dtype) if(isinstance(vals, (list, np.ndarray))) else np.array([vals], dtype=dtype)
        # if(not has_trait_dim and arr.ndim == 2 and arr.shape[1] == 1):
        #     arr = arr.flatten()
        #----------------------------------
        if(has_trait_dim):
            if(arr.ndim == 1):
                if(len(arr) == 1):
                    # print("return 1")
                    return np.repeat(arr, self.num_traits)
                elif(len(arr) == self.num_traits):
                    # print("return 2")
                    return arr # as is
                elif(len(arr) == self.num_types):
                    if(np.all(arr == arr[0])): # all elements equal
                        # print("return 3")
                        return np.full(shape=(self.num_traits,), fill_value=arr[0])
                    else:
                        # print("return 4")
                        return np.tile(arr.reshape(self.num_types, 1), (1, self.num_traits))
            elif(arr.ndim == 2):
                if(arr.shape[0] == self.num_types and arr.shape[1] == self.num_traits):
                    # print("return 5")
                    return arr # as is
                    # if(np.all(arr == arr[0])): # all rows equal
                        # print("return 5")
                    #     return arr[0]
                    # else:
                    #     print("return 6")
                    #     return arr # as is
        #----------------------------------
        else:
            if(arr.ndim == 1):
                if(len(arr) == 1):
                    # print("return 7")
                    return arr[0] # single val as scalar
                elif(len(arr) == self.num_types):
                    if(np.all(arr == arr[0])): # all elements equal
                        # print("return 8")
                        return arr[0] # single val as scalar
                    else:
                        # print("return 9")
                        return arr.reshape(self.num_types, 1)
            elif(arr.ndim == 2):
                if(arr.shape[0] == self.num_types and arr.shape[1] == 1):
                    # print("return 10")
                    return arr # as is

        #----------------------------------
        # If none of the above conditions met (hasn't returned by now):
        if(has_trait_dim):
            utils.error(f"Error in TypeSet.preprocess_params(): input with shape {arr.shape} does not correspond to the number of types ({self.num_types}) and/or traits ({self.num_traits}).")
        else:
            utils.error(f"Error in TypeSet.preprocess_params(): input with shape {arr.shape} does not correspond to the number of types ({self.num_types}) (has_trait_dim is {has_trait_dim}).")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update_phenotypic_costs(self): #, sigma=None, c=None, chi=None, J=None):
        # sigma = self.sigma if sigma is None else sigma
        # c     = self.c if c is None else c
        # chi   = self.chi if chi is None else chi
        # J     = self.J if J is None else J
        #----------------------------------
        costs = 0 + (self.c.ravel() if self.c.ndim == 2 else self.c)
        if(self.chi is not None):
            costs += np.sum(self.sigma * self.chi, axis=1)
        if(self.J is not None):
            costs += -1 * np.sum(self.sigma * np.dot(self.sigma, self.J), axis=1)
        #----------------------------------
        self.energy_costs = costs
        return self.energy_costs


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_mutant_phenotypes(self, sigma=None):
        sigma = self.sigma if sigma is None else sigma
        #----------------------------------
        mutations = np.tile(np.identity(sigma.shape[1]), reps=(sigma.shape[0], 1))
        sigma_mut = 1 * np.logical_xor( np.repeat(sigma, repeats=sigma.shape[1], axis=0), mutations )
        #----------------------------------
        return sigma_mut


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_mutant_set(self, has_type_ids=False, has_phylogeny=False): # sigma=None, J=None, 
        # print("\ngenerate_mutant_set")
        # sigma = self.sigma if sigma is None else sigma
        # J     = self.J if J is None else J
        #----------------------------------
        sigma_mut = self.generate_mutant_phenotypes()
        #---------------------------------- 
        b_mut     = np.repeat(self.b, repeats=self.sigma.shape[1], axis=0)   if self.b.ndim == 2   else self.b
        k_mut     = np.repeat(self.k, repeats=self.sigma.shape[1], axis=0)   if self.k.ndim == 2   else self.k
        eta_mut   = np.repeat(self.eta, repeats=self.sigma.shape[1], axis=0) if self.eta.ndim == 2 else self.eta
        l_mut     = np.repeat(self.l, repeats=self.sigma.shape[1], axis=0)   if self.l.ndim == 2   else self.l
        g_mut     = np.repeat(self.g, repeats=self.sigma.shape[1], axis=0)   if self.g.ndim == 2   else self.g
        c_mut     = np.repeat(self.c, repeats=self.sigma.shape[1], axis=0)   if self.c.ndim == 2   else self.c
        chi_mut   = np.repeat(self.chi, repeats=self.sigma.shape[1], axis=0) if self.chi.ndim == 2 else self.chi
        mu_mut    = np.repeat(self.mu, repeats=self.sigma.shape[1], axis=0)  if self.mu.ndim == 2  else self.mu

        # print("b_mut\n", b_mut, b_mut.shape)
        #----------------------------------
        mutant_set = TypeSet(sigma=sigma_mut, b=b_mut, k=k_mut, eta=eta_mut, l=l_mut, g=g_mut, c=c_mut, chi=chi_mut, J=self.J, mu=mu_mut, 
                             has_type_ids=has_type_ids, has_phylogeny=has_phylogeny, has_mutant_set=False)
        #----------------------------------
        self.mutant_set = mutant_set
        # return mutant_set


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def do_add_fromempty(self, new_type_set, new_type_idx):

    #     # print("self", self.sigma.shape)
    #     # print("idx ", new_type_idx)
    #     # print("new ", new_type_set.sigma.shape)
        
    #     sigma_new = np.empty(shape=(self.num_types+new_type_set.num_types, self.num_traits))
    #     sigma_new[:new_type_idx] = self.sigma[:new_type_idx]
    #     sigma_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.sigma
    #     sigma_new[new_type_idx+new_type_set.num_types:] = self.sigma[new_type_idx:]
    #     self.sigma = sigma_new

    #     b_new = np.empty(shape=(self.num_types+new_type_set.num_types, self.num_traits))
    #     b_new[:new_type_idx] = self.b[:new_type_idx]
    #     b_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.b
    #     b_new[new_type_idx+new_type_set.num_types:] = self.b[new_type_idx:]
    #     self.b = b_new

    #     k_new = np.empty(shape=(self.num_types+new_type_set.num_types, self.num_traits))
    #     k_new[:new_type_idx] = self.k[:new_type_idx]
    #     k_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.k
    #     k_new[new_type_idx+new_type_set.num_types:] = self.k[new_type_idx:]
    #     self.k = k_new

    #     l_new = np.empty(shape=(self.num_types+new_type_set.num_types, self.num_traits))
    #     l_new[:new_type_idx] = self.l[:new_type_idx]
    #     l_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.l
    #     l_new[new_type_idx+new_type_set.num_types:] = self.l[new_type_idx:]
    #     self.l = l_new

    #     eta_new = np.empty(shape=(self.num_types+new_type_set.num_types, self.num_traits))
    #     eta_new[:new_type_idx] = self.eta[:new_type_idx]
    #     eta_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.eta
    #     eta_new[new_type_idx+new_type_set.num_types:] = self.eta[new_type_idx:]
    #     self.eta = eta_new

    #     g_new = np.empty(shape=(self.num_types+new_type_set.num_types))
    #     g_new[:new_type_idx] = self.g[:new_type_idx]
    #     g_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.g
    #     g_new[new_type_idx+new_type_set.num_types:] = self.g[new_type_idx:]
    #     self.g = g_new

    #     c_new = np.empty(shape=(self.num_types+new_type_set.num_types))
    #     c_new[:new_type_idx] = self.c[:new_type_idx]
    #     c_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.c
    #     c_new[new_type_idx+new_type_set.num_types:] = self.c[new_type_idx:]
    #     self.c = c_new

    #     chi_new = np.empty(shape=(self.num_types+new_type_set.num_types, self.num_traits))
    #     chi_new[:new_type_idx] = self.chi[:new_type_idx]
    #     chi_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.chi
    #     chi_new[new_type_idx+new_type_set.num_types:] = self.chi[new_type_idx:]
    #     self.chi = chi_new

    #     mu_new = np.empty(shape=(self.num_types+new_type_set.num_types))
    #     mu_new[:new_type_idx] = self.mu[:new_type_idx]
    #     mu_new[new_type_idx:new_type_idx+new_type_set.num_types] = new_type_set.mu
    #     mu_new[new_type_idx+new_type_set.num_types:] = self.mu[new_type_idx:]
    #     self.mu = mu_new

    # def do_add_concatenate(self, new_type_set, new_type_idx):

    #     self.sigma = np.concatenate([self.sigma[:new_type_idx], new_type_set.sigma, self.sigma[new_type_idx:]])
    #     self.b = np.concatenate([self.b[:new_type_idx], new_type_set.b, self.b[new_type_idx:]])
    #     self.k = np.concatenate([self.k[:new_type_idx], new_type_set.k, self.k[new_type_idx:]])
    #     self.eta = np.concatenate([self.eta[:new_type_idx], new_type_set.eta, self.eta[new_type_idx:]])
    #     self.l = np.concatenate([self.l[:new_type_idx], new_type_set.l, self.l[new_type_idx:]])
    #     self.g = np.concatenate([self.g[:new_type_idx], new_type_set.g, self.g[new_type_idx:]])
    #     self.c = np.concatenate([self.c[:new_type_idx], new_type_set.c, self.c[new_type_idx:]])
    #     self.chi = np.concatenate([self.chi[:new_type_idx], new_type_set.chi, self.chi[new_type_idx:]])
    #     self.mu = np.concatenate([self.sigma[:new_type_idx], new_type_set.sigma, self.sigma[new_type_idx:]])

    # # def do_add_insert(self, new_type_set, new_type_idx):

    # #     new_sigma = np.insert(self.sigma, new_type_idx, new_type_set.sigma, axis=0)
    # #     new_b     = np.insert(self.b,     new_type_idx, new_type_set.b,     axis=0)
    # #     new_k     = np.insert(self.k,     new_type_idx, new_type_set.k,     axis=0)
    # #     new_eta   = np.insert(self.eta,   new_type_idx, new_type_set.eta,   axis=0)
    # #     new_l     = np.insert(self.l,     new_type_idx, new_type_set.l,     axis=0)
    # #     new_g     = np.insert(self.g,     new_type_idx, new_type_set.g,     axis=0)
    # #     new_c     = np.insert(self.c,     new_type_idx, new_type_set.c,     axis=0)
    # #     new_mu    = np.insert(self.mu,    new_type_idx, new_type_set.mu,    axis=0)
    # #     new_chi   = np.insert(self.chi,   new_type_idx, new_type_set.chi,   axis=0)

    # def do_add_insert(self, new_type_set, new_type_idx):

    #     self.sigma = np.insert(self.sigma, new_type_idx, new_type_set.sigma, axis=0)
    #     self.b     = np.insert(self.b,     new_type_idx, new_type_set.b,     axis=0)
    #     self.k     = np.insert(self.k,     new_type_idx, new_type_set.k,     axis=0)
    #     self.eta   = np.insert(self.eta,   new_type_idx, new_type_set.eta,   axis=0)
    #     self.l     = np.insert(self.l,     new_type_idx, new_type_set.l,     axis=0)
    #     self.g     = np.insert(self.g,     new_type_idx, new_type_set.g,     axis=0)
    #     self.c     = np.insert(self.c,     new_type_idx, new_type_set.c,     axis=0)
    #     self.mu    = np.insert(self.mu,    new_type_idx, new_type_set.mu,    axis=0)
    #     self.chi   = np.insert(self.chi,   new_type_idx, new_type_set.chi,   axis=0)


    def add_type(self, type_set=None, sigma=None, b=None, k=None, eta=None, l=None, g=None, c=None, chi=None, mu=None, index=None, parent_index=None, parent_id=None):
        new_type_idx = index if index is not None else self.num_types # default to adding to end of matrices
        ref_type_idx = new_type_idx - 1
        parent_idx   = np.where(self.type_ids==parent_id)[0] if parent_id is not None else parent_index
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
                                         c=c if c is not None else self.c[ref_type_idx],  
                                         chi=chi if chi is not None else self.chi[ref_type_idx],  
                                         mu=mu if mu is not None else self.mu[ref_type_idx],
                                         has_mutant_set=False, has_type_ids=False)
        # Check that the type set dimensions match the system dimensions:
        if(self.num_traits != new_type_set.num_traits): 
            utils.error(f"Error in TypeSet add_type(): The number of traits for added types ({new_type_set.num_traits}) does not match the number of type set traits ({self.num_traits}).")
        #----------------------------------
        
        #----------------------------------
        #----------------------------------
        #----------------------------------

        self.sigma = np.insert(self.sigma, new_type_idx, new_type_set.sigma, axis=0)


        self.b     = np.insert(self.b,   new_type_idx, new_type_set.b,   axis=0) if self.b.ndim == 2   else self.b
        self.k     = np.insert(self.k,   new_type_idx, new_type_set.k,   axis=0) if self.k.ndim == 2   else self.k
        self.eta   = np.insert(self.eta, new_type_idx, new_type_set.eta, axis=0) if self.eta.ndim == 2 else self.eta
        self.l     = np.insert(self.l,   new_type_idx, new_type_set.l,   axis=0) if self.l.ndim == 2   else self.l
        self.g     = np.insert(self.g,   new_type_idx, new_type_set.g,   axis=0) if self.g.ndim == 2   else self.g
        self.c     = np.insert(self.c,   new_type_idx, new_type_set.c,   axis=0) if self.c.ndim == 2   else self.c
        self.mu    = np.insert(self.mu,  new_type_idx, new_type_set.mu,  axis=0) if self.mu.ndim == 2  else self.mu
        self.chi   = np.insert(self.chi, new_type_idx, new_type_set.chi, axis=0) if self.chi.ndim == 2 else self.chi

        #----------------------------------
        #----------------------------------
        #----------------------------------

        # self.sigma = np.insert(self.sigma, new_type_idx, new_type_set.sigma, axis=0)
        # self.b     = np.insert(self.b,     new_type_idx, new_type_set.b,     axis=0)
        # self.k     = np.insert(self.k,     new_type_idx, new_type_set.k,     axis=0)
        # self.eta   = np.insert(self.eta,   new_type_idx, new_type_set.eta,   axis=0)
        # self.l     = np.insert(self.l,     new_type_idx, new_type_set.l,     axis=0)
        # self.g     = np.insert(self.g,     new_type_idx, new_type_set.g,     axis=0)
        # self.c     = np.insert(self.c,     new_type_idx, new_type_set.c,     axis=0)
        # self.mu    = np.insert(self.mu,    new_type_idx, new_type_set.mu,    axis=0)
        # self.chi   = np.insert(self.chi,   new_type_idx, new_type_set.chi,   axis=0)

        #----------------------------------

        self.energy_costs = np.insert(self.energy_costs, new_type_idx, new_type_set.energy_costs)
        # self.update_phenotypic_costs()
        #----------------------------------
        if(self.type_ids is not None):
            self.type_ids = np.insert(self.type_ids, new_type_idx, new_type_set.type_ids)
        # self.type_ids = np.array([self.get_type_id(i) for i in range(self.num_types)])
        #----------------------------------
        self.num_types = self.sigma.shape[0]
        #----------------------------------
        for i in range(new_type_idx, new_type_idx+new_type_set.num_types):
            new_lineage_id      = self.add_type_to_phylogeny(i, parent_idx, parent_id)
            self.lineage_ids    = np.insert(self.lineage_ids, i, new_lineage_id)
            self.parent_indices = np.insert(self.parent_indices, i, parent_idx)
        #----------------------------------
        # print("here")
        if(self.mutant_set is not None):
            # print("doin it")
            if(new_type_set.mutant_set is None):
                # print("\nnew_type_set.generate_mutant_set()")
                new_type_set.generate_mutant_set()
            # print(new_type_set.mutant_set.sigma.shape)
            self.mutant_set.add_type(new_type_set.mutant_set, index=new_type_idx*self.num_traits)
            # print(self.mutant_set.sigma.shape)
        # self.mutant_set = self.generate_mutant_set()
        #----------------------------------
        # print("check")
        # if(self.mutant_set is not None): 
            # print(self.b, self.b.shape)
            # print(new_type_set.b, new_type_set.b.shape)
            # print(self.mutant_set.b, self.mutant_set.b.shape)
        return


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_type_to_phylogeny(self, new_type_index, parent_index=None, parent_id=None):
        parent_index = np.where(self.type_ids==parent_id)[0] if parent_id is not None else parent_index
        #----------------------------------
        if(parent_index is None):
            new_lineage_id = str( len(self.phylogeny.keys())+1 )
            self.phylogeny.update({ new_lineage_id: {} })
        else:
            parent_lineage_id = self.lineage_ids[parent_index]
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
        # print("\nget_type")
        return TypeSet(sigma = self.sigma[type_idx], 
                        b    = self.b[type_idx].reshape(_num_types, self.num_traits)   if self.b.ndim == 2   else self.b, 
                        k    = self.k[type_idx].reshape(_num_types, self.num_traits)   if self.k.ndim == 2   else self.k, 
                        eta  = self.eta[type_idx].reshape(_num_types, self.num_traits) if self.eta.ndim == 2 else self.eta, 
                        l    = self.l[type_idx].reshape(_num_types, self.num_traits)   if self.l.ndim == 2   else self.l, 
                        g    = self.g[type_idx].reshape(_num_types, 1)                 if self.g.ndim == 2   else self.g, 
                        c    = self.c[type_idx].reshape(_num_types, 1)                 if self.c.ndim == 2   else self.c, 
                        chi  = self.chi[type_idx].reshape(_num_types, self.num_traits) if self.chi.ndim == 2 else self.chi, 
                        mu   = self.mu[type_idx].reshape(_num_types, 1)                if self.mu.ndim == 2  else self.mu,
                        lineage_ids = np.array(self.lineage_ids[type_idx]).flatten(),
                        has_mutant_set = False)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_type_id(self, index):
        # TODO: Include other (non-sigma) param vals back in the hash (broke after letting params be 1d):
        # return hash(tuple( self.sigma[index].tolist() + self.b[index].tolist() + self.k[index].tolist() + self.eta[index].tolist()
        #                    + self.l[index].tolist() + [self.g[index]] + [self.g[index]] + self.chi[index].tolist() + [self.mu[index]] ))
        return hash(tuple( self.sigma[index].tolist() ))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_mutant_indices(self, index):
        type_idx = utils.treat_as_list(index)
        # print( [i for i in range(self.mutant_set.num_types) if i // self.mutant_set.num_traits in type_idx] )
        # print( np.where(np.in1d(np.arange(0, self.mutant_set.num_types, 1)//self.mutant_set.num_traits, type_idx))[0] )
        return np.where(np.in1d(np.arange(0, self.mutant_set.num_types, 1)//self.mutant_set.num_traits, type_idx))[0]

    def get_parent_indices(self, index):
        mut_idx = utils.treat_as_list(index)
        return mut_idx // self.num_traits

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, index=None, type_id=None, include_mutants=False):
        # print("\nTypeSet.get_dynamics_params()")
        type_idx = np.where(np.in1d(self.type_ids, utils.treat_as_list(type_id))) if type_id is not None else index
        if(type_idx is None):
            type_idx = np.arange(0, self.num_types, 1)
        #----------------------------------
        # print(type_idx)
        mut_idx = self.get_mutant_indices(type_idx)
        # print(mut_idx)

        if(include_mutants):
            return {'num_types':    len(type_idx),
                    'num_mutants':  len(mut_idx),
                    'sigma':        np.concatenate([self.sigma[type_idx], self.mutant_set.sigma[mut_idx]]),
                    'b':            np.concatenate([self.b[type_idx],     self.mutant_set.b[mut_idx]])   if self.b.ndim == 2   else self.b,
                    'k':            np.concatenate([self.k[type_idx],     self.mutant_set.k[mut_idx]])   if self.k.ndim == 2   else self.k,
                    'eta':          np.concatenate([self.eta[type_idx],   self.mutant_set.eta[mut_idx]]) if self.eta.ndim == 2 else self.eta,
                    'l':            np.concatenate([self.l[type_idx],     self.mutant_set.l[mut_idx]])   if self.l.ndim == 2   else self.l,
                    'g':            np.concatenate([self.g[type_idx],     self.mutant_set.g[mut_idx]])   if self.g.ndim == 2   else self.g,
                    'c':            np.concatenate([self.c[type_idx],     self.mutant_set.c[mut_idx]])   if self.c.ndim == 2   else self.c,
                    'chi':          np.concatenate([self.chi[type_idx],   self.mutant_set.chi[mut_idx]]) if self.chi.ndim == 2 else self.chi,
                    'J':            self.J,
                    'mu':           np.concatenate([self.mu[type_idx],           self.mutant_set.mu[mut_idx]]) if self.mu.ndim == 2 else self.mu,
                    'energy_costs': np.concatenate([self.energy_costs[type_idx], self.mutant_set.energy_costs[mut_idx]])}
        else:
            return {'num_types':    self.num_types,
                    'sigma':        self.sigma,
                    'b':            self.b,
                    'k':            self.k,
                    'eta':          self.eta,
                    'l':            self.l,
                    'g':            self.g,
                    'c':            self.c,
                    'chi':          self.chi,
                    'J':            self.J,
                    'mu':           self.mu,
                    'energy_costs': self.energy_costs}

        # params = {'num_types':      len(type_idx),
        #             'sigma':        self.sigma[type_idx],
        #             'b':            self.b   if self.b.ndim < 2   else self.b[type_idx],
        #             'k':            self.k   if self.k.ndim < 2   else self.k[type_idx],
        #             'eta':          self.eta if self.eta.ndim < 2 else self.eta[type_idx],
        #             'l':            self.l   if self.l.ndim < 2   else self.l[type_idx],
        #             'g':            self.g   if self.g.ndim < 2   else self.g[type_idx],
        #             'c':            self.c   if self.c.ndim < 2   else self.c[type_idx],
        #             'chi':          self.chi if self.chi.ndim < 2 else self.chi[type_idx],
        #             'J':            self.J,
        #             'mu':           self.mu  if self.mu.ndim < 2  else self.mu[type_idx],
        #             'energy_costs': self.energy_costs[type_idx]}

        # if(include_mutants):
        #     params.update({
        #             'num_mutants':      self.mutant_set.num_types,
        #             'sigma_mut':        self.mutant_set.sigma[mut_idx],
        #             'b_mut':            self.mutant_set.b   if self.mutant_set.b.ndim < 2   else self.mutant_set.b[mut_idx],
        #             'k_mut':            self.mutant_set.k   if self.mutant_set.k.ndim < 2   else self.mutant_set.k[mut_idx],
        #             'eta_mut':          self.mutant_set.eta if self.mutant_set.eta.ndim < 2 else self.mutant_set.eta[mut_idx],
        #             'l_mut':            self.mutant_set.l   if self.mutant_set.l.ndim < 2   else self.mutant_set.l[mut_idx],
        #             'g_mut':            self.mutant_set.g   if self.mutant_set.g.ndim < 2   else self.mutant_set.g[mut_idx],
        #             'c_mut':            self.mutant_set.c   if self.mutant_set.c.ndim < 2   else self.mutant_set.c[mut_idx],
        #             'chi_mut':          self.mutant_set.chi if self.mutant_set.chi.ndim < 2 else self.mutant_set.chi[mut_idx],
        #             'J_mut':            self.mutant_set.J,
        #             'mu_mut':           self.mutant_set.mu  if self.mutant_set.mu.ndim < 2  else self.mutant_set.mu[mut_idx],
        #             'energy_costs_mut': self.mutant_set.energy_costs[mut_idx]})

        # print("dynamics params:\n", params)
        # exit()

        return params

        # if(include_mutants):
        #     return {'num_types':    len(type_idx),
        #             'num_mutants':  len(mut_idx),
        #             'sigma':        np.concatenate([self.sigma[type_idx], self.mutant_set.sigma[mut_idx]]),
        #             'b':            np.concatenate([self.b[type_idx],     self.mutant_set.b[mut_idx]]),
        #             'k':            np.concatenate([self.k[type_idx],     self.mutant_set.k[mut_idx]]),
        #             'eta':          np.concatenate([self.eta[type_idx],   self.mutant_set.eta[mut_idx]]),
        #             'l':            np.concatenate([self.l[type_idx],     self.mutant_set.l[mut_idx]]),
        #             'g':            np.concatenate([self.g[type_idx],     self.mutant_set.g[mut_idx]]),
        #             'c':            np.concatenate([self.c[type_idx],     self.mutant_set.c[mut_idx]]),
        #             'chi':          np.concatenate([self.chi[type_idx],   self.mutant_set.chi[mut_idx]]),
        #             'J':            self.J,
        #             'mu':           np.concatenate([self.mu[type_idx],           self.mutant_set.mu[mut_idx]]),
        #             'energy_costs': np.concatenate([self.energy_costs[type_idx], self.mutant_set.energy_costs[mut_idx]])}
        # else:
        #     return {'num_types':    self.num_types,
        #             'sigma':        self.sigma,
        #             'b':            self.b,
        #             'k':            self.k,
        #             'eta':          self.eta,
        #             'l':            self.l,
        #             'g':            self.g,
        #             'c':            self.c,
        #             'chi':          self.chi,
        #             'J':            self.J,
        #             'mu':           self.mu,
        #             'energy_costs': self.energy_costs}

        # # TODO: make possible to get multiple types by list of indices or ids
        # type_idx = np.where(self.type_ids==type_id)[0] if type_id is not None else index
        # if(type_idx is None):
        #     if(include_mutants):
        #         return {'num_types':    self.num_types,
        #                 'num_mutants':  self.mutant_set.num_types,
        #                 'sigma':        np.concatenate([self.sigma,        self.mutant_set.sigma]),
        #                 'b':            np.concatenate([self.b,            self.mutant_set.b]),
        #                 'k':            np.concatenate([self.k,            self.mutant_set.k]),
        #                 'eta':          np.concatenate([self.eta,          self.mutant_set.eta]),
        #                 'l':            np.concatenate([self.l,            self.mutant_set.l]),
        #                 'g':            np.concatenate([self.g,            self.mutant_set.g]),
        #                 'c':            np.concatenate([self.c,            self.mutant_set.c]),
        #                 'chi':          np.concatenate([self.chi,          self.mutant_set.chi]),
        #                 'J':            self.J,
        #                 'mu':           np.concatenate([self.mu,           self.mutant_set.mu]),
        #                 'energy_costs': np.concatenate([self.energy_costs, self.mutant_set.energy_costs])}
        #     else:
        #         return {'num_types':    self.num_types,
        #                 'sigma':        self.sigma,
        #                 'b':            self.b,
        #                 'k':            self.k,
        #                 'eta':          self.eta,
        #                 'l':            self.l,
        #                 'g':            self.g,
        #                 'c':            self.c,
        #                 'chi':          self.chi,
        #                 'J':            self.J,
        #                 'mu':           self.mu,
        #                 'energy_costs': self.energy_costs}
        #         # return (self.num_types, self.sigma, self.b, self.k, self.eta, self.l, self.g, self.c, self.chi, self.J, self.mu, self.energy_costs)
        # else:
        #     return {'num_types':    1,
        #             'sigma':        self.sigma[type_idx],
        #             'b':            self.b[type_idx],
        #             'k':            self.k[type_idx],
        #             'eta':          self.eta[type_idx],
        #             'l':            self.l[type_idx],
        #             'g':            self.g[type_idx],
        #             'c':            self.c[type_idx],
        #             'chi':          self.chi[type_idx],
        #             'J':            self.J[type_idx,:],
        #             'mu':           self.mu[type_idx],
        #             'energy_costs': self.energy_costs[type_idx]}
        #     # return (1, self.sigma[type_idx], self.b[type_idx], self.k[type_idx], self.eta[type_idx], self.l[type_idx], self.g[type_idx], self.c[type_idx], self.chi[type_idx], self.J[type_idx, :] if self.J is not None else None, self.mu[type_idx], self.energy_costs[type_idx])
        







        









