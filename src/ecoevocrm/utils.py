import sys
import numpy as np
import scipy
# from numba import jit

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SystemParameter():

    def __init__(self, values, num_types, num_traits, force_type_dim=False, force_trait_dim=False):

        self.num_types  = num_types
        self.num_traits = num_traits

        self.force_type_dim  = force_type_dim
        self.force_trait_dim = force_trait_dim

        if(not isinstance(values, (list, np.ndarray))):  # scalar
            if(force_type_dim and force_trait_dim):
                self._values = ExpandableArray( np.full(shape=(num_types, num_traits), fill_value=values) )
                self.has_type_dim  = True
                self.has_trait_dim = True
            elif(force_type_dim):
                self._values = np.full(shape=(num_types,), fill_value=values)
                self.has_type_dim  = True
                self.has_trait_dim = False
            elif(force_trait_dim):
                self._values = np.full(shape=(num_traits,), fill_value=values)
                self.has_type_dim  = False
                self.has_trait_dim = True
            else:
                self._values = np.array([values])[0] # makes this a np.int/float which has shape and ndim attributes
                self.has_type_dim  = False
                self.has_trait_dim = False
        else:  # list/array
            arr = np.array(values)
            if(arr.ndim == 1):
                if(len(arr) == num_traits):
                    if(force_type_dim):
                        if(np.all(arr == arr[0]) and not force_trait_dim):  # all elements equal
                            self._values = np.full(shape=(num_types,), fill_value=arr[0])
                            self.has_type_dim  = True
                            self.has_trait_dim = False
                        else:
                            self._values = ExpandableArray( np.tile(arr.reshape(1, self.num_traits), (self.num_types, 1)) )
                            self.has_type_dim  = True
                            self.has_trait_dim = True
                    else:
                        if(np.all(arr == arr[0]) and not force_trait_dim):  # all elements equal
                            self._values = arr[0]
                            self.has_type_dim  = False
                            self.has_trait_dim = False
                        else:
                            self._values = arr
                            self.has_type_dim  = False
                            self.has_trait_dim = True
                elif(len(arr) == num_types):
                    if(force_trait_dim):
                        if(np.all(arr == arr[0]) and not force_type_dim):  # all elements equal
                            self._values = np.full(shape=(num_traits,), fill_value=arr[0])
                            self.has_type_dim  = False
                            self.has_trait_dim = True
                        else:
                            self._values = ExpandableArray( np.tile(arr.reshape(self.num_types, 1), (1, self.num_traits)) )
                            self.has_type_dim  = True
                            self.has_trait_dim = True
                    else:
                        if(np.all(arr == arr[0]) and not force_type_dim):  # all elements equal
                            self._values = arr[0]
                            self.has_type_dim  = False
                            self.has_trait_dim = False
                        else:
                            self._values = arr
                            self.has_type_dim  = True
                            self.has_trait_dim = False
                else:
                    error(f"Error in SystemParameter.__init__(): input with shape {arr.shape} does not correspond to the number of types ({num_types}) or traits ({num_traits}).")
            elif(arr.ndim == 2):
                if(arr.shape[0] == num_types and arr.shape[1] == num_traits):
                    self._values = ExpandableArray( arr )  # as is

                    self.has_type_dim  = True
                    self.has_trait_dim = True
                else:
                    error(f"Error in SystemParameter.__init__(): input with shape {arr.shape} does not correspond to the number of types ({num_types}) and traits ({num_traits}).")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def shape(self):
        return self._values.shape

    @property
    def ndim(self):
        return self._values.ndim

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def values(self, type=None, trait=None, force_type_dim=False, force_trait_dim=False):
        type_idx  = treat_as_list(type) if type is not None else range(self.num_types)
        trait_idx = treat_as_list(trait) if trait is not None else range(self.num_traits)
        # - - - -
        if(self.ndim == 0):  # scalar
            if(force_type_dim and force_trait_dim):
                return np.full(shape=(len(type_idx), len(trait_idx)), fill_value=self._values)
            elif(force_type_dim):
                return np.full(shape=(len(type_idx),), fill_value=self._values)
            elif(force_trait_dim):
                return np.full(shape=(len(trait_idx),), fill_value=self._values)
            else:
                return self._values
        elif(self.ndim == 1):  # 1d array
            if(len(self._values) == self.num_traits):
                if(force_type_dim):
                    return np.tile(self._values[trait_idx].reshape(1, len(trait_idx)), (len(type_idx), 1))
                else:
                    return self._values[trait_idx]
            elif(len(self._values) == self.num_types):
                if(force_trait_dim):
                    return np.tile(self._values[type_idx].reshape(len([type_idx]), 1), (1, len(trait_idx)))
                else:
                    return self._values[type_idx]
            else:
                error(f"Error in SystemParameter.values(): shape of self._values does not match the number of types nor traits (shouldn't be allowed).")
        else:  # 2d ExpandableArray
            return self._values.values[type_idx][:, trait_idx]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_values(self, set_values, type=None, trait=None):
        # This function requires that the set_values match the dimensionality of the existing self._values (at least for now)
        # This function was hastily implemented so may have errors and/or unaccounted-for cases
        type_idx  = treat_as_list(type) if type is not None else range(self.num_types)
        trait_idx = treat_as_list(trait) if trait is not None else range(self.num_traits)
        # - - - -
        if(self.ndim == 0 and set_values.ndim == 0):  # scalar
            self._values = set_values
        elif(self.ndim == 1 and set_values.ndim == 1):  # 1d array
            if(len(self._values) == self.num_types):
                self._values[type_idx] = set_values
            elif(len(self._values) == self.num_traits):
                self._values[trait_idx] = set_values
        elif(self.ndim == 2 and set_values.ndim == 2):  # 2d ExpandableArray
            self._values._arr[type_idx, trait_idx] = set_values
        else:
            error(f"Error in SystemParameter.set_values(): This function currently requires that set_values matches the dimensionality of the existing self._values.")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_type(self, type, values_only=False):
        type_idx  = treat_as_list(type)
        # - - - -
        type_values = self.values()[type_idx] if self.has_type_dim else self.values()
        if(type_values.ndim == 2 and len(type_idx) == 1):
            type_values = type_values.ravel()
        # - - - -
        if(values_only):
            return type_values
        else:
            return SystemParameter(values=type_values, num_types=len(type_idx), num_traits=self.num_traits, force_type_dim=self.force_type_dim, force_trait_dim=self.force_trait_dim)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def combine(params_A, params_B, force_type_dim=False, force_trait_dim=False):
        try:
            if((params_A is None and params_B is not None) or (params_A._values is None and params_B._values is not None) or (0 in params_A.shape and not 0 in params_B.shape)): return params_B
            if((params_B is None and params_A is not None) or (params_B._values is None and params_A._values is not None) or (0 in params_B.shape and not 0 in params_A.shape)): return params_A
        except:
            pass
        # - - - -
        if(params_A.num_traits != params_B.num_traits):
            error(f"Error in SystemParameter.add(): The params_B have a different num_traits ({params_B.num_traits}) than the base params ({params_A.num_traits})")
        # - - - -
        valsA = params_A.values()
        valsB = params_B.values()
        comb_num_types       = params_A.num_types + params_B.num_types
        comb_num_traits      = params_A.num_traits
        comb_force_type_dim  = (force_type_dim | params_A.force_type_dim | params_B.force_type_dim)
        comb_force_trait_dim = (force_trait_dim | params_A.force_trait_dim | params_B.force_trait_dim)
        # - - - -
        # If both param sets are the same arrays:
        if(np.array_equal(valsA, valsB)):
            return SystemParameter(values=valsA, num_types=comb_num_types, num_traits=comb_num_traits, force_type_dim=comb_force_type_dim, force_trait_dim=comb_force_trait_dim)
        # - - - -
        # If all elements of both param sets are a constant value:
        if((valsA.ndim == 0 and np.all(valsB == valsA)) or (valsA.ndim > 0 and np.all(valsA == valsA[0]) and np.all(valsB == valsA[0]))):
            return SystemParameter(values=valsA[0], num_types=comb_num_types, num_traits=comb_num_traits, force_type_dim=comb_force_type_dim, force_trait_dim=comb_force_trait_dim)
        # - - - -
        # Otherwise, we need to make sure both param sets have type dimensions - and matching has_trait_dim - so we can concatenate them:
        if(not params_A.has_type_dim):
            valsA = valsA.reshape(1, params_A.num_traits) if params_A.has_trait_dim else valsA.reshape(1, 1)
            valsA = np.tile(valsA, (params_A.num_types, 1))
            valsA = valsA.ravel() if not params_A.has_trait_dim else valsA
        if(not params_B.has_type_dim):
            valsB = valsB.reshape(1, params_B.num_traits) if params_B.has_trait_dim else valsB.reshape(1, 1)
            valsB = np.tile(valsB, (params_B.num_types, 1))
            valsB = valsB.ravel() if not params_B.has_trait_dim else valsB
        if(params_A.has_trait_dim and not params_B.has_trait_dim):
            valsB = np.tile(valsB.reshape(params_B.num_types, 1), (1, params_B.num_traits))
        if(params_B.has_trait_dim and not params_A.has_trait_dim):
            valsA = np.tile(valsA.reshape(params_A.num_types, 1), (1, params_A.num_traits))
        return SystemParameter(values=np.concatenate([valsA, valsB]), num_types=comb_num_types, num_traits=comb_num_traits, force_type_dim=comb_force_type_dim, force_trait_dim=comb_force_trait_dim)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reorder(self, type_order):
        if(self.has_type_dim):
            if isinstance(self._values, ExpandableArray):
                self._values.reorder(type_order)
            else:
                self._values = self._values[type_order]
        return


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ExpandableArray():

    def __init__(self, arr, alloc_shape=None, dtype='float64', default_expand_factor=2):

        arr = np.atleast_2d(arr)

        self._shape = arr.shape

        self._alloc = alloc_shape if alloc_shape is not None else arr.shape

        self._arr = np.empty(shape=self._alloc, dtype=dtype)
        
        self._arr[:self._shape[0], :self._shape[1]] = arr

        self.dtype = dtype
        self.default_expand_factor = default_expand_factor

    @property
    def shape(self):
        return self._shape
    
    @property
    def alloc(self):
        return self._alloc

    @property
    def values(self):
        return self._arr[:self._shape[0], :self._shape[1]]

    @property
    def ndim(self):
        return self.values.ndim

    def expand_alloc(self, new_alloc):
        if(new_alloc[0] < self._alloc[0] or new_alloc[1] < self._alloc[1]):
            error("Error in ExpandableArray.expand_alloc(): new_alloc shape must be at least as large as current alloc shape in each dimension.")
        self._alloc = new_alloc
        exp_arr = np.empty(shape=self._alloc, dtype=self.dtype)
        exp_arr[:self._shape[0], :self._shape[1]] = self.values
        self._arr = exp_arr
        return self

    def add(self, added_arr, axis=0):
        added_arr = np.atleast_2d(added_arr)
        if(axis == 0):
            while(self._shape[0] + added_arr.shape[0] > self._alloc[0]):
                self.expand_alloc(new_alloc = (int(self._alloc[0]*self.default_expand_factor), self._alloc[1]))
            self._arr[self._shape[0]:self._shape[0]+added_arr.shape[0], :added_arr.shape[1]] = added_arr
            self._shape = (self._shape[0] + added_arr.shape[0], self._shape[1])
        elif(axis == 1):
            while(self._shape[1] + added_arr.shape[1] > self._alloc[1]):
                self.expand_alloc(new_alloc = (self._alloc[0], int(self._alloc[1]*self.default_expand_factor)))
            self._arr[:added_arr.shape[0], self._shape[1]:self._shape[1]+added_arr.shape[1]] = added_arr
            self._shape = (self._shape[0], self._shape[1] + added_arr.shape[1])
        return self
    
    def trim(self, alloc=None):
        self._alloc = self._shape if alloc is None else alloc
        self._arr   = self._arr[:self._alloc[0], :self._alloc[1]]
        return self

    def reorder(self, order):
        self._arr[:self._shape[0], :self._shape[1]] = self.values[order]
        return self


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def random_matrix(shape, mode, args={}, sparsity=0.0, symmetric=False, triangular=False, diagonal=None, ordered=False, shuffle=False, order_power=0, scale_range=None, seed=None):
    if(seed is not None):
        np.random.seed(seed)
    #--------------------------------
    # Generate random values according to one of the following random models:
    #--------------------------------
    if(mode == 'bernoulli'):
        M = np.random.binomial(n=1, p=(args['p'] if 'p' in args else 0.5), size=shape)
    elif(mode == 'binomial'):
        M = np.random.binomial(n=(args['n'] if 'n' in args else 1), p=(args['p'] if 'p' in args else 0.5), size=shape)
    elif(mode == 'uniform'):
        M = np.random.uniform(low=(args['min'] if 'min' in args else 0), high=(args['max'] if 'max' in args else 1), size=shape)
    elif(mode == 'normal'):
        M = np.random.normal(loc=(args['mean'] if 'mean' in args else 0), scale=(args['std'] if 'std' in args else 1), size=shape)
    elif(mode == 'logistic'):
        M = np.random.logistic(loc=(args['mean'] if 'mean' in args else 0), scale=(args['scale'] if 'scale' in args else 1), size=shape)
    elif(mode == 'exponential'):
        M = np.random.exponential(scale=(args['scale'] if 'scale' in args else 1), size=shape)
        M *=  np.random.choice([1, -1], size=shape)
    elif(mode == 'laplace'):
        M = np.random.laplace(loc=(args['mean'] if 'mean' in args else 0), scale=(args['scale'] if 'scale' in args else 1), size=shape)
    elif(mode == 'cauchy'):
        M = scipy.stats.cauchy.rvs(loc=(args['mean'] if 'mean' in args else 0), scale=(args['scale'] if 'scale' in args else 1), size=shape)
    elif(mode == 'exponential_normal'):
        loc   = args['mean'] if 'mean' in args else 0
        scale = args['scale'] if 'scale' in args else 1
        rate  = args['shape'] if 'shape' in args else args['rate'] if 'rate' in args else 1
        M = scipy.stats.exponnorm.rvs(K=1/(scale*rate), loc=loc, scale=rate, size=shape)
    elif(mode == 'gamma'):
        mean     = (args['mean'] if 'mean' in args else 0)
        coeffvar = (args['coeffvar'] if 'coeffvar' in args else args['cv'] if 'cv' in args else 0)
        scale    = mean * coeffvar ** 2
        shape    = mean / scale
        np.random.gamma(scale=scale, shape=shape, size=shape)
    elif(mode == 'tikhonov_sigmoid'):
        J_0    = args['J_0'] if 'J_0' in args else 0.2
        n_star = args['n_star'] if 'n_star' in args else 10
        delta  = args['delta'] if 'delta' in args else 3
        M = np.zeros(shape=shape)
        for i, j in np.ndindex(M.shape):
            if(i >= j):
                continue
            M[i,j] = np.random.normal( loc=0, scale=J_0*(1/(1 + np.exp((max(i+1, j+1) - n_star)/delta))) )   # +1s because i,j indices start at 0
    elif(mode == 'tikhonov_sigmoid_ordered'):
        J_0    = args['J_0'] if 'J_0' in args else 0.4
        n_star = args['n_star'] if 'n_star' in args else 10
        delta  = args['delta'] if 'delta' in args else 5
        M = np.zeros(shape=shape)
        vals   = [np.random.choice([1, -1]) * J_0/(1 + np.exp((i - n_star)/delta)) for i in range(int((M.shape[0]*M.shape[0]-M.shape[0])/2))]
        c = 0
        for j in range(M.shape[1]):
            for i in range(M.shape[0]):
                if(j <= i):
                    continue
                else:
                    M[i,j] = vals[c] * (1 if (j+i)%2==0 else -1)
                    c += 1
    elif(mode == 'choice'):
        M = np.random.choice(a=args['a'], size=shape)
    else:
        error(f"Error in random_matrix(): generator mode '{mode}' is not recognized.")
    #--------------------------------
    # Apply specified sparsity:
    # num_ = range(len(np.triu_indices(M.shape[0], k=0 if diagonal is not None else 1)[0])) if triangular else M.shape[1]*M.shape[0]
    # zeroed_indices = np.random.choice(, replace=False, size=int(M.shape[1]*M.shape[0]*sparsity))        
    if(triangular):
        active_indices   = np.triu_indices(M.shape[0], k=0 if diagonal is not None and diagonal != 0 else 1)
        zeroed_indices_i = np.random.choice(range(len(active_indices[0])), replace=False, size=int(len(active_indices[0])*sparsity))
        zeroed_indices   = (active_indices[0][zeroed_indices_i], active_indices[1][zeroed_indices_i])
        M[zeroed_indices] = 0
    else:
        zeroed_indices = np.random.choice(M.shape[1]*M.shape[0], replace=False, size=int(M.shape[1]*M.shape[0]*sparsity))
        M[np.unravel_index(zeroed_indices, M.shape)] = 0 
    #--------------------------------
    # Make symmetric, if applicable:
    if(symmetric):
        if(shape[0] != shape[1]):
            error(f"Error in random_matrix(): shape {shape} is not square and cannot be made symmetric.")
        M = np.tril(M) + np.triu(M.T, 1)
    #--------------------------------
    # Make triangular, if applicable:
    if(triangular):
        if(shape[0] != shape[1]):
            error(f"Error in random_matrix(): shape {shape} is not square and cannot be made triangular.")
        M *= 1 - np.tri(*M.shape, k=-1, dtype=np.bool)
    #--------------------------------
    # Set diagonal, if applicable:
    if(diagonal is not None):
        np.fill_diagonal(M, diagonal)
    #--------------------------------
    # Make ordered, if applicable:
    if(ordered):
        vals = np.array(sorted(M[M!=0], key=abs, reverse=True))
        c = 0
        for j in range(M.shape[1]):
            for i in range(M.shape[0]):
                if(M[i,j] != 0):
                    M[i,j] = vals[c]
                    c += 1

    if(shuffle): # I'm pretty sure this version works as desired (keep in mind order_power can be decimal, including between 0 and 1)
        vals = M[M!=0].copy()
        num_vals = len(vals)
        shuffled_vals = []
        while len(shuffled_vals) < num_vals:
            sel_i = np.random.choice(range(len(vals)), p=(np.abs(vals)**order_power)/(np.sum(np.abs(vals)**order_power))) if(len(vals) > 1) else 0
            shuffled_vals.append(vals[sel_i])
            vals = np.delete(vals, sel_i)
        c = 0
        for j in range(M.shape[1]):
            for i in range(M.shape[0]):
                if(M[i,j] != 0):
                    M[i,j] = shuffled_vals[c]
                    c += 1

    #--------------------------------
    # Scale values to desired range, if applicable:
    if(scale_range is not None):
        M[M != 0] = np.interp(M[M != 0], (M[M != 0].min(), M[M != 0].max()), (scale_range[0], scale_range[1]))
    #--------------------------------
    return M


def reshape(a, shape, prioritize_col_vector_if_1d=True, dtype='float64'):
    target_shape = shape
    if(isinstance(a, (list, np.ndarray)) and a.shape == target_shape):
        return a
    else:
        # Enforce that arr is a 2d numpy array:
        arr = np.array(a, dtype=dtype) if(isinstance(a, (list, np.ndarray))) else np.array([a], dtype=dtype)
        orig_shape = arr.shape
        if(arr.ndim == 1):
           arr = np.reshape(arr, (1, arr.size))
        # If a 1d array is given, tile it to match the system dimensions if possible:
        if(arr.shape[0] == 1):
           if((arr.shape[1] != target_shape[1] or (arr.shape[1] == target_shape[1] and prioritize_col_vector_if_1d)) and arr.shape[1] == target_shape[0]):
                arr = arr.T # make the row array a col array
           else:
                arr = np.tile(arr, (target_shape[0], 1))
        if(arr.shape[1] == 1):
           arr = np.tile(arr, (1, target_shape[1]))
        # Check if array was able to be reshaped to system dimensions:
        if(arr.shape != target_shape):
           if(shape is None):
                error(f"Error in reshape(): input with shape {orig_shape} could not be reshaped to the target dimensionality {target_shape}.")
        return  arr


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# @jit(nopython=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i, val in enumerate(vec):
        if val == item:
            return i
    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def binary_combinations(num_digits, exclude_all_zeros=False):
    import itertools
    combos = np.array([list(i) for i in itertools.product([0, 1], repeat=num_digits)])
    return combos if not exclude_all_zeros else combos[1:, :]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def treat_as_list(val):
    if(not isinstance(val, (list, np.ndarray)) and val is not None):
        return [val]
    elif(isinstance(val, (np.ndarray))):
        return val.ravel()
    else:
        return val


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def error(message, trigger_exit=True):
    print("\n"+message+"\n")
    if(trigger_exit):
        sys.exit()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_perturbations(vals, dist, args, mode, element_wise):
    # dist == constant: args={'val': ...}
    # dist == uniform:  args={'low': ..., 'high': ...}
    # dist == normal:   args={'mean': ..., 'std': ...}
    if(dist == 'constant'):
        perturb_vals = np.full_like(vals, fill_value=args['val']) if element_wise else args['val']
    elif(dist == 'uniform'):
        perturb_vals = np.random.uniform(low=args['low'], high=args['high'], size=(vals.shape if element_wise else 1))
    elif(dist == 'normal'):
        perturb_vals = np.random.normal(loc=args['mean'], scale=args['std'], size=(vals.shape if element_wise else 1))
    #----------------------------------
    if not element_wise or not isinstance(vals, (list, np.ndarray)):
        perturb_vals = perturb_vals.ravel()[0]
    #----------------------------------
    return perturb_vals


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sinusoid_series(T, dt=0.1, amplitude=1, period=2*np.pi, phase=0, shift=0, num_series=1, return_interp=True):
    amplitude = reshape(amplitude, shape=(1, num_series)).ravel()
    period    = reshape(period, shape=(1, num_series)).ravel()
    phase     = reshape(phase, shape=(1, num_series)).ravel()
    shift     = reshape(shift, shape=(1, num_series)).ravel()
    #--------------------------------
    t_series  = np.arange(0, T, step=dt)
    #--------------------------------
    y_series = []
    for i in range(num_series):
        y = ((amplitude[i] * np.sin(period[i] * (t_series + phase[i]))) + shift[i]).ravel()
        y_series.append(y)
    y_series = np.array(y_series)
    #--------------------------------
    if(return_interp):
        import scipy.interpolate
        return scipy.interpolate.interp1d(t_series, y_series)
    else:
        return y_series, t_series


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def brownian_series(T, dt=1, num_series=1, lamda=1, eta_mean=0, eta_std=1, k=0, y0=0, v0=0, return_interp=True, seed=None):
    _rng = np.random.default_rng(seed)
    #--------------------------------
    lamda    = reshape(lamda, shape=(1, num_series)).ravel()
    eta_mean = reshape(eta_mean, shape=(1, num_series)).ravel()
    eta_std  = reshape(eta_std, shape=(1, num_series)).ravel()
    k        = reshape(k, shape=(1, num_series)).ravel()
    y0       = reshape(y0, shape=(1, num_series)).ravel()
    v0       = reshape(v0, shape=(1, num_series)).ravel()
    #--------------------------------
    t_series  = np.arange(0, T+dt, step=dt)
    #--------------------------------
    y_series = []
    for i in range(num_series):
        y    = np.zeros_like(t_series)
        y[0] = y0[i]
        v      = np.zeros_like(t_series)
        v[0]   = v0[i]
        for t in range(len(t_series)-1):
            dv     = -lamda[i]*v[t] + np.random.normal(eta_mean[i], eta_std[i]) - k[i]*(y[t]-y[0])
            v[t+1] = v[t] + dv*dt
            dy     = v[t+1]
            y[t+1] = y[t] + dy*dt
        y_series.append(y)
    y_series = np.array(y_series)
    #--------------------------------
    if(return_interp):
        import scipy.interpolate
        return scipy.interpolate.interp1d(t_series, y_series)
    else:
        return y_series, t_series

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_boltzmann_temp_for_entropy(energy, target_entropy):
    def entropy_diff(beta, energy, target_entropy):
        boltzmann_dist = ( np.exp(-beta * energy.astype(np.float128)) / np.sum(np.exp(-beta * energy.astype(np.float128))) ).astype(np.float64)
        boltzmann_entropy = scipy.stats.entropy(boltzmann_dist)
        return np.abs(boltzmann_entropy - target_entropy)
    res = scipy.optimize.minimize(entropy_diff, x0=1, args=(energy, target_entropy), method='Nelder-Mead')
    beta_fit = res['x'][0]
    return beta_fit


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_stats(vals, prefix=None, include_last=False):
    _prefix = '' if prefix is None else prefix+'_'
    stats = {
                (_prefix+'mean'):     np.nanmean(vals),
                (_prefix+'median'):   np.nanmedian(vals),
                (_prefix+'min'):      np.min(vals),
                (_prefix+'max'):      np.max(vals),
                (_prefix+'std'):      np.nanstd(vals),
                # (_prefix+'var'):      np.nanvar(vals)
            }
    return stats

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def logistic_curve(x, m, k):
    return 1/(1 + np.exp(k*(x - m)))

def fit_logistic_curve(data_x, data_y, m_init=0, k_init=1, bounds=None, weights=None, interp_xmax=100):

    import sklearn.metrics
    
    def calc_logistic_rmse(logistic_params, data_x, data_y, weights=None):
        m, k        = logistic_params
        logistic_x  = np.arange(0, np.nanmax(data_x)*2, step=0.01, dtype=np.float128)
        logistic_y  = logistic_curve(logistic_x, m, k)
        logistic_fn = scipy.interpolate.interp1d(logistic_x, logistic_y)
        #----------
        rmse        = sklearn.metrics.mean_squared_error(y_true=logistic_fn(data_x), y_pred=data_y, sample_weight=weights)
        return rmse
    
    res = scipy.optimize.minimize(calc_logistic_rmse, x0=[m_init, k_init], args=(data_x, data_y, weights), method='Nelder-Mead', bounds=bounds)
    
    fit_logistic_params = res['x']
    fit_m               = fit_logistic_params[0]
    fit_k               = fit_logistic_params[1]
    
    fit_logistic_x  = np.arange(0, interp_xmax, step=0.01, dtype=np.float128)
    fit_logistic_y  = logistic_curve(fit_logistic_x, fit_m, fit_k)
    fit_logistic_fn = scipy.interpolate.interp1d(fit_logistic_x, fit_logistic_y)
    
    return (fit_logistic_fn, fit_m, fit_k)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def tanh_curve(x, m, k):
    return (1 + np.exp(-k*(x - m)))/(1 + np.exp(-k*(x - m)))

def fit_tanh_curve(data_x, data_y, m_init=0, k_init=1, bounds=None, weights=None, interp_xmax=100):

    import sklearn.metrics
    
    def calc_tanh_rmse(tanh_params, data_x, data_y, weights=None):
        m, k    = tanh_params
        tanh_x  = np.arange(0, np.nanmax(data_x)*2, step=0.01, dtype=np.float128)
        tanh_y  = tanh_curve(tanh_x, m, k)
        tanh_fn = scipy.interpolate.interp1d(tanh_x, tanh_y)
        #----------
        rmse    = sklearn.metrics.mean_squared_error(y_true=tanh_fn(data_x), y_pred=data_y, sample_weight=weights)
        return rmse
    
    res = scipy.optimize.minimize(calc_tanh_rmse, x0=[m_init, k_init], args=(data_x, data_y, weights), method='Nelder-Mead', bounds=bounds)
    
    fit_tanh_params = res['x']
    fit_m               = fit_tanh_params[0]
    fit_k               = fit_tanh_params[1]
    
    fit_tanh_x  = np.arange(0, interp_xmax, step=0.01, dtype=np.float128)
    fit_tanh_y  = tanh_curve(fit_tanh_x, fit_m, fit_k)
    fit_tanh_fn = scipy.interpolate.interp1d(fit_tanh_x, fit_tanh_y)
    
    return (fit_tanh_fn, fit_m, fit_k)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def unit_vector(vector):
    # Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def angle_between(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

















































































