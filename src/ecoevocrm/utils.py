import sys
import numpy as np

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

    def expand_alloc(self, new_alloc):
        if(new_alloc[0] < self._alloc[0] or new_alloc[1] < self._alloc[1]):
            utils.error("Error in ExpandableArray.expand_alloc(): new_alloc shape must be at least as large as current alloc shape in each dimension.")
        self._alloc = new_alloc
        exp_arr = np.empty(shape=self._alloc, dtype=self.dtype)
        exp_arr[:self._shape[0], :self._shape[1]] = self.values
        self._arr = exp_arr
        return self

    def add(self, added_arr, axis=0):
        added_arr = np.atleast_2d(added_arr)
        # print("> add")
        # print("> self.shape", self.shape, "self.alloc", self.alloc)
        # print("> added_arr", added_arr, added_arr.shape)
        if(axis == 0):
            # print("> axis 0")
            while(self._shape[0] + added_arr.shape[0] > self._alloc[0]):
                # print("> expand")
                self.expand_alloc(new_alloc = (int(self._alloc[0]*self.default_expand_factor), self._alloc[1]))
            self._arr[self._shape[0]:self._shape[0]+added_arr.shape[0], :added_arr.shape[1]] = added_arr
            self._shape = (self._shape[0] + added_arr.shape[0], self._shape[1])
        elif(axis == 1):
            while(self._shape[1] + added_arr.shape[1] > self._alloc[1]):
                self.expand_alloc(new_alloc = (self._alloc[0], int(self._alloc[1]*self.default_expand_factor)))
            self._arr[:added_arr.shape[0], self._shape[1]:self._shape[1]+added_arr.shape[1]] = added_arr
            self._shape = (self._shape[0], self._shape[1] + added_arr.shape[1])
        # print("> self.shape", self.shape, "self.alloc", self.alloc)
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

def random_matrix(shape, mode, args={}, sparsity=0.0, symmetric=False, triangular=False, diagonal=None):
    #--------------------------------
    # Generate random values according to one of the following random models:
    #--------------------------------
    if(mode == 'bernoulli'):
        M = np.random.binomial(n=1, p=(args['p'] if 'p' in args else 0.5), size=shape )
    elif(mode == 'binomial'):
        M = np.random.binomial(n=(args['n'] if 'n' in args else 1), p=(args['p'] if 'p' in args else 0.5), size=shape )
    elif(mode == 'uniform'):
        M = np.random.uniform(low=(args['min'] if 'min' in args else 0), high=(args['max'] if 'max' in args else 1), size=shape )
    elif(mode == 'normal'):
        M = np.random.normal(loc=(args['mean'] if 'mean' in args else 0), scale=(args['std'] if 'std' in args else 1), size=shape )
    elif(mode == 'tikhonov_sigmoid'):
        J_0    = args['J_0'] if 'J_0' in args else 0.2
        n_star = args['n_star'] if 'n_star' in args else 10
        delta  = args['delta'] if 'delta' in args else 3
        M = np.zeros(shape=shape)
        for i, j in np.ndindex(M.shape):
            if(i >= j):
                continue
            M[i,j] = np.random.normal( loc=0, scale=J_0*(1/(1 + np.exp((max(i,j) - n_star)/delta))) )
    else:
        error(f"Error in random_matrix(): generator mode '{mode}' is not recognized.")
    #--------------------------------
    # Apply specified sparsity:
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
                error(f"Error in reshape(): input with shape {orig_shape} could not be reshaped to the system dimensionality ({self.num_types} types, {self.num_resources} resources).")
           else:
                error(f"Error in reshape(): input with shape {orig_shape} could not be reshaped to the target dimensionality {target_shape}.")
        return  arr


def treat_as_list(val):
    if(not isinstance(val, (list, np.ndarray)) and val is not None):
        return [val]
    elif(isinstance(val, (np.ndarray))):
        return val.ravel()
    else:
        return val


def error(message, trigger_exit=True):
    print("\n"+message+"\n")
    if(trigger_exit):
        sys.exit()



















































































