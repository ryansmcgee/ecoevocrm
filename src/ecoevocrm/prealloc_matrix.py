import numpy as np
import ecoevocrm.utils as utils

class ExpandableArray():

    def __init__(self, arr, prealloc_shape=None, default_expand_factor=2):

        self._shape = arr.shape

        self._alloc = prealloc_shape if prealloc_shape is not None else arr.shape

        self._arr = np.zeros(shape=self._alloc)
        self._arr[:self._shape[0], :self._shape[1]] = arr

        self._default_expand_factor = default_expand_factor

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
        exp_arr = np.zeros(shape=self._alloc)
        exp_arr[:self._shape[0], :self._shape[1]] = self.values
        self._arr = exp_arr

    def add(self, added_arr, axis=0):
        added_arr = np.atleast_2d(added_arr)
        if(axis == 0):
            while(self._shape[0] + added_arr.shape[0] > self._alloc[0]):
                self.expand_alloc(new_alloc = (int(self._alloc[0]*self._default_expand_factor), self._alloc[1]))
            self._arr[self._shape[0]:self._shape[0]+added_arr.shape[0], :] = added_arr
            self._shape = (self._shape[0] + added_arr.shape[0], self._alloc[1])
        elif(axis == 1):
            while(self._shape[1] + added_arr.shape[1] > self._alloc[1]):
                self.expand_alloc(new_alloc = (self._alloc[0], int(self._alloc[1]*self._default_expand_factor)))
            self._arr[:, self._shape[1]:self._shape[1]+added_arr.shape[1]] = added_arr
            self._shape = (self._shape[0], self._shape[1] + added_arr.shape[1])
    
    def trim(self, alloc=None):
        self._alloc = self._shape if alloc is None else alloc
        self._arr   = self._arr[:self._alloc[0], :self._alloc[1]]


        



s = np.ones((1,4))
# print(s)
s = ExpandableArray(s)
print("shape:", s.shape, "alloc:", s.alloc)
print(s.values)

print("add_row")
s.add([2,2,2,2])
print("shape:", s.shape, "alloc:", s.alloc)
print(s.values)
s.add([3,3,3,3])
print("shape:", s.shape, "alloc:", s.alloc)
print(s.values)
s.add([4,4,4,4])
print("shape:", s.shape, "alloc:", s.alloc)
print(s.values)
s.add([5,5,5,5])
print("shape:", s.shape, "alloc:", s.alloc)
print(s.values)
s.add([6,6,6,6])
print("shape:", s.shape, "alloc:", s.alloc)
print(s.values)

s.add([[7,7,7,7],[8,8,8,8],[9,9,9,9]])
print("shape:", s.shape, "alloc:", s.alloc)
print(s.values)

print("trim")
s.trim()


print("shape:", s.shape, "alloc:", s.alloc)
print(s.values)