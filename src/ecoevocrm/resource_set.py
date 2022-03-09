import numpy as np

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ResourceSet():

    def __init__(self, num_resources=None,
                       rho   = 0,
                       tau   = 0,
                       omega = 1 ):

        # Determine the number of resources:
        if(isinstance(rho, (list, np.ndarray))):
            self.num_resources = len(rho)
        elif(isinstance(omega, (list, np.ndarray))):
            self.num_resources = len(omega)
        elif(isinstance(tau, (list, np.ndarray))):
            self.num_resources = len(tau)
        elif(num_resources is not None):
            self.num_resources  = num_resources
        else:
            utils.error("Error in ResourceSet __init__(): Number of resources must be specified by providing a) a value for num_resources, or b) lists for rho/tau/omega.")

        # Initialize resource parameters:
        self.rho   = utils.reshape(rho,   shape=(1, self.num_resources))
        self.tau   = utils.reshape(tau,   shape=(1, self.num_resources))
        self.omega = utils.reshape(omega, shape=(1, self.num_resources))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_resource(self, index=None, resource_id=None):
        # TODO: make possible to get multiple resources by list of indices or ids
        if(index is None and resource_id is None):
            utils.error(f"Error in TypeSet get_resource(): A resource index or resource id must be given.")
        resource_idx = np.where(self.resource_ids==resource_id)[0] if resource_id is not None else index
        return TypeSet(sigma=self.sigma[resource_idx], b=self.b[resource_idx], k=self.k[resource_idx], 
                        eta=self.eta[resource_idx], l=self.l[resource_idx], g=self.g[resource_idx], 
                        c=self.c[resource_idx], chi=self.chi[resource_idx], mu=self.mu[resource_idx],
                        lineage_ids=np.array(self.lineage_ids[resource_idx]).flatten())


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_resource_id(self, index):
        return hash((self.rho[index], self.tau[index], self.omega[index]))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_params(self, index=None, resource_id=None):
        # TODO: make possible to get multiple resources by list of indices or ids
        resource_idx = np.where(self.resource_ids==resource_id)[0] if resource_id is not None else index
        if(resource_idx is not None):
            return (self.rho[resource_idx], self.tau[resource_idx], self.omega[resource_idx])
        if(resource_idx is None):
            return (self.rho[:], self.tau[:], self.omega[:])


        
