import numpy as np
import scipy.interpolate

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ResourceSet():

    # Define Class constants:
    RESOURCE_INFLUX_CONSTANT          = 0
    RESOURCE_INFLUX_TEMPORAL          = 1

    def __init__(self, num_resources = None,
                       rho           = 0,
                       tau           = 1,
                       omega         = 1,
                       alpha         = 0,
                       theta         = 0,
                       phi           = 0,
                       D             = None ):

        # Determine the number of resources:
        if(isinstance(rho, (list, np.ndarray))):
            self.num_resources = len(rho)
        elif(isinstance(rho, scipy.interpolate.interpolate.interp1d)):
            self.num_resources = len(rho(0).ravel())
        elif(isinstance(omega, (list, np.ndarray))):
            self.num_resources = len(omega)
        elif(isinstance(tau, (list, np.ndarray))):
            self.num_resources = len(tau)
        elif(num_resources is not None):
            self.num_resources  = num_resources
        else:
            utils.error("Error in ResourceSet __init__(): Number of resources must be specified by providing a) a value for num_resources, or b) lists for rho/tau/omega.")

        # Initialize resource parameters:
        if(isinstance(rho, scipy.interpolate.interpolate.interp1d)):
            self._rho = rho
            self.resource_influx_mode = ResourceSet.RESOURCE_INFLUX_TEMPORAL
        else:
            self._rho = utils.reshape(rho, shape=(1, self.num_resources)).ravel()
            self.resource_influx_mode = ResourceSet.RESOURCE_INFLUX_CONSTANT
        self.tau   = utils.reshape(tau,   shape=(1, self.num_resources)).ravel()
        self.omega = utils.reshape(omega, shape=(1, self.num_resources)).ravel()
        self.alpha = utils.reshape(alpha, shape=(1, self.num_resources)).ravel()
        self.theta = utils.reshape(theta, shape=(1, self.num_resources)).ravel()
        self.phi   = utils.reshape(phi,   shape=(1, self.num_resources)).ravel()
        self.D     = utils.reshape(D,     shape=(self.num_resources, self.num_resources)) if D is not None else None


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def get_resource(self, index=None, resource_id=None):
    #     # TODO: make possible to get multiple resources by list of indices or ids
    #     if(index is None and resource_id is None):
    #         utils.error(f"Error in TypeSet get_resource(): A resource index or resource id must be given.")
    #     resource_idx = np.where(self.resource_ids==resource_id)[0] if resource_id is not None else index
    #     return TypeSet(sigma=self.sigma[resource_idx], b=self.b[resource_idx], k=self.k[resource_idx], 
    #                     eta=self.eta[resource_idx], l=self.l[resource_idx], g=self.g[resource_idx], 
    #                     c=self.c[resource_idx], chi=self.chi[resource_idx], mu=self.mu[resource_idx],
    #                     lineage_ids=np.array(self.lineage_ids[resource_idx]).flatten())


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def M(self):
        # M_ij = D_ij * w_j/w_i
        if(self.D is not None):
            W = np.tile(self.omega, (self.num_resources, 1))
            return self.D * W/W.T
        else:
            return None

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, vals):
        if(isinstance(vals, scipy.interpolate.interpolate.interp1d)):
            self._rho = vals
            self.resource_influx_mode = ResourceSet.RESOURCE_INFLUX_TEMPORAL
        else:
            self._rho = utils.reshape(vals, shape=(1, self.num_resources)).ravel()
            self.resource_influx_mode = ResourceSet.RESOURCE_INFLUX_CONSTANT


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_resource_id(self, index):
        return hash((self._rho[index], self.tau[index], self.omega[index], self.alpha[index], self.theta[index], self.phi[index]))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, index=None, resource_id=None):
        # TODO: make possible to get multiple resources by list of indices or ids
        resource_idx = np.where(self.resource_ids==resource_id)[0] if resource_id is not None else index
        if(resource_idx is None):
            return {'num_resources': self.num_resources,
                    'rho':           self._rho,
                    'tau':           self.tau,
                    'omega':         self.omega,
                    'alpha':         self.alpha,
                    'theta':         self.theta,
                    'phi':           self.phi,
                    'M':             self.M}
            # return (self.num_resources, self._rho, self.tau, self.omega, self.D)
        else:
            return {'num_resources': 1,
                    'rho':           self._rho[resource_idx],
                    'tau':           self.tau[resource_idx],
                    'omega':         self.omega[resource_idx],
                    'alpha':         self.alpha[resource_idx],
                    'theta':         self.theta[resource_idx],
                    'phi':           self.phi[resource_idx],
                    'M':             self.M[resource_idx,:]}
            # return (1, self._rho[resource_idx], self.tau[resource_idx], self.omega[resource_idx], self.D[resource_idx, :])
        


        
