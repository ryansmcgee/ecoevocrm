import numpy as np
import scipy.interpolate

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ResourceSet():

    # Define Class constants:
    RESOURCE_INFLUX_CONSTANT          = 0
    RESOURCE_INFLUX_TEMPORAL          = 1

    def __init__(self, num_resources    = None,
                       influx_rate      = 1,
                       decay_rate       = 1,
                       energy_content   = 1,
                       cross_production = None ):

        # Determine the number of resources:
        if(isinstance(influx_rate, (list, np.ndarray))):
            self.num_resources = len(influx_rate)
        elif(isinstance(influx_rate, scipy.interpolate.interpolate.interp1d)):
            self.num_resources = len(influx_rate(0).ravel())
        elif(isinstance(energy_content, (list, np.ndarray))):
            self.num_resources = len(energy_content)
        elif(isinstance(decay_rate, (list, np.ndarray))):
            self.num_resources = len(decay_rate)
        elif(num_resources is not None):
            self.num_resources  = num_resources
        else:
            utils.error("Error in ResourceSet __init__(): Number of resources must be specified by providing a) a value for num_resources, or b) lists for influx_rate/decay_rate/energy_content.")

        # Initialize resource parameters:
        if(isinstance(influx_rate, scipy.interpolate.interpolate.interp1d)):
            self._influx_rate = influx_rate
            self.resource_influx_mode = ResourceSet.RESOURCE_INFLUX_TEMPORAL
        else:
            self._influx_rate = utils.reshape(influx_rate, shape=(1, self.num_resources)).ravel()
            self.resource_influx_mode = ResourceSet.RESOURCE_INFLUX_CONSTANT
        self.decay_rate   = utils.reshape(decay_rate,   shape=(1, self.num_resources)).ravel()
        self.energy_content = utils.reshape(energy_content, shape=(1, self.num_resources)).ravel()
        self.cross_production     = utils.reshape(cross_production,     shape=(self.num_resources, self.num_resources)) if cross_production is not None else None


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
    def cross_production_energy(self):
        # M_ij = D_ij * energycontent_j/energycontent_i
        if(self.cross_production is not None):
            W = np.tile(self.energy_content, (self.num_resources, 1))
            return self.cross_production * W/W.T
        else:
            return None

    @property
    def influx_rate(self):
        return self._influx_rate

    @influx_rate.setter
    def influx_rate(self, vals):
        if(isinstance(vals, scipy.interpolate.interpolate.interp1d)):
            self._influx_rate = vals
            self.resource_influx_mode = ResourceSet.RESOURCE_INFLUX_TEMPORAL
        else:
            self._influx_rate = utils.reshape(vals, shape=(1, self.num_resources)).ravel()
            self.resource_influx_mode = ResourceSet.RESOURCE_INFLUX_CONSTANT


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_resource_id(self, index):
        return hash((self._influx_rate[index], self.decay_rate[index], self.energy_content[index]))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_dynamics_params(self, index=None, resource_id=None):
        # TODO: make possible to get multiple resources by list of indices or ids
        resource_idx = np.where(self.resource_ids==resource_id)[0] if resource_id is not None else index
        if(resource_idx is None):
            return {'num_resources': self.num_resources,
                    'influx_rate':           self._influx_rate,
                    'decay_rate':           self.decay_rate,
                    'energy_content':         self.energy_content,
                    'cross_production_energy':             self.cross_production_energy}
            # return (self.num_resources, self._influx_rate, self.decay_rate, self.energy_content, self.cross_production)
        else:
            return {'num_resources': 1,
                    'influx_rate':           self._influx_rate[resource_idx],
                    'decay_rate':           self.decay_rate[resource_idx],
                    'energy_content':         self.energy_content[resource_idx],
                    'cross_production_energy':             self.cross_production_energy[resource_idx, :]}
            # return (1, self._influx_rate[resource_idx], self.decay_rate[resource_idx], self.energy_content[resource_idx], self.cross_production[resource_idx, :])
        


        
