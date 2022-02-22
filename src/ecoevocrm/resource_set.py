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

        
