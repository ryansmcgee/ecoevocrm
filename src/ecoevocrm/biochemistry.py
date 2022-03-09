import numpy as np

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Biochemistry():

    def __init__(self, num_resources=None, J=None, D=None):

        # Determine the number of resources:
        if(isinstance(J, (np.ndarray)) and J.ndim == 2 and J.shape[0] == J.shape[1]):
            self.num_resources = J.shape[0]
        elif(isinstance(D, (np.ndarray)) and D.ndim == 2 and D.shape[0] == D.shape[1]):
            self.num_resources = D.shape[0]
        elif(num_resources is not None):
            self.num_resources  = num_resources
        else:
            utils.error("Error in Biochemistry __init__(): Number of resources must be specified by providing a) a square J matrix, b) a square D matrix, or c) a value for num_resources.")

        # Initialize biochemistry parameters:
        self.J = utils.reshape(J, shape=(self.num_resources, self.num_resources)) if J is not None else None
        self.D = utils.reshape(D, shape=(self.num_resources, self.num_resources)) if D is not None else None


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_params(self):
        return (self.J, self.D)

