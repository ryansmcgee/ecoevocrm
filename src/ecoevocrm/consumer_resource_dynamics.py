import numpy as np
from numba.experimental import jitclass

spec = []

@jitclass(spec)
class ConsumerResourceDynamics(object):

    def __init__(self):
        pass

    @staticmethod
    def rhs(t, variables, 
                    sigma, b, k, eta, g, l, c, chi, mu, energy_costs,
                    sigma_mut, b_mut, k_mut, eta_mut, g_mut, l_mut, c_mut, chi_mut, mu_mut, energy_costs_mut,
                    J, D, rho, tau, omega, resource_consumption_mode, resource_inflow_mode):

        _num_types, _num_resources = sigma.shape

        N_t = variables[:_num_types]
        R_t = variables[-1-_num_resources:-1]

        if(resource_consumption_mode == 0): # self.CONSUMPTION_MODE_FASTEQ):
            resource_demand      = np.sum(np.multiply(np.expand_dims(N_t, axis=1), sigma), axis=0)
            resource_consumption = np.multiply(sigma, b/(1 + (resource_demand/k)))
        elif(resource_consumption_mode == 1): # self.CONSUMPTION_MODE_LINEAR):
            resource_consumption = np.multiply(sigma, np.multiply(b, R_t))
        elif(resource_consumption_mode == 2): # self.CONSUMPTION_MODE_MONOD):
            resource_consumption = np.multiply(sigma, np.multiply(b, R_t/(R_t + k)))
        else:
            resource_consumption = np.zeros((_num_types, _num_resources))

        energy_uptake  = np.sum( np.multiply(omega, np.multiply((1-l), resource_consumption)), axis=1)

        energy_surplus = energy_uptake - energy_costs

        growth_rate    = np.multiply(g, energy_surplus)

        dNdt = np.multiply(N_t, growth_rate)

        if(resource_inflow_mode == 0):
            resource_inflow = np.zeros((1,_num_resources))
        elif(resource_inflow_mode == 1):
            resource_inflow = rho

        if(resource_inflow_mode == 0): # self.INFLOW_MODE_NONE):
            dRdt = np.zeros((1, _num_resources)).ravel()
        else:
            if(D is None):
                dRdt = ( resource_inflow - np.multiply(1/tau, R_t) - np.sum(np.multiply(np.expand_dims(N_t, axis=1), resource_consumption), axis=0) ).ravel()
            else:
                resource_secretion = np.divide(np.dot(np.multiply(omega, np.sum(np.multiply(np.expand_dims(N_t, axis=1), np.multiply(l, resource_consumption)), axis=0)), D.T), omega, where=(omega>0), out=np.zeros_like(omega)).ravel()
                dRdt = ( resource_inflow - np.multiply(1/tau, R_t) - np.sum(np.multiply(np.expand_dims(N_t, axis=1), resource_consumption), axis=0) + resource_secretion ).ravel()

        cumulative_mutation_propensity = variables[-1]
        # dCumPropMut = np.array([np.sum(self.mutation_propensities(_N_t, _R_t))])
        dCumPropMut = np.array([0])
        
        #------------------------------

        return np.concatenate((dNdt, dRdt, dCumPropMut))