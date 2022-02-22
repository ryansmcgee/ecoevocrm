import numpy as np

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_Lstar_types(system, Lstar='all', nonzero_abundance_only=True):
	# sigma = system.strain_pool.sigma if system is not None else strain_pool.sigma if strain_pool is not None else sigma
	# if(sigma is None):
	# 	utils.error(f"Error in coarse_graining get_Lstar_types(): A system, strain pool, or sigma matrix must be provided.")
	Lstar_vals = list(range(1, system.strain_pool.sigma.shape[1])) if Lstar == 'all' else utils.treat_as_list(Lstar)
	#------------------------------
	extant_type_indices = np.where(system.N_series[:,-1] > 0)[0] if nonzero_abundance_only else np.ones(system.strain_pool.num_types)
	# print(extant_type_indices)
	
	num_Lstar_types  = []
	Lstar_types_list = []
	for Lstar in Lstar_vals: 
		Lstar_types = np.unique(system.strain_pool.sigma[extant_type_indices, :Lstar], axis=0)
		num_Lstar_types.append(Lstar_types.shape[0])
		Lstar_types_list.append(Lstar_types)
		# print(Lstar)
		# print(Lstar_types)
		# print(Lstar_types.shape[0])
	return (Lstar_vals, num_Lstar_types, Lstar_types_list)



