import numpy as np

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_Lstar_types(system, Lstar='all', nonzero_abundance_only=True):
	Lstar_vals = list(range(1, system.type_set.sigma.shape[1])) if Lstar == 'all' else utils.treat_as_list(Lstar)
	#------------------------------
	extant_type_indices = np.where(system.N_series[:,-1] > 0)[0] if nonzero_abundance_only else np.ones(system.type_set.num_types)
	num_Lstar_types  = []
	Lstar_types_list = []
	for Lstar in Lstar_vals: 
		Lstar_types = np.unique(system.type_set.sigma[extant_type_indices, :Lstar], axis=0)
		num_Lstar_types.append(Lstar_types.shape[0])
		Lstar_types_list.append(Lstar_types)
	#------------------------------
	return (Lstar_vals, num_Lstar_types, Lstar_types_list)



