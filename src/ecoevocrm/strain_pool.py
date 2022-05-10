import numpy as np
import copy

import ecoevocrm.utils as utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_perturbed_systems(orig_system, reps=50, perturbation_args=None):
	if perturbation_args is None:
		perturbation_args =  { 'param': 'k', 
								'dist': 'normal', 
								'args': {'mean': 0, 'std': 0.1}, 
								'mode': 'multiplicative_proportional', 
								'element_wise': True }
	#----------------------------------
	perturbed_systems = []
	for rep in range(reps):
		perturbed_system = copy.deepcopy(orig_system).perturb(param=perturbation_args['param'], dist=perturbation_args['dist'], args=perturbation_args['args'], mode=perturbation_args['mode'], element_wise=perturbation_args['element_wise'])
		perturbed_systems.append(perturbed_system)
	#----------------------------------
	return perturbed_systems


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def generate_strain_pool(orig_system, rep_communities=50, perturbation_args=None, run_T=1e7):
	if perturbation_args is None:
		perturbation_args =  { 'param': 'k', 
								'dist': 'normal', 
								'args': {'mean': 0, 'std': 0.1}, 
								'mode': 'multiplicative_proportional', 
								'element_wise': True }
	#----------------------------------
	perturbed_systems = get_perturbed_systems(orig_system=orig_system, reps=rep_communities, perturbation_args=perturbation_args)
	#----------------------------------
	for i, perturbed_system in enumerate(perturbed_systems):
		print(f"Running dynamics for perturbation community {i+1}/{len(perturbed_systems)}\r", end="")
		perturbed_system.run(T=run_T)
	#----------------------------------
	strain_pool = copy.deepcopy(orig_system)
	strain_pool.set_type_abundance(type_index=range(strain_pool.type_set.num_types), abundance=0.0)
	for i in range(len(perturbed_systems)):
		strain_pool.combine(perturbed_systems[i])
		# print(f"{i+1}: {strain_pool.extant_type_set.sigma.shape[0]} unique extant types in strain pool.")
	#----------------------------------
	return (strain_pool, perturbed_systems)


# def generate_strain_pool(orig_system, rep_communities=50, perturbation_args=None, run_T=1e7):
# 	if perturbation_args is None:
# 		perturbation_args =  { 'param': 'k', 
# 								'dist': 'normal', 
# 								'args': {'mean': 0, 'std': 0.1}, 
# 								'mode': 'multiplicative_proportional', 
# 								'element_wise': True }
# 	#----------------------------------
# 	perturbed_systems = get_perturbed_systems(orig_system=orig_system, reps=rep_communities, perturbation_args=perturbation_args)
# 	#----------------------------------
	# for i, perturbed_system in enumerate(perturbed_systems):
	# 	print(f"Running dynamics for perturbation community {i+1}/{len(perturbed_systems)}")
	# 	perturbed_system.run(T=run_T)
	# 	if(i == 0):
	# 		strain_pool = copy.deepcopy(perturbed_systems[0])
	# 	else:
	# 		strain_pool.combine(perturbed_systems[i])
	# 	print(f"{perturbed_systems[i].extant_type_set.sigma.shape[0]} types in perturbed_system[{i}]; {strain_pool.extant_type_set.sigma.shape[0]} unique types in strain pool.")
# 	#----------------------------------
# 	return (strain_pool, perturbed_systems)



