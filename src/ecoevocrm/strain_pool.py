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
		print(f"Running dynamics for perturbation community {i+1}/{len(perturbed_systems)}")
		perturbed_system.run(T=run_T)
	#----------------------------------
	strain_pool = copy.deepcopy(perturbed_systems[0])
	for i in range(1, len(perturbed_systems)):
		strain_pool.combine(perturbed_systems[i])
	#----------------------------------
	return strain_pool





# def generate_strain_pool(orig_system, reps=50, run_T=1e7,
# 						 perturbation_args={'param': 'k', 
# 											'dist': 'normal', 
# 											'args': {'mean': 0, 'std': 0.1}, 
# 											'mode': 'multiplicative_proportional', 
# 											'element_wise': True} ):



# 	# print(orig_system.type_set.sigma.shape)
# 	# print(orig_system.mutant_set.sigma.shape)

# 	perturbed_systems = []

# 	for rep in range(reps):

# 		print("Strain pool rep", rep+1)

# 		perturbed_system = copy.deepcopy(orig_system).perturb(param=perturbation_args['param'], dist=perturbation_args['dist'], args=perturbation_args['args'], mode=perturbation_args['mode'], element_wise=perturbation_args['element_wise'])

# 		# print(perturbed_system.t_series)

# 		# print(perturbed_system.N)


# 		# print(perturbed_system.type_set.k)

# 		# print(perturbed_system.type_set.sigma.shape)
# 		# print(len(perturbed_system.extant_type_indices))
# 		# print(perturbed_system.mutant_set.sigma.shape)		

		

# 		perturbed_system.run(T=run_T)

# 		perturbed_systems.append(perturbed_system)

# 		# break

		

# 		# perturbed_systems.append(perturbed_system)

# 	# strain_pool = ConsumerResourceSystem.combine(perturbed_systems)

# 	return perturbed_systems

