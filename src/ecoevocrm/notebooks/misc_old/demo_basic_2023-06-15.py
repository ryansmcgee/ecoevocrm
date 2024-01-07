#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scipy


from ecoevocrm.consumer_resource_system import *
from ecoevocrm.consumer_resource_system import *
import ecoevocrm.utils as utils
import ecoevocrm.viz as viz
import ecoevocrm.coarse_graining as cg
import ecoevocrm.strain_pool


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# ----
# 
# ## Model parameters

# ### Traits

# Set the number of resources in the system:

num_resources = 10


# Define type(s) that will be present at the start of the simulation:
# 
# Here we define the initial population as consisting of only one type (which is the first type from the list of all possible types)

sigma_allpossible = utils.binary_combinations(num_resources, exclude_all_zeros=True)
sigma_allpossible = sigma_allpossible/sigma_allpossible.sum(axis=1, keepdims=1)


sigma = sigma_allpossible[0]


# The initial composition of the population, as encoded in the $\sigma$ matrix, looks like this:

viz.matrix_plot(sigma, linecolor='lightgray', linewidths=1)


# Set the initial abundance of each type:
#     
# (Here we set the initial abundance of each type to 1 individual)

N_init = np.ones(np.atleast_2d(sigma).shape[0])
N_init


# ### Costs

# ##### Define cost parameters for types:

# Set the baseline cost $\xi$:

xi = 0.1


# Set the cost per trait, $\chi_{i}$:

chi = 0.3


# Set the trait interaction costs by defining a matrix, $\\J_{ij}$, that encodes the cost for each pair of traits:

# J = None
J = utils.random_matrix((num_resources, num_resources), 'tikhonov_sigmoid', args={'n_star': 5, 'delta': 1}, triangular=True, diagonal=0,
                        seed=2)
viz.matrix_plot(J, vmin=-0.4, vmax=0.4)


# ### Environment

# Set the initial amount of each resource:

R_init = np.ones(num_resources)
viz.matrix_plot(R_init, vmin=0, vmax=1, cbar=False, linecolor='lightgray', linewidths=1)


# Set the influx of each resource:
# 
# (Here the last 5 resources have influx, while the first 5 resources have zero influx)

rho = np.ones(num_resources)
viz.matrix_plot(rho, vmin=0, vmax=1, cbar=False, linecolor='lightgray', linewidths=1)


# Set the time constant for resource dynamics (will be 1 in practically all cases)

tau = 1


# Set the resource dynamics mode used in the simulation. 
# 
# (In many basic scenarios we use 'fast resource equilibration' ('fasteq') dynamics)

resource_dynamics_mode='fasteq'


# ## Simulate model

system = Community(type_set=TypeSet(sigma=sigma, xi=xi, chi=chi, J=J, binarize_traits_J_cost_terms=True, mu=1e-10),
                   rho=rho, tau=tau,
                   N_init=N_init, R_init=R_init,
                   resource_dynamics_mode='explicit',
                   max_time_step=1e3)


T_total = 1e6


system.run(T=T_total)


fig, ax = plt.subplots(1, 1, figsize=(20, 8))
viz.stacked_abundance_plot(system, ax=ax, relative_abundance=True, apply_palette_depth=1, log_x_axis=True, color_seed=1)


fig, ax = plt.subplots(1, 1, figsize=(20, 8))
viz.resource_plot(system, ax=ax, stacked=False, relative=False, log_x_axis=True, log_y_axis=False)


viz.matrix_plot(system.extant_type_set.sigma)


# ---------
# ---------
