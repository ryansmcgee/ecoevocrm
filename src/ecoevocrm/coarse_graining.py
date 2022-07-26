import numpy as np
import scipy
import copy

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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_phylogenetic_group_abundances(system, phylogeny_depth, t=None, t_index=None, relative_abundance=False, mode='branchings'):
    t_idx = np.argmax(system.t_series >= t) if t is not None else t_index if t_index is not None else -1
    #----------------------------------
    lineageIDs             = np.array(system.type_set.lineage_ids)
    extant_type_indices    = system.get_extant_type_indices(t_index=t_idx)
    extant_type_lineageIDs = lineageIDs[extant_type_indices]
    abundances             = system.get_type_abundance(t_index=t_idx).ravel()
    total_abundance        = np.sum(abundances)
    # extant_type_abundances = system.get_type_abundance(type_index=extant_type_indices, t_index=t_idx).ravel()
    extant_type_abundances = abundances[extant_type_indices]
    #----------------------------------
    if(mode == 'branchings'):
        type_cladeIDs          = np.array(['.'.join(lid.split('.')[:phylogeny_depth]) for lid in extant_type_lineageIDs])# if lid.count('.') >= phylogeny_depth-1]
        unique_cladeIDs        = np.unique(type_cladeIDs)
        #------------------------------
        clade_abds_dict = {}
        for i, clade_id in enumerate(unique_cladeIDs):
            clade_abds_dict[clade_id] = np.sum( abundances[extant_type_indices[np.where(type_cladeIDs == clade_id)[0]]] )
            if(relative_abundance):
                clade_abds_dict[clade_id] /= total_abundance
    #----------------------------------
    if(mode == 'coalescings'):
        _tree = copy.deepcopy(system.type_set.phylogeny)
        #------------------------------
        def collapse_subtrees(tree_dict):
            for child_id, childs_subtree in tree_dict.items():
                if(len(childs_subtree) == 0):
                    continue
                else:
                    all_children_have_empty_subtrees = True
                    for grandchild_id, grandchilds_subtree in childs_subtree.items():
                        if(len(grandchilds_subtree) > 0):
                            all_children_have_empty_subtrees = False
                            break
                    if(all_children_have_empty_subtrees):
                        # collapse this child's subtree:
                        # print(f">> collapsing {child_id}")
                        tree_dict[child_id] = {}
                    else:
                        # traverse down into the child's subtree:
                        collapse_subtrees(childs_subtree)
        #..............................
        # Repeated rounds of childless subtree collapses   
        for d in range(1, phylogeny_depth+1):
            collapse_subtrees(_tree)
        #------------------------------    
        def collect_abundances(tree_dict, abds_dict):
            for child_id, childs_subtree in tree_dict.items():
                # print(child_id, "::", len(childs_subtree), "children,", ("extant" if child_id in extant_type_lineageIDs else "extinct"), ("interior node" if len(childs_subtree) > 0 else "leaf node"))
                if(len(childs_subtree) > 0):
                    # child_id is an interior node in the collapsed tree;
                    # abundance for this 'clade' is abundance of the type exactly matching this id
                    for i, extant_id in enumerate(extant_type_lineageIDs):
                        # check if extant_id mathces the child_id
                        if (child_id == extant_id):
                            if(extant_type_abundances[i] > 0):
                                abds_dict[child_id] = extant_type_abundances[i] / (np.sum(extant_type_abundances) if relative_abundance else 1)
                            # print ('\t', extant_id, "matches", child_id, "\tabd =", extant_type_abundances[i])
                            # break
                        # else:
                        #     print('\tskip', extant_id)
                    # continue traversing down the subtrees:
                    collect_abundances(childs_subtree, abds_dict)
                else:
                    # child_id is a leaf in the collapsed tree; 
                    # abundance for this clade is sum of all extant types that begin with this child_id
                    clade_abd = 0.0
                    for i, extant_id in enumerate(extant_type_lineageIDs):
                        # check if extant_id starts with the child_id
                        if (child_id == extant_id or extant_id[:len(child_id)+1] == child_id+'.'):
                        # if (child_id in extant_id and re.search("^" + child_id, extant_id)):
                            clade_abd += extant_type_abundances[i]
                            # print("extant_type_abundances", extant_type_abundances, extant_type_abundances.shape)
                            # print("extant_type_abundances[i]", extant_type_abundances[i])
                            # print("clade_abd", clade_abd)
                            # print ('\t', extant_id, "starts with", child_id, "\tabd =", extant_type_abundances[i])
                        # else:
                        #     print('\tskip', extant_id)
                    if(clade_abd > 0):
                        abds_dict[child_id] = clade_abd / (np.sum(extant_type_abundances) if relative_abundance else 1)
        #..............................
        clade_abds_dict = {}
        collect_abundances(_tree, clade_abds_dict)
    #----------------------------------
    return clade_abds_dict


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_functional_group_abundances(system, trait_subset, t=None, t_index=None, relative_abundance=False):
    t_idx = np.argmax(system.t_series >= t) if t is not None else t_index if t_index is not None else -1
    #----------------------------------
    extant_type_indices = system.get_extant_type_indices(t_index=t_idx)
    functypes           = system.type_set.sigma[extant_type_indices, :][:, trait_subset]
    unique_functypes    = np.unique(functypes, axis=0)
    abundances          = system.get_type_abundance(t_index=t_idx).ravel()
    total_abundance     = np.sum(abundances)
    #----------------------------------
    group_abds_dict = {}
    for j, group_trait_profile in enumerate(unique_functypes):
        group_id = np.array(['-' for i in range(system.type_set.num_traits)])
        # print("?  ", group_id)
        group_id[list(trait_subset)] = [str(int(i != 0)) for i in group_trait_profile]
        # print("?? ", group_id)
        group_id = ''.join(group_id.tolist())
        # print("???", group_id)
        group_abds_dict[group_id] = np.sum( abundances[extant_type_indices[np.where((functypes == group_trait_profile).all(axis=1))[0]]] )
        # print("@@@")
        if(relative_abundance):
            group_abds_dict[group_id] /= total_abundance
    #----------------------------------
    return group_abds_dict


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def turnover_metric(abundances_t0, abundances_tf, inverse=False):
    # Based on: https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2664.12959
    turnover = np.sum(np.power((abundances_t0 - abundances_tf), 2)) / ( np.sum(np.power(abundances_t0, 2)) + np.sum(np.power(abundances_tf, 2)) - np.sum(abundances_t0 * abundances_tf) )
    # if(turnover > 1 or turnover < 0):
        # print('abundances_t0\n', abundances_t0)
        # print('abundances_tf\n', abundances_tf)
        # print('(abundances_t0 - abundances_tf)\n', (abundances_t0 - abundances_tf))
        # print('np.power((abundances_t0 - abundances_tf), 2)\n', np.power((abundances_t0 - abundances_tf), 2))
        # print('np.sum(np.power((abundances_t0 - abundances_tf), 2))\n', np.sum(np.power((abundances_t0 - abundances_tf), 2)))
        # print('turnover', turnover)
        # exit()
    return turnover if not inverse else (1 - turnover)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def phylogenetic_group_turnover(system, phylogeny_depth, t0, tf, inverse=False, mode='branchings'):
    clade_abds_t0_dict = get_phylogenetic_group_abundances(system, phylogeny_depth, t=t0, relative_abundance=True, mode=mode)
    # print("clade_abds_t0_dict", clade_abds_t0_dict)
    clade_abds_tf_dict = get_phylogenetic_group_abundances(system, phylogeny_depth, t=tf, relative_abundance=True, mode=mode)
    for i, clade_id in enumerate(clade_abds_t0_dict.keys()):
        if(clade_id not in clade_abds_tf_dict):
            clade_abds_tf_dict[clade_id] = 0
    for i, clade_id in enumerate(clade_abds_tf_dict.keys()):
        if(clade_id not in clade_abds_t0_dict):
            clade_abds_t0_dict[clade_id] = 0
    # print('clade_abds_t0_dict', clade_abds_t0_dict)
    # print('clade_abds_tf_dict', clade_abds_tf_dict)
    #----------------------------------
    # print( sorted(clade_abds_t0_dict.keys()) )
    clade_abds_t0    = np.array([clade_abds_t0_dict[clade_id] for clade_id in sorted(clade_abds_t0_dict.keys())])
    # print(clade_abds_t0)
    # print('-.-')
    # exit()
    clade_abds_tf    = np.array([clade_abds_tf_dict[clade_id] for clade_id in sorted(clade_abds_tf_dict.keys())])
    clade_relabds_t0 = clade_abds_t0/np.sum(clade_abds_t0)
    clade_relabds_tf = clade_abds_tf/np.sum(clade_abds_tf)
    #----------------------------------
    tm = turnover_metric(clade_relabds_t0, clade_relabds_tf, inverse=inverse)
    # if(tm > 1 or tm < 0):
        # print(clade_abds_t0_dict)
        # print(clade_abds_t0)
        # print(clade_abds_tf_dict)
        # print(clade_abds_tf)
    return tm


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def functional_group_turnover(system, trait_subset, t0, tf, inverse=False):
    group_abds_t0_dict = get_functional_group_abundances(system, trait_subset, t=t0, relative_abundance=True)
    group_abds_tf_dict = get_functional_group_abundances(system, trait_subset, t=tf, relative_abundance=True)
    for i, group_id in enumerate(group_abds_t0_dict.keys()):
        if(group_id not in group_abds_tf_dict):
            group_abds_tf_dict[group_id] = 0
    for i, group_id in enumerate(group_abds_tf_dict.keys()):
        if(group_id not in group_abds_t0_dict):
            group_abds_t0_dict[group_id] = 0
    #----------------------------------
    group_abds_t0    = np.array([group_abds_t0_dict[group_id] for group_id in sorted(group_abds_t0_dict.keys())])
    group_abds_tf    = np.array([group_abds_tf_dict[group_id] for group_id in sorted(group_abds_tf_dict.keys())])
    group_relabds_t0 = group_abds_t0/np.sum(group_abds_t0)
    group_relabds_tf = group_abds_tf/np.sum(group_abds_tf)
    #----------------------------------
    tm = turnover_metric(group_relabds_t0, group_relabds_tf, inverse=inverse)
    # if(tm > 1 or tm < 0):
        # print(group_abds_t0_dict)
        # print(group_abds_t0)
        # print(group_abds_tf_dict)
        # print(group_abds_tf)
    return tm

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def functional_group_diversity(system, trait_subset, t=None, t_index=None, metric='shannon'):
#     t_idx = np.argmax(system.t_series >= t) if t is not None else t_index if t_index is not None else -1
#     #----------------------------------
#     if(metric == 'shannon'):
#         abundances = get_functional_group_abundances(system, trait_subset, t_index=t_idx, relative_abundance=True)
#         print("abundances", abundances)
#     #----------------------------------
#     else:
#         utils.error(f"Error in functional_group_diversity(): diversity metric '{metric}' is not recognized.")

def phylogenetic_group_diversity(system, phylogeny_depth, t=None, t_index=None, metric='shannon', mode='branchings'):
    t_idx = np.argmax(system.t_series >= t) if t is not None else t_index if t_index is not None else -1
    #----------------------------------
    if(metric == 'shannon'):
        abundances = list( get_phylogenetic_group_abundances(system, phylogeny_depth, t_index=t_idx, relative_abundance=True, mode=mode).values() )
        abundances = abundances/np.sum(abundances)
        entropy = scipy.stats.entropy(abundances)
        diversity = entropy
    #----------------------------------
    else:
        utils.error(f"Error in phylogenetic_group_diversity(): diversity metric '{metric}' is not recognized.")
    #----------------------------------
    return diversity

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def functional_group_diversity(system, trait_subset, t=None, t_index=None, metric='shannon'):
    t_idx = np.argmax(system.t_series >= t) if t is not None else t_index if t_index is not None else -1
    #----------------------------------
    if(metric == 'shannon'):
        abundances = list( get_functional_group_abundances(system, trait_subset, t_index=t_idx, relative_abundance=True).values() )
        abundances = abundances/np.sum(abundances)
        entropy = scipy.stats.entropy(abundances)
        diversity = entropy
    #----------------------------------
    else:
        utils.error(f"Error in functional_group_diversity(): diversity metric '{metric}' is not recognized.")
    #----------------------------------
    return diversity
    
            
    
    
    
    
    

            
            






