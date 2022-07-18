import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def matrix_plot(mat, ax=None, cmap=None, vmin=None, vmax=None, center=None, cbar=None, cbar_kws=None, cbar_ax=None,
                square=True, robust=False, linewidths=0, linecolor='white', 
                xticklabels='auto', yticklabels='auto', mask=None, annot=None, fmt='.2g', annot_kws=None):
    
    mat = np.atleast_2d(mat)

    if(cmap is None):
        if(np.any(mat < 0)):
            cmap   = 'RdBu'
            center = 0 if center == None else center
            vmin   = -1*np.max(np.abs(mat)) if vmin == None else vmin
            vmax   = np.max(np.abs(mat)) if vmax == None else vmax
        else:
            cmap = 'Greys'
            
    cbar = True if cbar is None and np.any((mat != 1) & (mat != 0)) else cbar

    with sns.axes_style('white'):
        ax = sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=center, 
                            robust=robust, annot=annot, fmt=fmt, annot_kws=annot_kws, 
                            linewidths=linewidths, linecolor=linecolor, 
                            cbar=cbar, cbar_kws=cbar_kws, cbar_ax=cbar_ax, square=square, 
                            xticklabels=xticklabels, yticklabels=yticklabels, mask=mask)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def color_types_by_phylogeny(type_set, palette='hls', root_color='#AAAAAA', highlight_clades='all', apply_palette_depth=1, shuffle_palette=True, 
                             color_step_start=0.13, color_step_slope=0.01, color_step_min=0.01):

    # TODO: Make the range of random updates to child color based on phenotype or fitness difference between parent and child

    num_palette_types = 0
    lineage_ids = np.asarray(type_set.lineage_ids)
    for lineage_id in lineage_ids:
        if(lineage_id.count('.') == apply_palette_depth):
            num_palette_types += 1

    palette = sns.color_palette(palette, num_palette_types)
    if(shuffle_palette):
        np.random.shuffle(palette)
    
    type_colors = [root_color for i in range(type_set.num_types)]

    if(isinstance(highlight_clades, str) and highlight_clades == 'all'):
        highlight_clades = list(type_set.phylogeny.keys())
    
    def color_subtree(d, parent_color, depth, next_palette_color_idx):
        if(not isinstance(d, dict) or not d):
            return
        parent_color_rgb   = tuple(int(parent_color.strip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) if ('#' in parent_color and len(parent_color)==7) else parent_color
        for lineage_id, descendants in d.items():
            type_idx       = np.argmax(lineage_ids == lineage_id)
            if(depth == apply_palette_depth):
                type_color = palette[next_palette_color_idx]
                next_palette_color_idx += 1
            elif(depth==0):
                type_color = parent_color_rgb
            elif(depth < apply_palette_depth):
                color_step_scale = max(color_step_start - color_step_slope*(depth-1), color_step_min)
                type_color = tuple([np.clip((parent_color_rgb[0] + np.random.uniform(low=-1*color_step_scale, high=color_step_scale)), 0, 1)]*3)
            else:
                color_step_scale = max(color_step_start - color_step_slope*(depth-1), color_step_min)
                type_color = tuple([np.clip((v + np.random.uniform(low=-1*color_step_scale, high=color_step_scale)), 0, 1) for v in parent_color_rgb])
            type_colors[type_idx] = type_color
            color_subtree(descendants, type_color, depth+1, next_palette_color_idx)
        
    color_subtree(type_set.phylogeny, parent_color=root_color, depth=0, next_palette_color_idx=0)

    if(not (isinstance(highlight_clades, str) and highlight_clades == 'all')):
        lineage_ids = np.asarray([lid+'.' for lid in lineage_ids])
        for i, color in enumerate(type_colors):
            if(not any(lineage_ids[i].startswith(str(highlight_id).strip('.')+'.') for highlight_id in highlight_clades)):
                type_colors[i] = [type_colors[i][0]]*3

    return type_colors


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def stacked_abundance_plot(system, ax=None, relative_abundance=False, t_max=None, t_downsample='default', log_x_axis=False,
                            type_colors=None, palette='hls', root_color='#AAAAAA', highlight_clades='all', apply_palette_depth=1, shuffle_palette=True, 
                            color_step_start=0.13, color_step_slope=0.01, color_step_min=0.01,
                            linewidth=None, edgecolor=None):

    if(type_colors is None):
        type_colors = color_types_by_phylogeny(system.type_set, palette=palette, root_color=root_color, highlight_clades=highlight_clades, apply_palette_depth=apply_palette_depth, shuffle_palette=shuffle_palette, color_step_start=color_step_start, color_step_slope=color_step_slope, color_step_min=color_step_min)

    if(t_max is None):
        t_max = np.max(system.t_series)

    if(t_downsample == 'default'):
        t_downsample = max(int((len(system.t_series)//10000)+1), 1)
    elif(t_downsample is None):
        t_downsample = 1
    
    ax = plt.axes() if ax is None else ax

    if(relative_abundance):
        ax.stackplot(system.t_series[system.t_series < t_max][::t_downsample], np.flip((system.N_series/np.sum(system.N_series, axis=0))[:, system.t_series < t_max][:, ::t_downsample], axis=0), baseline='zero', colors=type_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)
    else:
        ax.stackplot(system.t_series[system.t_series < t_max][::t_downsample], np.flip(system.N_series[:, system.t_series < t_max][:, ::t_downsample], axis=0), baseline='sym', colors=type_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)

    if(log_x_axis):
        ax.set_xscale('log')

    return ax


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resource_plot(system, ax=None, t_max=None, t_downsample='default', log_x_axis=False, log_y_axis=False, stacked=False, relative=False, resource_colors=None, palette='hls', linewidth=None, edgecolor=None, legend=True):

    if(t_max is None):
        t_max = np.max(system.t_series)

    if(t_downsample == 'default'):
        t_downsample = max(int((len(system.t_series)//10000)+1), 1)
    elif(t_downsample is None):
        t_downsample = 1

    resource_colors = sns.color_palette(palette, system.num_resources) if resource_colors is None else resource_colors
    
    ax = plt.axes() if ax is None else ax

    if(stacked):
        if(relative):
            ax.stackplot(system.t_series[system.t_series < t_max][::t_downsample], np.flip((system.R_series/np.sum(system.R_series, axis=0))[:, system.t_series < t_max][:, ::t_downsample], axis=0), baseline='zero', colors=resource_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)
        else:
            ax.stackplot(system.t_series[system.t_series < t_max][::t_downsample], np.flip(system.R_series[:, system.t_series < t_max][:, ::t_downsample], axis=0), baseline='sym', colors=resource_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)
    else:
        for i in range(system.num_resources):
            ax.plot(system.t_series[system.t_series < t_max][::t_downsample], system.R_series[:, system.t_series < t_max][i, ::t_downsample], color=resource_colors[i], label=f"resource {i+1}")
        if(legend):
            ax.legend()

    if(log_x_axis):
        ax.set_xscale('log')
    if(log_y_axis):
        ax.set_yscale('log')

    return ax


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Lstar_types_plot(system, ax=None, figsize=(7,5)):
    import ecoevocrm.coarse_graining as cg
    Lstar_types_data = cg.get_Lstar_types(system)

    ax = plt.axes() if ax is None else ax

    with sns.axes_style('white'):
        ax.plot(Lstar_types_data[0], Lstar_types_data[1])
    
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_xlabel('L$^*$')
        ax.set_ylabel('number of unique types')

        sns.despine()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def strainpool_plot(strainpool_system, type_weights, rank_cutoff=None, weight_cutoff=None, figsize=(10, 10), type_colors=None):
    
    type_set = strainpool_system.type_set
    
    rank_cutoff   = min(len(type_weights) if rank_cutoff is None else rank_cutoff, len(type_weights))
    weight_cutoff = max(np.min(type_weights) if weight_cutoff is None else weight_cutoff, 0)
    
    active_type_indices = np.argsort(type_weights)[::-1]
    
    active_type_indices = [i for i in active_type_indices if type_weights[i] > weight_cutoff][:rank_cutoff]
    
    trait_weights = np.sum(type_weights[active_type_indices, np.newaxis] * type_set.sigma[active_type_indices], axis=0)
    trait_weights = trait_weights/np.sum(trait_weights)
    
    #----------
    
    left    = 0.1
    bottom  = 0.1
    pheno_height  = 0.9
    pheno_width   = pheno_height * (type_set.num_traits/len(active_type_indices))
    histx_height  = 0.1
    histy_width   = 0.1
    costs_width   = 0.1
    spacing = 0.02

    rect_pheno = [left, bottom, pheno_width, pheno_height]
    rect_histx = [left, bottom + pheno_height + spacing, pheno_width, histx_height]
    rect_histy = [left + pheno_width + spacing, bottom, histy_width, pheno_height]
    rect_costs = [left + pheno_width + spacing + histy_width + spacing, bottom, costs_width, pheno_height]    

    fig = plt.figure(figsize=figsize)

    ax_pheno = fig.add_axes(rect_pheno)
    ax_histx = fig.add_axes(rect_histx, sharex=ax_pheno)
    ax_histy = fig.add_axes(rect_histy, sharey=ax_pheno)
    ax_costs = fig.add_axes(rect_costs, sharey=ax_pheno)
    
    sns.heatmap(type_set.sigma[active_type_indices], ax=ax_pheno, cmap='Greys', cbar=False)
    ax_pheno.set_xlabel('traits')
    ax_pheno.set_ylabel('types in strain pool')
    ax_pheno.set_yticks(0.5+np.array(range(len(active_type_indices))))
    ax_pheno.set_yticklabels(active_type_indices, rotation=0)
    
    ax_histy.barh(range(len(type_weights[active_type_indices])), type_weights[active_type_indices], 1, align='edge', edgecolor='white', alpha=0.5, color=type_colors[active_type_indices])
    ax_histy.tick_params(axis='y', left=True, labelleft=False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.set_xlabel('type weight')
    
    ax_histx.bar(range(len(trait_weights)), trait_weights, 1, align='edge', edgecolor='white', alpha=0.5)
    ax_histx.tick_params(axis='x', bottom=True, labelbottom=False)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.set_ylabel('trait weight')
    
    # print(active_type_indices)
    # print(type_set.energy_costs)
    # print(type_set.xi_cost_terms)
    # print(type_set.chi_cost_terms)
    # print(type_set.J_cost_terms)
    ax_costs.scatter(y=0.5+np.array(range(len(type_weights[active_type_indices]))), x=type_set.energy_costs[active_type_indices], marker='D', zorder=0, color="None", edgecolor='tab:red')
    ax_costs.scatter(y=0.5+np.array(range(len(type_weights[active_type_indices]))), x=type_set.xi_cost_terms[active_type_indices] if isinstance(type_set.xi_cost_terms, (list, np.ndarray)) else np.full_like(type_set.energy_costs[active_type_indices], type_set.xi_cost_terms), marker='|', color='tab:brown', zorder=98)
    ax_costs.scatter(y=0.5+np.array(range(len(type_weights[active_type_indices]))), x=type_set.chi_cost_terms[active_type_indices] if isinstance(type_set.chi_cost_terms, (list, np.ndarray)) else np.full_like(type_set.energy_costs[active_type_indices], type_set.chi_cost_terms), marker='|', color='#333333')
    ax_costs.scatter(y=0.5+np.array(range(len(type_weights[active_type_indices]))), x=type_set.J_cost_terms[active_type_indices] if isinstance(type_set.J_cost_terms, (list, np.ndarray)) else np.full_like(type_set.energy_costs[active_type_indices], type_set.J_cost_terms), marker='|', color='tab:purple', zorder=99)
    # ax_costs.scatter(y=0.5+np.array(range(len(type_weights[active_type_indices]))), x=strainpool_system.get_fitness(t=0)[active_type_indices], marker='s', zorder=1, color='None', edgecolor='tab:green')
    ax_costs.set_xlim(xmin=0, xmax=max(np.max(type_set.energy_costs), np.max(strainpool_system.get_fitness(t=0))))
    ax_costs.tick_params(axis='y', left=True, labelleft=False)
    ax_costs.spines['top'].set_visible(False)
    ax_costs.spines['right'].set_visible(False)
    ax_costs.set_xlabel('costs/fitness')

    fig.tight_layout()
    
    return fig, [ax_pheno, ax_histx, ax_histy, ax_costs]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def interp_series_plot(interp, t_vals, ax=None):
    L = interp(0).shape[0]
    ax = plt.axes() if ax is None else ax
    for i in range(L):
        ax.plot(t_vals, interp(t_vals)[i, :]) 
    ax.set_ylim(ymin=min(0, np.min(interp(t_vals))))



















