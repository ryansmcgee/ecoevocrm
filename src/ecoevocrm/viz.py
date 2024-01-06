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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, steps=256):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, steps)))
    return new_cmap

def concatenate_colormaps(cmap1, cmap2, steps=256):
    # sample colors from the light and dark colormaps
    colors1 = cmap1(np.linspace(0.0, 1.0, int(steps/2)))
    colors2 = cmap2(np.linspace(0.0, 1.0, int(steps/2)))
    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    cmap   = matplotlib.colors.LinearSegmentedColormap.from_list(f"{cmap1.name}-{cmap2.name}", colors)
    return cmap

def lightdark_cmap(color, cmin=0, cmax=1, steps=256, reverse=False):
    cmap_light = sns.light_palette(color, as_cmap=True)
    cmap_dark  = sns.dark_palette(color, as_cmap=True, reverse=True)
    cmap = concatenate_colormaps(cmap_light, cmap_dark, steps=steps)
    cmap = truncate_colormap(cmap, cmin, cmax, steps=steps)
    return cmap if not reverse else cmap.reversed()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def type_styles_by_phylogeny(type_set, base_color='#AAAAAA', clade_colors=None, color_tags=None, hatch_tags=None, vmin=0, vmax=1,
                              palette='hls', palette_depth=0, shuffle_palette=True, seed=None,
                              color_step_min=0.01, color_step_max=0.5, color_step_scale=1, color_step_dir='dark'):

    _rng = np.random.default_rng(seed)

    num_palette_types = 0
    lineageIDs = np.asarray(type_set.lineageIDs)
    for lineage_id in lineageIDs:
        if(lineage_id.count('.') == palette_depth):
            num_palette_types += 1

    palette = sns.color_palette(palette, num_palette_types)
    if(shuffle_palette):
        _rng.shuffle(palette)

    type_colors = [base_color for i in range(type_set.num_types)]

    #------------------------

    if(clade_colors is None):
        clade_colors = {}
        if(color_tags is not None):
            for lineage_id in type_set.lineageIDs:
                lineage_id_lastpart = lineage_id.split('.')[-1].split('[')[0]
                for tag in color_tags.keys():
                    if(tag in lineage_id_lastpart):
                        if(isinstance(color_tags[tag], dict)):
                            tag_count = lineage_id.count(tag)
                            clade_colors[lineage_id] = color_tags[tag][tag_count]
                        else:
                            clade_colors[lineage_id] = color_tags[tag]
                        break

    #------------------------

    def color_clade(d, parent_color, depth, next_palette_color_idx):
        if(not isinstance(d, dict) or not d):
            return

        if(isinstance(parent_color, str)):
            parent_color = matplotlib.colors.to_rgb(parent_color)

        for lineage_id, descendants in d.items():
            type_idx   = np.argmax(lineageIDs == lineage_id)
            parent_idx = type_set.get_progenitor_indices(type_idx)[0]

            type_cost   = type_set.energy_costs[type_idx]
            parent_cost = type_set.energy_costs[parent_idx] if parent_idx is not None else None
            cost_diff   = np.abs(parent_cost - type_cost) if parent_idx is not None else None

            type_cmap  = None
            type_color = None
            if(depth==0):
                type_color = parent_color
            if(depth == palette_depth):
                type_color = palette[next_palette_color_idx]
                next_palette_color_idx += 1
            if(lineage_id in clade_colors):
                clade_color = clade_colors[lineage_id]
                if(isinstance(clade_color, matplotlib.colors.LinearSegmentedColormap)):
                    type_cmap  = clade_color
                    type_color = type_cmap( (type_cost - vmin)/(vmax - vmin) )  # here clade_color is a cmap
                else:
                    type_color = clade_color
            if(type_color is None):  # type_color did not meet any of the criteria above:
                if(isinstance(parent_color, matplotlib.colors.LinearSegmentedColormap)):
                    type_cmap  = parent_color
                    type_color = type_cmap( (type_cost - vmin)/(vmax - vmin) )  # here parent_color is a cmap
                else:
                    color_step = min(max(cost_diff, color_step_min), color_step_max)
                    color_step_coeff = -color_step_scale if color_step_dir == 'dark' else color_step_scale if color_step_dir == 'light' else _rng.choice([-color_step_scale, color_step_scale])
                    if(depth < palette_depth):
                        type_color = tuple([np.clip((parent_color[0] + color_step_coeff*color_step), 0, 1)]*3)
                    else:
                        type_color = tuple([np.clip((v + color_step_coeff*color_step), 0, 1) for v in parent_color])

            #--------------------
            type_colors[type_idx] = type_color
            pass_color = type_color if type_cmap is None else type_cmap
            color_clade(descendants, pass_color, depth+1, next_palette_color_idx)

    color_clade(type_set.phylogeny, parent_color=base_color, depth=0, next_palette_color_idx=0)

    #------------------------

    type_hatches = ['' for u in range(type_set.num_types)]
    if(hatch_tags is not None):
        for u, lineage_id in enumerate(type_set.lineageIDs):
            lineage_id_lastpart = lineage_id.split('.')[-1].split('[')[0]
            for tag in hatch_tags.keys():
                if(tag in lineage_id_lastpart):
                    if(isinstance(hatch_tags[tag], dict)):
                        tag_count = lineage_id.count(tag)
                        type_hatches[u] = hatch_tags[tag][tag_count]
                    else:
                        type_hatches[u] = hatch_tags[tag]
                    break

    #------------------------
    return type_colors, type_hatches

#OLD:
# def assign_type_colors(type_set, base_color='#AAAAAA', clade_colors={}, palette='hls', palette_depth=0, shuffle_palette=True, seed=None,
#                        color_step_min=0.01, color_step_max=0.5, color_step_scale=1, color_step_dir='dark'):
#
#     _rng = np.random.default_rng(seed)
#
#     num_palette_types = 0
#     lineageIDs = np.asarray(type_set.lineageIDs)
#     for lineage_id in lineageIDs:
#         if(lineage_id.count('.') == palette_depth):
#             num_palette_types += 1
#
#     palette = sns.color_palette(palette, num_palette_types)
#     if(shuffle_palette):
#         _rng.shuffle(palette)
#
#     type_colors = [base_color for i in range(type_set.num_types)]
#
#     def color_clade(d, parent_color, depth, next_palette_color_idx):
#         if(not isinstance(d, dict) or not d):
#             return
#
#         # if('#' in parent_color and len(parent_color)==7):
#         #     parent_color = tuple(int(parent_color.strip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
#         if(isinstance(parent_color, str)):
#             parent_color = matplotlib.colors.to_rgb(parent_color)
#
#         for lineage_id, descendants in d.items():
#             type_idx   = np.argmax(lineageIDs == lineage_id)
#             parent_idx = type_set.get_progenitor_indices(type_idx)[0]
#
#             type_cost   = type_set.energy_costs[type_idx]
#             parent_cost = type_set.energy_costs[parent_idx] if parent_idx is not None else None
#             cost_diff   = np.abs(parent_cost - type_cost) if parent_idx is not None else None
#
#             type_cmap  = None
#             type_color = None
#             if(depth==0):
#                 type_color = parent_color
#             if(depth == palette_depth):
#                 type_color = palette[next_palette_color_idx]
#                 next_palette_color_idx += 1
#             if(lineage_id in clade_colors):
#                 clade_color = clade_colors[lineage_id]
#                 if(isinstance(clade_color, matplotlib.colors.LinearSegmentedColormap)):
#                     type_cmap  = clade_color
#                     type_color = type_cmap(type_cost)  # here clade_color is a cmap
#                 else:
#                     type_color = clade_color
#             if(type_color is None):  # type_color did not meet any of the criteria above:
#                 if(isinstance(parent_color, matplotlib.colors.LinearSegmentedColormap)):
#                     type_cmap  = parent_color
#                     type_color = type_cmap(type_cost)  # here parent_color is a cmap
#                 else:
#                     color_step = min(max(cost_diff, color_step_min), color_step_max)
#                     color_step_coeff = -color_step_scale if color_step_dir == 'dark' else color_step_scale if color_step_dir == 'light' else _rng.choice([-color_step_scale, color_step_scale])
#                     if(depth < palette_depth):
#                         type_color = tuple([np.clip((parent_color[0] + color_step_coeff*color_step), 0, 1)]*3)
#                     else:
#                         type_color = tuple([np.clip((v + color_step_coeff*color_step), 0, 1) for v in parent_color])
#
#
#             #--------------------
#             type_colors[type_idx] = type_color
#             pass_color = type_color if type_cmap is None else type_cmap
#             color_clade(descendants, pass_color, depth+1, next_palette_color_idx)
#
#     color_clade(type_set.phylogeny, parent_color=base_color, depth=0, next_palette_color_idx=0)
#
#     return type_colors
#
#
# # EVEN OLDER VERSION BELOW:
# def color_types_by_phylogeny(type_set, palette='hls', root_color='#AAAAAA', highlight_clades='all', apply_palette_depth=1, shuffle_palette=True,
#                              color_step_start=0.13, color_step_slope=0.01, color_step_min=0.01, color_seed=None):
#
#     color_seed = np.random.randint(low=0, high=1e9) if color_seed is None else color_seed
#     np.random.seed(color_seed)
#
#     num_palette_types = 0
#     lineageIDs = np.asarray(type_set.lineageIDs)
#     for lineage_id in lineageIDs:
#         if(lineage_id.count('.') == apply_palette_depth):
#             num_palette_types += 1
#
#     palette = sns.color_palette(palette, num_palette_types)
#     if(shuffle_palette):
#         np.random.shuffle(palette)
#
#     type_colors = [root_color for i in range(type_set.num_types)]
#
#     if(isinstance(highlight_clades, str) and highlight_clades == 'all'):
#         highlight_clades = list(type_set.phylogeny.keys())
#
#     def color_subtree(d, parent_color, depth, next_palette_color_idx):
#         if(not isinstance(d, dict) or not d):
#             return
#         parent_color_rgb   = tuple(int(parent_color.strip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) if ('#' in parent_color and len(parent_color)==7) else parent_color
#         for lineage_id, descendants in d.items():
#             type_idx       = np.argmax(lineageIDs == lineage_id)
#             if(depth == apply_palette_depth):
#                 type_color = palette[next_palette_color_idx]
#                 next_palette_color_idx += 1
#             elif(depth==0):
#                 type_color = parent_color_rgb
#             elif(depth < apply_palette_depth):
#                 color_step_scale = max(color_step_start - color_step_slope*(depth-1), color_step_min)
#                 type_color = tuple([np.clip((parent_color_rgb[0] + np.random.uniform(low=-1*color_step_scale, high=color_step_scale)), 0, 1)]*3)
#             else:
#                 color_step_scale = max(color_step_start - color_step_slope*(depth-1), color_step_min)
#                 type_color = tuple([np.clip((v + np.random.uniform(low=-1*color_step_scale, high=color_step_scale)), 0, 1) for v in parent_color_rgb])
#             type_colors[type_idx] = type_color
#             color_subtree(descendants, type_color, depth+1, next_palette_color_idx)
#
#     color_subtree(type_set.phylogeny, parent_color=root_color, depth=0, next_palette_color_idx=0)
#
#     if(not (isinstance(highlight_clades, str) and highlight_clades == 'all')):
#         lineageIDs = np.asarray([lid+'.' for lid in lineageIDs])
#         for i, color in enumerate(type_colors):
#             if(not any(lineageIDs[i].startswith(str(highlight_id).strip('.')+'.') for highlight_id in highlight_clades)):
#                 type_colors[i] = [type_colors[i][0]]*3
#
#     return type_colors


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def abundance_plot(community, ax=None, type_colors=None, type_hatches=None, relative_abundance=False, stacked=True, baseline='sym', t_max=None, t_downsample='default', log_x_axis=False,
                   base_color='#AAAAAA', clade_colors=None, color_tags=None, hatch_tags=None, vmin=0, vmax=1, palette='hls', palette_depth=0, shuffle_palette=True, color_step_min=0.01, color_step_max=0.5, color_step_scale=1, color_step_dir='dark', seed=None,
                   linewidth=None, edgecolor=None):

    if(type_colors is None or type_hatches is None):
        _type_colors, _type_hatches = type_styles_by_phylogeny(community.type_set, base_color=base_color, clade_colors=clade_colors, color_tags=color_tags, hatch_tags=hatch_tags, vmin=vmin, vmax=vmax,
                                               palette=palette, palette_depth=palette_depth, shuffle_palette=shuffle_palette, seed=seed,
                                               color_step_min=color_step_min, color_step_max=color_step_max, color_step_scale=color_step_scale, color_step_dir=color_step_dir)
        type_colors  = _type_colors if type_colors is None else type_colors
        type_hatches = _type_hatches if type_hatches is None else type_hatches

    if(t_max is None):
        t_max = np.max(community.t_series)

    if(t_downsample == 'default'):
        t_downsample = max(int((len(community.t_series)//10000)+1), 1)
    elif(t_downsample is None):
        t_downsample = 1

    ax = plt.axes() if ax is None else ax

    if(stacked):
        if(relative_abundance):
            stacks = ax.stackplot(community.t_series[community.t_series < t_max][::t_downsample], np.flip((community.N_series/np.sum(community.N_series, axis=0))[:, community.t_series < t_max][:, ::t_downsample], axis=0), baseline='zero', colors=type_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)
        else:
            stacks = ax.stackplot(community.t_series[community.t_series < t_max][::t_downsample], np.flip(community.N_series[:, community.t_series < t_max][:, ::t_downsample], axis=0), baseline=baseline, colors=type_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)

        if(type_hatches is not None):
            for stack, hatch in zip(stacks, type_hatches[::-1]):
                stack.set_hatch(hatch)

    else:
        for u in range(community.num_types):
            ax.plot(community.t_series[community.t_series < t_max][::t_downsample], community.N_series[:, community.t_series < t_max][u, ::t_downsample], color=type_colors[u])

    if(log_x_axis):
        ax.set_xscale('log')

    ax.grid(False)

    return ax


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def attributes_plot(type_set, ax=None, plot_traits=True, plot_consumption_rate=True, plot_mutation_rate=True, plot_segregation_rate=True, plot_transfer_rate_donor=True, plot_transfer_rate_recip=True,
                    trait_colors=None):

    ax = plt.axes() if ax is None else ax

    ax.set_ylim(0, type_set.num_types)
    ax.set_xlim(0, type_set.num_traits)
    ax.set_aspect('equal')
    ax.set_ylim(ax.get_ylim()[::-1])

    if(plot_traits):
        traits = type_set.traits
        trait_colors = trait_colors if trait_colors is not None and len(trait_colors) == traits.shape[1] else ['k']*traits.shape[1]
        for u in range(traits.shape[0]):
            for i in range(traits.shape[1]):
                if(traits[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor='w', facecolor=trait_colors[i]))

    if(plot_consumption_rate):
        consumption_rate = type_set._params['consumption_rate'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(consumption_rate.shape[0]):
            for i in range(consumption_rate.shape[1]):
                if(consumption_rate[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor='peru', facecolor='none', hatch='...', alpha=0.5))

    if(plot_mutation_rate):
        mutation_rate = type_set._params['mutation_rate'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(mutation_rate.shape[0]):
            for i in range(mutation_rate.shape[1]):
                if(mutation_rate[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor='limegreen', facecolor='none', hatch='---', alpha=0.5))

    if(plot_segregation_rate):
        segregation_rate = type_set._params['segregation_rate'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(segregation_rate.shape[0]):
            for i in range(segregation_rate.shape[1]):
                if(segregation_rate[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor='gold', facecolor='none', hatch='|||', alpha=0.5))

    if(plot_transfer_rate_donor):
        transfer_rate_donor = type_set._params['transfer_rate_donor'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(transfer_rate_donor.shape[0]):
            for i in range(transfer_rate_donor.shape[1]):
                if(transfer_rate_donor[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor='mediumpurple', facecolor='none', hatch='///', alpha=0.5))

    if(plot_transfer_rate_recip):
        transfer_rate_recip = type_set._params['transfer_rate_recip'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(transfer_rate_recip.shape[0]):
            for i in range(transfer_rate_recip.shape[1]):
                if(transfer_rate_recip[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor='orchid', facecolor='none', hatch='\\\\\\', alpha=0.5))

    for u in range(traits.shape[0]):
        for i in range(traits.shape[1]):
            ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=2, edgecolor='w', facecolor='none'))

    return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def phylogeny_plot(system, ax=None, y_axis='index', log_x_axis=True, show_lineageIDs=True, show_phenotypes=True, annot_extinct=False,
                   type_colors=None, palette='hls', root_color='#AAAAAA', highlight_clades='all', apply_palette_depth=1, shuffle_palette=True, 
                   color_step_start=0.13, color_step_slope=0.01, color_step_min=0.01):
    
    if(type_colors is None):
        type_colors = color_types_by_phylogeny(system.type_set, palette=palette, root_color=root_color, highlight_clades=highlight_clades, apply_palette_depth=apply_palette_depth, shuffle_palette=shuffle_palette, color_step_start=color_step_start, color_step_slope=color_step_slope, color_step_min=color_step_min)
    
    ax = plt.axes() if ax is None else ax
    
    for i in range(system.num_types)[::-1]:
        
        try:
            
            abd_series = system.N_series[i, :]
        
            tidx_birth = (abd_series != 0).argmax(axis=0)
            t_birth    = system.t_series[tidx_birth]

            tidx_death = np.nonzero(abd_series)[0][-1]
            t_death    = system.t_series[tidx_death]

            parent_idx = system.type_set.parent_indices[i]

            N_total_end = np.sum(system.N_series[:, -1])
            N_i_end     = system.N_series[i, -1]
            
            ypos_i      = system.type_set.energy_costs[i] if y_axis == 'cost' else -i
            ypos_parent = system.type_set.energy_costs[parent_idx] if y_axis == 'cost' else -parent_idx
            
            ax.plot([t_birth, t_death], [ypos_i, ypos_i], color=type_colors[i], lw=0.5) 

            if(parent_idx is not None):
                ax.plot([t_birth, t_birth], [ypos_parent, ypos_i], color=type_colors[parent_idx], ls='--', lw=0.5, zorder=-99)

            if(N_i_end > 0 or annot_extinct):
                
                if(N_i_end > 0):
                    ax.plot([t_death, t_death+t_death*0.2], [ypos_i, ypos_i], color='#999999', ls=':', lw=0.5)
                    ax.scatter(t_death+t_death*0.2, ypos_i, color=type_colors[i], s=1000*(N_i_end/N_total_end), zorder=90)
                
                if(show_lineageIDs):

                    ax.annotate(system.type_set.lineageIDs[i] 
                                    + ('  ' + ''.join(['X' if system.type_set.sigma[i][j] > 0 else '-' for j in range(system.type_set.num_traits)]) if show_phenotypes else '')
                                    + ('  ' + "{0:.6f}".format(system.type_set.xi.ravel()[i] if isinstance(system.type_set.xi, np.ndarray) else system.type_set.xi ))
                                    , 
                                xy=(t_death+t_death*0.35, ypos_i), color=type_colors[i], fontsize=2, xycoords='data', annotation_clip=False)

        except:
            pass

    if(log_x_axis):
        ax.set_xscale('log')
        
    ax.set_xlabel("time")
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resource_plot(system, ax=None, t_max=None, t_downsample='default', log_x_axis=False, log_y_axis=False, stacked=False, relative=False, resource_colors=None, palette='terrain', linewidth=None, edgecolor=None, legend=True):

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def tree_plot(system, ax=None, t_max=None, log_x_axis=False, log_y_axis=False, onlyOriginMutations=False, translucentLaterMutations=True): #I PLAN TO PUT THIS HERE
    
    #Potential further changes:
    #-Can graph other quantities belonging to types besides cost on the y-axis (Alter the lines with "###" in them)
    #-Can change the layaring of plot elements
    
    if(t_max is None):
        t_max = np.max(system.t_series)
    
    ax = plt.axes() if ax is None else ax
    
    typeValueDict = {"0" : system.type_set.energy_costs[0]} ###can change this from energy_costs to change the y-axis

    typeIndex=0
    listOfPairedListsH = [] 
    for typeSeries in system.N_series:
        xpointsH = [] #horizontal
        ypointsH = [] #horizontal
        pairList = [xpointsH,ypointsH]
        listOfPairedListsH.append(pairList) #storage for graphing horizontal lines

        firstTimeIndex = 0
        if(typeSeries[firstTimeIndex] != 0): #horizontal lines for first time step 
            listOfPairedListsH[typeIndex][0].append(system.t_series[0])
            value = system.type_set.energy_costs[typeIndex] ###can change this from energy_costs to change the y-axis
            listOfPairedListsH[typeIndex][1].append(value)

        typeIndex = typeIndex + 1

        
    timeIndex = 0
    for time in system.t_series:
        typeIndex = 0
        for typeSeries in system.N_series:
            if(timeIndex > 1): 
                if(typeSeries[timeIndex] != 0 or typeSeries[timeIndex-1] != 0): #horizontal lines
                    listOfPairedListsH[typeIndex][0].append(time)
                    value = system.type_set.energy_costs[typeIndex] ###can change this from energy_costs to change the y-axis
                    listOfPairedListsH[typeIndex][1].append(value)
                if(typeSeries[timeIndex-1]==0 and typeSeries[timeIndex] != 0): #vertical lines and their markers
                    parentTypeIndex = system.type_set.parent_indices[typeIndex]
                    parentValue = system.type_set.energy_costs[parentTypeIndex] ###can change this from energy_costs to change the y-axis
                    xpointsV = [time, time] #vertical
                    ypointsV = [value, parentValue] #vertical
                    if((value in typeValueDict.values()) == False): #only clearly graphing vertical lines and their markers if the value isnt stored
                        plt.plot(xpointsV, ypointsV) #, color="blue"
                        plt.plot(time, value, marker="o", markersize=4, color="magenta", alpha=0.2)
                        plt.plot(time, parentValue, marker="o", markersize=4, color="black", alpha=0.2)
                    if((value in typeValueDict.values()) == True): #faintly graphing vertical lines and their markers if the value is stored
                        if(onlyOriginMutations == False and translucentLaterMutations == True): #find the best way to do this
                            plt.plot(xpointsV, ypointsV, color="blue", alpha=0.09)
                            plt.plot(time, value, marker="o", markersize=4, color="magenta", alpha=0.09)
                            plt.plot(time, parentValue, marker="o", markersize=4, color="black", alpha=0.09)
                        if(onlyOriginMutations == False and translucentLaterMutations == False):
                            plt.plot(xpointsV, ypointsV)
                            plt.plot(time, value, marker="o", markersize=4, color="magenta", alpha=0.09)
                            plt.plot(time, parentValue, marker="o", markersize=4, color="black", alpha=0.09)
                            
                    typeValueDict[typeIndex] = value #Storing type,value pair so can graph a sinlge vertical line to a horizontal line

                if(typeSeries[timeIndex-1] != 0 and typeSeries[timeIndex]==0): 
                    plt.plot(time, value, marker="x", markersize=6, color="red", alpha=0.2) #marking deaths #Transparancy allows for seeing if multiple deaths are occuring at the same time

                    if(typeIndex in typeValueDict and value in typeValueDict.values()): #Removing a type,value pair upon death 
                        del typeValueDict[typeIndex]

            typeIndex = typeIndex + 1
        timeIndex = timeIndex + 1

    typeIndex = 0
    for typeSeries in system.N_series: #plotting horizontal lines
        plt.plot(listOfPairedListsH[typeIndex][0], listOfPairedListsH[typeIndex][1])
        typeIndex = typeIndex + 1
        

    if(log_x_axis):
        ax.set_xscale('log')
    if(log_y_axis):
        ax.set_yscale('log')

    return ax




















