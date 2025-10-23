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
        if(np.all(np.isin(mat, [0, 1])) or np.all(np.isin(mat, [-1, 1]))):
            cmap = 'Greys'
        elif(np.any(mat < 0)):
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
                              color_step_mode=None, color_step_dir='random', color_step_start=0.13, color_step_slope=0.01,
                              color_step_min=0.01, color_step_max=0.25, color_step_scale=1):

    _rng = np.random.default_rng(seed)

    num_palette_types = 0
    lineageIDs = np.asarray(type_set.lineageIDs)
    for lineage_id in lineageIDs:
        if(lineage_id.count('.') <= palette_depth):
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
                lineage_id_lastpart = lineage_id.split('.')[-1].split('[')[0].split('(')[0]
                for tag in color_tags.keys():
                    if(isinstance(color_tags[tag], dict)):
                        if(tag in lineage_id):
                            tag_count = lineage_id.count(tag)
                            clade_colors[lineage_id] = color_tags[tag][tag_count]
                            break
                    else:
                        if(tag in lineage_id_lastpart):
                            clade_colors[lineage_id] = color_tags[tag]
                            break




    #------------------------

    color_step_mode = color_step_mode if color_step_mode is not None else 'depth' if not clade_colors else 'cost'

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
                    if(color_step_mode == 'depth'):
                        # color_step_scale = max(color_step_start - color_step_slope*(depth-1), color_step_min)
                        # type_color = tuple([np.clip((parent_color_rgb[0] + np.random.uniform(low=-1*color_step_scale, high=color_step_scale)), 0, 1)]*3)
                        color_step = min(max(_rng.uniform(low=0, high=color_step_start-color_step_slope*(depth-1)), color_step_min), color_step_max)
                    elif(color_step_mode == 'cost'):
                        color_step = min(max(cost_diff, color_step_min), color_step_max)
                    else:
                        color_step       = 0
                    # - - - -
                    color_step_coeff = -color_step_scale if color_step_dir == 'dark' else color_step_scale if color_step_dir == 'light' else _rng.choice([-color_step_scale, color_step_scale])
                    # - - - -
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
            lineage_id_lastpart = lineage_id.split('.')[-1].split('[')[0].split('(')[0]
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def abundance_plot(community, ax=None, type_colors=None, type_hatches=None, relative_abundance=False, stacked=True, baseline='sym', t_min=0, t_max=None, t_downsample='default', log_x_axis=False, log_y_axis=False,
                   base_color='#AAAAAA', clade_colors=None, color_tags=None, hatch_tags=None, vmin=0, vmax=1, palette='hls', palette_depth=0, shuffle_palette=True,
                   color_step_min=0.01, color_step_max=0.5, color_step_scale=1, color_step_dir='random', seed=None,
                   linewidth=None, edgecolor=None):

    if(type_colors is None):# or type_hatches is None):
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

    t_mask = (t_min <= community.t_series) & (community.t_series <= t_max)
    t_series = community.t_series[t_mask][::t_downsample]
    if(relative_abundance):
        N_series = (community.N_series/np.sum(community.N_series, axis=0))[:, t_mask][:, ::t_downsample]
    else:
        N_series = community.N_series[:, t_mask][:, ::t_downsample]

    if(stacked):
        if(relative_abundance):
            stacks = ax.stackplot(t_series, np.flip(N_series, axis=0), baseline='zero', colors=type_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)
        else:
            stacks = ax.stackplot(t_series, np.flip(N_series, axis=0), baseline=baseline, colors=type_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)

        if(type_hatches is not None):
            for stack, hatch in zip(stacks, type_hatches[::-1]):
                stack.set_hatch(hatch)

    else:
        for u in range(community.num_types):
            ax.plot(t_series, N_series[u], color=type_colors[u])

    if(log_x_axis):
        ax.set_xscale('log')
    if(log_y_axis):
        ax.set_yscale('log')

    ax.grid(False)

    plt.tight_layout()

    return ax


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def attributes_plot(type_set, ax=None, show_traits=True, type_colors=None, trait_colors=None,
                    hatch_consumption_rate='...', hatch_mutation_rate='|||', hatch_segregation_rate='---', hatch_transfer_rate_donor='///', hatch_transfer_rate_recip='\\\\\\',
                    color_consumption_rate='peru', color_mutation_rate='limegreen', color_segregation_rate='gold', color_transfer_rate_donor='mediumpurple', color_transfer_rate_recip='orchid', annot_alpha=1):

    ax = plt.axes() if ax is None else ax

    ax.set_ylim(0, type_set.num_types)
    ax.set_xlim(0, type_set.num_traits)
    ax.set_aspect('equal')
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xticks([i+0.5 for i in range(type_set.num_traits)])
    ax.set_yticks([i+0.5 for i in range(type_set.num_types)])
    ax.set_xticklabels([i for i in range(type_set.num_traits)])
    ax.set_yticklabels([i for i in range(type_set.num_types)])

    _type_colors  = None
    _trait_colors = None
    if(show_traits):
        traits = type_set.traits
        if(type_colors is not None and len(type_colors) == traits.shape[0]):
            _type_colors = type_colors
        elif(trait_colors is not None and len(trait_colors) == traits.shape[1]):
            _trait_colors = trait_colors
        for u in range(traits.shape[0]):
            for i in range(traits.shape[1]):
                if(traits[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor='w', facecolor=(_type_colors[u] if _type_colors is not None else _trait_colors[i] if _trait_colors is not None else 'k')))

    if(hatch_consumption_rate):
        consumption_rate = type_set._params['consumption_rate'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(consumption_rate.shape[0]):
            for i in range(consumption_rate.shape[1]):
                if(consumption_rate[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor=color_consumption_rate, facecolor='none', hatch='...', alpha=annot_alpha))

    if(hatch_mutation_rate):
        mutation_rate = type_set._params['mutation_rate'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(mutation_rate.shape[0]):
            for i in range(mutation_rate.shape[1]):
                if(mutation_rate[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor=color_mutation_rate, facecolor='none', hatch='|||', alpha=annot_alpha))

    if(hatch_segregation_rate):
        segregation_rate = type_set._params['segregation_rate'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(segregation_rate.shape[0]):
            for i in range(segregation_rate.shape[1]):
                if(segregation_rate[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor=color_segregation_rate, facecolor='none', hatch='---', alpha=annot_alpha))

    if(hatch_transfer_rate_donor):
        transfer_rate_donor = type_set._params['transfer_rate_donor'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(transfer_rate_donor.shape[0]):
            for i in range(transfer_rate_donor.shape[1]):
                if(transfer_rate_donor[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor=color_transfer_rate_donor, facecolor='none', hatch='///', alpha=annot_alpha))

    if(hatch_transfer_rate_recip):
        transfer_rate_recip = type_set._params['transfer_rate_recip'].values(force_type_dim=True, force_trait_dim=True)
        for u in range(transfer_rate_recip.shape[0]):
            for i in range(transfer_rate_recip.shape[1]):
                if(transfer_rate_recip[u, i] != 0):
                    ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=1, edgecolor=color_transfer_rate_recip, facecolor='none', hatch='\\\\\\', alpha=annot_alpha))

    for u in range(traits.shape[0]):
        for i in range(traits.shape[1]):
            ax.add_patch(matplotlib.patches.Rectangle((i, u), 1, 1, linewidth=2, edgecolor='w', facecolor='none'))

    return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def phylogeny_plot(community, ax=None, y_axis='index', log_x_axis=True, annot_lineageIDs=True, annot_traits=False, annot_extinct=True,
                   type_colors=None, base_color='#AAAAAA', clade_colors=None, color_tags=None, hatch_tags=None, vmin=0, vmax=1,
                   palette='hls', palette_depth=0, shuffle_palette=True, color_step_min=0.01, color_step_max=0.5, color_step_scale=1, color_step_dir='random', seed=None,
                   linewidth=1, annot_fontsize=8):

    if(type_colors is None):
        type_colors = type_styles_by_phylogeny(community.type_set, base_color=base_color, clade_colors=clade_colors, color_tags=color_tags, hatch_tags=hatch_tags, vmin=vmin, vmax=vmax,
                                               palette=palette, palette_depth=palette_depth, shuffle_palette=shuffle_palette, seed=seed,
                                               color_step_min=color_step_min, color_step_max=color_step_max, color_step_scale=color_step_scale, color_step_dir=color_step_dir)
    
    ax = plt.axes() if ax is None else ax
    
    for i in range(community.num_types)[::-1]:
        
        try:
            
            abd_series = community.N_series[i, :]
        
            tidx_birth = (abd_series != 0).argmax(axis=0)
            t_birth    = community.t_series[tidx_birth]

            tidx_death = np.nonzero(abd_series)[0][-1]
            t_death    = community.t_series[tidx_death]

            progenitor_index, progenitor_class = community.type_set.get_progenitor_indices(i, return_progenitor_class=True)
            progenitor_index = progenitor_index[0]
            progenitor_class = progenitor_class[0]
            donor_index = community.type_set.transfer_donor_indices[i]

            N_total_end = np.sum(community.N_series[:, -1])
            N_i_end     = community.N_series[i, -1]
            
            ypos_i      = community.type_set.energy_costs[i] if y_axis == 'cost' else -i

            ax.plot([t_birth, t_death], [ypos_i, ypos_i], color=type_colors[i], lw=linewidth)

            if(progenitor_index is not None):
                ypos_parent = community.type_set.energy_costs[progenitor_index] if y_axis == 'cost' else -progenitor_index

                ax.plot([t_birth, t_birth], [ypos_parent, ypos_i], color=type_colors[donor_index if donor_index is not None else progenitor_index],
                        ls=':' if progenitor_class == 'segregation' else (0, (5,1)) if progenitor_class == 'transfer' else '-', lw=linewidth, zorder=-99)

            if(N_i_end > 0 or annot_extinct):
                
                if(N_i_end > 0):
                    # ax.plot([t_death, t_death+t_death*0.2], [ypos_i, ypos_i], color='#999999', ls=':', lw=linewidth) # <-- this is the dashed line to annotations for surviving types
                    # ax.scatter(t_death+t_death*0.2, ypos_i, color=type_colors[i], s=1000*(N_i_end/N_total_end), zorder=90) # <-- this is the end dot scaled to abundance
                    ax.scatter(t_death, ypos_i, color=type_colors[i], s=10, zorder=90)
                
                if(annot_lineageIDs):
                    ax.annotate(community.type_set.lineageIDs[i]
                                    + (f"    [{''.join(['1' if community.type_set.traits[i][j] > 0 else '0' for j in range(community.type_set.num_traits)])}]" if annot_traits else ''),
                                xy=(t_death+t_death*0.1, ypos_i-0.0), va='center', color=type_colors[i], fontsize=annot_fontsize, xycoords='data', annotation_clip=False)

        except:
            pass

    if(log_x_axis):
        ax.set_xscale('log')
        
    ax.set_xlabel("time")
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.grid(False)

    plt.tight_layout()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resource_plot(community, ax=None, t_max=None, t_downsample='default', log_x_axis=False, log_y_axis=False, stacked=False, relative=False, resource_colors=None, palette='terrain', linewidth=None, edgecolor=None, legend=True):

    if(t_max is None):
        t_max = np.max(community.t_series)

    if(t_downsample == 'default'):
        t_downsample = max(int((len(community.t_series)//10000)+1), 1)
    elif(t_downsample is None):
        t_downsample = 1

    resource_colors = sns.color_palette(palette, community.num_resources) if resource_colors is None else resource_colors
    
    ax = plt.axes() if ax is None else ax

    if(stacked):
        if(relative):
            ax.stackplot(community.t_series[community.t_series < t_max][::t_downsample], np.flip((community.R_series/np.sum(community.R_series, axis=0))[:, community.t_series < t_max][:, ::t_downsample], axis=0), baseline='zero', colors=resource_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)
        else:
            ax.stackplot(community.t_series[community.t_series < t_max][::t_downsample], np.flip(community.R_series[:, community.t_series < t_max][:, ::t_downsample], axis=0), baseline='sym', colors=resource_colors[::-1], linewidth=linewidth, edgecolor=edgecolor)
    else:
        for i in range(community.num_resources):
            ax.plot(community.t_series[community.t_series < t_max][::t_downsample], community.R_series[:, community.t_series < t_max][i, ::t_downsample], color=resource_colors[i], label=f"resource {i+1}")
        if(legend):
            ax.legend()

    if(log_x_axis):
        ax.set_xscale('log')
    if(log_y_axis):
        ax.set_yscale('log')

    ax.grid(False)

    return ax


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Lstar_types_plot(community, ax=None, figsize=(7,5)):
    import ecoevocrm.coarse_graining as cg
    Lstar_types_data = cg.get_Lstar_types(community)

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

def interp_series_plot(interp, t_vals, ax=None, colors=None):
    L = interp(0).shape[0]
    ax = plt.axes() if ax is None else ax
    for i in range(L):
        ax.plot(t_vals, interp(t_vals)[i, :], c=colors[i] if colors is not None else None)
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




















