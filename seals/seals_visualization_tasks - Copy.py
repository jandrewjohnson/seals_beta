from matplotlib import colors as colors
from matplotlib import pyplot as plt
import numpy as np
import hazelbean as hb
import os
import seals_utils
import pandas as pd

from seals_visualization_functions import *

def visualization(p):
    # Just to create folder
    pass

def coarse_change_with_class_change_underneath(passed_p=None):
    if passed_p is None:
        global p 
    else:
        p = passed_p

    if p.run_this:


        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                if p.scenario_type !=  'baseline':
                    max_plotting_size = 200000
                    lulc_baseline_array = hb.as_array_resampled_to_size(p.base_year_lulc_path, max_plotting_size)

                    # By default, this will select 4 zones from different parts of the list to plot full change matrices. This is slow.
                    # You can override this to plot all here:
                    zones_to_plot = 'first' # one of first, all, or four

                    for year_c, year in enumerate(p.years):
                        target_allocation_zones_dir = os.path.join(p.allocations_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones')
                        seals_utils.load_blocks_list(p, target_allocation_zones_dir)

                        if year_c == 0:
                            previous_year = p.key_base_year
                        else:
                            previous_year = p.years[year_c - 1]
                        
                        if zones_to_plot == 'all':
                            target_zones = p.global_processing_blocks_list
                        elif zones_to_plot == 'four':
                            target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                        elif zones_to_plot == 'first':
                            target_zones = [p.global_processing_blocks_list[0]]
                        else:
                            raise ValueError('zones_to_plot must be one of first, all, or four')

                        # Make sure the target zones are in the right format
                        for c, row in enumerate(target_zones):
                                target_zones[c] = str(row[0] + '_' + row[1])

                        for target_zone in target_zones:            
                            ha_diff_from_previous_year_dir_to_plot = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year)) 
                            allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', target_zone, 'allocation') 
                            lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                            lulc_projected_array = None

                            if previous_year == p.key_base_year:
                                lulc_previous_year_path = os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_baseline_' + p.model_label + '_' + str(p.year) + '.tif')

                                lulc_previous_year_array = None     # For deffered loading                       
                            else:
                                previous_allocation_dir_to_plot = allocation_dir_to_plot.replace('\\','/').replace('/' + str(year) + '/', '/' + str(previous_year) + '/')
                                lulc_previous_year_path = os.path.join(previous_allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif')
                                lulc_previous_year_array = None
                            
                            for class_id, class_label in zip(p.lulc_correspondence_dict['dst_ids'], p.lulc_correspondence_dict['dst_labels']): 
                                
                                filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                                scaled_proportion_to_allocate_path = os.path.join(ha_diff_from_previous_year_dir_to_plot, filename)
                                output_path = os.path.join(p.cur_dir, str(year) + '_' + target_zone + '_' + class_label + '_projected_expansion_and_contraction.png')

                                if hb.path_exists(scaled_proportion_to_allocate_path) and not hb.path_exists(output_path):
                                    hb.log('Plotting ' + output_path)
                                    if lulc_projected_array is None:
                                        lulc_projected_array = hb.as_array_resampled_to_size(lulc_projected_path, max_plotting_size)


                                    if lulc_previous_year_array is None:
                                        lulc_previous_year_array = hb.as_array_resampled_to_size(lulc_previous_year_path, max_plotting_size)
                                    change_array = hb.as_array(scaled_proportion_to_allocate_path)
                                    
                                    show_class_expansions_vs_change_underneath(lulc_previous_year_array, lulc_projected_array, class_id, change_array, output_path,
                                                                    title='Class ' + class_label + ' projected expansion and contraction on coarse change')


def coarse_change_with_class_change(passed_p=None):
    if passed_p is None:
        global p 
    else:
        p = passed_p

    if p.run_this:


        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                if p.scenario_type !=  'baseline':
                    max_plotting_size = 200000
                    lulc_baseline_array = hb.as_array_resampled_to_size(p.base_year_lulc_path, max_plotting_size)

                    # By default, this will select 4 zones from different parts of the list to plot full change matrices. This is slow.
                    # You can override this to plot all here:
                    zones_to_plot = 'first' # one of first, all, or four

                    for year_c, year in enumerate(p.years):
                        target_allocation_zones_dir = os.path.join(p.allocations_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones')
                        seals_utils.load_blocks_list(p, target_allocation_zones_dir)

                        if year_c == 0:
                            previous_year = p.key_base_year
                        else:
                            previous_year = p.years[year_c - 1]
                        
                        if zones_to_plot == 'all':
                            target_zones = p.global_processing_blocks_list
                        elif zones_to_plot == 'four':
                            target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                        elif zones_to_plot == 'first':
                            target_zones = [p.global_processing_blocks_list[0]]
                        else:
                            raise ValueError('zones_to_plot must be one of first, all, or four')

                        # Make sure the target zones are in the right format
                        for c, row in enumerate(target_zones):
                                target_zones[c] = str(row[0] + '_' + row[1])

                        for target_zone in target_zones:            
                            ha_diff_from_previous_year_dir_to_plot = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year)) 
                            allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', target_zone, 'allocation') 
                            lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                            lulc_projected_array = None

                            if previous_year == p.key_base_year:
                                lulc_previous_year_path = os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(previous_year) + '.tif')

                                lulc_previous_year_array = None     # For deffered loading                       
                            else:
                                previous_allocation_dir_to_plot = allocation_dir_to_plot.replace('\\','/').replace('/' + str(year) + '/', '/' + str(previous_year) + '/')
                                lulc_previous_year_path = os.path.join(previous_allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif')
                                lulc_previous_year_array = None
                            
                            for class_id, class_label in zip(p.lulc_correspondence_dict['dst_ids'], p.lulc_correspondence_dict['dst_labels']): 
                                
                                filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                                scaled_proportion_to_allocate_path = os.path.join(allocation_dir_to_plot, filename)
                                output_path = os.path.join(p.cur_dir, str(year) + '_' + target_zone + '_' + class_label + '_projected_expansion_and_contraction.png')

                                if hb.path_exists(scaled_proportion_to_allocate_path) and not hb.path_exists(output_path):
                                    hb.log('Plotting ' + output_path)
                                    if lulc_projected_array is None:
                                        lulc_projected_array = hb.as_array_resampled_to_size(lulc_projected_path, max_plotting_size)


                                    if lulc_previous_year_array is None:
                                        lulc_previous_year_array = hb.as_array_resampled_to_size(lulc_previous_year_path, max_plotting_size)
                                    change_array = hb.as_array(scaled_proportion_to_allocate_path)
                                    
                                    show_class_expansions_vs_change(lulc_previous_year_array, lulc_projected_array, class_id, change_array, output_path,
                                                                    title='Class ' + class_label + ' projected expansion and contraction on coarse change')

def target_zones_matrices_pngs(passed_p=None):
    if passed_p is None:
        global p 
    else:
        p = passed_p

    # START HERE: Document how i separates the chnage matrices and change matrices pngs into content/visualization. Then
    # add a a simple LULC plot. This might involve pulling in geoecon code.

    if p.run_this:
        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                classes_that_might_change = p.changing_class_indices
                if p.scenario_type !=  'baseline':
                    for c, year in enumerate(p.years):
                        full_change_matrix_no_diagonal_path = os.path.join(p.full_change_matrices_dir, str(year), 'full_change_matrix_no_diagonal.tif')
                        full_change_matrix_no_diagonal_auto_png_path = os.path.join(p.cur_dir, str(year) + '_full_change_matrix_no_diagonal_auto.png')
                        if not hb.path_exists(full_change_matrix_no_diagonal_auto_png_path) or not hb.path_exists(full_change_matrix_no_diagonal_path):
                            n_classes = len(classes_that_might_change)

                            fig, ax = plt.subplots()
                            fig.set_size_inches(10, 8)
                            
                            # Get the CR_width_height of the zone(s) we want to plut


                            zones_to_plot = 'first' # one of first, all, or four
                            if zones_to_plot == 'all':
                                target_zones = p.global_processing_blocks_list
                                offsets = [p.coarse_blocks_list[0]]
                                offsets = [[int(i) for i in j] for j in target_zones]
                            
                            elif zones_to_plot == 'four':
                                target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                                offsets = [p.coarse_blocks_list[0]]
                                offsets = [[int(i) for i in j] for j in target_zones]
                            

                            
                            elif zones_to_plot == 'first':
                                target_zones = [p.global_processing_blocks_list[0]]
                                offsets = [p.coarse_blocks_list[0]]
                                offsets = [[int(i) for i in offsets[0]]]
                            else:
                                raise ValueError('zones_to_plot must be one of first, all, or four')

                            
                            # full_change_matrix_no_diagonal = hb.as_array(full_change_matrix_no_diagonal_path)
                            
                            for offset in offsets:
                                full_change_matrix_no_diagonal = hb.load_geotiff_chunk_by_cr_size(full_change_matrix_no_diagonal_path, offset)

                                for year in p.years:

                                    current_lulc_filename = 'change_matrix_' + str(year) + '.tif'
                                    current_change_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label)
                                    # title = 'LULC ' + p.exogenous_label + ' ' + p.climate_label + ' ' + p.model_label + ' ' + p.counterfactual_label + ' ' + str(year)
                                    # title = title.title()

                                    hb.create_directories(current_change_dir)

                                    full_change_matrix_no_diagonal_png_path = os.path.join(current_change_dir, current_lulc_filename)                            
                            
                            if np.sum(full_change_matrix_no_diagonal) > 0:
                                # Plot the heatmap
                                vmin = np.min(full_change_matrix_no_diagonal)
                                vmax = np.max(full_change_matrix_no_diagonal)
                                im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin + 1, vmax=vmax))

                                # Create colorbar
                                cbar = ax.figure.colorbar(im, ax=ax)
                                cbar.set_label('Number of cells changed from class ROW to class COL', size=10)
                                # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

                                # We want to show all ticks...
                                ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
                                ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0]))
                                # ... and label them with the respective list entries.

                                row_labels = []
                                col_labels = []

                                coarse_match_n_rows = hb.get_shape_from_dataset_path(p.aoi_ha_per_cell_coarse_path)[0]
                                coarse_match_n_cols = hb.get_shape_from_dataset_path(p.aoi_ha_per_cell_coarse_path)[1]
                                for i in range(n_classes * (coarse_match_n_rows)):
                                    class_id = i % n_classes
                                    row_labels.append(str(class_id))
                                    
                                for i in range(n_classes * (coarse_match_n_cols)):
                                    class_id = i % n_classes
                                    col_labels.append(str(class_id))                            

                                trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction

                                for i in range(coarse_match_n_rows):
                                    ann = ax.annotate('Zone ' + str(i + 1), xy=(-3.5, i / coarse_match_n_rows + .5 / coarse_match_n_rows), xycoords=trans)
                                    # ann = ax.annotate('Class ' + str(i + 1), xy=(-2.5, i / p.coarse_match.n_rows + .5 / p.coarse_match.n_rows), xycoords=trans)
                                    ann = ax.annotate('Zone ' + str(i + 1), xy=(i * (coarse_match_n_rows + 1) + .25 * coarse_match_n_rows, 1.05), xycoords=trans)  #
                                    # ann = ax.annotate('MgII', xy=(-2, 1 / (i * n_classes + n_classes / 2)), xycoords=trans)
                                    # plt.annotate('This is awesome!',
                                    #              xy=(-.1, i * n_classes + n_classes / 2),
                                    #              xycoords='data',
                                    #              textcoords='offset points',
                                    #              arrowprops=dict(arrowstyle="->"))

                                # ax.set_xticklabels(col_labels)
                                # ax.set_yticklabels(row_labels)

                                # Let the horizontal axes labeling appear on top.
                                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                                # Rotate the tick labels and set their alignment.
                                plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")

                                # im, cbar = heatmap(full_change_matrix_no_diagonal, row_labels, col_labels, ax=ax,
                                #                    cmap="YlGn", cbarlabel="harvest [t/year]")
                                # Turn spines off and create white grid.
                                for edge, spine in ax.spines.items():
                                    spine.set_visible(False)

                                ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1] + 1) - .5, minor=True)
                                ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0] + 1) - .5, minor=True)
                                ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
                                ax.tick_params(which="minor", bottom=False, left=False)



                                # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

                                major_gridline = False
                                for i in range(n_classes * coarse_match_n_rows + 1):
                                    try:
                                        if i % n_classes == 0:
                                            major_gridline = i
                                        else:
                                            major_gridline = False
                                    except:
                                        major_gridline = 0

                                    if major_gridline is not False:
                                        xloc = major_gridline - .5
                                        yloc = major_gridline - .5
                                        ax.axvline(x=xloc, color='grey')
                                        ax.axhline(y=yloc, color='grey')

                                # ax.axvline(x = 0 - .5, color='grey')
                                # ax.axvline(x = 4 - .5, color='grey')
                                # ax.axhline(y = 0 - .5, color='grey')
                                # ax.axhline(y = 4 - .5, color='grey')

                                # plt.title('Cross-zone change matrix')
                                # ax.cbar_label('Number of cells changed from class ROW to class COL')
                                plt.savefig(full_change_matrix_no_diagonal_png_path)

                                vmax = np.max(full_change_matrix_no_diagonal)
                                # full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.png')
                                # fig, ax = plt.subplots()
                                # im = ax.imshow(full_change_matrix_no_diagonal)
                                # ax.axvline(x=.5, color='red')
                                # ax.axhline(y=.5, color='yellow')
                                # plt.title('Draw a line on an image with matplotlib')

                                # plt.savefig(full_change_matrix_no_diagonal_png_path)

                                
                                from hazelbean.visualization import full_show_array
                                full_show_array(full_change_matrix_no_diagonal, output_path=full_change_matrix_no_diagonal_auto_png_path, cbar_label='Number of changes from class R to class C per tile', title='Change matrix mosaic',
                                                num_cbar_ticks=2, vmin=0, vmid=vmax / 10.0, vmax=vmax, color_scheme='ylgnbu')

def full_change_matrices_pngs(passed_p=None):
    if passed_p is None:
        global p 
    else:
        p = passed_p

    # START HERE: Document how i separates the chnage matrices and change matrices pngs into content/visualization. Then
    # add a a simple LULC plot. This might involve pulling in geoecon code.

    if p.run_this:
        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                classes_that_might_change = p.changing_class_indices
                if p.scenario_type !=  'baseline':
                    for c, year in enumerate(p.years):
                        full_change_matrix_no_diagonal_path = os.path.join(p.full_change_matrices_dir, str(year), 'full_change_matrix_no_diagonal.tif')
                        full_change_matrix_no_diagonal_auto_png_path = os.path.join(p.cur_dir, str(year) + '_full_change_matrix_no_diagonal_auto.png')
                        if not hb.path_exists(full_change_matrix_no_diagonal_auto_png_path) or not hb.path_exists(full_change_matrix_no_diagonal_path):
                            n_classes = len(classes_that_might_change)

                            fig, ax = plt.subplots()
                            fig.set_size_inches(10, 8)
                            
                            full_change_matrix_no_diagonal = hb.as_array(full_change_matrix_no_diagonal_path)
                            if np.sum(full_change_matrix_no_diagonal) > 0:
                                # Plot the heatmap
                                vmin = np.min(full_change_matrix_no_diagonal)
                                vmax = np.max(full_change_matrix_no_diagonal)
                                im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin + 1, vmax=vmax))

                                # Create colorbar
                                cbar = ax.figure.colorbar(im, ax=ax, shrink=.6)
                                cbar.set_label('Number of cells changed from class ROW to class COL', size=10)
                                # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

                                # We want to show all ticks...
                                # ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
                                # ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
                                ax.set_xticks([])
                                ax.set_yticks([])
                                # ... and label them with the respective list entries.

                                # row_labels = []
                                # col_labels = []

                                coarse_match_n_rows = hb.get_shape_from_dataset_path(p.aoi_ha_per_cell_coarse_path)[0]
                                coarse_match_n_cols = hb.get_shape_from_dataset_path(p.aoi_ha_per_cell_coarse_path)[1]
                                # for i in range(n_classes * (coarse_match_n_rows)):
                                #     class_id = i % n_classes
                                #     # row_labels.append(str(class_id))
                                    
                                # for i in range(n_classes * (coarse_match_n_cols)):
                                #     class_id = i % n_classes
                                #     # col_labels.append(str(class_id))                            

                                # trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction

                                # plot_zone_labels = False # Must manually build locations
                                # if plot_zone_labels:
                                #     for i in range(coarse_match_n_rows):
                                #         ann = ax.annotate('Zone ' + str(i + 1), xy=(-3.5, i / coarse_match_n_rows + .5 / coarse_match_n_rows), xycoords=trans)
                                #         # ann = ax.annotate('Class ' + str(i + 1), xy=(-2.5, i / p.coarse_match.n_rows + .5 / p.coarse_match.n_rows), xycoords=trans)
                                #     for i in range(coarse_match_n_cols):
                                #         ann = ax.annotate('Zone ' + str(i + 1), xy=(i * (coarse_match_n_cols + 1) + .25 * coarse_match_n_cols, 1.05), xycoords=trans)  #
                                #         # ann = ax.annotate('MgII', xy=(-2, 1 / (i * n_classes + n_classes / 2)), xycoords=trans)
                                #         # plt.annotate('This is awesome!',
                                #         #              xy=(-.1, i * n_classes + n_classes / 2),
                                #         #              xycoords='data',
                                #         #              textcoords='offset points',
                                #         #              arrowprops=dict(arrowstyle="->"))
                                # plot_ticks = 1
                                # if plot_ticks:
                                #     # ax.set_xticklabels(col_labels)
                                #     # ax.set_yticklabels(row_labels)

                                #     # Let the horizontal axes labeling appear on top.
                                #     ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)

                                #     # Rotate the tick labels and set their alignment.
                                #     # plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")

                                #     # ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1] + 1) - .5, minor=True)
                                #     # ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0] + 1) - .5, minor=True)
                                #     # ax.tick_params(which="minor", bottom=False, left=False)

                                # im, cbar = heatmap(full_change_matrix_no_diagonal, row_labels, col_labels, ax=ax,
                                #                    cmap="YlGn", cbarlabel="harvest [t/year]")
                                # Turn spines off and create white grid.
                                for edge, spine in ax.spines.items():
                                    spine.set_visible(False)

                                ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
                                

                                full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, str(year) + '_full_change_matrix_no_diagonal.png')
                                # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

                                major_gridline = False
                                for i in range(n_classes * coarse_match_n_rows + 1):
                                    try:
                                        if i % n_classes == 0:
                                            major_gridline = i
                                        else:
                                            major_gridline = False
                                    except:
                                        major_gridline = 0

                                    if major_gridline is not False:
                                        xloc = major_gridline - .5
                                        yloc = major_gridline - .5
                                        # ax.axvline(x=xloc, color='grey')
                                        ax.axhline(y=yloc, color='grey')

                                major_gridline = False
                                for i in range(n_classes * coarse_match_n_cols + 1):
                                    try:
                                        if i % n_classes == 0:
                                            major_gridline = i
                                        else:
                                            major_gridline = False
                                    except:
                                        major_gridline = 0

                                    if major_gridline is not False:
                                        xloc = major_gridline - .5
                                        yloc = major_gridline - .5
                                        ax.axvline(x=xloc, color='grey')
                                        # ax.axhline(y=yloc, color='grey')


                                # # plt.title('Cross-zone change matrix')
                                # ax.cbar_label('Number of cells changed from class ROW to class COL')
                                plt.savefig(full_change_matrix_no_diagonal_png_path)

                                vmax = np.max(full_change_matrix_no_diagonal)
                                # full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.png')
                                # fig, ax = plt.subplots()
                                # im = ax.imshow(full_change_matrix_no_diagonal)
                                # ax.axvline(x=.5, color='red')
                                # ax.axhline(y=.5, color='yellow')
                                # plt.title('Draw a line on an image with matplotlib')

                                # plt.savefig(full_change_matrix_no_diagonal_png_path)

                                
                                from hazelbean.visualization import full_show_array
                                full_show_array(full_change_matrix_no_diagonal, output_path=full_change_matrix_no_diagonal_auto_png_path, cbar_label='Number of changes from class R to class C per tile', title='Change matrix mosaic',
                                                num_cbar_ticks=2, vmin=0, vmid=vmax / 10.0, vmax=vmax, color_scheme='ylgnbu')


## HYBRID FUNCTION
def plot_generation(p, generation_id):
    projected_lulc_path = os.path.join(p.optimized_seals_run_dir, 'gen' + str(generation_id).zfill(6) + '_predicted_lulc.tif')
    p.projected_lulc_af = hb.ArrayFrame(projected_lulc_path)
    p.overall_similarity_plot_af = hb.ArrayFrame(os.path.join(p.optimized_seals_run_dir, 'gen' + str(generation_id).zfill(6) + '_overall_similarity_plot.tif'))

    overall_similarity_sum = np.sum(p.overall_similarity_plot_af.data)
    for i in p.change_class_labels:
        difference_metric_path = os.path.join(p.optimized_seals_run_dir, 'gen' + str(generation_id).zfill(6) + '_class_' + str(i - 1) + '_similarity.tif')
        difference_metric = hb.as_array(difference_metric_path)
        change_array = hb.as_array(p.coarse_change_paths[i - 1])

        annotation_text = """Class 
similarity:

""" + str(round(np.sum(difference_metric))) + """


Weighted
class
similarity:

""" + str(round(np.sum(difference_metric) / np.count_nonzero(np.where((p.projected_lulc_af.data == i) & (p.baseline_lulc_af.data != i), 1, 0)), 3)) + """


Overall
similarity
sum:

""" + str(round(np.sum(overall_similarity_sum), 3)) + """
"""

        output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_class_' + str(i) + '_observed_vs_projected.png')
        show_lulc_class_change_difference(p.baseline_lulc_af.data, p.observed_lulc_af.data, p.projected_lulc_af.data, i, difference_metric,
                                          change_array, annotation_text, output_path)

        output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_class_' + str(i) + '_projected_expansion_and_contraction.png')
        show_class_expansions_vs_change(p.baseline_lulc_af.data, p.projected_lulc_af.data, i, change_array, output_path, title='Class ' + str(i) + ' projected expansion and contraction on coarse change')

    output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_lulc_comparison_and_scores.png')
    show_overall_lulc_fit(p.baseline_lulc_af.data, p.observed_lulc_af.data, p.projected_lulc_af.data, p.overall_similarity_plot_af.data, output_path, title='Overall LULC and fit')


def plot_final_run():
    global p
    if p.run_this:
        if not getattr(p, 'final_generation_id', None):
            p.final_generation_id = 0
        plot_generation(p, p.final_generation_id)

        p.plot_change_matrices = 1
        if p.plot_change_matrices:
            from matplotlib import colors as colors
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 8)

            # Plot the heatmap
            vmin = np.min(full_change_matrix_no_diagonal)
            vmax = np.max(full_change_matrix_no_diagonal)
            im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin + 1, vmax=vmax))

            # Create colorbar
            import matplotlib.ticker as ticker

            cbar = ax.figure.colorbar(im, ax=ax, format=ticker.FuncFormatter(lambda x, p : int(x)))
            cbar.set_label('Number of cells changed from class ROW to class COL', size=10)

            # Set ticks...
            ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
            ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0]))

            # Create labels for each coarse zone indexed by i and j
            row_labels = []
            col_labels = []
            for i in range(n_classes * p.chunk_coarse_match.n_rows):
                class_id = i % n_classes
                coarse_grid_cell_counter = int(i / n_classes)
                row_labels.append(str(class_id))
                col_labels.append(str(class_id))

            trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction

            for i in range(p.chunk_coarse_match.n_rows):
                ann = ax.annotate('Zone i=' + str(i + 1), xy=(-3.5, (p.chunk_coarse_match.n_rows - i) / p.chunk_coarse_match.n_rows - .5 / p.chunk_coarse_match.n_rows), xycoords=trans)
                ann = ax.annotate('Zone j=' + str(i + 1), xy=(i * (p.chunk_coarse_match.n_rows + 1) + .25 * p.chunk_coarse_match.n_rows, 1.05), xycoords=trans)  #

            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(row_labels)

            # Let the horizontal axes labeling appear on top.
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

            # Turn spines off and create white grid.
            for edge, spine in ax.spines.items():
                spine.set_visible(False)

            ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1] + 1) - .5, minor=True)
            ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0] + 1) - .5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

            full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'fcmnd.png')
            # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

            major_gridline = False
            for i in range(n_classes * p.chunk_coarse_match.n_rows + 1):
                try:
                    if i % n_classes == 0:
                        major_gridline = i
                    else:
                        major_gridline = False
                except:
                    major_gridline = 0

                if major_gridline is not False:
                    xloc = major_gridline - .5
                    yloc = major_gridline - .5
                    ax.axvline(x=xloc, color='grey')
                    ax.axhline(y=yloc, color='grey')

            plt.savefig(full_change_matrix_no_diagonal_png_path)

            vmax = np.max(full_change_matrix_no_diagonal)
            # full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagona_auto.png')

            # Not really necessary but decent exampe of auto plot.
            # hb.full_show_array(full_change_matrix_no_diagonal, output_path=full_change_matrix_no_diagonal_png_path, cbar_label='Number of changes from class R to class C per tile', title='Change matrix mosaic',
            #                    num_cbar_ticks=2, vmin=0, vmid=vmax / 10.0, vmax=vmax, color_scheme='ylgnbu')


def lulc_pngs(p):
    if p.run_this:
        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            

            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                # Build a dict for the lulc labels
                labels_dict = dict(zip(p.all_class_indices, p.all_class_labels))
               
                # this acountcs for the fact that the way the correspondence is loaded is not necessarily in the numerical order
                indices_to_labels_dict = dict(sorted(labels_dict.items()))

                for year in p.years:
                    if p.scenario_type ==  'baseline':
                        current_lulc_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(p.key_base_year) + '.tif'
                        title = 'LULC ' + p.exogenous_label + ' ' + p.model_label + ' ' + str(p.key_base_year)
                        title = title.title()
                    else:
                        current_lulc_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif'
                        title = 'LULC ' + p.exogenous_label + ' ' + p.climate_label + ' ' + p.model_label + ' ' + p.counterfactual_label + ' ' + str(year)
                        title = title.title()


                    max_plotting_size = 1000000
                    
                    current_lulc_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, current_lulc_filename)
                    current_output_path = os.path.join(p.cur_dir, current_lulc_filename.replace('.tif', '.png'))
                    if not hb.path_exists(current_output_path):
                    
                        current_lulc_array = hb.as_array_resampled_to_size(current_lulc_path, max_plotting_size)

                        plot_array_as_seals7_lulc(current_lulc_array, output_path=current_output_path, title=title, indices_to_labels_dict=indices_to_labels_dict)



def coarse_fine_with_report(passed_p=None):
    if passed_p is None:
        global p 
    else:
        p = passed_p

    if p.run_this:


        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                if p.scenario_type !=  'baseline':
                    max_plotting_size = 200000
                    lulc_baseline_array = hb.as_array_resampled_to_size(p.base_year_lulc_path, max_plotting_size)

                    # By default, this will select 4 zones from different parts of the list to plot full change matrices. This is slow.
                    # You can override this to plot all here:
                    zones_to_plot = 'four' # one of first, all, or four

                    for year_c, year in enumerate(p.years):
                        target_allocation_zones_dir = os.path.join(p.allocations_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones')
                        seals_utils.load_blocks_list(p, target_allocation_zones_dir)

                        if year_c == 0:
                            previous_year = p.key_base_year
                        else:
                            previous_year = p.years[year_c - 1]
                        

                        if p.plotting_level >= 100:
                            zones_to_plot = 'all'
                        elif p.plotting_level >= 30:
                            zones_to_plot = 'four'
                        elif p.plotting_level >= 20:
                            zones_to_plot = 'first'
                        else:
                            zones_to_plot = 'none'

                        if zones_to_plot == 'all':
                            target_zones = p.global_processing_blocks_list
                        elif zones_to_plot == 'four':
                            target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                        elif zones_to_plot == 'first':
                            target_zones = [p.global_processing_blocks_list[0]]
                        else:
                            target_zones = []
                            hb.debug('No zones to plot.')

                        # Make sure the target zones are in the right format
                        for c, row in enumerate(target_zones):
                                target_zones[c] = str(row[0] + '_' + row[1])

                        for target_zone in target_zones:            
                            allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', target_zone, 'allocation') 
                            lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                            lulc_projected_array = None
                            
                            ha_per_cell_coarse_path = os.path.join(allocation_dir_to_plot, 'block_ha_per_cell_coarse.tif')
                            ha_per_cell_coarse_array = hb.as_array(ha_per_cell_coarse_path)
                            
                            ha_per_cell_fine_path = os.path.join(allocation_dir_to_plot, 'block_ha_per_cell_fine.tif')
                            ha_per_cell_fine_array = hb.as_array(ha_per_cell_fine_path)
                            
                            # lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.model_label + '_' + str(p.year) + '.tif')
                            # lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_simplification_label + '_baseline_' + p.model_label + '_' + str(p.year) + '.tif')

                            if previous_year == p.key_base_year:
                                lulc_previous_year_path = os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.model_label + '_' + str(previous_year) + '.tif')

                                lulc_previous_year_array = None     # For deffered loading                       
                            else:
                                previous_allocation_dir_to_plot = allocation_dir_to_plot.replace('\\','/').replace('/' + str(year) + '/', '/' + str(previous_year) + '/')
                                lulc_previous_year_path = os.path.join(previous_allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif')
                                lulc_previous_year_array = None
                            
                            if p.plotting_level >= 30:
                                do_class_specific_plots = True
                            else:
                                do_class_specific_plots = False

                            if do_class_specific_plots:
                                for class_id, class_label in zip(p.changing_class_indices, p.changing_class_labels): 
                                    
                                    filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                                    scaled_proportion_to_allocate_path = os.path.join(allocation_dir_to_plot, filename)
                                    output_path = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), str(target_zone), class_label + '_projected_expansion_and_contraction.png')

                                    if hb.path_exists(scaled_proportion_to_allocate_path) and not hb.path_exists(output_path):
                                        hb.log('Plotting ' + output_path)
                                        hb.create_directories(output_path)
                                        if lulc_projected_array is None:
                                            lulc_projected_array = hb.as_array_resampled_to_size(lulc_projected_path, max_plotting_size)


                                        if lulc_previous_year_array is None:
                                            lulc_previous_year_array = hb.as_array_resampled_to_size(lulc_previous_year_path, max_plotting_size)
                                        change_array = hb.as_array(scaled_proportion_to_allocate_path)

                                        
                                        show_specific_class_expansions_vs_change_with_numeric_report_and_validation(lulc_previous_year_array, lulc_projected_array, class_id, class_label, change_array, ha_per_cell_coarse_array, 
                                                                                            ha_per_cell_fine_array, allocation_dir_to_plot, output_path,
                                                                                            title='Class ' + class_label + ' projected expansion and contraction on coarse change')
            
                            if p.plotting_level >= 20:
                                do_all_class_plots = True
                            else:
                                do_all_class_plots = False
                            
                            if do_all_class_plots:
                                change_array_paths = []
                                for class_id, class_label in zip(p.changing_class_indices, p.changing_class_labels): 
                                    filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                                    scaled_proportion_to_allocate_path = os.path.join(allocation_dir_to_plot, filename)
                                    change_array_paths.append(scaled_proportion_to_allocate_path)
                                    
                                if lulc_projected_array is None:
                                    lulc_projected_array = hb.as_array_resampled_to_size(lulc_projected_path, max_plotting_size)


                                if lulc_previous_year_array is None:
                                    lulc_previous_year_array = hb.as_array_resampled_to_size(lulc_previous_year_path, max_plotting_size)
                                change_array = hb.as_array(scaled_proportion_to_allocate_path)
                                
                                output_path = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), str(target_zone), 'all_classes_projected_expansion_and_contraction.png')
                                if not hb.path_exists(output_path):
                                    hb.create_directories(output_path)
                                    show_all_class_expansions_vs_change_with_numeric_report_and_validation(lulc_previous_year_array, lulc_projected_array, p.changing_class_indices, 
                                                                                                    p.changing_class_labels, change_array_paths, ha_per_cell_coarse_array, 
                                                                                                    ha_per_cell_fine_array, allocation_dir_to_plot, output_path, 
                                                                                                    title='Projected expansion and contraction on coarse change')



## HYBRID FUNCTION
def simpler_plot_generation(p):
    # LEARNING POINT: I had assigned these as p.projected_lulc_af, which because they were project level, means they couldn't be deleted as intermediates.
    projected_lulc_path = os.path.join(p.cur_dir, 'projected_lulc.tif')
    projected_lulc_af = hb.ArrayFrame(projected_lulc_path)
    overall_similarity_plot_af = hb.ArrayFrame(os.path.join(p.cur_dir, 'overall_similarity_plot.tif'))
    lulc_baseline_af = hb.ArrayFrame(p.lulc_baseline_path)
    # p.lulc_observed_af = hb.ArrayFrame(p.lulc_observed_path)
    lulc_observed_af = hb.ArrayFrame(p.lulc_t2_path)


    coarse_change_paths = hb.list_filtered_paths_nonrecursively(p.coarse_change_dir, include_extensions='.tif')
    scaled_proportion_to_allocate_paths = []
    for path in coarse_change_paths:
        scaled_proportion_to_allocate_paths.append(os.path.join(p.coarse_change_dir, os.path.split(path)[1]))

    overall_similarity_sum = np.sum(overall_similarity_plot_af.data)
    for i in p.change_class_labels:
        difference_metric_path = os.path.join(p.cur_dir, 'class_' + str(i - 1) + '_similarity_plots.tif')
        difference_metric = hb.as_array(difference_metric_path)


        change_array = hb.as_array(scaled_proportion_to_allocate_paths[i - 1])

        annotation_text = """Class 
similarity:

""" + str(round(np.sum(difference_metric))) + """
.

Weighted
class
similarity:

""" + str(round(np.sum(difference_metric) / np.count_nonzero(np.where((projected_lulc_af.data == i) & (lulc_baseline_af.data != i), 1, 0)), 3)) + """


Overall
similarity
sum:

""" + str(round(np.sum(overall_similarity_sum), 3)) + """
"""

        # hb.pp(hb.enumerate_array_as_odict(p.lulc_baseline_af.data))
        # hb.pp(hb.enumerate_array_as_odict(p.lulc_observed_af.data))
        # hb.pp(hb.enumerate_array_as_odict(p.projected_lulc_af.data))

        output_path = os.path.join(p.cur_dir, 'class_' + str(i) + '_observed_vs_projected.png')
        show_lulc_class_change_difference(lulc_baseline_af.data, lulc_observed_af.data, projected_lulc_af.data, i, difference_metric,
                                          change_array, annotation_text, output_path)

        output_path = os.path.join(p.cur_dir, 'class_' + str(i) + '_projected_expansion_and_contraction.png')
        show_class_expansions_vs_change(lulc_baseline_af.data, projected_lulc_af.data, i, change_array, output_path, title='Class ' + str(i) + ' projected expansion and contraction on coarse change')

    output_path = os.path.join(p.cur_dir, 'lulc_comparison_and_scores.png')
    show_overall_lulc_fit(lulc_baseline_af.data, lulc_observed_af.data, projected_lulc_af.data, overall_similarity_plot_af.data, output_path, title='Overall LULC and fit')

    overall_similarity_plot_af = None
