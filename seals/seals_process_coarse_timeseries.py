import os
import hazelbean as hb
import numpy as np
import pandas as pd
import multiprocessing
from matplotlib import pyplot as plt
import geopandas as gpd
from hazelbean import netcdf
from seals_utils import download_google_cloud_blob
from hazelbean import netcdf
from hazelbean.netcdf import describe_netcdf, extract_global_netcdf
import config
from hazelbean import utils
from hazelbean import pyramids

import seals_utils

def coarse_change(p):
    # Just to create folder
    pass

def download_base_data(p):
    task_note = """" 
Download the base data. Unlike other tasks, this task puts the files into a tightly defined directory structure rooted at  p.base_data_dir    
    """
    if p.run_this:

        # Generated based on if required files actually exist.
        p.required_base_data_urls = []
        p.required_base_data_dst_paths = []

        print('replaced by p.get_path')
        # flattened_list = hb.flatten_nested_dictionary(p.required_base_data_paths, return_type='values')

        # hb.debug('Script requires the following Base Data to be in your base_data_dir\n' + hb.pp(p.required_base_data_paths, return_as_string=True))
        # for path in p.required_base_data_paths:
        #     if not hb.path_exists(path) and not path == 'use_generated' and not path == 'calibration_task':
        #         hb.log('Path did not exist, so adding it to urls to download: ' + str(path))

        #         # HACK, should have made this cleaner

        #         if p.base_data_dir in path:

        #             url_from_path =  path.split(os.path.split(p.base_data_dir)[1])[1].replace('\\', '/')
        #             url_from_path = 'base_data' + url_from_path

        #             p.required_base_data_urls.append(url_from_path)
        #             p.required_base_data_dst_paths.append(path)

        # if len(p.required_base_data_urls) > 0:
        #     for c, blob_url in enumerate(p.required_base_data_urls):

        #         # The data_credentials_path file needs to be given to the user with a specific service account email attached. Generated via the gcloud CMD line, described in new_computer.
        #         filename = os.path.split(blob_url)[1]
        #         dst_path = p.required_base_data_dst_paths[c]
        #         if not hb.path_exists(dst_path): # Check one last time to ensure that it wasn't added twice.
        #             download_google_cloud_blob(p.input_bucket_name, blob_url, p.data_credentials_path, dst_path)



def lulc_as_coarse_states(p):
    """This task is not needed for the BASE seals workflow but is for either calibration runs, or runs that are based on adjusting the inputs to match existing LULC."""
    p.L.warning('This task doesnt speed up enough when using testing. consider adding it to base data.')
    """For the purposes of calibration, create change-matrices for each coarse grid-cell based on two observed ESA lulc maps.
    Does something similar to prepare_lulc"""

    from hazelbean.calculation_core.cython_functions import calc_change_matrix_of_two_int_arrays
   
    if p.run_this:
        p.ha_per_cell_coarse = hb.ArrayFrame(p.global_ha_per_cell_course_path)
        p.coarse_match = hb.ArrayFrame(p.global_ha_per_cell_course_path)


        # TODO This needs to be fixed so that it calculates on the reclassification in use (currently it's using simplified hardcoded but we need it to shift to)
        output_arrays = np.zeros((len(p.class_indices), p.coarse_match.shape[0], p.coarse_match.shape[1]))
        calc_change_matrix = False
        numpy_output_path = os.path.join(p.cur_dir, 'change_matrices.npy')
        if not hb.path_exists(numpy_output_path) and calc_change_matrix:
            t1 = hb.ArrayFrame(p.training_start_year_simplified_lulc_path)
            t2 = hb.ArrayFrame(p.training_end_year_simplified_lulc_path)
            fine_cells_per_coarse_cell = round((p.ha_per_cell_coarse.cell_size / t1.cell_size) ** 2)
            aspect_ratio = t1.num_cols / p.coarse_match.num_cols
            for r in range(p.coarse_match.num_rows):
                hb.log('Processing observed change row', r)
                for c in range(p.coarse_match.num_cols):
                    t1_subarray = t1.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                    t2_subarray = t2.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                    # ha_per_cell_subarray = p.ha_per_cell_coarse.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]

                    ha_per_cell_coarse_this_subarray = p.ha_per_cell_coarse.data[r, c]
                    change_matrix, counters = calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.class_indices)
                    # Potentially unused relic from prepare_lulc
                    full_change_matrix = np.zeros((len(p.class_indices), len(p.class_indices)))
                    vector = seals_utils.calc_change_vector_of_change_matrix(change_matrix)

                    ha_per_cell_this_subarray = p.ha_per_cell_coarse.data[r, c] / fine_cells_per_coarse_cell

                    if vector:
                        for i in p.class_indices:
                            output_arrays[i - 1, r, c] = vector[i - 1] * ha_per_cell_this_subarray
                    else:
                        output_arrays[i, r, c] = 0.0

            for c, class_label in enumerate(p.class_labels):
                output_path = os.path.join(p.cur_dir, class_label + '_observed_change.tif')
                hb.save_array_as_geotiff(output_arrays[c], output_path, p.coarse_match.path)
            hb.save_array_as_npy(output_arrays, numpy_output_path)

            # Stores all of the classes in a 3d array ready for validation exercises below.
            change_3d = hb.load_npy_as_array(numpy_output_path)

        # Sometimes you don't want the change but need the actual state maps (ala luh) implied by a given ESA map.
        # Here calculates a cython function that downscales a fine_categorical to a stack of coarse_continuous 3d
        # Test that this is equivilent to change_3d


        p.base_year_simplified_lulc_path = p.global_esa_simplified_lulc_paths_by_year[p.baseline_years[0]]

        p.observed_lulc_paths_to_calculate_states = [ p.base_year_simplified_lulc_path]
        # p.observed_lulc_paths_to_calculate_states = [p.global_esa_simplified_lulc_paths_by_year[2000], p.global_esa_simplified_lulc_paths_by_year[2015], p.base_year_simplified_lulc_path]

        p.years_to_calculate_states = p.baseline_years
        # p.years_to_calculate_states = [2000, 2015] + p.baseline_years
        p.observed_state_paths = {}
        for year in p.years_to_calculate_states:
            p.observed_state_paths[year] = {}
            for class_label in p.class_labels + p.nonchanging_class_labels:
                p.observed_state_paths[year][class_label] = os.path.join(p.cur_dir, hb.file_root(p.global_esa_simplified_lulc_paths_by_year[year]) + '_state_' + str(class_label) + '_observed.tif')

        global_bb = hb.get_bounding_box(p.observed_lulc_paths_to_calculate_states[0])
        # TODOO Here incorporate test-mode bb.
        # stitched_bb = hb.get_bounding_box()

        for c, year in enumerate(p.years_to_calculate_states):
            if not hb.path_exists(p.observed_state_paths[year][p.class_labels[0]], verbose=0):
            # if not all([hb.path_exists(i) for i in p.observed_state_paths[year]]):

                fine_path = p.observed_lulc_paths_to_calculate_states[c]
                hb.log('Calculating coarse_state_stack from ' + fine_path)

                output_dir = p.cur_dir

                fine_input_array = hb.load_geotiff_chunk_by_bb(fine_path, global_bb, datatype=5)
                coarse_match_array = hb.load_geotiff_chunk_by_bb(p.coarse_match_path, global_bb, datatype=6)

                chunk_edge_length = int(fine_input_array.shape[0] / coarse_match_array.shape[0])

                max_value_to_summarize = 8
                values_to_summarize = np.asarray(p.class_indices + p.nonchanging_class_indices, dtype=np.int32)

                import hazelbean.calculation_core
                coarse_state_3d = hb.calculation_core.cython_functions.calculate_coarse_state_stack_from_fine_classified(fine_input_array,
                                                                      coarse_match_array,
                                                                      values_to_summarize,
                                                                      max_value_to_summarize)

                c = 0
                for k, v in p.observed_state_paths[year].items():
                    # Convert a count of states to a proportion of grid-cell
                    a = coarse_state_3d[c].astype(np.float32) / np.float32(chunk_edge_length ** 2)
                    hb.save_array_as_geotiff(a, v, p.coarse_match_path, data_type=6)
                    c += 1

        # Clip back to AOI


        c = 0
        for k, v in p.observed_state_paths[year].items():
            hb.load_geotiff_chunk_by_bb(v, p.bb, output_path=hb.suri(v, 'aoi'))




def coarse_extraction(p):
    doc = """Create a empty folder dir. This will hold all of the coarse intermediate outputs, such as per-year changes in lu hectarage. Naming convention matches source. After reclassification this will be in destination conventions.  """
    if p.run_this:
         
        
        # if p.report_netcdf_read_analysis:
            
        for index, row in list(p.scenarios_df.iterrows()):
            seals_utils.assign_df_row_to_object_attributes(p, row)

            hb.log('Extracting coarse states for scenario ' + str(index) + ' of ' + str(len(p.scenarios_df)) + ' with row ' + str([i for i in row]))
            hb.debug('Analyzing row:\n' + str(row))

            if p.scenario_type == 'baseline':

                if hb.path_exists(os.path.join(p.input_dir, p.coarse_projections_input_path)):
                    src_nc_path = os.path.join(p.input_dir, p.coarse_projections_input_path)
                elif hb.path_exists(os.path.join(p.base_data_dir, p.coarse_projections_input_path)):
                    src_nc_path = os.path.join(p.base_data_dir, p.coarse_projections_input_path)
                else:
                    hb.log('Could not find ' + str(p.coarse_projections_input_path) + ' in either ' + str(p.input_dir) + ' or ' + str(p.base_data_dir))

                dst_dir = os.path.join(p.cur_dir, p.exogenous_label, p.model_label)

                adjustment_dict = {
                    'time': row['time_dim_adjustment'],  # eg +850 or *5+14 eg
                }

                filter_dict = {
                    'time': p.years,
                }

                if not hb.path_exists(dst_dir):
                    extract_global_netcdf(src_nc_path, dst_dir, adjustment_dict, filter_dict, skip_if_exists=True, verbose=0)
            else:
                
                if hb.path_exists(os.path.join(p.input_dir, p.coarse_projections_input_path)):
                    src_nc_path = os.path.join(p.input_dir, p.coarse_projections_input_path)
                elif hb.path_exists(os.path.join(p.base_data_dir, p.coarse_projections_input_path)):
                    src_nc_path = os.path.join(p.base_data_dir, p.coarse_projections_input_path)
                else:
                    hb.log('No understandible input_source.')
               
                dst_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label)



                adjustment_dict = {
                    'time': row['time_dim_adjustment'],  # or *5+14 eg
                }

                filter_dict = {
                    'time': p.years,
                }
                if not hb.path_exists(dst_dir):
                    extract_global_netcdf(src_nc_path, dst_dir, adjustment_dict, filter_dict, skip_if_exists=True, verbose=0)

                
def coarse_simplified_proportion(p):
    task_note = """This function converts the extracted geotiffs from the source 
classification to the the destination classification, potentially aggregating classes as it goes. """\
    
    if p.run_this:
        
        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)
            hb.log('Converting coarse_extraction to simplified proportion for scenario ' + str(index) + ' of ' + str(len(p.scenarios_df)) + ' with row ' + str([i for i in row]))


            
            # p.coarse_correspondence_path = hb.get_first_extant_path(p.coarse_correspondence_path, [p.input_dir, p.base_data_dir])
            p.coarse_correspondence_dict = hb.utils.get_reclassification_dict_from_df(p.coarse_correspondence_path, 'src_id', 'dst_id', 'src_label', 'dst_label')
            
            # if hb.path_exists(os.path.join(p.input_dir, p.coarse_correspondence_path)):
            #     p.coarse_correspondence_dict = hb.utils.get_reclassification_dict_from_df(os.path.join(p.input_dir, p.coarse_correspondence_path), 'src_id', 'dst_id', 'src_label', 'dst_label')
            # elif hb.path_exists(os.path.join(p.base_data_dir, p.coarse_correspondence_path)):
            #     p.coarse_correspondence_dict = hb.utils.get_reclassification_dict_from_df(os.path.join(p.base_data_dir, p.coarse_correspondence_path), 'src_id', 'dst_id', 'src_label', 'dst_label')
            # else:
            #     raise NameError('Unable to find ' + p.coarse_correspondence_path)

            if p.scenario_type == 'baseline':

                for year in p.years:
                    
                    dst_dir = os.path.join(p.cur_dir, p.exogenous_label, p.model_label, str(year))
                    hb.create_directories(dst_dir)


                    for k, v in p.coarse_correspondence_dict['dst_to_src_reclassification_dict'].items():
                        # pos = list(p.coarse_correspondence_dict['dst_to_src_reclassification_dict'].keys()).index(k)
                        output_array = None


                        # dst_class_label = list(p.coarse_correspondence_dict['dst_labels'])[pos]
                        
                        dst_class_label = p.coarse_correspondence_dict['dst_ids_to_labels'][k]
                        dst_path = os.path.join(dst_dir, str(dst_class_label) + '_prop_' + p.exogenous_label + '_' + p.model_label + '_' + str(year) + '.tif')

                        if not hb.path_exists(dst_path):
                            
                            # Notice that here implies the relationship from src to simplified is many to one.
                            for c, i in enumerate(v):

                                if p.lc_class_varname == 'all_variables':
                                    src_class_label = p.coarse_correspondence_dict['src_ids_to_labels'][i]
                                    # src_class_label = p.coarse_correspondence_dict['src_labels'][c]
                                    src_dir = os.path.join(p.coarse_extraction_dir, p.exogenous_label, p.model_label, 'time_' + str(year))

                                    src_path = os.path.join(src_dir, src_class_label + '.tif')
                                    ndv = hb.get_ndv_from_path(src_path)
                                    coarse_shape = hb.get_shape_from_dataset_path(src_path)
                                    if output_array is None:
                                        output_array = np.zeros(coarse_shape, dtype=np.float64)
                                    input_array = hb.as_array(src_path)
                                    output_array += np.where(input_array != ndv, input_array, 0)
                                else:
                                    # Then the lc_class vars have been embeded as dimensions... grrrr
                                    src_class_label = p.coarse_correspondence_dict['src_ids_to_labels'][i]
                                    # src_class_label = p.coarse_correspondence_dict['src_labels'][c]
                                    src_dir = os.path.join(p.coarse_extraction_dir, p.exogenous_label, p.model_label, 'time_' + str(year), p.dimensions[-1] + '_' + str(i))

                                    src_path = os.path.join(src_dir, p.lc_class_varname + '.tif')
                                    ndv = hb.get_ndv_from_path(src_path)
                                    coarse_shape = hb.get_shape_from_dataset_path(src_path)
                                    if output_array is None:
                                        output_array = np.zeros(coarse_shape, dtype=np.float64)
                                    input_array = hb.as_array(src_path)
                                    output_array += np.where(input_array != ndv, input_array, 0)

                            hb.save_array_as_geotiff(output_array, dst_path, src_path)
                            output_array = None
            else:
                for year in p.years:
                    
                    dst_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                    hb.create_directories(dst_dir)


                    for k, v in p.coarse_correspondence_dict['dst_to_src_reclassification_dict'].items():
                        # pos = list(p.coarse_correspondence_dict['dst_to_src_reclassification_dict'].keys()).index(k)
                        output_array = None
                        dst_class_label = p.coarse_correspondence_dict['dst_ids_to_labels'][k]
                        # dst_class_label = list(p.coarse_correspondence_dict['dst_labels'])[pos]
                        dst_path = os.path.join(dst_dir, str(dst_class_label) + '_prop_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')

                        if not hb.path_exists(dst_path):
                            
                            # Notice that here implies the relationship from src to simplified is many to one.
                            for c, i in enumerate(v):

                                if p.lc_class_varname == 'all_variables':
                                    src_class_label = p.coarse_correspondence_dict['src_ids_to_labels'][i]
                                    # src_class_label = p.coarse_correspondence_dict['src_labels'][c]
                                    src_dir = os.path.join(p.coarse_extraction_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, 'time_' + str(year))

                                    src_path = os.path.join(src_dir, src_class_label + '.tif')
                                    ndv = hb.get_ndv_from_path(src_path)
                                    coarse_shape = hb.get_shape_from_dataset_path(src_path)
                                    if output_array is None:
                                        output_array = np.zeros(coarse_shape, dtype=np.float64)
                                    input_array = hb.as_array(src_path)
                                    output_array += np.where(input_array != ndv, input_array, 0)
                                else:
                                    # Then the lc_class vars have been embeded as dimensions... grrrr
                                    src_class_label = p.coarse_correspondence_dict['src_ids_to_labels'][i]
                                    # src_class_label = p.coarse_correspondence_dict['src_labels'][c]
                                    src_dir = os.path.join(p.coarse_extraction_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, 'time_' + str(year), p.dimensions[-1] + '_' + str(i))

                                    src_path = os.path.join(src_dir, p.lc_class_varname + '.tif')
                                    ndv = hb.get_ndv_from_path(src_path)

                                    coarse_shape = hb.get_shape_from_dataset_path(src_path)
                                    if output_array is None:
                                        output_array = np.zeros(coarse_shape, dtype=np.float64)
                                    input_array = hb.as_array(src_path)
                                    output_array += np.where(input_array != ndv, input_array, 0)                        
                            hb.save_array_as_geotiff(output_array, dst_path, src_path)
                            output_array = None


def coarse_simplified_ha_difference_from_base_year(p):

    if p.run_this:      

        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)
            hb.log('Converting coarse_extraction to simplified proportion for scenario ' + str(index) + ' of ' + str(len(p.scenarios_df)) + ' with row ' + str([i for i in row]))
            
            if hb.path_exists(os.path.join(p.input_dir, p.coarse_correspondence_path)):
                p.lulc_correspondence_dict = hb.utils.get_reclassification_dict_from_df(os.path.join(p.input_dir, p.coarse_correspondence_path), 'src_id', 'dst_id', 'src_label', 'dst_label')
            elif hb.path_exists(os.path.join(p.base_data_dir, p.coarse_correspondence_path)):
                p.lulc_correspondence_dict = hb.utils.get_reclassification_dict_from_df(os.path.join(p.base_data_dir, p.coarse_correspondence_path), 'src_id', 'dst_id', 'src_label', 'dst_label')
            else:
                raise NameError('Unable to find ' + p.coarse_correspondence_path)
                        
            cell_size = hb.get_cell_size_from_path(os.path.join(p.base_data_dir, p.coarse_projections_input_path))
            
            correct_global_ha_per_cell_path = p.ha_per_cell_paths[cell_size * 3600.0] 
            correct_aoi_ha_per_cell_path = os.path.join(p.intermediate_dir, 'project_aoi', 'pyramids', 'aoi_ha_per_cell_' + str(int(cell_size * 3600.0)) + '.tif')
            hb.clip_raster_by_bb(correct_global_ha_per_cell_path, p.bb, correct_aoi_ha_per_cell_path)
            ha_per_cell_array = hb.as_array(correct_aoi_ha_per_cell_path)
            

            if p.scenario_type != 'baseline':


                for c, i in enumerate(p.coarse_correspondence_dict['dst_ids']):
                   
                    # pos = list(coarse_correspondence_dict['values'].keys()).index(k)


                    output_array = None
                    dst_class_label = p.coarse_correspondence_dict['dst_labels'][c]


                    # Get the input_path to the baseline_reference_label from the scenarios_df
                    baseline_reference_label = row['baseline_reference_label']
                    baseline_reference_row = p.scenarios_df.loc[p.scenarios_df['scenario_label'] == baseline_reference_label]
                    baseline_exogenous_label = baseline_reference_row['exogenous_label'].values[0]
                    baseline_reference_model = baseline_reference_row['model_label'].values[0]
                    
                    base_year = int(row['key_base_year'])

                    base_year_dir = os.path.join(p.coarse_simplified_proportion_dir, baseline_exogenous_label, baseline_reference_model, str(base_year))
                    base_year_path = os.path.join(base_year_dir, str(dst_class_label) + '_prop_' + baseline_exogenous_label + '_' + baseline_reference_model + '_' + str(base_year) + '.tif')
                    # base_year_path = os.path.join(base_year_dir, p.lulc_simplification_label + '_' + v + '.tif')

                    for year in p.years:
                    
                        src_dir = os.path.join(p.coarse_simplified_proportion_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                        src_path = os.path.join(src_dir, str(dst_class_label) + '_prop_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')

                        dst_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                        hb.create_directories(dst_dir)

                        dst_path = os.path.join(dst_dir, dst_class_label + '_' + str(year) + '_' + str(base_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif')

                        if not hb.path_exists(dst_path):

                            input_array = hb.load_geotiff_chunk_by_bb(src_path, p.bb)
                            input_ndv = hb.get_ndv_from_path(src_path)

                            base_year_array = hb.load_geotiff_chunk_by_bb(base_year_path, p.bb)
                            base_year_ndv = hb.get_ndv_from_path(base_year_path)

                            input_array = np.where(input_array == input_ndv, 0, input_array)
                            base_year_array = np.where(base_year_array == base_year_ndv, 0, base_year_array)                            

                            if input_array.shape != base_year_array.shape:
                                raise NameError('input_array.shape != base_year_array.shape: ' + str(input_array.shape) + ' != ' + str(base_year_array.shape) + '. This means that the coarse definition of the scenario that you are subtracting from the coarse definition of the baseline is mixing resolutions. You probably want to resample one of the two layers first.') 
                            current_array = (input_array - base_year_array) * ha_per_cell_array

                            hb.save_array_as_geotiff(current_array, dst_path, correct_aoi_ha_per_cell_path)
                    


def coarse_simplified_ha_difference_from_previous_year(p):
    task_documentation = """Calculates LUH2_simplified difference from base year"""
    if p.run_this:      

        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)
            hb.log('Converting coarse_extraction to simplified proportion for scenario ' + str(index) + ' of ' + str(len(p.scenarios_df)) + ' with row ' + str([i for i in row]))
            
            if hb.path_exists(os.path.join(p.input_dir, p.coarse_correspondence_path)):
                p.lulc_correspondence_dict = hb.utils.get_reclassification_dict_from_df(os.path.join(p.input_dir, p.coarse_correspondence_path), 'src_id', 'dst_id', 'src_label', 'dst_label')
            elif hb.path_exists(os.path.join(p.base_data_dir, p.coarse_correspondence_path)):
                p.lulc_correspondence_dict = hb.utils.get_reclassification_dict_from_df(os.path.join(p.base_data_dir, p.coarse_correspondence_path), 'src_id', 'dst_id', 'src_label', 'dst_label')
            else:
                raise NameError('Unable to find ' + p.coarse_correspondence_path)
                        
            cell_size = hb.get_cell_size_from_path(os.path.join(p.base_data_dir, p.coarse_projections_input_path))
            
            correct_global_ha_per_cell_path = p.ha_per_cell_paths[cell_size * 3600.0] 
            correct_aoi_ha_per_cell_path = os.path.join(p.intermediate_dir, 'project_aoi', 'pyramids', 'aoi_ha_per_cell_' + str(int(cell_size * 3600.0)) + '.tif')
            hb.clip_raster_by_bb(correct_global_ha_per_cell_path, p.bb, correct_aoi_ha_per_cell_path)
            ha_per_cell_array = hb.as_array(correct_aoi_ha_per_cell_path)
            

            

            if p.scenario_type != 'baseline':


                for c, i in enumerate(p.coarse_correspondence_dict['dst_ids']):
                   
                    # pos = list(coarse_correspondence_dict['values'].keys()).index(k)


                    output_array = None
                    dst_class_label = p.coarse_correspondence_dict['dst_labels'][c]


                    # Get the input_path to the baseline_reference_label from the scenarios_df
                    baseline_reference_label = row['baseline_reference_label']
                    baseline_reference_row = p.scenarios_df.loc[p.scenarios_df['scenario_label'] == baseline_reference_label]
                    baseline_exogenous_label = baseline_reference_row['exogenous_label'].values[0]
                    baseline_reference_model = baseline_reference_row['model_label'].values[0]
                    current_starting_year = None
                    previous_year = None

                    # Process the current row and its corresponding baseline to get the full set of years involved
                    
                    
                    # starting_year = int(row['key_base_year'])
                    # base_year = int(row['key_base_year'])

                    # base_year_dir = os.path.join(p.coarse_simplified_proportion_dir, baseline_exogenous_label, baseline_reference_model, str(base_year))
                    # base_year_path = os.path.join(base_year_dir, str(dst_class_label) + '_prop_' + baseline_exogenous_label + '_' + baseline_reference_model + '_' + str(base_year) + '.tif')
                    # # base_year_path = os.path.join(base_year_dir, p.lulc_simplification_label + '_' + v + '.tif')

                    for year in p.years:


                        if current_starting_year is None:
                            base_year = int(row['key_base_year'])
                            current_starting_year = base_year
                            current_starting_year_dir = os.path.join(p.coarse_simplified_proportion_dir, baseline_exogenous_label, baseline_reference_model, str(current_starting_year))
                            current_starting_year_path = os.path.join(current_starting_year_dir, str(dst_class_label) + '_prop_' + baseline_exogenous_label + '_' + baseline_reference_model + '_' + str(current_starting_year) + '.tif')
                        else:

                            
                            current_starting_year = previous_year
                            current_starting_year_dir = os.path.join(p.coarse_simplified_proportion_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(current_starting_year))
                            # current_starting_year_dir = os.path.join(p.coarse_simplified_proportion_dir, baseline_exogenous_label, baseline_reference_model, str(current_starting_year))
                            current_starting_year_path = os.path.join(current_starting_year_dir, str(dst_class_label) + '_prop_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(current_starting_year) + '.tif')
                            # current_starting_year_path = os.path.join(current_starting_year_dir, str(dst_class_label) + '_prop_' + baseline_exogenous_label + '_' + baseline_reference_model + '_' + str(current_starting_year) + '.tif')
                            
                        current_ending_year_src_dir = os.path.join(p.coarse_simplified_proportion_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                        current_ending_year_src_path = os.path.join(current_ending_year_src_dir, str(dst_class_label) + '_prop_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')

                        current_ending_year_dst_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                        hb.create_directories(current_ending_year_dst_dir)

                        current_ending_year_dst_path = os.path.join(current_ending_year_dst_dir, dst_class_label + '_' + str(year) + '_' + str(current_starting_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif')

                        if not hb.path_exists(current_ending_year_dst_path):

                            ending_year_array = hb.load_geotiff_chunk_by_bb(current_ending_year_src_path, p.bb)
                            ending_year_ndv = hb.get_ndv_from_path(current_ending_year_src_path)

                            starting_year_array = hb.load_geotiff_chunk_by_bb(current_starting_year_path, p.bb)
                            starting_year_ndv = hb.get_ndv_from_path(current_starting_year_path)

                            ending_year_array = np.where(ending_year_array == ending_year_ndv, 0, ending_year_array)
                            starting_year_array = np.where(starting_year_array == starting_year_ndv, 0, starting_year_array)                            

                            if ending_year_array.shape != starting_year_array.shape:
                                raise NameError('ending_year_array.shape != starting_year_array.shape: ' + str(ending_year_array.shape) + ' != ' + str(starting_year_array.shape) + '. This means that the coarse definition of the scenario that you are subtracting from the coarse definition of the baseline is mixing resolutions. You probably want to resample one of the two layers first.') 
                            current_array = (ending_year_array - starting_year_array) * ha_per_cell_array

                            hb.save_array_as_geotiff(current_array, current_ending_year_dst_path, correct_aoi_ha_per_cell_path)

                        previous_year = year
                    
