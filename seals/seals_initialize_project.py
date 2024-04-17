# initialize_project defines the run() command for the whole project, which takes the project object as its only function.

import hazelbean as hb
# conda_envs_with_cython = hb.check_which_conda_envs_have_library_installed('cython')
# print(conda_envs_with_cython)


import os, sys
import hazelbean as hb
from hazelbean import cloud_utils

import seals_main
import seals_generate_base_data
import seals_process_coarse_timeseries
import seals_visualization_tasks
import config
from seals_utils import download_google_cloud_blob

def run(p):   
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)
       
    # initialize and set all basic variables. Sadly this is still needed even for a SEALS run until it's extracted.
    # p.combined_block_lists_paths = None # This will be smartly determined in either calibration or allocation


    ###--------------- Additional options (could make advanced options in the UI eventually) ----------------

    # Determine if overviews should be written.
    p.write_global_lulc_overviews_and_tifs = True    

    # Specifies which sigmas should be used in a gaussian blur of the class-presence tifs in order to regress on adjacency.
    # Note this will have a huge impact on performance as full-extent gaussian blurs for each class will be generated for
    # each sigma.
    # TODOO Figure out how this relates to the coefficients csv. I could probably derive these values from that csv.
    p.gaussian_sigmas_to_test = [1, 5]

    # There are still multiple ways to do the allocation. Unless we input a fully-defined
    # change matrix, there will always be ambiguities. One way of lessening them is to 
    # switch from the default allocation method (just do positive allocation requests)
    # to one that also increases the total goal when some other class
    # goes on it. Allowing it leads to a greater amount of the requested allocation
    # happening, but it can lead to funnylooking total-flip cells.
    p.allow_contracting = 0

    # Change how many generations of training to allow. A generation is an exhaustive search so relatievely few generations are required to get to a point
    # where no more improvements can be found.
    p.num_generations = 1

    # If True, will load that which was calculated in the calibration run.
    p.use_calibration_created_coefficients = 0

    
    # Sometimes runs fail mid run. This checks for that and picks up where there is a completed file for that zone. 
    # However doing so can cause confusing cache-invalidation situations for troubleshooting so it's off by default.
    p.skip_created_downscaling_zones = 0

    # For testing,it may be useful to just run the first element of each iterator for speed.
    p.run_only_first_element_of_each_iterator = 0


    ##--------------- Pyramid path references.  ----------------
    
    # To easily convert between per-ha and per-cell terms, these very accurate ha_per_cell maps are defined.
    p.ha_per_cell_10sec_ref_path = os.path.join('pyramids', "ha_per_cell_10sec.tif")
    p.ha_per_cell_300sec_ref_path = os.path.join('pyramids', "ha_per_cell_300sec.tif")
    p.ha_per_cell_900sec_ref_path = os.path.join('pyramids', "ha_per_cell_900sec.tif")
    p.ha_per_cell_1800sec_ref_path = os.path.join('pyramids', "ha_per_cell_1800sec.tif")
    p.ha_per_cell_3600sec_ref_path = os.path.join('pyramids', "ha_per_cell_3600sec.tif")


    p.ha_per_cell_ref_paths = {}
    p.ha_per_cell_ref_paths[10.0] = p.ha_per_cell_10sec_ref_path
    p.ha_per_cell_ref_paths[300.0] = p.ha_per_cell_300sec_ref_path
    p.ha_per_cell_ref_paths[900.0] = p.ha_per_cell_900sec_ref_path
    p.ha_per_cell_ref_paths[1800.0] = p.ha_per_cell_1800sec_ref_path
    p.ha_per_cell_ref_paths[3600.0] = p.ha_per_cell_3600sec_ref_path

    # To easily convert between per-ha and per-cell terms, these very accurate ha_per_cell maps are defined.
    p.ha_per_cell_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_10sec.tif")
    p.ha_per_cell_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_300sec.tif")
    p.ha_per_cell_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_900sec.tif")
    p.ha_per_cell_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_1800sec.tif")
    p.ha_per_cell_3600sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_3600sec.tif")

    p.ha_per_cell_paths = {}
    p.ha_per_cell_paths[10.0] = p.ha_per_cell_10sec_path
    p.ha_per_cell_paths[300.0] = p.ha_per_cell_300sec_path
    p.ha_per_cell_paths[900.0] = p.ha_per_cell_900sec_path
    p.ha_per_cell_paths[1800.0] = p.ha_per_cell_1800sec_path
    p.ha_per_cell_paths[3600.0] = p.ha_per_cell_3600sec_path

    # The ha per cell paths also can be used when writing new tifs as the match path.
    p.match_10sec_path = p.ha_per_cell_10sec_path
    p.match_300sec_path = p.ha_per_cell_300sec_path
    p.match_900sec_path = p.ha_per_cell_900sec_path
    p.match_1800sec_path = p.ha_per_cell_1800sec_path
    p.match_3600sec_path = p.ha_per_cell_3600sec_path

    p.match_paths = {}
    p.match_paths[10.0] = p.match_10sec_path
    p.match_paths[300.0] = p.match_300sec_path
    p.match_paths[900.0] = p.match_900sec_path
    p.match_paths[1800.0] = p.match_1800sec_path
    p.match_paths[3600.0] = p.match_3600sec_path

    p.ha_per_cell_column_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_10sec.tif")
    p.ha_per_cell_column_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_300sec.tif")
    p.ha_per_cell_column_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_900sec.tif")
    p.ha_per_cell_column_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_1800sec.tif")
    p.ha_per_cell_column_3600sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_3600sec.tif")

    # If you're willing to assume the world is a sphere, it's faster to just load the columns
    p.ha_per_cell_column_paths = {}
    p.ha_per_cell_column_paths[10.0] = p.ha_per_cell_column_10sec_path
    p.ha_per_cell_column_paths[300.0] = p.ha_per_cell_column_300sec_path
    p.ha_per_cell_column_paths[900.0] = p.ha_per_cell_column_900sec_path
    p.ha_per_cell_column_paths[1800.0] = p.ha_per_cell_column_1800sec_path
    p.ha_per_cell_column_paths[3600.0] = p.ha_per_cell_column_3600sec_path

    # On the stitched_lulc_simplified_scenarios task, optionally clip it to the aoi. Be aware that this
    # means you can no longer user it in Pyramid-style operations (basically all besides zonal stats).
    p.clip_to_aoi = 1
    
    ### ------------------- SET UNUSED ATTRIBUTES TO NONE ------------------- ###
    
    if not hasattr(p, 'subset_of_blocks_to_run'):
        p.subset_of_blocks_to_run = None # No subset


    ### ------------------- Build paths to download ------------------- ###
    p.static_regressor_paths = {}
    p.static_regressor_paths['sand_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'sand_percent.tif')
    p.static_regressor_paths['silt_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'silt_percent.tif')
    p.static_regressor_paths['soil_bulk_density'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_bulk_density.tif')
    p.static_regressor_paths['soil_cec'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_cec.tif')
    p.static_regressor_paths['soil_organic_content'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_organic_content.tif')
    p.static_regressor_paths['strict_pa'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'strict_pa.tif')
    p.static_regressor_paths['temperature_c'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'temperature_c.tif')
    p.static_regressor_paths['travel_time_to_market_mins'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'travel_time_to_market_mins.tif')
    p.static_regressor_paths['wetlands_binary'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'wetlands_binary.tif')
    p.static_regressor_paths['alt_m'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'alt_m.tif')
    p.static_regressor_paths['carbon_above_ground_mg_per_ha_global'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'carbon_above_ground_mg_per_ha_global.tif')
    p.static_regressor_paths['clay_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'clay_percent.tif')
    p.static_regressor_paths['ph'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'ph.tif')
    p.static_regressor_paths['pop'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'pop.tif')
    p.static_regressor_paths['precip_mm'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'precip_mm.tif')

    # # Create list of paths that need to be downloaded
    # p.required_base_data_paths = []
    # p.required_base_data_paths.append(p.countries_iso3_path)
    # p.required_base_data_paths.append(p.base_year_lulc_path)
    # p.required_base_data_paths.append(p.calibration_parameters_source)
    # p.required_base_data_paths.append(p.coarse_projections_input_path)
    # p.required_base_data_paths.append(p.lulc_correspondence_path)
    # p.required_base_data_paths.append(p.coarse_correspondence_path)

    # p.required_base_data_paths.extend(p.static_regressor_paths.values())


    p.fine_resolution_degrees = hb.pyramid_compatible_resolutions[p.fine_resolution_arcseconds]
    p.coarse_resolution_degrees = hb.pyramid_compatible_resolutions[p.coarse_resolution_arcseconds]
    p.fine_resolution = p.fine_resolution_degrees
    p.coarse_resolution = p.coarse_resolution_degrees

    p.fine_ha_per_cell_path = p.ha_per_cell_paths[p.fine_resolution_arcseconds]
    p.fine_match_path = p.fine_ha_per_cell_path

    p.coarse_ha_per_cell_path = p.ha_per_cell_paths[p.coarse_resolution_arcseconds]
    p.coarse_match_path = p.coarse_ha_per_cell_path




    ###-----------------  GTAP-specific parameters.  -----------------
    ### Keep this for when I integrate back in with GTAP.
    # TODOO This is still based on the file below, which was from Purdue. It is a vector of 300sec gridcells and should be replaced with continuous vectors
    # p.gtap37_aez18_input_vector_path = os.path.join(p.base_data_dir, "pyramids", "GTAP37_AEZ18.gpkg")
    # p.use_calibration_from_zone_centroid_tile = 1
    # 
    # p.calibration_zone_polygons_path = os.path.join(p.gtap37_aez18_input_vector_path)  # Only needed if use_calibration_from_zone_centroid_tile us True.    
    # if p.is_gtap1_run:
    #     # because I don't yet auto-generate the cmf files and other GTAP modelled inputs, and instead just take the files out of the zipfile Uris
    #     # provides, I still have to follow his naming scheme. This list comprehension converts a policy_scenario_label into a gtap1 or gtap2 label.
    #     p.gtap1_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_noES' for i in p.policy_scenario_labels]
    # #     p.gtap2_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_allES' for i in p.policy_scenario_labels]
    # p.gtap_combined_policy_scenario_labels = ['BAU', 'BAU_rigid', 'PESGC', 'SR_Land', 'PESLC', 'SR_RnD_20p', 'SR_Land_PESGC', 'SR_PESLC', 'SR_RnD_20p_PESGC', 'SR_RnD_PESLC', 'SR_RnD_20p_PESGC_30']
    # p.gtap_just_bau_label = ['BAU']
    # p.gtap_bau_and_30_labels = ['BAU', 'SR_RnD_20p_PESGC_30']
    # p.luh_labels = ['no_policy']
    # This is a zipfile I received from URIS that has all the packaged GTAP files ready to run. Extract these to a project dir.
    # p.gtap_aez_invest_release_string = '04_20_2021_GTAP_AEZ_INVEST'
    # p.gtap_aez_invest_zipfile_path = os.path.join(p.base_data_dir, 'gtap_aez_invest_releases', p.gtap_aez_invest_release_string + '.zip')
    # p.gtap_aez_invest_code_dir = os.path.join(p.script_dir, 'gtap_aez', p.gtap_aez_invest_release_string)
    
    ##### RECOMPILE CYTHON FILE #####
    # If you require compiling the cython code, set the name of a conda/mamba environment here. 
    # If unset, it will use whatever conda/mamba environment you used to run the script.
    # If you don't know what this means, just leave it as None and it will use the default environment.
    # However, depending on how you installed this, you might need to compile the cython code regardless.
    # You will know this is the case if the code fails with "cannot import seals_cython_functions"
    # or something simmilar.
    # Note that to actually do the recompilation of the C code, you will need to follow the instructions from the following error message 
    # error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
    
    # Get the name of the current environment
    
    
    p.execute()


def project_aoi(p):
    """
    Generate the area of interest (AOI) of the current project based on the inputs defined in the run file.

    This task must be run first because it defines how all subsequent data will be extracted (based on the bounding box of the AOI)

    """ 
    download_urls = {}   
    # download_urls[p.countries_iso3_path] = p.countries_iso3_path.split(p.base_data_dir)[1].replace('\\', '/')
    # download_urls[p.ha_per_cell_paths[p.fine_resolution_arcseconds]] = p.ha_per_cell_paths[p.fine_resolution_arcseconds].split(p.base_data_dir)[1].replace('\\', '/')
    # download_urls[p.ha_per_cell_paths[p.coarse_resolution_arcseconds]] = p.ha_per_cell_paths[p.coarse_resolution_arcseconds].split(p.base_data_dir)[1].replace('\\', '/')
    p.countries_iso3_path
    p.countries_iso3_path = p.get_path(p.countries_iso3_path)
    p.ha_per_cell_coarse_path = p.get_path(p.ha_per_cell_ref_paths[p.coarse_resolution_arcseconds])
    p.ha_per_cell_fine_path = p.get_path(p.ha_per_cell_ref_paths[p.fine_resolution_arcseconds])
    
    # TODO This references something in initialize_project in seals. But, we only download the ones that are needed given the resolutions
    # p.ha_per_cell_paths[p.fine_resolution_arcseconds] = p.get_path(p.ha_per_cell_paths[p.fine_resolution_arcseconds])
 
    # for download_path, download_url in download_urls.items():
    #     if not hb.path_exists(download_path): # Check one last time to ensure that it wasn't added twice.
    #         cloud_utils.download_google_cloud_blob(p.input_bucket_name, download_url, p.data_credentials_path, download_path)
    
    
    # Note that here there is a little bit more logic outside the run_this block. But it only references the things that it assumes had been made sometime else.
    if isinstance(p.aoi, str):
        if p.aoi == 'global':
            p.aoi_path = p.countries_iso3_path
            p.aoi_label = 'global'
            p.bb_exact = hb.global_bounding_box
            p.bb = p.bb_exact

            ### TODO Start here. aoi_ha... is the clipped, but still need to have the global one seperate so it can download it.
            p.aoi_ha_per_cell_coarse_path = p.get_path(p.ha_per_cell_ref_paths[p.coarse_resolution_arcseconds])
            p.aoi_ha_per_cell_fine_path = p.get_path(p.ha_per_cell_ref_paths[p.fine_resolution_arcseconds])
        
        elif isinstance(p.aoi, str):
            if len(p.aoi) == 3: # Then it might be an ISO3 code. For now, assume so.
                p.aoi_path = os.path.join(p.cur_dir, 'aoi_' + str(p.aoi) + '.gpkg')
                p.aoi_label = p.aoi
            else: # Then it's a path to a shapefile.
                p.aoi_path = p.aoi
                p.aoi_label = os.path.splitext(os.path.basename(p.aoi))[0]

            for current_aoi_path in hb.list_filtered_paths_nonrecursively(p.cur_dir, include_strings='aoi'):
                if current_aoi_path != p.aoi_path:
                    raise NameError('There is more than one AOI in the current directory. This means you are trying to run a project in a new area of interst in a project that was already run in a different area of interest. This is not allowed! You probably want to create a new project directory and set the p = hb.ProjectFlow(...) line to point to the new directory.')
                

            if not hb.path_exists(p.aoi_path):
                hb.extract_features_in_shapefile_by_attribute(p.countries_iso3_path, p.aoi_path, 'iso3', p.aoi.upper())
            p.bb_exact = hb.spatial_projection.get_bounding_box(p.aoi_path)
            p.bb = hb.pyramids.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.processing_resolution_arcseconds)
            p.aoi_ha_per_cell_fine_path = os.path.join(p.cur_dir, 'pyramids', 'aoi_ha_per_cell_fine.tif')
            p.aoi_ha_per_cell_coarse_path = os.path.join(p.cur_dir, 'pyramids', 'aoi_ha_per_cell_coarse.tif')
        else:
            p.bb_exact = hb.spatial_projection.get_bounding_box(p.aoi_path)
            p.bb = hb.pyramids.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.processing_resolution_arcseconds)
            p.aoi_ha_per_cell_fine_path = os.path.join(p.cur_dir, 'pyramids', 'aoi_ha_per_cell_fine.tif')
            p.aoi_ha_per_cell_coarse_path = os.path.join(p.cur_dir, 'pyramids', 'aoi_ha_per_cell_coarse.tif')
    else:
        raise NameError('Unable to interpret p.aoi.')


    if p.run_this:

        if isinstance(p.aoi, str):

            if p.aoi == 'global':
                pass

            elif isinstance(p.aoi, str):
                if len(p.aoi) == 3:                     
                    pass
                else:
                    pass

                if not hb.path_exists(p.aoi_ha_per_cell_fine_path):
                    hb.create_directories(p.aoi_ha_per_cell_fine_path)
                    hb.clip_raster_by_bb(p.ha_per_cell_paths[p.fine_resolution_arcseconds], p.bb, p.aoi_ha_per_cell_fine_path)

                if not hb.path_exists(p.aoi_ha_per_cell_coarse_path):
                    hb.clip_raster_by_bb(p.ha_per_cell_paths[p.coarse_resolution_arcseconds], p.bb, p.aoi_ha_per_cell_coarse_path)

        else:
            raise NameError('Unable to interpret p.aoi.')

def build_task_tree_by_name(p, task_tree_name):
    full_task_tree_name = 'build_' + task_tree_name + '_task_tree'
    target_function = globals()[full_task_tree_name]
    print('Launching SEALS. Building task tree: ' + task_tree_name)

    target_function(p)

# def build_simple_allocation_run_task_tree(p):
#     # This is the default task tree that is run on a new seals installation.

#     ##### Preprocessing #####

#     # Define the project AOI
#     p.project_aoi_task = p.add_task(project_aoi)

#     # Download the base data (note that it saves to p.base_data_dir and thus doesn't create its own dir)
#     p.base_data_task = p.add_task(seals_process_coarse_timeseries.download_base_data, creates_dir=False)


#     ##### FINE PROCESSED INPUTS #####

#     # Make folder for all generated data.
#     p.fine_processed_inputs_task = p.add_task(seals_generate_base_data.fine_processed_inputs)

#     # Generate some simple gaussian kernels for later convolutions
#     p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.fine_processed_inputs_task, creates_dir=False)

#     # Clip the fine LULC to the project AOI
#     p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.fine_processed_inputs_task, creates_dir=False)

#     # Simplify the LULC
#     p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.fine_processed_inputs_task, creates_dir=False)

#     # Convert the simplified LULC into 1 binary presence map for each simplified LUC
#     p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.fine_processed_inputs_task, creates_dir=False)

#     # Convolve the lulc_binaries # TODOO Precache this for performance
#     p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.fine_processed_inputs_task, creates_dir=False)

#     # # Write a LOCAL version of the regression starting values based on the new data generated.
#     # p.local_data_regression_starting_values_task = p.add_task(seals_generate_base_data.local_data_regressors_starting_values)


#     ##### COARSE CHANGE #####
#     # Make folder for all generated data.
#     p.coarse_change_task = p.add_task(seals_process_coarse_timeseries.coarse_change, skip_existing=0)

#     # Extract coarse change from source
#     p.extraction_task = p.add_task(seals_process_coarse_timeseries.coarse_extraction, parent=p.coarse_change_task, run=1, skip_existing=0)

#     # Reclassify coarse source to simplified scheme 
#     p.coarse_simplified_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_proportion, parent=p.coarse_change_task, skip_existing=0)

#     # Calculate LUH2_simplified difference from base year
#     p.coarse_simplified_ha_difference_from_base_year_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha_difference_from_base_year, parent=p.coarse_change_task, skip_existing=0)


#     ##### ALLOCATION #####

#     # Iterate through different scenarios to be allocated.
#     p.allocations_task = p.add_iterator(seals_main.allocations, run_in_parallel=0, skip_existing=0)

#     # Define the zones overwhich parallel allocation should be run
#     p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocations_task, run_in_parallel=p.run_in_parallel, skip_existing=0)

#     # Actually do the allocation
#     p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=1)



#     ##### STITCH ZONES #####

#     # Stitch the simplified LULC tiles back together
#     p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios)

#     # # Reclassify the stitched simplified to original classification.
#     # p.stitched_lulc_esa_scenarios_task =  p.add_task(seals_main.stitched_lulc_esa_scenarios)

#     # p.post_run_visualization_task = p.add_task(seals_visualization_tasks.post_run_visualization)


def build_allocation_run_task_tree(p):
    # This is the default task tree that is run on a new seals installation.

    ##### Preprocessing #####

    # Define the project AOI
    p.project_aoi_task = p.add_task(project_aoi)

    # Download the base data (note that it saves to p.base_data_dir and thus doesn't create its own dir)
    p.base_data_task = p.add_task(seals_process_coarse_timeseries.download_base_data, creates_dir=False)


    ##### FINE PROCESSED INPUTS #####

    # Make folder for all generated data.
    p.fine_processed_inputs_task = p.add_task(seals_generate_base_data.fine_processed_inputs)

    # Generate some simple gaussian kernels for later convolutions
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Clip the fine LULC to the project AOI
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Simplify the LULC
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Convert the simplified LULC into 1 binary presence map for each simplified LUC
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Convolve the lulc_binaries # TODOO Precache this for performance
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.fine_processed_inputs_task, creates_dir=False)

    # # Write a LOCAL version of the regression starting values based on the new data generated.
    # p.local_data_regression_starting_values_task = p.add_task(seals_generate_base_data.local_data_regressors_starting_values)


    ##### COARSE CHANGE #####
    # Make folder for all generated data.
    p.coarse_change_task = p.add_task(seals_process_coarse_timeseries.coarse_change, skip_existing=0)

    # Extract coarse change from source
    p.extraction_task = p.add_task(seals_process_coarse_timeseries.coarse_extraction, parent=p.coarse_change_task, run=1, skip_existing=0)

    # Reclassify coarse source to simplified scheme 
    p.coarse_simplified_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_proportion, parent=p.coarse_change_task, skip_existing=0)

    # Calculate LUH2_simplified difference from base year
    p.coarse_simplified_ha_difference_from_previous_year_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha_difference_from_previous_year, parent=p.coarse_change_task, skip_existing=0)


    ##### ALLOCATION #####

    # Iterate through different scenarios to be allocated.
    p.allocations_task = p.add_iterator(seals_main.allocations, run_in_parallel=0, skip_existing=0)

    # Define the zones overwhich parallel allocation should be run
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocations_task, run_in_parallel=p.run_in_parallel, skip_existing=0)

    # Actually do the allocation
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=1)



    ##### STITCH ZONES #####

    # Stitch the simplified LULC tiles back together
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios)

    # # Reclassify the stitched simplified to original classification.
    # p.stitched_lulc_esa_scenarios_task =  p.add_task(seals_main.stitched_lulc_esa_scenarios)

    # p.post_run_visualization_task = p.add_task(seals_visualization_tasks.post_run_visualization)


def build_visualization_task_tree(p):
    # This is the default task tree that is run on a new seals installation.

    ##### Preprocessing #####

    # Define the project AOI
    p.project_aoi_task = p.add_task(project_aoi)

    # Download the base data (note that it saves to p.base_data_dir and thus doesn't create its own dir)
    p.base_data_task = p.add_task(seals_process_coarse_timeseries.download_base_data, creates_dir=False, run=0)


    ##### FINE PROCESSED INPUTS #####

    # Make folder for all generated data.
    p.fine_processed_inputs_task = p.add_task(seals_generate_base_data.fine_processed_inputs, run=0)

    # Generate some simple gaussian kernels for later convolutions
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.fine_processed_inputs_task, creates_dir=False, run=0)

    # Clip the fine LULC to the project AOI
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.fine_processed_inputs_task, creates_dir=False, run=0)

    # Simplify the LULC
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.fine_processed_inputs_task, creates_dir=False, run=0)

    # Convert the simplified LULC into 1 binary presence map for each simplified LUC
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.fine_processed_inputs_task, creates_dir=False, run=0)

    # Convolve the lulc_binaries # TODOO Precache this for performance
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.fine_processed_inputs_task, creates_dir=False, run=0)

    # # Write a LOCAL version of the regression starting values based on the new data generated.
    # p.local_data_regression_starting_values_task = p.add_task(seals_generate_base_data.local_data_regressors_starting_values)


    ##### COARSE CHANGE #####
    # Make folder for all generated data.
    p.coarse_change_task = p.add_task(seals_process_coarse_timeseries.coarse_change, skip_existing=0, run=1)

    # Extract coarse change from source
    p.extraction_task = p.add_task(seals_process_coarse_timeseries.coarse_extraction, parent=p.coarse_change_task, skip_existing=0, run=0)

    # Reclassify coarse source to simplified scheme 
    p.coarse_simplified_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_proportion, parent=p.coarse_change_task, skip_existing=0, run=0)

    # Calculate LUH2_simplified difference from base year
    p.coarse_simplified_ha_difference_from_previous_year_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha_difference_from_previous_year, parent=p.coarse_change_task, skip_existing=0, run=0)


    ##### ALLOCATION #####

    # Iterate through different scenarios to be allocated.
    p.allocations_task = p.add_iterator(seals_main.allocations, run_in_parallel=0, skip_existing=0, run=0)

    # Define the zones overwhich parallel allocation should be run
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocations_task, run_in_parallel=p.run_in_parallel, skip_existing=0, run=0)

    # Actually do the allocation
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=1, run=0)



    ##### STITCH ZONES #####

    # Stitch the simplified LULC tiles back together
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios, run=0)

    # # Reclassify the stitched simplified to original classification.
    # p.stitched_lulc_esa_scenarios_task =  p.add_task(seals_main.stitched_lulc_esa_scenarios)

    # p.post_run_visualization_task = p.add_task(seals_visualization_tasks.post_run_visualization)


    ##### VIZUALIZE EXISTING DATA #####
    # Make folder for all visualizations.
    p.visualization_task = p.add_task(seals_visualization_tasks.visualization, skip_existing=0, run=1)

    # For each class, plot the coarse and fine data
    p.coarse_change_with_class_change_task = p.add_task(seals_visualization_tasks.coarse_change_with_class_change, parent=p.visualization_task, run=1)

    p.coarse_fine_with_report_task = p.add_task(seals_visualization_tasks.coarse_fine_with_report, parent=p.visualization_task, run=1)

    # For each class, plot the coarse and fine data
    p.create_full_change_matrices_task = p.add_task(seals_main.full_change_matrices, parent=p.visualization_task, run=1)
    
    # For each class, plot the coarse and fine data
    p.target_zones_matrices_task = p.add_task(seals_main.target_zones_matrices, parent=p.visualization_task, run=1)
    
    # For each class, plot the coarse and fine data
    p.plot_full_change_matrices_pngs_task = p.add_task(seals_visualization_tasks.full_change_matrices_pngs, parent=p.visualization_task, run=1)
    
    # For each class, plot the coarse and fine data
    p.plot_target_zones_matrices_pngs_task = p.add_task(seals_visualization_tasks.target_zones_matrices_pngs, parent=p.visualization_task, run=1)

    # Simple plot of the PNGs.
    p.lulc_pngs_task = p.add_task(seals_visualization_tasks.lulc_pngs, parent=p.visualization_task, run=1)

def build_allocation_and_visualization_run_task_tree(p):
    
    # Define the project AOI
    p.project_aoi_task = p.add_task(project_aoi)

    # Download the base data (note that it saves to p.base_data_dir and thus doesn't create its own dir)
    p.base_data_task = p.add_task(seals_process_coarse_timeseries.download_base_data, creates_dir=False)


    ##### FINE PROCESSED INPUTS #####

    # Make folder for all generated data.
    p.fine_processed_inputs_task = p.add_task(seals_generate_base_data.fine_processed_inputs)

    # Generate some simple gaussian kernels for later convolutions
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Clip the fine LULC to the project AOI
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Simplify the LULC
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Convert the simplified LULC into 1 binary presence map for each simplified LUC
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Convolve the lulc_binaries # TODOO Precache this for performance
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.fine_processed_inputs_task, creates_dir=False)

    # # Write a LOCAL version of the regression starting values based on the new data generated.
    # p.local_data_regression_starting_values_task = p.add_task(seals_generate_base_data.local_data_regressors_starting_values)


    ##### COARSE CHANGE #####
    # Make folder for all generated data.
    p.coarse_change_task = p.add_task(seals_process_coarse_timeseries.coarse_change, skip_existing=0)

    # Extract coarse change from source
    p.extraction_task = p.add_task(seals_process_coarse_timeseries.coarse_extraction, parent=p.coarse_change_task, run=1, skip_existing=0)

    # Reclassify coarse source to simplified scheme 
    p.coarse_simplified_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_proportion, parent=p.coarse_change_task, skip_existing=0)

    # Calculate LUH2_simplified difference from base year
    p.coarse_simplified_ha_difference_from_previous_year_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha_difference_from_previous_year, parent=p.coarse_change_task, skip_existing=0)


    ##### ALLOCATION #####

    # Iterate through different scenarios to be allocated.
    p.allocations_task = p.add_iterator(seals_main.allocations, run_in_parallel=0, skip_existing=0)

    # Define the zones overwhich parallel allocation should be run
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocations_task, run_in_parallel=p.run_in_parallel, skip_existing=0)

    # Actually do the allocation
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=1)



    ##### STITCH ZONES #####

    # Stitch the simplified LULC tiles back together
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios)

    # # Reclassify the stitched simplified to original classification.
    # p.stitched_lulc_esa_scenarios_task =  p.add_task(seals_main.stitched_lulc_esa_scenarios)

    # p.post_run_visualization_task = p.add_task(seals_visualization_tasks.post_run_visualization)

    ##### VIZUALIZE EXISTING DATA #####
    # Make folder for all visualizations.
    p.visualization_task = p.add_task(seals_visualization_tasks.visualization, skip_existing=0, run=1)

    # For each class, plot the coarse and fine data
    p.coarse_change_with_class_change_task = p.add_task(seals_visualization_tasks.coarse_change_with_class_change, parent=p.visualization_task, run=1)

    p.coarse_fine_with_report_task = p.add_task(seals_visualization_tasks.coarse_fine_with_report, parent=p.visualization_task, run=1)

    # For each class, plot the coarse and fine data
    p.create_full_change_matrices_task = p.add_task(seals_main.full_change_matrices, parent=p.visualization_task, run=1)
    
    # For each class, plot the coarse and fine data
    p.target_zones_matrices_task = p.add_task(seals_main.target_zones_matrices, parent=p.visualization_task, run=1)
    
    # For each class, plot the coarse and fine data
    p.plot_full_change_matrices_pngs_task = p.add_task(seals_visualization_tasks.full_change_matrices_pngs, parent=p.visualization_task, run=1)
    
    # For each class, plot the coarse and fine data
    # p.plot_target_zones_matrices_pngs_task = p.add_task(seals_visualization_tasks.target_zones_matrices_pngs, parent=p.visualization_task, run=1)

    # Simple plot of the PNGs.
    p.lulc_pngs_task = p.add_task(seals_visualization_tasks.lulc_pngs, parent=p.visualization_task, run=1)


def build_global_allocation_and_visualization_run_task_tree(p):
    # This is the default task tree that is run on a new seals installation.

    ##### Preprocessing #####

    # Define the project AOI
    p.project_aoi_task = p.add_task(project_aoi)

    # Download the base data (note that it saves to p.base_data_dir and thus doesn't create its own dir)
    p.base_data_task = p.add_task(seals_process_coarse_timeseries.download_base_data, creates_dir=False)


    ##### FINE PROCESSED INPUTS #####

    # Make folder for all generated data.
    p.fine_processed_inputs_task = p.add_task(seals_generate_base_data.fine_processed_inputs)

    # Generate some simple gaussian kernels for later convolutions
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Clip the fine LULC to the project AOI
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Simplify the LULC
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Convert the simplified LULC into 1 binary presence map for each simplified LUC
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Convolve the lulc_binaries # TODOO Precache this for performance
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.fine_processed_inputs_task, creates_dir=False)

    # # Write a LOCAL version of the regression starting values based on the new data generated.
    # p.local_data_regression_starting_values_task = p.add_task(seals_generate_base_data.local_data_regressors_starting_values)


    ##### COARSE CHANGE #####
    # Make folder for all generated data.
    p.coarse_change_task = p.add_task(seals_process_coarse_timeseries.coarse_change, skip_existing=0)

    # Extract coarse change from source
    p.extraction_task = p.add_task(seals_process_coarse_timeseries.coarse_extraction, parent=p.coarse_change_task, run=1, skip_existing=0)

    # Reclassify coarse source to simplified scheme 
    p.coarse_simplified_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_proportion, parent=p.coarse_change_task, skip_existing=0)

    # Calculate LUH2_simplified difference from base year
    p.coarse_simplified_ha_difference_from_previous_year_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha_difference_from_previous_year, parent=p.coarse_change_task, skip_existing=0)


    ##### ALLOCATION #####

    # Iterate through different scenarios to be allocated.
    p.allocations_task = p.add_iterator(seals_main.allocations, run_in_parallel=0, skip_existing=0)

    # Define the zones overwhich parallel allocation should be run
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocations_task, run_in_parallel=p.run_in_parallel, skip_existing=0)

    # Actually do the allocation
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=1)



    ##### STITCH ZONES #####

    # Stitch the simplified LULC tiles back together
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios)

    # # Reclassify the stitched simplified to original classification.
    # p.stitched_lulc_esa_scenarios_task =  p.add_task(seals_main.stitched_lulc_esa_scenarios)

    # p.post_run_visualization_task = p.add_task(seals_visualization_tasks.post_run_visualization)

    ##### VIZUALIZE EXISTING DATA #####
    # Make folder for all visualizations.
    p.visualization_task = p.add_task(seals_visualization_tasks.visualization, skip_existing=0, run=1)

    # For each class, plot the coarse and fine data
    p.coarse_change_with_class_change_task = p.add_task(seals_visualization_tasks.coarse_change_with_class_change, parent=p.visualization_task, run=0)

    # For each class, plot the coarse and fine data
    p.create_full_change_matrices_task = p.add_task(seals_main.full_change_matrices, parent=p.visualization_task, run=0)
    
    # For each class, plot the coarse and fine data
    p.target_zones_matrices_task = p.add_task(seals_main.target_zones_matrices, parent=p.visualization_task, run=0)
    
    # For each class, plot the coarse and fine data
    p.plot_full_change_matrices_pngs_task = p.add_task(seals_visualization_tasks.full_change_matrices_pngs, parent=p.visualization_task, run=0)
    
    # For each class, plot the coarse and fine data
    p.plot_target_zones_matrices_pngs_task = p.add_task(seals_visualization_tasks.target_zones_matrices_pngs, parent=p.visualization_task, run=0)

    # Simple plot of the PNGs.
    p.lulc_pngs_task = p.add_task(seals_visualization_tasks.lulc_pngs, parent=p.visualization_task, run=1)
 


def build_allocation_and_quick_visualization_run_task_tree(p):

    # Define the project AOI
    p.project_aoi_task = p.add_task(project_aoi)

    # # Download the base data (note that it saves to p.base_data_dir and thus doesn't create its own dir)
    # p.base_data_task = p.add_task(seals_process_coarse_timeseries.download_base_data, creates_dir=False)


    ##### FINE PROCESSED INPUTS #####

    # Make folder for all generated data.
    p.fine_processed_inputs_task = p.add_task(seals_generate_base_data.fine_processed_inputs)

    # Generate some simple gaussian kernels for later convolutions
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Clip the fine LULC to the project AOI
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Simplify the LULC
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Convert the simplified LULC into 1 binary presence map for each simplified LUC
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.fine_processed_inputs_task, creates_dir=False)

    # Convolve the lulc_binaries # TODOO Precache this for performance
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.fine_processed_inputs_task, creates_dir=False)

    # # Write a LOCAL version of the regression starting values based on the new data generated.
    # p.local_data_regression_starting_values_task = p.add_task(seals_generate_base_data.local_data_regressors_starting_values)


    ##### COARSE CHANGE #####
    # Make folder for all generated data.
    p.coarse_change_task = p.add_task(seals_process_coarse_timeseries.coarse_change, skip_existing=0)

    # Extract coarse change from source
    p.extraction_task = p.add_task(seals_process_coarse_timeseries.coarse_extraction, parent=p.coarse_change_task, run=1, skip_existing=0)

    # Reclassify coarse source to simplified scheme 
    p.coarse_simplified_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_proportion, parent=p.coarse_change_task, skip_existing=0)

    # Calculate LUH2_simplified difference from base year
    p.coarse_simplified_ha_difference_from_previous_year_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha_difference_from_previous_year, parent=p.coarse_change_task, skip_existing=0)


    ##### ALLOCATION #####

    # Iterate through different scenarios to be allocated.
    p.allocations_task = p.add_iterator(seals_main.allocations)

    # Define the zones overwhich parallel allocation should be run
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocations_task)

    # Actually do the allocation
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=1)



    ##### STITCH ZONES #####

    # Stitch the simplified LULC tiles back together
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios)

    # # Reclassify the stitched simplified to original classification.
    # p.stitched_lulc_esa_scenarios_task =  p.add_task(seals_main.stitched_lulc_esa_scenarios)

    # p.post_run_visualization_task = p.add_task(seals_visualization_tasks.post_run_visualization)

    ##### VIZUALIZE EXISTING DATA #####
    # Make folder for all visualizations.
    p.visualization_task = p.add_task(seals_visualization_tasks.visualization, skip_existing=0, run=1)

    # For each class, plot the coarse and fine data
    p.coarse_change_with_class_change_task = p.add_task(seals_visualization_tasks.coarse_change_with_class_change, parent=p.visualization_task, run=0)

    # For each class, plot the coarse and fine data
    p.plot_full_change_matrices_task = p.add_task(seals_main.full_change_matrices, parent=p.visualization_task, run=0)
    
    # For each class, plot the coarse and fine data
    p.plot_full_change_matrices_task = p.add_task(seals_visualization_tasks.full_change_matrices_pngs, parent=p.visualization_task, run=0)

    # Simple plot of the PNGs.
    p.lulc_pngs_task = p.add_task(seals_visualization_tasks.lulc_pngs, parent=p.visualization_task, run=1)

    p.coarse_fine_with_report_task = p.add_task(seals_visualization_tasks.coarse_fine_with_report, parent=p.visualization_task, run=0)

def build_complete_run_task_tree(p):
    ## OUT OF DATE, but should be replicated
    p.project_aoi_task = p.add_task(project_aoi)
    p.base_data_task = p.add_task(seals_process_coarse_timeseries.download_base_data, creates_dir=False,                                                         run=1, skip_existing=0)
    p.regressors_starting_values_task = p.add_task(seals_generate_base_data.regressors_starting_values,                                             run=1, skip_existing=0)
    p.generated_data_task = p.add_task(seals_generate_base_data.generated_data,                                                                     run=1, skip_existing=0)
    p.aoi_vector_task = p.add_task(seals_generate_base_data.aoi_vector, parent=p.generated_data_task, creates_dir=False,                            run=1, skip_existing=0)
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.generated_data_task, creates_dir=False,                              run=1, skip_existing=0)
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.generated_data_task, creates_dir=False,        run=1, skip_existing=0)
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.generated_data_task, creates_dir=False,                      run=1, skip_existing=0)
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.generated_data_task, creates_dir=False,              run=1, skip_existing=0)
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.generated_data_task, creates_dir=False,              run=1, skip_existing=0)
    p.local_data_regression_starting_values_task = p.add_task(seals_generate_base_data.local_data_regressors_starting_values,                       run=1, skip_existing=0)
    p.luh2_extraction_task = p.add_task(seals_process_coarse_timeseries.luh2_extraction,                                                                         run=1, skip_existing=0)
    p.luh2_difference_from_base_year_task = p.add_task(seals_process_coarse_timeseries.luh2_difference_from_base_year,                                           run=1, skip_existing=0)
    p.luh2_as_simplified_proportion_task = p.add_task(seals_process_coarse_timeseries.luh2_as_simplified_proportion,                                                     run=1, skip_existing=0)
    p.simplified_difference_from_base_yea_task = p.add_task(seals_process_coarse_timeseries.simplified_difference_from_base_year,                                        run=1, skip_existing=0)
    p.calibration_generated_inputs_task = p.add_task(seals_main.calibration_generated_inputs,                                                       run=1, skip_existing=0)
    p.calibration_task = p.add_iterator(seals_main.calibration, run_in_parallel=1,                                                                  run=1, skip_existing=0)
    p.calibration_prepare_lulc_task = p.add_task(seals_main.calibration_prepare_lulc, parent=p.calibration_task,                                    run=1, skip_existing=0)
    p.calibration_change_matrix_task = p.add_task(seals_main.calibration_change_matrix, parent=p.calibration_task,                                  run=1, skip_existing=0)
    p.calibration_zones_task = p.add_task(seals_main.calibration_zones, parent=p.calibration_task,                                                  run=1, skip_existing=0, logging_level=20)
    p.calibration_plots_task = p.add_task(seals_main.calibration_plots, parent=p.calibration_task,                                                  run=1, skip_existing=0)
    p.combined_trained_coefficients_task = p.add_task(seals_main.combined_trained_coefficients,                                                     run=1, skip_existing=0)
    p.allocations_task = p.add_iterator(seals_main.allocations, run_in_parallel=0,                                                                  run=1, skip_existing=0)
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocations_task, run_in_parallel=1,                             run=1, skip_existing=0)
    p.allocation_change_matrix_task = p.add_task(seals_main.allocation_change_matrix, parent=p.allocation_zones_task,                               run=1, skip_existing=0)
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task,                                                           run=1, skip_existing=0)
    p.allocation_exclusive_task = p.add_task(seals_main.allocation_exclusive, parent=p.allocation_zones_task,                                       run=0, skip_existing=0)
    p.allocation_from_change_matrix_task = p.add_task(seals_main.allocation_from_change_matrix, parent=p.allocation_zones_task,                     run=0, skip_existing=0)
    p.change_pngs_task = p.add_task(seals_main.change_pngs, parent=p.allocation_zones_task,                                                         run=0, skip_existing=0)
    p.change_exclusive_pngs_task = p.add_task(seals_main.change_exclusive_pngs, parent=p.allocation_zones_task,                                     run=0, skip_existing=0)
    p.change_from_change_matrix_pngs_task = p.add_task(seals_main.change_from_change_matrix_pngs, parent=p.allocation_zones_task,                   run=0, skip_existing=0)
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios,                                           run=1, skip_existing=0)
    p.stitched_lulc_esa_scenarios_task =  p.add_task(seals_main.stitched_lulc_esa_scenarios,                                                        run=1, skip_existing=0)

    return p


def build_global_carbon_by_scenario_task_tree(p):
    # Example of a post-run task tree
    p.project_aoi_task = p.add_task(project_aoi)
    # NOTE. Because p.stitched_lulc_esa_scenarios_dir is referenced in carbon_comparison, need to call but not run the corresponding task.
    p.simplified_difference_from_base_year_task =  p.add_task(seals_process_coarse_timeseries.simplified_difference_from_base_year,                 run=0, skip_existing=0)
    p.stitched_lulc_esa_scenarios_task =  p.add_task(seals_main.stitched_lulc_esa_scenarios,                                                        run=0, skip_existing=0)
    p.stitched_lulc_simplified_scenarios_task =  p.add_task(seals_main.stitched_lulc_simplified_scenarios,                                          run=0, skip_existing=0)
    p.global_carbon_by_scenario_task = p.add_task(seals_main.global_carbon_by_scenario,                                                             run=1, skip_existing=0)
    p.carbon_comparison_task = p.add_task(seals_main.carbon_comparison,                                                                             run=1, skip_existing=0)

