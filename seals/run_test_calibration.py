import os
import hazelbean as hb

main = ''
if __name__ == '__main__':

    # This should be the only path a run file needs to set. Everything is relative to this (or the source code dir)
    p = hb.ProjectFlow(r'../../projects/test_seals_calibration')

    # import gtap_invest_main
    import seals_main
    import seals_process_coarse_timeseries
    from gtap_invest.visualization import visualization

    # initialize and set all basic variables. Sadly this is still needed even for a SEALS run until it's extracted.
    gtap_invest_main.initialize_paths(p)

    # Test mode is a much smaller set of scenarios and spatial tiles
    p.test_mode = True

    # In order to apply this code to the magpie model, I set this option to either
    # use the GTAP-shited LUH data (as was done in the WB feedback model)
    # or to instead use the outputs of some other extraction functions with
    # no shifting logic. This could be scaled to different interfaces
    # when models have different input points.
    p.is_gtap1_run = False
    p.is_magpie_run = False
    p.is_calibration_run = True
    p.is_standard_seals_run = False

    p.adjust_baseline_to_match_magpie_2015 = False

    # Run configuration options
    p.num_workers = 14 # None sets it to max available.
    p.cython_reporting_level = 3
    p.output_writing_level = 5 # >=2 writes chunk-baseline lulc
    p.build_overviews_and_stats = 0 # For later fast-viewing, this can be enabled to write ovr files and geotiff stats files.
    p.write_projected_coarse_change_chunks = 1 # in the SEALS allocation, for troubleshooting, it can be useful to see what was the coarse allocation input.

    # TASK to figure out: draw a task tree consistent both with magpie needing esa-magpie 2015 calibration AND
    # gtap needing an extra base-year of 2021 AND gtap being a 3-layer allocation with SSP2 (which should be made interchangeable with Magpie)

    # Scenarios configuration: We projected 2014 to 2021 so that we could apply the policy in 2021 (which needed knowing the state of the economy in that point).
    p.training_start_year = 2000

    p.base_year = 2015
    if p.is_gtap1_run:
        p.policy_base_year = 2021
        p.base_years = [p.base_year, p.policy_base_year]
    elif p.is_magpie_run:
        p.base_years = [p.base_year]
    elif p.is_calibration_run:
        p.base_years = [p.base_year]

    # TODOO For magpie integration: make this an override.
    p.baseline_labels = ['baseline']
    p.baseline_coarse_state_paths = {}
    p.baseline_coarse_state_paths['baseline'] = {}
    p.baseline_coarse_state_paths['baseline'][p.base_year] = os.path.join(p.input_dir, "SSP2_BiodivPol_LPJmL5_2021-05-21_15.08.06", "cell.land_0.5_share_to_seals_SSP2_BiodivPol_LPJmL5.nc")

    # Basic iteration over years is possible/partly implemented. For WB purposes we just had 2030.
    p.scenario_years = [2050]

    # Scenarios are defined by a combination of meso-level focusing layer that defines coarse LUC and Climate with the policy scenarios (or just scenarios) below.
    # Partially implemented logic of easily rerunning with different rcp/ssps.
    p.luh_scenario_labels = ['rcp45_ssp2', 'rcp85_ssp5']

    # These are the POLICY scenarios. The model will iterate over these as well.
    p.gtap_combined_policy_scenario_labels = ['BAU', 'BAU_rigid', 'PESGC', 'SR_Land', 'PESLC', 'SR_RnD_20p', 'SR_Land_PESGC', 'SR_PESLC',  'SR_RnD_20p_PESGC', 'SR_RnD_PESLC', 'SR_RnD_20p_PESGC_30']
    p.gtap_just_bau_label = ['BAU']
    p.gtap_bau_and_30_labels = ['BAU', 'SR_RnD_20p_PESGC_30']
    p.luh_labels = ['no_policy']

    p.magpie_policy_scenario_labels = [
        'SSP2_BiodivPol_LPJmL5',
        'SSP2_BiodivPol+ClimPol_LPJmL5',
        'SSP2_BiodivPol+ClimPol+NCPpol_LPJmL5',
        'SSP2_ClimPol_LPJmL5',
        'SSP2_NPI_base_LPJmL5',
    ]

    p.magpie_test_policy_scenario_labels = [
        # 'SSP2_BiodivPol_LPJmL5',
        # 'SSP2_BiodivPol_ClimPol_LPJmL5',
        'SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5',
        # 'SSP2_ClimPol_LPJmL5',
        # 'SSP2_NPI_base_LPJmL5',
    ]

    # Scenarios are defined by a combination of meso-level focusing layer that defines coarse LUC and Climate with the policy scenarios (or just scenarios) below.
    p.magpie_scenario_coarse_state_paths = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol_LPJmL5_2021-05-21_15.08.06", "cell.land_0.5_share_to_seals_SSP2_BiodivPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol+ClimPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol+ClimPol_LPJmL5_2021-05-21_15.09.32", "cell.land_0.5_share_to_seals_SSP2_BiodivPol+ClimPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol+ClimPol+NCPpol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol+ClimPol+NCPpol_LPJmL5_2021-05-21_15.10.54", "cell.land_0.5_share_to_seals_SSP2_BiodivPol+ClimPol+NCPpol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_ClimPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_ClimPol_LPJmL5_2021-05-21_15.12.19", "cell.land_0.5_share_to_seals_SSP2_ClimPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_NPI_base_LPJmL5'] = os.path.join(p.input_dir, "SSP2_NPI_base_LPJmL5_2021-05-21_15.05.56", "cell.land_0.5_share_to_seals_SSP2_NPI_base_LPJmL5.nc")

    p.gtap_scenario_coarse_state_paths = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2030] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2030]['BAU'] = hb.luh_scenario_states_paths['rcp45_ssp2']
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2050]['BAU'] = hb.luh_scenario_states_paths['rcp45_ssp2']

    p.luh_scenario_coarse_state_paths = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2030] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2030] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2050] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2030]['no_policy'] = hb.luh_scenario_states_paths['rcp45_ssp2']
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2050]['no_policy'] = hb.luh_scenario_states_paths['rcp45_ssp2']
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2030]['no_policy'] = hb.luh_scenario_states_paths['rcp85_ssp5']
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2050]['no_policy'] = hb.luh_scenario_states_paths['rcp85_ssp5']

    if p.is_magpie_run:
        p.scenario_coarse_state_paths = p.magpie_scenario_coarse_state_paths
    elif p.is_gtap1_run:
        p.scenario_coarse_state_paths = p.gtap_scenario_coarse_state_paths
    elif p.is_calibration_run:
        p.scenario_coarse_state_paths = p.luh_scenario_coarse_state_paths

    if p.test_mode:
        if p.is_magpie_run:
            p.policy_scenario_labels = p.magpie_test_policy_scenario_labels
        elif p.is_gtap1_run:
            p.policy_scenario_labels = p.gtap_bau_and_30_labels
        elif p.is_calibration_run:
            p.policy_scenario_labels = p.luh_labels
    else:
        if p.is_magpie_run:
            p.policy_scenario_labels = p.magpie_policy_scenario_labels
        elif p.is_gtap1_run:
            p.policy_scenario_labels = p.gtap_combined_policy_scenario_labels
        elif p.is_calibration_run:
            p.policy_scenario_labels = p.luh_labels

    if p.is_gtap1_run:
        # HACK, because I don't yet auto-generate the cmf files and other GTAP modelled inputs, and instead just take the files out of the zipfile Uris
        # provides, I still have to follow his naming scheme. This list comprehension converts a policy_scenario_label into a gtap1 or gtap2 label.
        p.gtap1_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_noES' for i in p.policy_scenario_labels]
        p.gtap2_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_allES' for i in p.policy_scenario_labels]



    # This is a zipfile I received from URIS that has all the packaged GTAP files ready to run. Extract these to a project dir.
    p.gtap_aez_invest_release_string = '04_20_2021_GTAP_AEZ_INVEST'
    p.gtap_aez_invest_zipfile_path = os.path.join(p.model_base_data_dir, 'gtap_aez_invest_releases', p.gtap_aez_invest_release_string + '.zip')
    p.gtap_aez_invest_code_dir = os.path.join(p.script_dir, 'gtap_aez', p.gtap_aez_invest_release_string)

    # Associate each luh, year, and policy scenario with a set of seals input parameters. This can be used if, for instance, the policy you
    # are analyzing involves focusing land-use change into certain types of gridcells.


    p.gtap_pretrained_coefficients_path_dict = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['BAU'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['BAU'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['PESGC'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['RnD'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_Land'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['PESLC'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_Land_PESGC'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_PESLC'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESGC'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESLC'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p_PESGC'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESGC_30'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints_and_protected_areas.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p_PESGC_30'] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'input', 'gtap_trained_coefficients_combined_with_constraints_and_protected_areas.xlsx')

    p.magpie_pretrained_coefficients_path_dict = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015]['baseline'] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015]['baseline'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')

    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'] = {}
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050] = {}
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_ClimPol_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_NPI_base_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_NPI_base_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')


    # The model relies on both ESACCI defined for the base year and a simplified LULC (7 classes) for the SEALS downscaling step, defined here).
    p.base_year_lulc_path = os.path.join(hb.SEALS_BASE_DATA_DIR, 'lulc_esa', 'full', 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-' + str(p.base_year) + '-v2.0.7.tif')
    p.base_year_simplified_lulc_path = os.path.join(hb.SEALS_BASE_DATA_DIR, 'lulc_esa', 'simplified', 'lulc_esa_simplified_' + str(p.base_year) + '.tif')
    p.lulc_training_start_year_10sec_path = os.path.join(hb.SEALS_BASE_DATA_DIR, 'lulc_esa', 'simplified', 'lulc_esa_simplified_' + str(p.training_start_year) + '.tif')

    # Provided by GTAP team.
    # TODOO This is still based on the file below, which was from Purdue. It is a vector of 300sec gridcells and should be replaced with continuous vectors
    p.gtap37_aez18_input_vector_path = os.path.join(p.model_base_data_dir, "region_boundaries\GTAPv10_AEZ18_37Reg.shp")

    # GTAP-InVEST has two resolutions: fine (based on ESACCI) and coarse (based on LUH), though actually the coarse does change when using magpie 30min.
    p.fine_resolution = hb.get_cell_size_from_path(p.match_10sec_path)
    p.coarse_resolution = hb.get_cell_size_from_path(p.scenario_coarse_state_paths[p.luh_scenario_labels[0]][p.scenario_years[0]][p.policy_scenario_labels[0]])
    p.coarse_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.coarse_resolution]

    p.coarse_match = hb.ArrayFrame(p.coarse_match_path)

    # Sometimes runs fail mid run. This checks for that and picks up where there is a completed file for that zone.
    p.skip_created_downscaling_zones = 1

    # The SEALS-simplified classes are defined here, which can be iterated over. We also define what classes are shifted by GTAP's endogenous land-calcualtion step.
    p.class_indices = [1, 2, 3, 4, 5] # These are the indices of classes THAT CAN EXPAND/CONTRACT
    p.nonchanging_class_indices = [6, 7] # These add other lulc classes that might have an effect on LUC but cannot change themselves (e.g. water, barren)
    p.regression_input_class_indices = p.class_indices + p.nonchanging_class_indices

    p.class_labels = ['urban', 'cropland', 'grassland', 'forest', 'nonforestnatural',]
    p.nonchanging_class_labels = ['water', 'barren_and_other']
    p.regression_input_class_labels = p.class_labels + p.nonchanging_class_labels

    p.shortened_class_labels = ['urban', 'crop', 'past', 'forest', 'other',]

    p.class_indices_that_differ_between_ssp_and_gtap = [2, 3, 4,]
    p.class_labels_that_differ_between_ssp_and_gtap = ['cropland', 'grassland', 'forest',]

    # A little awkward, but I used to use integers and list counting to keep track of the actual lulc class value. Now i'm making it expicit with dicts.
    p.class_indices_to_labels_correspondence = dict(zip(p.class_indices, p.class_labels))
    p.class_labels_to_indices_correspondence = dict(zip(p.class_labels, p.class_indices))

    # Used for (optional) calibration of seals.
    p.calibrate = True
    p.num_generations = 2

    if p.is_gtap1_run:
        p.pretrained_coefficients_path_dict = p.gtap_pretrained_coefficients_path_dict
    elif p.is_magpie_run:
        p.pretrained_coefficients_path_dict = p.magpie_pretrained_coefficients_path_dict
    elif p.is_calibration_run:
        p.pretrained_coefficients_path_dict = 'use_generated' # TODOO Make this point somehow to the generated one.


    # If calibrate of SEALS is done, here are some starting coefficient guesses to speed it up.
    p.spatial_regressor_coefficients_path = os.path.join(p.input_dir, "spatial_regressor_starting_coefficients.xlsx")

    p.static_regressor_paths = {}
    p.static_regressor_paths['sand_percent'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'sand_percent.tif')
    p.static_regressor_paths['silt_percent'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'silt_percent.tif')
    p.static_regressor_paths['soil_bulk_density'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'soil_bulk_density.tif')
    p.static_regressor_paths['soil_cec'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'soil_cec.tif')
    p.static_regressor_paths['soil_organic_content'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'soil_organic_content.tif')
    p.static_regressor_paths['strict_pa'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'strict_pa.tif')
    p.static_regressor_paths['temperature_c'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'temperature_c.tif')
    p.static_regressor_paths['travel_time_to_market_mins'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'travel_time_to_market_mins.tif')
    p.static_regressor_paths['wetlands_binary'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'wetlands_binary.tif')
    p.static_regressor_paths['alt_m'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'alt_m.tif')
    p.static_regressor_paths['carbon_above_ground_mg_per_ha_global'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'carbon_above_ground_mg_per_ha_global.tif')
    p.static_regressor_paths['clay_percent'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'clay_percent.tif')
    p.static_regressor_paths['ph'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'ph.tif')
    p.static_regressor_paths['pop'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'pop.tif')
    p.static_regressor_paths['precip_mm'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'precip_mm.tif')


    # TODOO Make these be selected by the base_year, policy_base_year inputs.
    p.training_start_year_lulc_path = hb.global_esa_lulc_paths_by_year[p.training_start_year]
    p.training_end_year_lulc_path = hb.global_esa_lulc_paths_by_year[p.base_year]
    p.base_year_lulc_path = hb.global_esa_lulc_paths_by_year[p.base_year]

    p.training_start_year_seals7_lulc_path = hb.global_esa_seals7_lulc_paths_by_year[p.training_start_year]
    p.training_end_year_seals7_lulc_path = hb.global_esa_seals7_lulc_paths_by_year[p.base_year]
    p.base_year_seals7_lulc_path = hb.global_esa_seals7_lulc_paths_by_year[p.base_year]

    # SEALS results will be tiled on top of output_base_map_path, filling in areas potentially outside of the zones run (e.g., filling in small islands that were skipped_
    p.output_base_map_path = p.base_year_simplified_lulc_path

    if p.test_mode:
        p.stitch_tiles_to_global_basemap = 0
        if p.is_gtap1_run:
            run_1deg_subset = 1
            run_5deg_subset = 0
            magpie_subset = 0
        elif p.is_magpie_run:
            run_1deg_subset = 0
            run_5deg_subset = 0
            magpie_subset = 1
        elif p.is_calibration_run:
            run_1deg_subset = 1
            run_5deg_subset = 0
            magpie_subset = 0

    else:
        p.stitch_tiles_to_global_basemap = 1
        run_1deg_subset = 0
        run_5deg_subset = 0
        magpie_subset = 0

    # If a a subset is defined, set its tiles here.
    if run_1deg_subset:
        p.processing_block_size = 1.0  # arcdegrees
        p.subset_of_blocks_to_run = [15526]  # wisconsin
        # p.subset_of_blocks_to_run = [15526, 15526 + 180 * 1, 15526 + 180 * 2]  # wisconsin
        # p.subset_of_blocks_to_run = [
        #     15526, 15526 + 180 * 1, 15526 + 180 * 2,
        #     15527, 15527 + 180 * 1, 15527 + 180 * 2,
        #     15528, 15528 + 180 * 1, 15528 + 180 * 2,
        # ]  # wisconsin
        p.force_to_global_bb = False
    elif run_5deg_subset:
        p.processing_block_size = 5.0  # arcdegrees
        p.subset_of_blocks_to_run = [476, 476 + 1 + (36 * 2), 476 + 3 + (36 * 4), 476 + 9 + (36 * 8), 476 + 1 + (36 * 25)]  # Montana
        p.force_to_global_bb = False
    elif magpie_subset:
        p.processing_block_size = 5.0  # arcdegrees
        # p.subset_of_blocks_to_run = [476, 476 + 1 + (36 * 2)]
        # p.subset_of_blocks_to_run = [476 + 9 + (36 * 8)]  # Montana
        p.subset_of_blocks_to_run = [476]
        p.force_to_global_bb = False
    else:
        p.subset_of_blocks_to_run = None
        p.processing_block_size = 5.0  # arcdegrees
        p.force_to_global_bb = True


    ## ADD TASKS to project_flow task tree, then below set if they should run and/or be skipped if existing.
    p.luh2_extraction = p.add_task(seals_process_coarse_timeseries.luh2_extraction, skip_existing=1)
    p.luh2_difference_from_base_year = p.add_task(seals_process_coarse_timeseries.luh2_difference_from_base_year, skip_existing=1)
    p.luh2_as_seals7_proportion = p.add_task(seals_process_coarse_timeseries.luh2_as_seals7_proportion, skip_existing=1)
    p.seals7_difference_from_base_year = p.add_task(seals_process_coarse_timeseries.seals7_difference_from_base_year, skip_existing=1)

    p.calibration_generated_inputs_task = p.add_task(seals_main.calibration_generated_inputs, skip_existing=1)
    p.calibration_task = p.add_iterator(seals_main.calibration, run_in_parallel=1, skip_existing=1)
    p.calibration_prepare_lulc_task = p.add_task(seals_main.calibration_prepare_lulc, parent=p.calibration_task, skip_existing=1)
    p.calibration_zones_task = p.add_task(seals_main.calibration_zones, parent=p.calibration_task, skip_existing=1)

    p.luh_allocations = p.add_iterator(seals_main.policy_scenario_allocations, run_in_parallel=False, skip_existing=1)
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.luh_allocations, run_in_parallel=True, skip_existing=1)
    p.prepare_lulc_task = p.add_task(seals_main.prepare_lulc, parent=p.allocation_zones_task, skip_existing=1)
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=1)

    p.stitched_lulcs_task = p.add_task(seals_main.stitched_lulcs, skip_existing=1)
    p.map_esa_simplified_back_to_esa_task = p.add_task(seals_main.map_esa_simplified_back_to_esa)



    p.execute()


