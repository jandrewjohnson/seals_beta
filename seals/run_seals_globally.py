import os, sys
import seals_utils
import seals_initialize_project
import hazelbean as hb
import pandas as pd
from seals_utils import download_google_cloud_blob

main = ''
if __name__ == '__main__':

    ### ------- ENVIRONMENT SETTINGS -------------------------------

    # Users should only need to edit lines in this ENVIRONMENT SETTINGS section
    # Everything is relative to these (or the source code dir).
    # Specifically, 
    # 1. ensure that the project_dir makes sense for your machine
    # 2. ensure that the base_data_dir makes sense for your machine
    # 3. ensure that the data_credentials_path points to a valid credentials file, if relevant
    # 4. ensure that the input_bucket_name points to a cloud bucket you have access to, if relevant

    # A ProjectFlow object is created from the Hazelbean library to organize directories and enable parallel processing.
    # project-level variables are assigned as attributes to the p object (such as in p.base_data_dir = ... below)
    # The only agrument for a project flow object is where the project directory is relative to the current_working_directory.
    user_dir = os.path.expanduser('~')        
    extra_dirs = ['seals', 'projects']

    # The project_name is used to name the project directory below. Also note that
    # ProjectFlow only calculates tasks that haven't been done yet, so adding 
    # a new project_name will give a fresh directory and ensure all parts
    # are run.
    project_name = 'seals_global_project'

    # The project-dir is where everything will be stored, in particular in an input, intermediate, or output dir
    # IMPORTANT NOTE: This should not be in a cloud-synced directory (e.g. dropbox, google drive, etc.), which
    # will either make the run fail or cause it to be very slow. The recommended place is (as coded above)
    # somewhere in the users's home directory.
    project_dir = os.path.join(user_dir, os.sep.join(extra_dirs), project_name)

    p = hb.ProjectFlow(project_dir)

    # The ProjectFlow object p will manage all tasks to be run, enables parallelization over spatial tiles or model runs,
    # manages directories, and provies a central place to store project-level variables (as attributes of p) that
    # works between tasks and between parallel threads. For instance, here we define the local variables above
    # to ProjectFlow attributes.
    p.user_dir = user_dir
    p.project_name = project_name
    p.project_dir = project_dir

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    # Additionally, if you're clever, you can move files generated in your tasks to the right base_data_dir
    # directory so that they are available for future projects and avoids redundant processing.
    # NOTE THAT the final directory has to be named base_data to match the naming convention on the google cloud bucket.
    # As with the project dir, this should be a non-cloud-synced directory, and ideally on a fast NVME SSD drive,
    # as this is primarily io-bound.
    p.base_data_dir = os.path.join('C:/Users/jajohns/Files/base_data')
    # p.base_data_dir = os.path.join('G:/My Drive/Files/base_data')

    # In order for SEALS to download using the google_cloud_api service, you need to have a valid credentials JSON file.
    # Identify its location here. If you don't have one, email jajohns@umn.edu. The data are freely available but are very, very large
    # (and thus expensive to host), so I limit access via credentials.
    p.data_credentials_path = '..\\api_key_credentials.json'

    # There are different versions of the base_data in gcloud, but best open-source one is 'gtap_invest_seals_2023_04_21'
    p.input_bucket_name = 'gtap_invest_seals_2023_04_21'

    # If you want to run SEALS with the run.py file in a different directory (ie in the project dir)
    # then you need to add the path to the seals directory to the system path.
    custom_seals_path = None
    if custom_seals_path is not None: # G:/My Drive/Files/Research/seals/seals_dev/seals
        sys.path.insert(0, custom_seals_path)

    # SEALS will run based on the scenarios defined in a scenario_definitions.csv
    # If you have not run SEALS before, SEALS will generate it in your project's input_dir.
    # A useful way to get started is to to run SEALS on the test data without modification
    # and then edit the scenario_definitions.csv to your project needs.
    # Some of the other test files use different scenario definition csvs 
    # to illustrate the technique. If you point to one of these 
    # (or any one CSV that already exists), SEALS will not generate a new one.
    # The avalable example files in the default_inputs include:
    # - test_three_scenario_defininitions.csv
    # - test_scenario_defininitions_multi_coeffs.csvs
    
    p.scenario_definitions_path = os.path.join(p.input_dir, 'global_scenario_definitions.csv')

    # Set defaults and generate the scenario_definitions.csv if it doesn't exist.
    if not hb.path_exists(p.scenario_definitions_path):
        # There are several possibilities for what you might want to set as the default.
        # Choose accordingly by uncommenting your desired one. The set of
        # supported options are
        # - set_attributes_to_dynamic_default (primary one)
        # - set_attributes_to_dynamic_many_year_default
        # - set_attributes_to_default # Deprecated

        seals_utils.set_attributes_to_dynamic_default(p) # Default option


        # Optional overrides for us in intitla scenarios
        p.aoi = 'RWA'

        # seals_utils.set_attributes_to_dynamic_default(p)
        # Once the attributes are set, generate the scenarios csv and put it in the input_dir.
        seals_utils.generate_scenarios_csv_and_put_in_input_dir(p)
        p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
    else:
        # Read in the scenarios csv and assign the first row to the attributes of this object (in order to setup additional 
        # project attributes like the resolutions of the fine scale and coarse scale data)
        p.scenarios_df = pd.read_csv(p.scenario_definitions_path)

        # Because we've only read the scenarios file, set the attributes
        # to what is in the first row.
        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)
            break # Just get first for initialization.
            
    # Set which type of run this is. This is used to determine which tasks to run.
    # Options are 'allocation_run', 'allocation_and_visualization_run', 'calibration_run'
    run_type = 'global_allocation_and_visualization_run'
    # run_type = 'allocation_and_quick_visualization_run'

    ### ------- DOWNLOAD SETUP FILES ------------------------------------

    # Validate all file paths. This is where the project-specific files exit (such as scenarios.csv which might be in the input_dir),
    # so thus here we overwrite the default attributes above.
    path_attribute_names_to_validate = ['scenario_definitions_path', 'coarse_projections_input_path', 'lulc_correspondence_path', 'coarse_correspondence_path', 'calibration_parameters_source', 'base_year_lulc_path', ]
    for path_attribute_name in path_attribute_names_to_validate:
        check_path = getattr(p, path_attribute_name)

        # Note here that i break my standard extant_path appraoch because it also checks seals/default_inputs in the base data. Not sure if i like this.
        extant_path = hb.get_first_extant_path(check_path, [p.input_dir, p.base_data_dir, os.path.join(p.base_data_dir, 'seals', 'default_inputs')])
        setattr(p, path_attribute_name, extant_path)

    ### START HERE, Remove downloading of scenarios_defintions_path and ensure that the auto-generated version correctly writes.
    # Initial download of setup files
    # download_paths = [p.scenario_definitions_path]
    # for path in download_paths:
    #     if not hb.path_exists(path) and path is not None: # Check one last time to ensure that it wasn't added twice.
    #         url = 'base_data' + '/' +  hb.path_to_url(path, os.path.split(p.base_data_dir)[1])
    #         download_google_cloud_blob(p.input_bucket_name, url, p.data_credentials_path, path)




    # SEALS has two resolutions: fine and coarse. In most applications, fine is 10 arcseconds (~300m at equator, based on ESACCI)
    # and coarse is based on IAM results that are 900 arcseconds (LUH2) or 1800 arcseconds (MAgPIE). Note that there is a coarser-yet
    # scale possible from e.g. GTAP-determined endogenous LUC. This is excluded in the base SEALS config.
    # These are set based on the input data, so here we will analyze two files in the input csv
    # and set the resolutions based on that, but first we need to download the files.
    download_paths = [p.base_year_lulc_path, p.coarse_projections_input_path, p.calibration_parameters_source, p.coarse_correspondence_path, p.lulc_correspondence_path]
    for path in download_paths:
        path = hb.get_first_extant_path(path, [p.input_dir, p.base_data_dir])
        if not hb.path_exists(path): # Check one last time to ensure that it wasn't added twice.
        # if not hb.path_exists(path): # Check one last time to ensure that it wasn't added twice.

            # Notice that we take set local_path_to_strip to NOT include the final dir 'base_data' in the p.base_data_dir
            # because the cloud bucket has that file dir in it's root.
            url =  hb.path_to_url(path, os.path.split(p.base_data_dir)[0])
            download_google_cloud_blob(p.input_bucket_name, url, p.data_credentials_path, path)

    ### ------- SCENARIO DERIVED ATTRIBUTES ---------------------------------

    # To support loading an extent by AOI, define a country-iso3 vector here.
    p.countries_iso3_path = os.path.join(p.base_data_dir, 'pyramids', 'countries_iso3.gpkg')

    # Some variables need further processing into attributes,
    # like parsing a correspondence csv into a dict.
    seals_utils.set_derived_attributes(p)

    # To run a much faster version for code-testing purposes, enable test_mode. Selects a much smaller set of scenarios and spatial tiles. Will change the AOI to a small country.
    p.test_mode = 0

    # Run a version of the test mode (fast) that also outputs tons of diagnostics, intermediates, and plots. Not feasible to plot these for the globe.
    p.test_and_report_mode = 0


    p.fine_resolution = hb.get_cell_size_from_path(p.base_year_lulc_path)
    p.coarse_resolution = hb.get_cell_size_from_path(p.coarse_projections_input_path)
    p.fine_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.fine_resolution]
    p.coarse_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.coarse_resolution]

    p.processing_block_size = 4.0 # In degrees. Must be in pyramid_compatible_resolutions
    p.processing_resolution_arcseconds = p.processing_block_size * 3600.0 # MUST BE FLOAT
    p.processing_resolution = p.processing_block_size


    # calibration_parameters_override_dict can be used in specific scenarios to e.g., not allow expansion of cropland into forest by overwriting the default calibration. Note that I haven't set exactly how this
    # will work if it is specific to certain zones or a single-zone that overrides all. The latter would probably be easier.
    # If the DF has a value, override. If it is None or "", keep from parameters source.
    p.calibration_parameters_override_dict = {}
    # p.calibration_parameters_override_dict['rcp45_ssp2'][2030]['BAU'] = os.path.join(p.input_dir, 'calibration_overrides', 'prevent_cropland_expansion_into_forest.xlsx')

    # Configure the logger that captures all the information generated.
    p.L = hb.get_logger('test_run_seals')

    # UNIMPLEMENTED OPTIONS
    # Set the training start year and end year. These years will be used for calibrating the model. Once calibrated, project forward
    # from the base_year (which could be the same as the training_end_year but not necessarily).
    # If the training years are given, it will generate convolutions for them over the whole AOI, which may take a lot of time.
    # p.training_start_year = None
    # p.training_end_year = None

    # For GTAP-enabled runs, we project the economy from the latest GTAP reference year to the year in which a
    # policy is made so that we can apply the policy to a future date. Set that policy year here. (Only affects runs if p.is_gtap_run is True)

    # SEALS Operates on a simplified reclassification of LULC maps. Optionally specify this here
    # otherwise it will assume you are using the default esa to seals7 simplification.

    # For whatever simplification mapping you choose, you will need to specify a correspondence between the src and the simplified classification
    # By default SEALS uses a built in esa to seals7 correspondenc

    # Calibrating the model can be VERY time consuming. This attribute lets you point it to a different
    # data source for the coefficient parameters. Otherwise, you can set it to 'calibration_task', but
    # this requires that you at least have a calibration directory filled with trained coefficients
    # for the zones you are running.

    ### ------- RUNTIME OPTIONS ------------------------------------------------------

    p.build_overviews_and_stats = 0  # For later fast-viewing, this can be enabled to write ovr files and geotiff stats files. NYI anywhere.

    # TEST-specific options to speed up further. Operates as overwrite.
    p.force_to_global_bb = True
    if p.test_mode:
        p.force_to_global_bb = False

    if p.test_mode:
        p.processing_block_size = 1.0 # In degrees. Must be in pyramid_compatible_resolutions
        p.processing_resolution = p.processing_block_size
        p.processing_resolution_arcseconds = p.processing_block_size * 3600.0 # MUST BE FLOAT

    p.run_in_parallel = 1
    if p.test_mode:
        p.run_in_parallel = 0

    p.plotting_level = 0
    if p.test_mode:
        if p.test_and_report_mode:
            p.plotting_level = 22

    p.cython_reporting_level = 0
    if p.test_mode:
        if p.test_and_report_mode:
            p.cython_reporting_level = 22

    p.calibration_cython_reporting_level = 0
    if p.test_mode:
        if p.test_and_report_mode:
            p.calibration_cython_reporting_level = 22

    p.output_writing_level = 0  # >=2 writes chunk-baseline lulc
    if p.test_mode:
        if p.test_and_report_mode:
            p.output_writing_level = 22

    p.write_projected_coarse_change_chunks = 0  # in the SEALS allocation, for troubleshooting, it can be useful to see what was the coarse allocation input.
    if p.test_mode:
        if p.test_and_report_mode:
            p.write_projected_coarse_change_chunks = 1

    p.write_calibration_generation_arrays = 0  # in the SEALS allocation, for troubleshooting, it can be useful to see what was the coarse allocation input.
    if p.test_mode:
        if p.test_and_report_mode:
            p.write_calibration_generation_arrays = 1

    # Because convolving a lulc map is so computationally expensive, you may set this option to manually set which year of convolutions to use. If this is None (the default), it will convolve the base year.
    p.years_to_convolve_override = None
    # p.years_to_convolve_override = [2014]

     # None sets it to max available. Otherwise, set to an integer. # Note that on windows, there is an arcane error
     # that leads to python not being able to handle more than 63 concurrent threads:
     # "need at most 63 handles, got a sequence of length 129". This should only be hit by threadripper CPUs 
     # at the moment, but is a strong reason to go fully linux.
    p.num_workers = 60 

    # Choose which set of tasks to run and build the task tree accordingly.
    # ProjectFlow generates directories in your intermediate_dir based on the names of the tasks
    # It also makes a reference to other tasks' directories by postpending '_dir'
    # to the name of the task and saving it as an attribute to p. E.x.,
    # p.fine_processed_inputs_dir. These attributes will be set if a task is loaded into the tree, 
    # even if it does not run.

    # MOVE THIS TO HB? No, cause then it wouldnt have task functions available as globals
    seals_initialize_project.build_task_tree_by_name(p, run_type)
    
    seals_initialize_project.run(p)

    result = 'Done!'


