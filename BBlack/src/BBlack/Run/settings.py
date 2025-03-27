import os
import json
import BBlack.Run.advanced_params as AP
import BBlack.Run.getting_started as GS


def Make_param_file():
    """---------------------------------------------------"""

    """        *** Main Code, should not change ***       """

    """  1- Set the directory for all intermediate and definitive results  """

    if not os.path.exists('Run/' + GS.name_of_project_folder):
        os.mkdir('Run/' + GS.name_of_project_folder)

    """  2- Gather and save the parameter used in the study  """

    param_dictionary = {'name_of_project_folder': GS.name_of_project_folder,
                        'redshift_range': GS.redshift_range,
                        'catalog_size': GS.catalog_size,
                        'observables': GS.observables,
                        'co_type': GS.co_type,
                        'astro_model_list': GS.astro_model_list,
                        'observing_runs': GS.observing_runs,
                        'compute_multi_channel': GS.compute_multi_channel,
                        'n_cpu_max': GS.n_cpu_max,
                        'observable_variation': GS.observable_variation,
                        'overwrite': {'astromodel': GS.rerun_catalog_computation,
                                      'samples': GS.rerun_samples_generation,
                                      'bayesian_analysis': GS.rerun_bayesian_analysis},
                        'results': {'cleaning': GS.run_data_cleaning,
                                    'plots': GS.run_plots}
                        }
    advParams = {"AM_params": {'input_parameters': AP.input_parameters, 'spin_model': AP.spin_model},
                 "sampling_params": {'size': AP.sampling_size,
                                     'number_of_walkers': AP.sampling_number_of_walkers,
                                     'chain_length': AP.sampling_chain_length,
                                     'bandwidth_KDE': AP.sampling_bandwidth_KDE},
                 "detector_params": {'detectors_avail': AP.detectors_avail,
                                     'psd_attributes': AP.psd_attributes},
                 "event_selection": {'pastro_limit': AP.pAstroLimit,
                                     'far_limit': AP.farLimit,
                                     'snr_limit': AP.snrLimit,
                                     'runs_param': AP.available_obs_runs},
                 "bayes_model_params": {'waveform_approximant': AP.bayes_model_processing_waveform_approximant,
                                        'bandwidth_KDE': AP.bayes_model_processing_bandwidth_KDE,
                                        'likelihood_option': AP.bayes_option_compute_likelihood,
                                        'multi_channel_option': AP.bayes_option_multichannel},
                 'post_processing_params': {'observable_range': AP.observable_range},
                 'Cosmo_model': AP.cosmo_model
                 }

    set(GS.name_of_project_folder, param_dictionary, advParams)


def set(_projectFolder, _paramDictionnary, _advParamDictionnary):
    try:
        # Validate inputs
        if not isinstance(_paramDictionnary, dict) or not isinstance(_advParamDictionnary, dict):
            raise ValueError("Both _paramDictionnary and _advParamDictionnary must be dictionaries.")

        # Merge dictionaries
        output = {**_paramDictionnary, **_advParamDictionnary}
        json_object = json.dumps(output, indent=4)  # Pretty print JSON with a standard indent level

        # Define file paths
        base_params_path = os.path.join('Run', 'Params.json')
        project_folder_path = os.path.join('Run', _projectFolder)
        project_params_path = os.path.join(project_folder_path, 'Params.json')

        # Ensure 'Run' and project folder exist
        os.makedirs('Run', exist_ok=True)
        os.makedirs(project_folder_path, exist_ok=True)

        # Write to base Params.json
        with open(base_params_path, "w") as file:
            file.write(json_object)

        # Write to project-specific Params.json
        with open(project_params_path, "w") as file:
            file.write(json_object)

        print(f"Successfully wrote Params.json to {base_params_path} and {project_params_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")

def clean():
    params = json.load(open('Run/Params.json', 'r'))
    for am in params['astro_model_list'].keys():
        os.remove('Run/' + params['name_of_project_folder']+'/'+ am +'_AM.pickle')
    for det in params['detector_list'].keys():
        os.remove('Run/' + params['name_of_project_folder']+'/'+ det +'_DET.pickle')
    for net in params['network_list'].keys():
        os.remove('Run/' + params['name_of_project_folder'] + '/' + net + '_NET.pickle')
    os.remove('Run/Params.json')