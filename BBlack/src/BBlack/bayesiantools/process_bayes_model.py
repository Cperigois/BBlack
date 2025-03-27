import numpy as np
import pandas as pd
import os
import json
import importlib.resources
from BBlack.GWtools.detector import DetectorGW
from BBlack.bayesiantools.bayes_model import BayesModel

# Import parameter file
with importlib.resources.open_text("BBlack.Run", "Params.json") as f:
    params = json.load(f)


def process_bayes_model(astro_model):
    # Make sure directories are created
    if not os.path.exists("Run/" + params['name_of_project_folder'] + "/Bayes_Models/"):
        os.mkdir("Run/" + params['name_of_project_folder'] + "/Bayes_Models/")
    if not os.path.exists("Run/" + params['name_of_project_folder'] + "/Bayes_Models/Efficiency/"):
        os.mkdir("Run/" + params['name_of_project_folder'] + "/Bayes_Models/Efficiency")
    if not os.path.exists("Run/" + params['name_of_project_folder'] + "/Bayes_Models/Match_model/"):
        os.mkdir("Run/" + params['name_of_project_folder'] + "/Bayes_Models/Match_model")

    far_limit = params['event_selection']['far_limit']
    snr_limit = params['event_selection']['snr_limit']
    pastro_limit = params['event_selection']['pastro_limit']
    approximant = params['bayes_model_params']["waveform_approximant"]  # waveform approximant
    bw_method = params['bayes_model_params']["bandwidth_KDE"]  # KDE bandwidth to use

    for obs in params['observing_runs']:
        # Initialise observing run
        # read and select events following user criteria
        run_info = pd.read_csv('AuxiliaryFiles/observing_runs_info/' + obs + '_events.csv')
        run_info = run_info[(run_info['far'] < far_limit) &
                            (run_info['SNR'] > snr_limit) &
                            (run_info['p_astro_' + params['co_type']] > pastro_limit)]
        run_size = len(run_info.name)
        run_info.to_csv("Run/" + params['name_of_project_folder'] + '/selection_from_' + obs + '.dat', sep='\t',
                        index=None)
        event_list = run_info.name
        # extract other params set by the user
        n_cpu = np.max([run_size, params['n_cpu_max']])  # number of CPUs

        # Initialise detector
        detector_name = params['event_selection']['runs_param'][obs]['detector']  # detector name
        detector = DetectorGW(detector_name, params['event_selection']['runs_param'][obs]['delta_freq'])

        for var in params['observable_variation'].keys():
            bayes_model_name = astro_model.name + '_' + obs + '_' + var
            # Initialise Bayesian model
            bayes_model = BayesModel(name=bayes_model_name,
                                     astro_model=astro_model,
                                     observing_run_name=obs,
                                     event_list=event_list,
                                     detector=detector,
                                     variation=var)
            file_exist = (os.path.isfile(bayes_model.file_name_match) &
                          os.path.isfile(bayes_model.file_name_efficiency))
            if (not file_exist) or params['overwrite']['bayesian_analysis']:
                bayes_model.compute_model_efficiency(astro_model.sample_file_name, n_cpu=n_cpu, approximant=approximant)

                # Compute the matching term for all the events of the observing run
                if n_cpu > run_size:
                    bayes_model.model_matching(n_cpu=run_size,
                                               bw_method=bw_method)
                else:
                    bayes_model.model_matching(n_cpu=n_cpu, bw_method=bw_method)
                bayes_model.save()
                print('Done! ', params['name_of_project_folder'], ' ', obs, ' ', var)
            else:
                bayes_model.load()
                print('Done! ', params['name_of_project_folder'], ' ', obs, ' ', var)
                print('Files already exist and are not recomputed \nto recompute the bayesian analysis \nset the '
                      'parameter rerun_bayesian_analysis to True')
