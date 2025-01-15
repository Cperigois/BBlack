import time
from BBlack.GWtools.detector import DetectorGW
import concurrent.futures
from BBlack.astrotools.utility_functions import berti_pdet_fit, mc_q_to_m1_m2, parallel_array_range, clean_path, f_merg, \
    jump_mix_frac
import astropy.cosmology as cosmo
import pycbc.waveform
from itertools import chain
import numpy as np
import scipy.stats
import pandas as pd
import os
import json
import BBlack.bayesiantools.bayes_model as BM
import BBlack.astrotools.AstroModel as AM
import BBlack.GWtools.detector as Detector
import BBlack.GWtools.gw_event as GWE

os.environ['MKL_NUM_THREADS'] = '1'  # this command prevents Python from multithreading
# (useful especially for Demoblack machine!)
params = json.load(open('Run/Params.json', 'r'))


def compute_likelihood(astro_model):
    bayes_opt = params['bayes_model_params']['likelihood_option']
    results = {}
    for var in params['observable_variation'].keys():
        # Computation of likelihood
        subresults = {}
        log_likelihood = 0.0
        print('***   ', astro_model.name, '   ***   ', var, '   ***')
        for obs in params['observing_runs']:
            det_name = params['event_selection']['runs_param'][obs]['detector']
            detector = Detector.DetectorGW(det_name, params['event_selection']['runs_param'][obs]['delta_freq'])
            bm_name = astro_model.name + '_' + obs + '_' + var
            df = pd.read_csv('Run/' + params['name_of_project_folder'] + '/selection_from_' + obs + '.dat',
                             index_col=None, sep='\t')
            bayes_model = BM.BayesModel(name=bm_name, astro_model=astro_model, observing_run_name=obs,
                                        detector=detector,
                                        variation=var, read_match=True, read_eff=True)

            # Get relevant terms for computation
            print(bayes_model.match_model.describe())
            integral_match_model = np.log(bayes_model.match_model.int).sum()
            detection_efficiency = bayes_model.efficiency
            n_sources = bayes_model.n_sources
            n_obs = len(bayes_model.match_model.index)

            # Compute log-likelihood
            log_likelihood_obs = compute_log_likelihood(bayes_opt, integral_match_model, n_obs,
                                                        n_sources, detection_efficiency)
            print(f"Log-likelihood for observing run {obs} with variation {var} : {log_likelihood_obs}")
            subresults[obs] = log_likelihood_obs
            log_likelihood += log_likelihood_obs
        subresults['total'] = log_likelihood
        results[var] = subresults
        print("The value of the total log-likelihood with variation " + var + " is : " + str(log_likelihood))
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Likelihoods/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Likelihoods/")
    json_object = json.dumps(results, indent=len(results.keys()))
    with open('Run/' + params['name_of_project_folder'] + '/Likelihoods/' + astro_model.name + '_likelihoods.json',
              "w") as file:
        file.write(json_object)  # encode dict into JSON
    return results


def multichannel_analysis(name):
    if not os.path.exists('Run/' + params['name_of_project_folder']+'/Multichannel_analysis'):
        os.mkdir('Run/' + params['name_of_project_folder']+'/Multichannel_analysis')
    channel_list = params['compute_multi_channel'][name]
    bayes_opt = params["bayes_model_params"]['multi_channel_option']
    n_mcmc = 100000  # total stops for the MCMC chain
    scale_jump = 0.01  # scale used for the gaussian jump
    name_file = "Run/"+params['name_of_project_folder']+"/Multichannel_analysis/Multichannel_ana_" + name + ".dat"
    models_dict = {}
    observing_runs = {}
    observing_runs_list = params['observing_runs']
    for var in params['observable_variation']:
        for channel in range(len(channel_list)):
            for obs in range(len(observing_runs_list)):
                astro_model = AM.AstroModel(name=channel_list[channel])
                det_name = params['event_selection']['runs_param'][observing_runs_list[obs]]['detector']
                detector = Detector.DetectorGW(det_name,
                                               params['event_selection']['runs_param'][observing_runs_list[obs]][
                                                   'delta_freq'])
                bm_name = astro_model.name + '_' + observing_runs_list[obs] + '_' + var
                models_dict[(channel, obs)] = BM.BayesModel(name=bm_name, astro_model=astro_model,
                                                            observing_run_name=observing_runs_list[obs],
                                                            detector=detector,
                                                            variation=var, read_match=True, read_eff=True)
        # Initialisation
        mcmc_chain = np.zeros(
            (n_mcmc, len(channel_list) + 1))  # initialise chain, last column contains log-likelihood

        # -------------------------------------------      Main code       -------------------------------------------------

        # -----------------     Starting MCMC point     -----------------

        # Generate a value of hyperparameter hyp randomly within the prior ranges
        mix_frac_ini = np.random.dirichlet(np.ones(len(channel_list)), 1)[0]

        # Compute log-likelihood
        log_likelihood_ini = 0.0
        for obs in range(len(observing_runs_list)):
            df = pd.read_csv('Run/' + params['name_of_project_folder'] + '/selection_from_' + observing_runs_list[obs] +
                             '.dat', sep='\t', index_col=None)
            events_list = df.name
            detection_efficiency = 0.0
            n_sources = 0.0
            n_obs = len(df.name)
            match_sources = []
            for i in range(len(channel_list)):
                detection_efficiency += mix_frac_ini[i] * models_dict[(i, obs)].efficiency
                n_sources += mix_frac_ini[i] * models_dict[(i, obs)].n_sources
                if i == 0:
                    match_sources = mix_frac_ini[i] * models_dict[(i, obs)].match_model.int
                else:
                    match_sources += mix_frac_ini[i] * models_dict[(i, obs)].match_model.int
            integral_match_model = np.log(match_sources).sum()

            log_likelihood_ini += compute_log_likelihood(bayes_opt, integral_match_model, n_obs, n_sources,
                                                         detection_efficiency)

        # Record the initial point values and assign them to current position of the chain
        mcmc_chain[0, :-1] = mix_frac_ini
        mcmc_chain[-1] = log_likelihood_ini
        mix_frac_cur = mix_frac_ini
        log_likelihood_cur = log_likelihood_ini

        # -----------------     Main chain     -----------------

        start = time.perf_counter()
        accept = 1
        for n in range(n_mcmc - 1):
            if n % 25000 == 0:
                print("Completion : " + str(100.0 * float(n) / float(n_mcmc)) + " %")

            # Jump proposal for mixing fraction
            mix_frac_jump = jump_mix_frac(mix_frac_cur, scale_jump)

            # Compute log-likelihood
            log_likelihood_jump = 0.0
            for obs in range(len(observing_runs)):
                detection_efficiency = 0.0
                n_sources = 0.0
                n_obs = len(df.name)

                match_sources = []
                for i in range(len(channel_list)):
                    detection_efficiency += mix_frac_jump[i] * models_dict[(i, obs)].efficiency
                    n_sources += mix_frac_jump[i] * models_dict[(i, obs)].n_sources
                    if i == 0:
                        match_sources = mix_frac_jump[i] * models_dict[(i, obs)].match_model.int
                    else:
                        match_sources += mix_frac_jump[i] * models_dict[(i, obs)].match_model.int
                integral_match_model = np.log(match_sources).sum()

                log_likelihood_jump += compute_log_likelihood(bayes_opt, integral_match_model,
                                                              n_obs, n_sources, detection_efficiency)

            # Accept jump with probability Metropolis-Hastings ratio
            Hratio = np.exp(log_likelihood_jump - log_likelihood_cur)
            if Hratio > 1:
                mix_frac_cur = mix_frac_jump
                log_likelihood_cur = log_likelihood_jump
                accept += 1
            else:
                beta = np.random.uniform()
                if Hratio > beta:
                    mix_frac_cur = mix_frac_jump
                    log_likelihood_cur = log_likelihood_jump
                    accept += 1

            # Update chain values
            mcmc_chain[n + 1, :-1] = mix_frac_cur
            mcmc_chain[n + 1, -1] = log_likelihood_cur

        # Take 1 out of 200 points to ensure a good independance of the samples
        mcmc_chain = mcmc_chain[::200]

        # Create the file containing the output
        print("Acceptance rate is : {} %".format(100.0 * float(accept) / float(n_mcmc)))
        finish = time.perf_counter()
        print(f'Finished in {round(finish - start, 2)} second(s)')

        header_file = "\t".join(["f_" + f for f in channel_list]) + "\t" + "Likelihood" + "\n"

        with open(name_file, "w") as fileout:
            fileout.write("\t".join(["f_" + f for f in channel_list]) + "\t" + "Likelihood" + "\n")  # header
            np.savetxt(fileout, mcmc_chain, delimiter='\t', fmt='%.4f')


def compute_log_likelihood(bayes_opt, integral_match_model, n_obs, n_sources, detection_efficiency):
    """This function computes the log-likelihood according to the option selected for a number of observed
    that were observed during a given observing run

    Parameters
    ----------
    bayes_opt : str
        Choice for the form of the log-likelihood
    integral_match_model : float
        Sum of the logarithm of the integral matching model with distribution over all observed events
    n_obs : int
        Number of observed events.
    n_sources : int
        Number of sources that is predicted by the model
    detection_efficiency : float
        Value of the detection efficiency

    Returns
    -------
    log_likelihood : float of tuples
        Value of the log-likelihood
    """

    log_likelihood = None
    if bayes_opt == "MatchOnly":
        log_likelihood = integral_match_model
    elif bayes_opt == "NoRate":
        log_likelihood = integral_match_model - float(n_obs) * np.log(detection_efficiency)
    elif bayes_opt == "All":
        log_likelihood = integral_match_model - detection_efficiency * n_sources + float(n_obs) * np.log(n_sources)
    else:
        print("Choose a Bayesian option between All, NoRate and MatchOnly.")
        exit()

    return log_likelihood
