import os
import sys
from Project_Modules.spin_model import SpinModel
from Project_Modules.astro_model import AstroModel
from Project_Modules.detector import DetectorGW
from Project_Modules.observing_run import ObservingRun
from Project_Modules.bayes_model import BayesModel
from astro_model_param import return_astro_param
os.environ['MKL_NUM_THREADS'] = '1' # this command prevents Python from multithreading
                                    #(useful especiallfy for Demoblack machine!)


if __name__ == '__main__':



    # -------------------------------------------      User input       ------------------------------------------------

    n_cpu = 1  # number of CPUs
    approximant = "IMRPhenomPv2"  # waveform approximant
    bw_method = 0.075  # KDE bandwidth to use
    detector_name = "Livingston_O1"  # detector name
    obs_run_name = "O1"  # observing run name

    # -------------------------------------------      Main code       -------------------------------------------------

    # Make sure directories are created
    if not os.path.exists("Bayes_Models/"):
        os.mkdir("Bayes_Models/")
    if not os.path.exists("Bayes_Models/Efficiency/"):
        os.mkdir("Bayes_Models/Efficiency")
    if not os.path.exists("Bayes_Models/Match_model/"):
        os.mkdir("Bayes_Models/Match_model")

    # Get all the parameters set in astro_model_param.py
    _, astro_param, co_param, mag_gen_param, name_spin_model = return_astro_param(sys.argv)

    # Initialise spin and astro model
    spin_model = SpinModel(name_model=name_spin_model, mag_gen_param=mag_gen_param)
    astro_model = AstroModel(astro_model_parameters=astro_param, co_parameters=co_param, spin_model=spin_model,
                             load_cat=True, load_mrd=True)

    # Initialise observing run
    observing_run = ObservingRun(obs_run_name, read_data_posterior=True, read_data_prior=True,
                                 co_only=astro_model.astro_model_parameters["co_type"])

    # Initialise detector
    detector = DetectorGW(detector_name)

    # Initialise Bayesian model
    bayes_model = BayesModel(astro_model=astro_model, observing_run=observing_run, detector=detector)

    # Compute detection efficiency using samples generated by generate_samples.py
    sample_file_name = "sampling_" + "_".join([astro_model.map_name_par[x] + "_" +
                                               str(astro_model.astro_model_parameters[x]) for x in
                                               astro_model.astro_model_parameters]) + ".dat"
    bayes_model.compute_model_efficiency("Samples/" + sample_file_name, n_cpu=n_cpu, approximant=approximant)

    # Compute the matching term for all the events of the observing run
    if n_cpu > observing_run.n_det[astro_model.astro_model_parameters["co_type"]]:
        bayes_model.model_matching(n_cpu=observing_run.n_det[astro_model.astro_model_parameters["co_type"]],
                                   bw_method=bw_method)
    else:
        bayes_model.model_matching(n_cpu=n_cpu, bw_method=bw_method)
    print(obs_run_name, ' ', astro_model.astro_model_parameters["co_type"])