import pickle

from BBlack.GWtools.detector import DetectorGW
import concurrent.futures
from BBlack.astrotools.utility_functions import berti_pdet_fit, mc_q_to_m1_m2, parallel_array_range, clean_path, f_merg
import astropy.cosmology as cosmo
import pycbc.waveform
from itertools import chain
import numpy as np
import scipy.stats
import pandas as pd
import os
import json
import BBlack.astrotools.AstroModel as AM
import BBlack.GWtools.detector as Detector
import BBlack.GWtools.gw_event as GWE

params = json.load(open('Run/Params.json', 'r'))


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
        detector = Detector.DetectorGW(detector_name, params['event_selection']['runs_param'][obs]['delta_freq'])

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


class BayesModel:

    def __init__(self, name, astro_model, observing_run_name, detector, variation, read_match=False,
                 read_eff=False, event_list=None):
        """Creates an instance of BayesModel using an astrophysical model, an observing run and a GW detector object.

        Parameters
        ----------
        astro_model : AstroModel object
            Astrophysical model considered
        observing_run : ObservingRun object
            Observing run considered
        detector : DetectorGW object
            Detector considered
        read_match : bool
            If True, reads the matching model values from file (default = False)
        read_eff : bool
            If True, reads the detection efficiency of the model from file (default = False)
        path_dir_int : str
            Name of the directory where the match of the model can be found
        path_dir_eff : str
            Name of the directory where the detection efficiency of the model can be found
        """
        self.name = name
        # Check that the astro model is loaded with the good parameters
        if (not os.path.exists('Run/' + params['name_of_project_folder'] + '/' + self.name + '_BM.pickle')
                or params['overwrite']['bayesian_analysis']):
            if isinstance(astro_model, AM.AstroModel):
                if not astro_model.loaded_flag["mrd"]:
                    astro_model.read_merger_rate_file()
            self.astro_model = astro_model
            #        else:
            #            raise TypeError("\nError: astro_model is not of expected type.\n"
            #                            "Please pass an instance of class AstroModel() in input.")
            self.observing_run_name = observing_run_name
            self.run_params = params['event_selection']['runs_param'][observing_run_name]
            self.event_list = event_list
            self.variation = variation

            if isinstance(detector, DetectorGW):
                self.detector = detector
            else:
                raise TypeError("\nError: detector is not of expected type.\n"
                                "Please pass an instance of class DetectorGW() in input.")

            # Set directories
            self.path_dir_int = "Run/" + params['name_of_project_folder'] + "/Bayes_Models/Match_model/"
            self.path_dir_eff = "Run/" + params['name_of_project_folder'] + "/Bayes_Models/Efficiency/"
            self.file_name_match = (self.path_dir_int + self.astro_model.name + "_" + self.observing_run_name + "_" +
                                    self.variation + ".dat")
            self.file_name_efficiency = (self.path_dir_eff + self.astro_model.name + "_" + self.observing_run_name + "_"
                                         + self.variation + ".dat")

            # Number of sources in detector-frame (integrate merger rate in detector frame on the redshift range).
            # Careful, this assumes that the merger rate file only contains values for the range of redshift considered
            # in the Bayesian analysis.
            self.n_sources = self.run_params['duration'] * np.sum(self.astro_model.data_mrd['mr_df']) \
                             * round(self.astro_model.data_mrd['z'][1] - self.astro_model.data_mrd['z'][0], 5)

        else:
            self.load()
        # Get the values for model's match with events
        self.match_model = {}

        if read_match:
            self.read_match_model()

        # Get the values for model's efficiency
        self.efficiency = None
        self.n_det = None

        if read_eff:
            self.read_efficiency()
            self.n_det = self.n_sources * self.efficiency

    def read_efficiency(self):
        """Reads the value of the detection efficiency of the model from a file.
        """

        # Check that the file exists at the path given
        if not os.path.isfile(self.file_name_efficiency):
            raise FileNotFoundError(f"The efficiency file was not found in {self.file_name_efficiency}")

        # Read the efficiency from file (single value)
        self.efficiency = np.loadtxt(self.file_name_efficiency)

    def read_match_model(self):
        """Read the model matching for all the events of the observing run.
        """

        # Check that the file exists at the given path
        if not os.path.isfile(self.file_name_match):
            raise FileNotFoundError(f"The match model file was not found in {self.file_name_match}")

        # Read the match values for each event
        self.match_model = pd.read_csv(self.file_name_match, index_col=None, sep = '\t', header = None)
        #with open(self.file_name_match) as filein:
        #    for line in filein:
        #        line.strip("\n")
        #        (key, val) = line.split("\t")
        #        if key in self.event_list:
        #            self.match_model[key] = float(val)

    def compute_snr(self, args):
        """Compute the optimal SNR for an ensemble of binaries sampled from the catalog.

        Parameters
        ----------
        args : tuple
            Tuple of values containing the data and approximant to use

        Returns
        -------
        opt_snr : numpy array
            Values of the optimal SNR for the set of input binaries
        """

        # Unpack arguments
        data_sample, approximant = args

        # Compute luminosity distance in Mpc
        if "ld" not in data_sample:
            data_sample["ld"] = cosmo.Planck15.luminosity_distance(data_sample["z"])

        # If (m1,m2) is not present, compute them from Mc and q
        if "m1" or "m2" not in data_sample:
            data_sample["m1"], data_sample["m2"] = mc_q_to_m1_m2(data_sample["Mc"], data_sample["q"])

        # Set some detector parameters
        delta_f = self.detector.delta_freq
        low_freq = self.detector.low_freq
        high_freq = self.detector.high_freq

        # Compute the optimal SNR
        opt_snr = np.zeros(len(data_sample)) + 1.e-10
        # Check if the merger frequency occurs in the LVK band
        f_merger = f_merg(data_sample['m1'], data_sample['m2'], 0, data_sample['z'])
        subdata = data_sample[f_merger > low_freq]
        opt_snr = [
            pycbc.filter.matchedfilter.sigma(pycbc.waveform.get_fd_waveform(approximant=approximant,
                                                                            mass1=m1 * (1. + z), mass2=m2 * (1. + z),
                                                                            spin1x=0., spin1y=0., spin1z=0.,
                                                                            spin2x=0., spin2y=0., spin2z=0.,
                                                                            delta_f=delta_f, f_lower=low_freq,
                                                                            distance=ld,
                                                                            inclination=0., f_ref=20.)[0],
                                             psd=self.detector.psd_data,
                                             low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
            for m1, m2, ld, z in zip(subdata["m1"], subdata["m2"], subdata["ld"], subdata["z"])]
        return opt_snr

    def compute_model_efficiency(self, name_file_samples, n_cpu=4, rho_thr=8.0, approximant=None):
        """Function that computes the model efficiency. It requres to have pre-generated a set of samples that are
        reprsentative of the astrophysical model associated with the model.

        Parameters
        ----------
        name_file_samples : str
            Name of the file containing the parameters of the binaries sampled from the astro model
        n_cpu : int
            Number of CPU to use
        rho_thr : float
            Value of the SNR threshold to use to compute w
        approximant : str
            Name of the waveform approximant to use

        Returns
        -------
        mean_det_prob : float
            Model detection efficiency
        """
        if not os.path.exists(self.file_name_efficiency):
            # Set the waveform approximant
            if approximant is not None:
                if approximant not in pycbc.waveform.fd_approximants():
                    raise ValueError(f"The approximant is not valid and must be taken "
                                     f"from {pycbc.waveform.fd_approximants()}")
            else:
                approximant = "IMRPhenomPv2"

            # Read the samples from the input file
            df = pd.read_csv(
                'Run/' + params['name_of_project_folder'] + '/Samples/' + self.astro_model.sample_file_name,
                delimiter="\t")
            length = len(df.z)

            # Get the division limit for data
            ranges = parallel_array_range(length, n_cpu)
            args = ((df[r[0]:r[1]], approximant) for r in ranges)

            # Compute in parallel the values of optimal SNR associated with the input samples
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(self.compute_snr, args)

            rho_opt = np.array([x for x in chain.from_iterable(results)])
            w = rho_thr / rho_opt
            w = [x if x < 1.0 else 1.0 for x in w]

            # Implement the fit from Berti to pdet(w) and compute pdet
            pdet = berti_pdet_fit()
            kappas = pdet(w)
            mean_det_prob = np.mean(kappas)
            self.efficiency = mean_det_prob
            # Write the value of the detection efficiency in a file
            with open(self.file_name_efficiency, "w") as fileout:
                fileout.write(str(mean_det_prob) + "\n")
        else:
            self.read_efficiency()
        return self.efficiency

    def model_matching_one_cpu(self, args):
        """This function is called by model_matching to run in parallel the model match computation

        Parameters
        ----------
        args : tuple
            Tuple of values containing the ranges (tuples) and bandwidth method to use for KDE

        Returns
        -------
        int_event : dict
            Dictionary that associates one GW event with the integral match of the event with the model
        """

        # Unpack arguments
        ranges, bw_method = args

        # Initialize the output
        int_event = {}

        # Generate model KDE
        kde_model = scipy.stats.gaussian_kde(
            np.array(
                [self.astro_model.data_cat[x] for x in params['observable_variation'][self.variation]['observables']]),
            bw_method=bw_method)

        # Get the list of events associated with this CPU
        list_events = self.event_list[ranges[0]:ranges[1]]

        # Loop over the list of events
        for event_name in list_events:
            # Get GW event posterior and prior data from LVK
            event = GWE.GwEvent(name=event_name)
            data_post = event.data_post
            data_prior = event.data_prior

            # Generate prior KDE
            kde_prior = scipy.stats.gaussian_kde(np.array([data_prior[x]
                                                           for x in params['observable_variation'][self.variation][
                                                               'observables']]),
                                                 bw_method="scott")

            # Compute the KDE for both the model and prior
            values_kde_prior = kde_prior(
                np.array([data_post[x] for x in params['observable_variation'][self.variation]['observables']]))
            values_kde_model = kde_model(
                np.array([data_post[x] for x in params['observable_variation'][self.variation]['observables']]))

            # Compute the integral match value
            int_event[event_name] = np.sum(values_kde_model / values_kde_prior) / len(data_post)

        return int_event

    def model_matching(self, n_cpu, bw_method):
        """This function generates the value of the match of the model with all the events in the observing run
        associated in the BayesModel.

        Parameters
        ----------
        n_cpu : int
            Number of CPU to use in the analysis
        bw_method : float
            Value for the bandwidth of the KDE to use

        Returns
        -------
        int_event : dict
            Dictionary that associates one GW event with the integral match of the event with the model
        """

        # Check that the catalog of AstroModel has been loaded
        if not self.astro_model.loaded_flag["cat"]:
            raise ValueError("Catalog data was not loaded when creating AstroModel object"
                             "Impossible to run model_matching.")

        # Create the set of arguments to pass to each CPU
        length = len(self.event_list)
        ranges = parallel_array_range(length, n_cpu)
        args = ((r, bw_method) for r in ranges)

        # Run the parallel computation
        with concurrent.futures.ProcessPoolExecutor() as executor:

            # Compute in parallel
            results = executor.map(self.model_matching_one_cpu, args)

            # Associate all the results into one dictionary
            int_event = {}
            for res in results:
                int_event.update(res)

        # Write the results in a file
        with open(self.file_name_match, "w") as fileout:
            for k, v in int_event.items():
                fileout.write(str(k) + "\t" + str(v) + "\n")

        return self.file_name_match

    def load(self):
        """try load self.name.txt"""
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_BM.pickle', 'rb')
        data_pickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(data_pickle)

    def save(self):
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_BM.pickle', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
