import os
import BBlack.astrotools.utility_functions as UF
import os.path
import pandas as pd
import numpy as np
from astropy.cosmology import Planck15
import BBlack.astrotools.auxiliary_cosmorate as auxiliary_cosmorate
import astropy.units as u
import random
import concurrent.futures
import emcee
import scipy.stats
import re
import json
import pickle
import pickletools
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

params = json.load(open('Run/Params.json', 'r'))


def initialization():
    for model in params['astro_model_list'].keys():
        astromodel = AstroModel(name=params['astro_model_list'][model]['name'],
                                path_to_MRD=params['astro_model_list'][model]['path_to_MRD'],
                                path_to_catalogs=params['astro_model_list'][model]['path_to_catalogs'])


class AstroModel:
    def __init__(self, name, path_to_catalogs=None, path_to_MRD=None, observables=params['observables'],
                 spins='InCat', duration=1, catsize_opt="Fixed"):
        """Create an instance of your model.
         Parameters
         ----------
         cat_name : str
             Name of the reshaped catalogue in the folder catalogue
         duration : int of float
             Duration of your supposed catalogue, ie your initial catalogue is showing all sources arriving on the
             detector in X years.
             (default duration = 1)
         original_cat_path : str
             Path to your original catalogue
         sep_cat: str
            Used to read your original catalogue with pandas " ", "," or "\t"
         index_column: bool
            Used to read your original catalogue with pandas. True if you have a column with indexes in your original file.
            (default = None)
         spin_option : str
            Choose an option to eventually generate the spin among {"Zeros", "Isotropic", "Dynamic"}. Default is 'Zeros'
         flags: dict
            Dictionary of a possible flag column in your catalogue (Can be used to distinguish the type of binary, the
            formation channel...)
         """

        # Set class variables
        self.name = name
        self.sample_file_name = "sampling_" + self.name + ".dat"
        self.pkl_file = 'Run/' + params['name_of_project_folder'] + '/' + self.name + '.pickle'
        if (not os.path.exists('Run/' + params['name_of_project_folder'] + '/' + self.name + '_AM.pickle')) or (
                params['overwrite']['astromodel'] == True):
            self.observables = observables
            self.spin_option = spins
            self.file_mrd = params['astro_model_list'][name]['path_to_MRD']
            self.file_mrd_output = 'Run/' + params[
                'name_of_project_folder'] + '/Astro_Models/MergerRateDensity/Mrd_' + name + '.dat'
            self.path_to_catalogs = params['astro_model_list'][name]['path_to_catalogs']
            self.file_catalogs_output = ('Run/' + params['name_of_project_folder'] +
                                         '/Astro_Models/Catalogs/_' + name + '.dat')
            self.catsize_opt = catsize_opt
            self.duration = duration
            self.creation_flag = {"cat": False, "mrd": False}
            self.loaded_flag = {"cat": False, "mrd": False}
            self.sample_file_name = "sampling_" + self.name + ".dat"
            self.process_astro_model()
            self.read_catalog_file()
            self.read_merger_rate_file()
            self.save()
        else:
            self.load()
            self.read_catalog_file()
            self.read_merger_rate_file()

    def process_astro_model(self):
        # -------------------------------------      Main code       ---------------------------------------------------
        # Make sure directories are created
        if not os.path.exists('Run/' + params['name_of_project_folder']):
            os.mkdir('Run/' + params['name_of_project_folder'])
        if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/"):
            os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/")
        if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs/"):
            os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs")
        if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/MergerRateDensity/"):
            os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/MergerRateDensity")

        # CosmoRate processing
        auxiliary_cosmorate.process_cosmorate(path_dir_cr=self.path_to_catalogs)

        # Create the merger rate file
        self.create_merger_rate_file(range_z=params['redshift_range'], delimiter="\t")

        # Create the merger rate file
        # model_astro.create_merger_rate_file(dir_cosmorate=dir_cosmo_rate, range_z=range_z)
        if self.catsize_opt == "Fixed":
            self.catsize = params['catalog_size']
        else:
            self.catsize = self.sources_in_tobs_time(self.duration)
        # Create the catalog
        self.create_catalog_file()

    def read_merger_rate_file(self, delimiter='\t'):
        """Read the merger rate file and stores data.

        Parameters
        ----------
        dir_mrd : str
            If specified, function will look in this directory to read merger rate file, instead of reading in
            path set during instanciation (default = None)
        delimiter : str
            Delimiter used to separate columns in merger rate file (default = "\t")
        """

        # Check file existence
        if not os.path.isfile(self.file_mrd_output):
            raise FileNotFoundError("\nThe file for the merger rate density of the model could not be found.\n"
                                    "\t 1) If the merger rate file was not created, run method "
                                    "create_merger_rate_file()\n"
                                    "\t 2) If the merger rate file was created, check that the file "
                                    "is located at {}".format(self.file_mrd_output))
        # Read file
        data_mrd = pd.read_csv(self.file_mrd_output, sep=delimiter, index_col=None)

        # Update instance variables
        self.loaded_flag["mrd"] = True
        self.data_mrd = data_mrd

    def create_merger_rate_file(self, range_z, delimiter="\t"):
        """Create the merger rate file using the information provided by CosmoRate. In current version, the file
        has only with two columns : redshift and merger rate density source frame; and the file is found by
        matching the regex 'MRD'.
        To compute the merger rate, we currently make use of Planck15 cosmology.

        Parameters
        ----------
        dir_cosmorate : str
            Path towards the location of the merger rate file of CosmoRate
        range_z : float
            Maximum range of redshift for the model. Must be a multiple of CosmoRate redshift division
        delimiter : str
            Delimiter used to separate columns in merger rate file (default = "\t")
        """

        if not os.path.isfile(self.file_mrd):
            raise FileNotFoundError("\nThe source file for the merger rate density of CosmoRate could not be found.\n"
                                    "Check that the file is in {} and that there is only one file containing the "
                                    "string 'MRD'".format(self.file_mrd))

        # Read file
        data_original = np.loadtxt(self.file_mrd, skiprows=1)

        # Check that range_z is a multiple of deltaz of cosmoRate
        redshift = data_original[:, 0]
        delta_z = round(redshift[1] - redshift[0], 5)
        if not (range_z / delta_z).is_integer():
            raise ValueError("Error: the redshift range {} is not a multiple of the redshift interval {} from "
                             "CosmoRate.".format(range_z, delta_z))

        # Compute merger rate density and merger rate in detector-frame. Planck 15 cosmology is used here
        mrd_source_frame = data_original[:, 1]
        mrd_detector_frame = np.array([mrd_s * (1.0 / (1.0 + z)) for mrd_s, z in zip(mrd_source_frame, redshift)])
        dvc_dz = np.array([4. * np.pi * Planck15.differential_comoving_volume(z).to(u.Gpc ** 3 / u.sr).value
                           for z in redshift])
        mr_detector_frame = np.array([dvc * mr_df for dvc, mr_df in zip(dvc_dz, mrd_detector_frame)])

        # Create and write file.
        with open(self.file_mrd_output, "w") as fileout:
            fileout.write(delimiter.join(["z", "mrd_sf", "mrd_df", "mr_df"]) + "\n")
            for z, mrd_sf, mrd_df, mr_df in zip(redshift, mrd_source_frame, mrd_detector_frame, mr_detector_frame):
                if z <= range_z:
                    fileout.write("{0:.4f} {4} {1:.4f} {4} {2:.4f} {4} {3:.4f} \n"
                                  "".format(z, mrd_sf, mrd_df, mr_df, delimiter))
                else:
                    return

    def read_catalog_file(self, delimiter="\t"):
        """Read the catalog file and stores data.

        Parameters
        ----------
        dir_cat : str
            If specified, function will look in this directory to read catalog file, instead of reading in path
            set during instanciation (default = None)
        delimiter : str
            Delimiter used to separate columns in catalog file (default = "\t")
        """

        # Set path and namefiles
        # if dir_cat is not None:
        #    path = clean_path(dir_cat)
        # else:
        #    path = self.dir_cat
        # file_cat = self.cat_file

        # Check file existence
        if not os.path.isfile(self.file_catalogs_output):
            raise FileNotFoundError("\nThe file for the model's catalog could not be found.\n"
                                    "\t 1) If the catalog was not created, run method "
                                    "create_catalog_file()\n"
                                    "\t 2) If the catalogfile was created, check that the file "
                                    "is located at {}".format(self.file_catalogs_output))

        # Read catalog
        df = pd.read_csv(self.file_catalogs_output, delimiter=delimiter)
        if any(x not in df.columns for x in self.observables):
            raise KeyError("One of the parameter you are trying to access is not present in the catalog. This results "
                           "from the fact that create_catalog_file was run with a different set of parameters.\n. "
                           "Please re-run create_catalog_file() with the set {}".format(self.observables))
        data_cat = df[self.observables]

        # Update instance variables
        self.loaded_flag["cat"] = True
        # self.dir_cat = path
        self.data_cat = data_cat

    def create_catalog_file(self, delimiter="\t", input_catname_beg='all_vcm_', input_catname_end='_50.dat'):
        """Create the catalog file using the information from CosmoRate and previously created merger rate
        density file.
        In current version, CosmoRate files have the name "identifier_file + "_" + str(i) + "_50.dat" where i
        refers to each redshift bin. It this changes, the code below needs to be updated.

        Parameters
        ----------
        dir_cosmorate : str
            Path towards the location of the catalog file of CosmoRate
        num_cat : int
            Number of sources wanted for the catalog
        frac_hier : float
            Fraction of hierarchical mergers (used for specific spin model)
        overwrite : bool
            If set to True, erase and rewrite the catalog if the catalog already existed (default = False)
        delimiter : str
            Delimiter used to separate columns in catalog file (default = "\t")
        """

        # Set name of catalog fileparams['name_of_project_folder']+'/Astro_Models/MergerRateDensity/Mrd_'+self.name+'.dat'
        self.file_catalogs_output = ('Run/' + params['name_of_project_folder'] +
                                     '/Astro_Models/Catalogs/Catalog_' + self.name + '.dat')

        # Check if CosmoRate files were processed before by looking at log-file
        # dir_cosmorate = clean_path(dir_cosmorate)
        # log_file = LogFileCR(dir_cosmorate)
        # if not log_file.status["header_rewritten"]:
        #    raise ValueError("Ran CosmoRate pre-processing routines before creating catalog file.")

        # Read merger file if it was not done before
        self.read_merger_rate_file()

        # Read merger rate, and create a cumulative sum from it. Then randomly generate points in [0,1] and use
        # cdf values as bins. The number that fall in each bin, is then the number of sources associated with
        # the redshift bin
        mr_df = self.data_mrd['mr_df']

        cdf = np.append(0.0, np.cumsum(mr_df / mr_df.sum()))
        counts, bin_edges = np.histogram(np.random.rand(self.catsize), bins=cdf)
        # Initiate dataframe that will contains catalog values
        df_final = pd.DataFrame(columns=self.observables)

        # Get the names of catalog files from CosmoRate
        dir_catfile = self.path_to_catalogs + "catalogs/"
        n = os.listdir(dir_catfile)
        regex = re.search('([A-Za-z0-9_]*_)\d+(_[A-Za-z0-9]*.dat)', n[0])
        print("*******  START : CATALOG CREATION  *******")

        # Loop over redshift bins
        for i, c in enumerate(counts[:len(counts) - 1]):

            # Read CosmoRate catalog. So far the name of catalog files is  "identifier_file_i_50.dat"
            # If CosmoRate changes, updates this part too.

            cat_source_name = dir_catfile + regex.group(1) + str(i + 1) + regex.group(2)
            df = pd.read_csv(cat_source_name, delimiter="\t")
            df.rename(columns=params['AM_params']['input_parameters'], inplace=True)

            # Map to Mc, q if they are selected as parameters
            if "Mc" in self.observables or "q" in self.observables:
                df["Mc"], df["q"] = UF.m1_m2_to_mc_q(df["m1"], df["m2"])

            if "m1" in self.observables or "m2" in self.observables:
                df["m1"], df["m2"] = UF.mc_q_to_m1_m2(df["Mc"], df["q"])

            if "Mt" in self.observables and "Mt" not in df.columns:
                df["Mt"] = df["m1"] + df["m2"]

            if self.spin_option != 'InCat':
                df['chi1'], df['costheta1'], df['chi2'], df['costheta2'] = self.generate_spins()

            else:
                if 'costheta1' not in df.columns:
                    df['costheta1'] = np.cos(df['theta1'])
                    df['costheta2'] = np.cos(df['theta2'])

            # Compute chi_effective

            if "chieff" in self.observables:
                df['chieff'] = (df['chi1'] * df['costheta1'] * df["m1"] + df['chi2'] * df['costheta2'] * df["m2"]) / (
                        df["m1"] + df["m2"])

            # Compute luminosity distance if not already in CosmoRate
            # if "Dl" in self.observables and "Dl" not in df.columns:
            #    df["Dl"] = Cosmo.luminosity_distance(df["z"]).value

            if "chip" in self.observables and "chip" not in df.columns:
                df["chip"] = self.comp_chip(m1=df["m1"], m2=df["m2"], cos_theta_1=df["costheta1"],
                                            cos_theta_2=df['costheta2'], chi1=df['chi1'],
                                            chi2=df['chi2'])
            #  Select only relevant parameters
            df = df[self.observables]
            # Concatenate dataframe with correct number of elements
            for _ in range(int(c / len(df))):
                df_final = pd.concat([df_final, df])
            ind_list = random.sample(range(len(df)), c % len(df))
            df_final = pd.concat([df_final, df.iloc[ind_list]])

        # Write dataframe to file
        df_final.to_csv(self.file_catalogs_output, sep=delimiter, index=False, float_format="%.4f")
        print("*******  END : CATALOG CREATION  *******")

    def sample_catalog(self, n_walkers, n_chain, bw_method, n_cpu):
        """Routine to generate samples from the catalogs of the astrophysical model. In particular, this is used
        to compute the efficiency of the model (or VT).
        The samples are generated using MCMC with emcee. In current version, the entire MCMC chain is exported
        as samples, implying some correlation left in the samples.

        Parameters
        ----------
        n_walkers : int
            Number of walkers for the MCMC chain, must be superior to twice the dimension of the problem at hand.
        n_chain : int
            Length of the chain generated.
        bw_method : float or str
            Bandwidth method used for kernel density estimation.
        n_cpu : int
            Number of CPU selected to run the routine.

        Returns
        -------
        samples : 'pandas dataframe'
            Samples generated by the routine
        """

        # Create arguments to be set to multiple CPUs
        args = ((n_walkers, n_chain, bw_method, cpu) for cpu in range(n_cpu))

        # Print a warning that user should dismiss the warning error from emcee
        print("WARNING : Do not pay attention of the warning from the infinite with the "
              "log. This is due to emcee, and is not affecting the end computation.")
        # Main function
        if n_cpu > 1:  # Parallel option

            # Parallel computation
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(self.sample_catalog_one_cpu, args)

            # Concatenate the samples from the various CPUs
            samples_final = pd.DataFrame()
            for df in results:
                if df is not None:
                    samples_final = pd.concat([samples_final, df], ignore_index=True)
            samples_final = samples_final.reset_index(drop=True)

        else:  # Single CPU
            samples_final = self.sample_catalog_one_cpu(*args)

        return samples_final

    def sample_catalog_one_cpu(self, args):
        """This function generates samples for one CPU.

        Parameters
        ----------
        args : tuple
            This tuple contains the same argument than the ones of sample_catalog. This tuple format is used to
            make concurrent futures work.

        Returns
        -------
        samples : pandas dataframe
            Samples generated by this CPU.
        """

        n_walkers, n_chain, bw_method, cpu = args
        n_dim = len(self.observables)
        print(f"cpu {cpu}, args {args}")

        # Different seeds for each CPU
        np.random.seed()

        # Compute KDE
        kde = scipy.stats.gaussian_kde(np.array([self.data_cat[x] for x in self.observables]),
                                       bw_method=bw_method)

        # No idea why, but log_prob does not work on scighera
        def log_prob(x, kde_in):
            return np.log(kde_in(x))

        # Draw a random point in the range set by the 99% credible interval
        min_quantile = np.array(self.data_cat.quantile(0.005))
        max_quantile = np.array(self.data_cat.quantile(0.995))
        p0 = min_quantile + (max_quantile - min_quantile) * np.random.rand(n_walkers, n_dim)

        # Define sampler
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=[kde])

        # Burn-in
        state_ini = sampler.run_mcmc(p0, 50)
        sampler.reset()

        # Main chain
        sampler.run_mcmc(state_ini, n_chain, progress=True)

        # Flatten and restrict chains in the min/max range
        min_np = np.array(self.data_cat.min())
        max_np = np.array(self.data_cat.max())
        samples = UF.flatten_restrict_range_output_emcee(sampler, self.observables, min_np, max_np)
        return samples

    def hist(self, var, ax=None, bins=50, logx=False, logy=False, range_x=None, range_y=None,
             save=False, namefile=None, show=True):
        """Histogram routine for the event parameter. Either do a 1d or 2d histograms depending on inputs.

        Parameters
        ----------
        var : str or list of str
            Name of variable(s)
        ax : matplotlib.axes.Axes object
            If specified, use the axis to plot the figure (multiple plots). If None, create a new figure
            (default = None)
        bins : int
            Number of bins to use for the plot (default = 50)
        logx : bool
            If True, set the x-axis logarithmic (default = False)
        logy : bool
            If True, set the y-axis logarithmic (default = False)
        range_x : tuple
            If specified, use this range for y-axis. (default = None)
        range_y : tuple
            If specified, use this range for y-axis in the 2d case. Need to also set range_x at the same time
            (default = None)
        save : bool
            If True, save the figure  (default = False)
        namefile: str
            Name of the file to save if save was set to true (default = None)
        show : bool
            If true, disply the graph (default = True)
        """

        # Load posterior or prior data
        if not self.loaded_flag["cat"]:
            raise ValueError("Catalog data are not loaded")

        if type(var) == str:  # 1d histogram
            title = None
            gf.hist_1d(self.data_cat, var, ax=ax, bins=bins, title=title, logx=logx, logy=logy,
                       range_x=range_x, save=save, namefile=namefile, show=show)
        elif type(var) == list and len(var) == 1:  # 1d histogram
            title = None
            gf.hist_1d(self.data_cat, var[0], ax=ax, bins=bins, title=title, logx=logx, logy=logy,
                       range_x=range_x, save=save, namefile=namefile, show=show)
        elif type(var) == list and len(var) == 2:  # 2d histograms
            title = None
            gf.hist_2d(self.data_cat, var[0], var[1], ax=ax, bins=bins, title=title, logx=logx, logy=logy,
                       range_x=range_x, range_y=range_y, save=save, namefile=namefile, show=show)
        else:
            raise NotImplementedError("Option not implemented. Use corner() for such set of variables.")

    def corner(self, var_select=None, save=False, quantiles=None):
        """Corner plot for catalog variable. It uses the package corner.py, with minimum functionnality as
        some features seem to need some fixing.

        Parameters
        ----------
        var_select : list of str
            List of variables considered for the corner plot. If None, use loaded instance variables
            (default = None)
        save : bool
            If True, save the figure.
        quantiles : list of float
            List of quantiles that appear as lines in 1d-histograms of the corner plot.
        """

        if not self.loaded_flag["cat"]:
            raise ValueError("Catalog data are not loaded.")
        data = self.data_cat

        # Select the appropriate variables
        if var_select is not None:
            check_inputlist_with_accessible_values(var_select, "var_select", self.co_parameters, "event_par")
        else:
            var_select = self.co_parameters

        title = "CornerPlot_" + "".join(var_select) + "_" + self.name_model
        gf.corner(data, title, var_select=var_select, save=save, quantiles=quantiles)

    def check_sample_hist(self, name_file_samples, var, ax=None, range_x=None, logy=False):

        # Check that catalog is indeed loaded
        if not self.loaded_flag["cat"]:
            raise ValueError("Catalog data are not loaded.")

        # Read samples
        if os.path.isfile(name_file_samples):
            raise FileNotFoundError(f"File {name_file_samples} could not be found")
        data_sample = pd.read_csv(name_file_samples, delimiter="\t")

        # Set axis for the plot
        if ax is None:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
        else:
            if not isinstance(ax, Axes):
                raise TypeError("ax must be a Matplotlib Axes object.")

        # Check variable is accessible in catalog and sample file
        if var not in self.data_cat.columns:
            raise KeyError(f"{var} not in catalog files.")
        if var not in data_sample.columns:
            raise KeyError(f"{var} not in samples.")

        # Do the plot
        self.hist(var=var, ax=ax, bins=50, range_x=range_x, show=False, logy=logy)
        ax.hist(data_sample[var], density=True, lw=3, histtype="step", bins=50, range=range_x)

        # Set the legend
        ax.legend(["Model", "Sample"], fontsize=20)

    def sources_in_tobs_time(self, tobs):
        """This function computes the predicted number of observations for the model in a given observation time.

        Parameters
        ----------
        tobs : float
            Observation time in year.

        Returns
        -------
        n_sources : float
            Number of sources predicted by the model in an obsevation time of tobs years.
        """

        # Check that observation time is a float
        # if type(tobs) != float:
        #    raise TypeError("Observation time tobs must be a float.")
        tobs = np.rint(tobs).astype(int)
        # Check that the merger rate density is loaded
        if not self.loaded_flag["mrd"]:
            self.read_merger_rate_file()

        # Compute number of sources. Be careful, the range of redshift used will be the one in the merger rate density
        # file
        # print(self.data_mrd['mr_df'], ' ',self.data_mrd['z'][1], ' ',  self.data_mrd['z'][0])
        # n_sources = tobs * np.sum(self.data_mrd['mr_df']) * round(self.data_mrd['z'][1] - self.data_mrd['z'][0], 5)
        n_sources = tobs * np.sum(self.data_mrd['mr_df']) * round(0.1500 - 0.0500, 5)

        return n_sources

    def generate_spins(self):

        if self.spin_option == 'Rand_Isotropic':
            sigmaSpin = 0.1
            v1_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            v2_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            v3_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            V1 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            v1_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            v2_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            v3_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            V2 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            costheta1 = np.ones(self.catsize)
            costheta2 = np.ones(self.catsize)

        elif self.spin_option == 'Rand_Dynamics':
            sigmaSpin = 0.1
            v1_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            v2_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            v3_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            V1 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            v1_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            v2_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            v3_L = np.random.normal(0.0, sigmaSpin, size=self.catsize)
            V2 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            costheta1 = 2. * np.random.uniform(0.0, 1.0, size=self.catsize) - 1.0
            costheta2 = 2. * np.random.uniform(0.0, 1.0, size=self.catsize) - 1.0

        elif self.spin_option == 'Zeros':
            V1 = np.zeros(self.catsize)
            V2 = np.zeros(self.catsize)
            costheta1 = np.zeros(self.catsize)
            costheta2 = np.zeros(self.catsize)
        else:
            print(
                'Choose a spin model (Rand_Isotropic or Rand_Dynamics) or specify InCat if your catalogue already contain the spins.')
        return V1, costheta1, V2, costheta2

    def comp_chieff(m1, m2, chi1, chi2, cos_theta_1, cos_theta_2):
        """This function computes the chi effective for source(s).

        Parameters
        ----------
        m1 : float or numpy array
            Mass of the 1st binary component
        m2 : float or numpy array
            Mass of the 2nd binary component
        chi1 : float or numpy array
            Spin magnitude of the 1st binary component
        chi2 : float or numpy_array
            Spin magnitude of the 2nd binary component
        cos_theta_1 : float or numpy array
            Cosine angle betwen angular momentum and spin of 1st component
        cos_theta_2 : float or numpy array
            Cosine angle betwen angular momentum and spin of 2nd component

        Returns
        -------
        chieff : float or numpy array
            Chi effective
        """

        chieff = (chi1 * cos_theta_1 * m1 + chi2 * cos_theta_2 * m2) / (m1 + m2)

        return chieff

    def comp_chip(self, m1, m2, chi1, chi2, cos_theta_1, cos_theta_2):
        """This function computes the chi effective for source(s).

        Parameters
        ----------
        m1 : float or numpy array
            Mass of the 1st binary component
        m2 : float or numpy array
            Mass of the 2nd binary component
        chi1 : float or numpy array
            Spin magnitude of the 1st binary component
        chi2 : float or numpy_array
            Spin magnitude of the 2nd binary component
        cos_theta_1 : float or numpy array
            Cosine angle betwen angular momentum and spin of 1st component
        cos_theta_2 : float or numpy array
            Cosine angle betwen angular momentum and spin of 2nd component

        Returns
        -------
        chip : float or numpy array
            Chi precessing
        """

        chip1 = (2. + (3. * m2) / (2. * m1)) * chi1 * m1 * m1 * (1. - cos_theta_1 * cos_theta_1) ** 0.5
        chip2 = (2. + (3. * m1) / (2. * m2)) * chi2 * m2 * m2 * (1. - cos_theta_2 * cos_theta_2) ** 0.5
        chipmax = np.maximum(chip1, chip2)
        chip = chipmax / ((2. + (3. * m2) / (2. * m1)) * m1 * m1)
        return chip

    def generate_samples(self):

        if (not os.path.exists("Run/" + params['name_of_project_folder'] + "/Samples/" + self.sample_file_name) or
                params['overwrite']['samples']):
            num_samples = params['sampling_params']['size']  # 100 #10000  # number of samples wanted
            n_cpu = params['n_cpu_max']  # number of CPUs
            n_walkers = params['sampling_params']['number_of_walkers']  # number of MCMC walkers
            n_chain = params['sampling_params']['chain_length']  # 50 #500  # length of MCMC chain
            bw_method = params['sampling_params']['bandwidth_KDE']  # KDE bandwidth to use

            # Generate num_samples from the astro_model using MCMC
            samples_final = pd.DataFrame()
            n = 0
            while n < num_samples:
                samples = self.sample_catalog(n_walkers=n_walkers, n_chain=n_chain,
                                              bw_method=bw_method, n_cpu=n_cpu)
                samples_final = pd.concat([samples_final, samples], ignore_index=True)
                n = len(samples_final)

            # Make sure directories for samples are created
            if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Samples/"):
                os.mkdir('Run/' + params['name_of_project_folder'] + "/Samples/")

            # Export the samples to a file
            samples_final.to_csv("Run/" + params['name_of_project_folder'] + "/Samples/" + self.sample_file_name,
                                 sep="\t", index=False, float_format="%.4f")
        self.save()

    def load(self):
        """try load self.name.txt"""
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_AM.pickle', 'rb')
        data_pickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(data_pickle)

    def save(self):
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_AM.pickle', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
