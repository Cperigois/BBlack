import datetime
import os
import re
import pandas as pd

from BBlack.astrotools.utility_functions import check_inputlist_with_accessible_values, clean_path


class GwEvent:

    # List of currently available parameters in the files.
    co_param = ["m1", "m2", "Mc", "Mt", "q", "chieff", "z", "chip"]

    def __init__(self, name, read_posterior=True, read_prior=True, path_posterior=None, path_prior=None,
                 event_par=None):
        """This function creates an instance of GwEvent, by setting the name and path towards data.

        Parameters
        ----------
        name_event : str
            Name of the GW event selected. It has to be teh nomenclature of the LVC.
        read_posterior : bool
            If True, read the posterior data of the event (default = True)
        read_prior : bool
            If True, read the prior data of the event (default = False)
        path_posterior : str
            If provided, sets a new path towards the posterior data. If None, use the default path. (default = None)
        path_prior : str
            If provided, sets a new path towards the prior data. If None, use the default path. (default = None)
        event_par : list of str
            If provided, selects the set of parameters provided. If None, set it to the default values.
            (default : None)
        """

        self.name = name
        self.flags_loaded = {"post": False, "prior": False}

        # Set the date for the event. It assumes that the name of the event is GW/YEAR/MONTH/DAY (LIGO nomenclature)
        match = re.match(r"GW(\d{2})(\d{2})(\d{2})", self.name)
        year, month, day = (2000 + int(match[1]), int(match[2]), int(match[3]))
        self.date_event = datetime.date(year, month, day)

        # Set the choice for the event's parameters.
        if event_par is None:
            self.event_par = self.co_param[:]
        else:
            check_inputlist_with_accessible_values(event_par, "event_par", self.co_param, "co_param")
            self.event_par = event_par

        # Set the path towards posterior data
        if path_posterior is None:
            path_posterior = "AuxiliaryFiles/LVC_data/Posterior/"

        # Set the path towards prior data
        if path_prior is None:
            path_prior = "AuxiliaryFiles/LVC_data/Prior/"

        # If selected, load the posterior data using the path
        self.data_post = None
        if read_posterior:
            self.data_post = self.read_posterior_data(path_posterior)
            self.flags_loaded["post"] = True

        # If selected, load the prior data using the path
        self.data_prior = None
        if read_prior:
            self.data_prior = self.read_prior_data(path_prior)
            self.flags_loaded["prior"] = True

    def read_posterior_data(self, path_dir="AuxiliaryFiles/LVC_data/Posterior/"):
        """This function reads and returns the posterior data for the given GW event.

        Parameters
        ----------
        path_dir : str
            Path to directory where the posterior data are.

        Returns
        -------
        data_event_post : pandas dataframe
            Dataframe containing the posterior data for the GW event.
        """

        # Set the name of the file and check for existence
        name_file = clean_path(path_dir) + self.name + "_post.dat"
        if not os.path.isfile(name_file):
            raise FileNotFoundError(f"File for posterior data not found at {name_file}")

        # Read the data
        data_event_post = pd.read_csv(path_dir + self.name + "_post.dat", delimiter="\t")
        data_event_post = data_event_post[self.event_par]
        self.flags_loaded["post"] = True

        return data_event_post

    def read_prior_data(self, path_dir="auxiliary_files/LVC_data/Prior/"):
        """This function reads and returns the posterior data for the given GW event.
        Parameters
        ----------
        path_dir : str
            Path to directory where the posterior data are.

        Returns
        -------
        data_event_prior : pandas dataframe
            Dataframe containing the prior data for the GW event.
        """

        # Set the name of the file and check for existence
        name_file = clean_path(path_dir) + self.name + "_prior.dat"
        if not os.path.isfile(name_file):
            raise FileNotFoundError(f"File for prior data not found at {name_file}")

        # Read the data and set flag to True
        data_event_prior = pd.read_csv(path_dir + self.name + "_prior.dat", delimiter="\t")
        data_event_prior = data_event_prior[self.event_par]
        self.flags_loaded["prior"] = True

        return data_event_prior

    def __str__(self):
        """Sets the appearance of the print function for the class.

        Returns
        -------
        return_string : str
            String that appears when using print() on the object
        """

        return_string = f"Event name : {self.name_event}"
        if self.flags_loaded["post"]:
            return_string = "\n".join([return_string,
                                       f"Length of posterior data : {self.data_post.shape[0]}."])
        if self.flags_loaded["prior"]:
            return_string = "\n".join([return_string,
                                       f"Length of prior data : {self.data_prior.shape[0]}."])

        return return_string

    def hist(self, var, prior=False, ax=None, bins=50, logx=False, logy=False, range_x=None,
             range_y=None, save=False, namefile=None, show=True):
        """Histogram routine for the event parameter. Either do a 1d or 2d histograms depending on inputs.

        Parameters
        ----------
        var : str or list of str
            Name of variable(s)
        prior : bool
            If True, plot prior values instead of posterior data (default = False)
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
        if prior:
            if not self.flags_loaded["prior"]:
                raise ValueError("Prior data are not loaded.")
            data = self.data_prior
        else:
            if not self.flags_loaded["post"]:
                raise ValueError("Posterior data are not loaded.")
            data = self.data_post

        if type(var) == str or (type(var) == list and len(var) == 1):  # 1d histogram
            title = None
            gf.hist_1d(data, var, ax=ax, bins=bins, title=title, logx=logx, logy=logy, range_x=range_x, save=save,
                       namefile=namefile, show=show)
        elif type(var) == list and len(var) == 2:  # 2d histograms
            title = None
            gf.hist_2d(data, var[0], var[1], ax=ax, bins=bins, title=title, logx=logx, logy=logy, range_x=range_x,
                       range_y=range_y, save=save, namefile=namefile, show=show)
        else:
            raise NotImplementedError("Option not implemented. Use corner() for such set of variables.")

    def corner(self, var_select=None, prior=False, save=False, quantiles=None):
        """Corner plot for selected parameters. It uses the package corner.py, with minimum functionnality as
        some features seem to need some fixing.

        Parameters
        ----------
        var_select : list of str
            List of variables considered for the corner plot. If None, use loaded instance variables
            (default = None)
        prior : bool
            If True, plot prior values instead of posterior data (default = False)
        save : bool
            If True, save the figure.
        quantiles : list of float
            List of quantiles that appear as lines in 1d-histograms of the corner plot.
        """

        # Load posterior or prior data
        if prior:
            if not self.flags_loaded["prior"]:
                raise ValueError("Prior data are not loaded.")
            data = self.data_prior
        else:
            if not self.flags_loaded["post"]:
                raise ValueError("Posterior data are not loaded.")
            data = self.data_post

        # Select the appropriate variables
        if var_select is not None:
            check_inputlist_with_accessible_values(var_select, "var_select", self.event_par, "event_par")
        else:
            var_select = self.event_par

        title = "CornerPlot_" + "".join(var_select) + "_" + self.name_event
        gf.corner(data, title, var_select=var_select, save=save, quantiles=quantiles)

