import matplotlib.pyplot as plt
import pycbc.psd
import os
import numpy as np
from BBlack.astrotools.utility_functions import clean_path
from decimal import Decimal
import BBlack.Run.advanced_params as params


class DetectorGW:



    def __init__(self, name, delta_freq=None):
        """Function that creates an instance of a detector object. In particular, it sets the value for the PSD of the
        detector

        Parameters
        ----------
        name : str
            Name of selected detector
        delta_freq : float
            Value of frequency interval chosen
        low_freq : float
            Minimum frequency of the PSD
        high_freq : float
            Maximum frequency of the PSD
        path_dir_psd : str
            Path towards the PSD file if the PSD is imported from a file
        """

        if name not in param.detectors_avail:
            raise ValueError("Select a detector in list {}".format(self.detectors_avail))

        # Assign name and get psd attributes from dictionary psd_attributes
        self.name = name
        self.psd_name = param.psd_attributes[self.name]["psd_name"]
        self.in_pycbc = param.psd_attributes[self.name]["in_pycbc"]
        self.min_freq = param.psd_attributes[self.name]["min_freq"]
        self.max_freq = param.psd_attributes[self.name]["max_freq"]
        self.delta_freq_min = param.psd_attributes[self.name]["delta_freq_min"]

        # Set the frequency interval
        if delta_freq is not None:
            self.check_data_freq(delta_freq)
            self.delta_freq = delta_freq
        else:
            self.delta_freq = 0.025

        # Set the low frequency cutoff.
        if low_freq < self.min_freq:
            raise ValueError(f"low_freq {low_freq} is inferior to minimum frequency {self.min_freq}")
        if Decimal(str(low_freq)) % Decimal(str(self.delta_freq)) != Decimal('0.0'):
            raise ValueError(f"low_freq {low_freq} is not a multiple of delta_freq {self.delta_freq}.")
        self.low_freq = low_freq

        # Set the high frequency cutoff
        if high_freq > self.max_freq:
            raise ValueError(f"high_freq {high_freq} is superior to maximum frequency {self.max_freq}")
        if high_freq <= self.low_freq:
            raise ValueError(f"high_freq {high_freq} is inferior to low_freq {self.low_freq}")
        if Decimal(str(high_freq)) % Decimal(str(self.delta_freq)) != Decimal('0.0'):
            raise ValueError(f"high_freq {high_freq} is not a multiple of delta_freq {self.delta_freq}.")
        self.high_freq = high_freq

        # Compute the length using max_freq. Note that the psd generated also contain values that are inferior to
        # self.low_freq. However, PyCBC will ignore values below the low_frequency cutoff when doing SNR computation.
        self.length = int(self.high_freq / self.delta_freq) + 1

        # Read psd data
        self.psd_data = self.read_psd_data(path_dir_psd=path_dir_psd)

    def check_data_freq(self, delta_freq):
        """This function checks that the value given for delta_freq is valid.

        Parameters
        ----------
        delta_freq : float
            Value of delta_freq to check
        """

        # Raises an error if delta_freq is inferior to the minimum
        if delta_freq < self.delta_freq_min:
            raise ValueError(f"delta_freq {delta_freq} is inferior to the minimum {self.delta_freq_min} for "
                             f"psd {self.psd_name}.")

        # If the PSD is read from a file, the value for delta_freq has to be a multiple of the smallest
        # delta_freq of the file
        if not self.in_pycbc:
            if Decimal(str(delta_freq)) % Decimal(str(self.delta_freq_min)) != Decimal('0.0'):
                raise ValueError(f"delta_freq {delta_freq} must be a mutliple of {self.delta_freq_min} for "
                                 f"psd {self.psd_name}.")

    def read_psd_data(self, path_dir_psd=None):
        """This function sets the values for the PSE on the interval of frequency chosen.

        Parameters
        ----------
        path_dir_psd : str
            Path towards the PSD file if the PSD is imported from a file
        """

        if self.in_pycbc:
            # Read psd file
            psd_data = pycbc.psd.from_string(psd_name=self.psd_name, length=self.length, delta_f=self.delta_freq,
                                  low_freq_cutoff=self.low_freq)
        else:
            # Set the path towards psd files
            if path_dir_psd is None:
                path_dir_psd = "auxiliary_files/PSDs/"
            else:
                path_dir_psd = clean_path(path_dir_psd)

            # Check that file exists
            namefile = path_dir_psd + self.psd_name + ".dat"
            if not os.path.isfile(namefile):
                raise FileNotFoundError(f"Psd file was not found at {namefile}")

            # Read psd file
            psd_data = pycbc.psd.read.from_txt(filename=namefile, length=self.length, delta_f=self.delta_freq,
                                               low_freq_cutoff=self.low_freq, is_asd_file=False)

        return psd_data

    def plot_psd(self):
        """This function displays the values of the psd.
        """

        # Create the array of frequency
        frequency = np.arange(0.0, self.high_freq+self.delta_freq, self.delta_freq)

        # Create the figure and the plot
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.set_yscale("log")
        ax.plot(frequency, self.psd_data, lw=3)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD detector {}".format(self.name))

        # Show the plot
        plt.show()