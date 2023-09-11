import datetime
"""
This file contain all the parameters to be used for the analysis.
It's divides in three main parts respectively dedicated to the 3 main packages : \
            AstroModel, BaesianAnalysis, GravitationalWave
"""

"""             *** ASTROMODEL ***           """

"""
Parameters of the user input catalogue. Please do not change the left column.
{<Input catalogue names> : <Names for output catalogues>}
"""

input_parameters = {
    "m1": "m1",   # mass of compact object 1
    "mrem1" : "m1",
    "Mass1" : "m1",

    "m2": "m2",   # mass of compact object 2
    "mrem2": "m2",
    "Mass2": "m2",

    "chi1": "chi1",   # spin magnitude of compact object 1
    "th1": "theta1",   # angle between angular momentum and spin for compact object 1
    "cos_nu1": "costheta1",   # cosine of tilt 1st supernova
    "cmu1" : "costheta1",   # cosine of tilt 1st supernova

    "chi2": "chi2",   # spin magnitude of compact object 2
    "th2": "theta2",   # angle between angular momentum and spin for compact object 2
    "cos_nu2": "costheta2",   # cosine of tilt 2nd supernova
    "cmu2" : "costheta2",   # cosine of tilt 2nd supernova

    "z_merg": "z",   # redshift at merger
    "z_form": "zForm",   # redshift at merger

    "ID": "id",   # Id of the binary
    "time_delay [yr]": "timeDelay",   # time delay
    "mZAMS1" : "mzams1",   # Zero Age Main Sequence mass of the 1st component
    "mZAMS2" : "mzams2",   # Zero Age Main Sequence mass of the 2nd component
    "M_SC" : "m_sc",   # mass of star cluster
    "Z progenitor" : "z_progenitor",
    "Channel": "Channel",
    "Ng" : "Ng",
    "tSN1" : "tsn1",
    "tSN2" : "tsn2"
    }

CompactObjectParametersAvailable = ['m1', 'm2', 'Mc', 'q', 'z', 'chieff', 'chip']

"""             *** Sampling parameters ***           """

"""
Parameters for the sampling of the catalogue.
"""

samplingNumberWalkers = 16  # number of MCMC walkers
samplingNumberChain = 500  # 50 #500  # length of MCMC chain
samplingBandwidthKDE = 0.075  # KDE bandwidth to use


"""             *** Bayes Model Processing ***           """

"""
Parameters for the computation of the match and the efficiency.
"""

bayesModelProcessingWaveformApproximant = "IMRPhenomPv2"  # waveform approximant the fastest beeing "IMRPhenomD" but not accounting for precession
bayesModelProcessingBandWidthKDE = 0.075  # KDE bandwidth to use


"""             *** GW Observations ***           """

"""
Detectors and psd available in the program.
"""

# List of accessible detectors from text file
detectors_avail = ["Livingston_O1", "Livingston_O2", "Livingston_O3a", "Livingston_O3b", "Hanford_O1", "Hanford_O2",
                   "Hanford_O3a", "Virgo_O2", "Virgo_O3a", "LIGO_Design", "ET_Design"]

# For psd read from files, the values were set when constructing the files
# For pycbc psd, the min and max frequency can be modified, as long as it is understood where the model breaks
# For delta_freq_min for pycbc psd, tests showed that a value of 0.01 was generating malloc error, which is the
# reason why the minimum value was set to 0.015.
psd_attributes = {
    "Livingston_O1": {"psd_name": "Livingston_O1_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.75,
                      "delta_freq_min": 0.025},
    "Livingston_O2": {"psd_name": "Livingston_O2_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.75,
                      "delta_freq_min": 0.025},
    "Livingston_O3a": {"psd_name": "Livingston_O3a_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.75,
                       "delta_freq_min": 0.025},
    "Livingston_O3b": {"psd_name": "Livingston_O3a_psd", "in_pycbc": False, "min_freq": 10.0, "max_freq": 5000.0,
                       "delta_freq_min": 0.025},
    "Hanford_O1": {"psd_name": "Hanford_O1_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.75,
                   "delta_freq_min": 0.025},
    "Hanford_O2": {"psd_name": "Hanford_O2_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.75,
                   "delta_freq_min": 0.025},
    "Hanford_O3a": {"psd_name": "Hanford_O3a_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.75,
                    "delta_freq_min": 0.025},
    "Virgo_O2": {"psd_name": "Virgo_O2_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.75,
                 "delta_freq_min": 0.025},
    "Virgo_O3a": {"psd_name": "Virgo_O3a_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.75,
                  "delta_freq_min": 0.025},
    "LIGO_Design": {"psd_name": "aLIGODesignSensitivityP1200087", "in_pycbc": True, "min_freq": 0.01,
                    "max_freq": 2048.0, "delta_freq_min": 0.015},
    "ET_Design": {"psd_name": "EinsteinTelescopeP1600143", "in_pycbc": True, "min_freq": 0.01,
                  "max_freq": 2048.0, "delta_freq_min": 0.015}
    }

"""
Parameters of the different runs which can be used for the analysis.
This part should not be changed except for adding future runs.
{<Run_name> dictionnary : {'Sensitivity': must correspond to keys of psd_attributes dictionnary, "Duration": in yr, "start_date": starting date of the run, "end_date": ending date of the run}
"""

runs_avail = ['O1', 'O2', 'O3a', 'O3b']


runs_attributes = {
    'O1': {'Sensitivity' : 'Livingston_O1',
      "Duration" : 0.1331,  # 48.6 days (arxiv 1606.04856, section 2, page 8)
      "start_date": datetime.date(2015, 9, 12),
      "end_date": datetime.date(2016, 1, 19)},

    'O2': {'Sensitivity' : 'Livingston_O2',
      "Duration" : 0.3231,  # 118 a days (arxiv 1811.12907, section 2B, page 4),
      "start_date": datetime.date(2016, 11, 30),
      "end_date": datetime.date(2017, 8, 25)},

    'O3a': {'Sensitivity' : 'Livingston_O3a',
       "Duration" : 0.2230,  # 81.4  days (arxiv 2010.14527, section 2, page 10)
       "start_date": datetime.date(2019, 4, 1),
       "end_date": datetime.date(2019, 10, 1)},

    'O3b': {'Sensitivity' : 'Livingston_O3a',
       "Duration" : 0.2053,  # 81.4  days (arxiv 2010.14527, section 2, page 10)
       "start_date": datetime.date(2019, 11, 1),
       "end_date": datetime.date(2020, 3, 27)}
    }







