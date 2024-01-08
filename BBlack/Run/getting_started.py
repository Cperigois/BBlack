import BBlack.Run.advanced_params as AP
import sys
import os
import json
sys.path.append('../')
import BBlack.astrotools.AstroModel as AM
import BBlack.GWtools.detector as Detector

"""----------------------TO FILL----------------------"""

"""             *** GENERIC PARAMETERS ***            """

name_of_project_folder = 'Example'  # Careful, the name there HAS TO BE the SAME as the name of the folder
redshift_range = 15
catalog_size = 200  # Minimum suggested 3000, 200 used for the example
observables = ['Mc', 'q', 'z']  # Choose among ['Mc', 'q', 'z', 'chip', 'chieff']
n_cpu_max = 12 # Number maximal of cpu used by the code

from_cosmorate = True  # If your files are standard output from cosmorate this parameter should be True, else, should be
# ... False
param_dictionary = {'name_of_project_folder': name_of_project_folder,
                    'redshift_range': redshift_range,
                    'catalog_size': catalog_size,
                    'observables': observables
                    }
AP.set(name_of_project_folder, param_dictionary, AP.advParams)


"""               *** ASTROMODELS ***                 """
"""
        class AstroModel:
        Parameters: 
        Optional: spinModel: among:
                  'InCat'(by default option) means that the spins are already in your catalogues
                  'Rand_Isotropic': Build random aligned spins with magnitude from a maxwellian law sigma = 0.1
                  'Rand_Dynamics': Build random misaligned spins with magnitude from a maxwellian law sigma = 0.1
                  'Zeros' (default is 'InCat', assuming that your spins are in your initial catalog
"""

example_1 = AM.AstroModel(name="Example1", path_to_catalogs="./../BBHs_m01/",
                          path_to_MRD='./../BBHs_m01/MRD_spread_9Z_40_No_MandF2017_0.3_No_No_0.dat')

example_2 = AM.AstroModel(name="Example2", path_to_catalogs="./../BBHs_m02/",
                          path_to_MRD='./../BBHs_m02/MRD_spread_9Z_40_No_MandF2017_0.3_No_No_0.dat')

astro_model_list = [example_1.name, example_2.name]

"""               *** GW DATA ***                 """
"""
        Set the runs you want to use
        List od available detectors : 
"""

observing_runs = {'O1': {'detector': 'Livingston_O1', 'deltafreq': 1.0},
                  'O2': {'detector': 'Livingston_O2', 'deltafreq': 1.0},
                  'O3a': {'detector': 'Livingston_O3a', 'deltafreq': 1.0},
                  'O3b': {'detector': 'Livingston_O3b', 'deltafreq': 1.0}}


"""               *** Bayesian options ***                 """
"""
        Choose if you want to compute multichannel analysis
"""
compute_multi_channel = {'1VS2': [example_1.name, example_2.name]}  # To be kept empty to disable channel analysis

"""---------------------------------------------------"""

"""        *** Main Code, should not change ***       """

"""  1- Set the directory for all intermediate and definitive results  """

if not os.path.exists(name_of_project_folder):
    os.mkdir(name_of_project_folder)

"""  2- Gather and save the parameter used in the study  """

param_dictionary = {'name_of_project_folder': name_of_project_folder,
                    'redshift_range': redshift_range,
                    'catalog_size': catalog_size,
                    'observables': observables,
                    'astro_model_list': astro_model_list,
                    'observing_runs': observing_runs,
                    'compute_multi_channel': compute_multi_channel,
                    'n_cpu_max': n_cpu_max
                    }
AP.set(name_of_project_folder, param_dictionary, AP.advParams)

"""  4- Clean folder and remove useless files  """
AP.clean()
