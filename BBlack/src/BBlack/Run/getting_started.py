
import sys


sys.path.append('../')
"""----------------------TO FILL----------------------"""

"""             *** GENERIC PARAMETERS ***            """

name_of_project_folder = 'Example'  # Careful, the name there HAS TO BE the SAME as the name of the folder
redshift_range = 15
catalog_size = 200  # Minimum suggested 3000, 200 used for the example
observables = ['Mc', 'q', 'z', 'chieff', 'chip']  # Choose among ['Mc', 'q', 'z', 'chip', 'chieff']
n_cpu_max = 4  # Number maximal of cpu used by the code
co_type = 'BBH'  # To be chosen among 'BBH', 'BNS', 'NSBH'
from_cosmorate = True  # If your files are standard output from cosmorate this parameter should be True, else, should be
# ... False


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
astromodel_1 = {'name': "Example1",
                'path_to_catalogs': "./../BBHs_m01/",
                'path_to_MRD': './../BBHs_m01/MRD_spread_9Z_40_No_MandF2017_0.3_No_No_0.dat'}

astromodel_2 = {'name': "Example2",
                'path_to_catalogs': "./../BBHs_m02/",
                'path_to_MRD': './../BBHs_m02/MRD_spread_9Z_40_No_MandF2017_0.3_No_No_0.dat'}

astro_model_list = {astromodel_1['name']: astromodel_1,
                    astromodel_2['name']: astromodel_2}
rerun_catalog_computation = False
rerun_samples_generation = False

"""               *** GW DATA ***                 """
"""
        Set the runs you want to use
        List of available detectors : 
"""
observing_runs = ['O1', 'O2', 'O3a', 'O3b']

"""               *** Bayesian options ***                 """
"""
        Choose option for bayesian analysis
"""
rerun_bayesian_analysis = True
observable_variation = {'all': {'observables': observables},
                        'no_spin': {'observables': ['Mc', 'q', 'z']}}  # Choose among ['Mc', 'q', 'z', 'chip', 'chieff']
compute_multi_channel = {
    '1VS2': [astromodel_1['name'], astromodel_2['name']]}  # To be kept empty to disable channel analysis

"""               *** Post processing ***                 """
"""
        Choose if you want to compute multichannel analysis
"""
run_data_cleaning = False
run_plots = False


