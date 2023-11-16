import advanced_params as AP
import sys
import os
import json
sys.path.append('../')
import BBlack.astrotools.AstroModel as AM



"""----------------------TO FILL----------------------"""

"""             *** GENERIC PARAMETERS ***            """

name_of_project_folder = 'Savings' # Careful, the name there HAS TO BE the SAME as the name of the folder
redshift_range = 15
catalog_size = 200 # Minimum suggested 3000, 200 used for the example
observables = ['Mc', 'q', 'z'] # Choose among ['Mc', 'q', 'z', 'chip', 'chieff']

from_cosmorate = True # If your files are standerd output from cosmorate this parameter should be True, else, should be
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
example_1 = AM.AstroModel(name="Example1", path_to_catalogs = "./../BBHs_m01/catalogs/",
                          path_to_MRD='./../../BBHs_m01/MRD_spread_9Z_40_No_MandF2017_0.3_No_No_0.dat')

example_2 = AM.AstroModel(name="Example2", path_to_catalogs = "./../BBHs_m02/catalogs/",
                          path_to_MRD='./../../BBHs_m02/MRD_spread_9Z_40_No_MandF2017_0.3_No_No_0.dat')


astro_model_list = [example_1, example_2]
compute_multi_channel = {'1VS2' : [example_1, example_2]} # Can be empty

"""---------------------------------------------------"""

"""        *** Main Code, should not change ***       """

"""  1- Set the directory for all intermediate and definitiv results  """

if not os.path.exists(name_of_project_folder):
    os.mkdir(name_of_project_folder)

"""  2- Gather and save the parameter used in the study  """

param_dictionary = {'name_of_project_folder': name_of_project_folder,
                   'redshift_range': redshift_range,
                   'catalog_size': catalog_size,
                   'observables': observables,
                   'astro_model_list': astro_model_list,
                   'compute_multi_channel': compute_multi_channel
                   }

AP.set(name_of_project_folder, param_dictionary, AP.advParams)

"""  3- Run the analysis  """
"""  3-  a) prepare all astromodels  """

for am in astro_model_list:
    am.process_astro_model()

"""  3-  b) re-sample the catalogs  """


"""  3-  c) compute the match  """


"""  3-  d) compute the likelihoods  """


"""  3-  e) If asked compute the multichannel analysis  """

if len(compute_multi_channel.keys())>0 :
    for mc_ana in compute_multi_channel.keys() :
        print('***   ',mc_ana, '   ***')

"""  4- Clean folder and remove useless files  """
AP.clean()



