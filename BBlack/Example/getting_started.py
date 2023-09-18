import advanced_params as AP
import sys
sys.path.append('../')
import astrotools.AstroModel as AM



"""----------------------TO FILL----------------------"""

"""             *** GENERIC PARAMETERS ***            """

name_of_project_folder = 'Example' # Careful, the name there HAS TO BE the SAME as the name of the folder
redshiftRange = 15
catalogSize = 200 # Minimum suggested 3000, 200 used for the example
observables = ['Mc', 'q', 'z'] # Choose among ['Mc', 'q', 'z', 'chip', 'chieff']

fromCosmorate = True # If your files are standerd output from cosmorate this parameter should be True, else, should be False


"""               *** ASTROMODELS ***                 """

Example1 = AM.AstroModel(name = "Example1", path2Catalogs = "./Data/BBHs_m01/",spinmodel path2Mrd = './Data/BBHs_m01/MRD.dat')# Optional: spinModel among 'InCat', 'Rand_Isotropic', 'Rand_Dynamics', 'Zeros' (default is 'InCat', assuming that your spins are in your initial catalog.

Example2 = AM.AstroModel(name = "Example2", path2Catalogs = "./Data/BBHs_m02/", path2Mrd = './Data/BBHs_m02/MRD.dat')


astroModelList = [Example1, Example2]
ComputeMultiChannel = {'1VS2' : [Example1, Example2]} # Can be empty

"""---------------------------------------------------"""

"""        *** Main Code, should not change ***       """

paramDictionary = {'nameOfProjectFolder': nameOfProjectFolder,
                   'redshiftRange': redshiftRange,
                   'catalogSize': catalogSize,
                   'observables': observables,
                   'pAstroLimit': pAstroLimit,
                   'farLimit': farLimit,
                   'snrLimit': snrLimit,
                   'astroModelList': astroModelList,
                   'ComputeMultiChannel': ComputeMultiChannel
                   }

AP.set(nameOfProjectFolder, paramDictionary, AP.advParams)



