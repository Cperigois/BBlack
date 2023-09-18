from astro_model import AstroModel as AM
import advanced_params as AP


"""----------------------TO FILL----------------------"""

"""             *** GENERIC PARAMETERS ***            """

nameOfProjectFolder = 'Example'
redshiftRange = 15
catalogSize = 200 # Minimum suggested 3000, 200 used for the example
observables = ['Mc', 'q', 'z'] # Choose among ['Mc', 'q', 'z', 'chip', 'chieff']

pAstroLimit = 0.9 # use only GW events with a pastro > pAstroLimit
farLimit = 0.25 # use only GW events with a far < farLimit
snrLimit = 0 # use only GW events with a snr > snrLimit

fromCosmorate = True


"""               *** ASTROMODELS ***                 """

Example1 = AM.AstroModel(name = "Example1", path2Catalogs = "./Data/BBHs_m01/", path2Mrd = './Data/BBHs_m01/MRD.dat')# Optional: spinModel among 'InCat', 'Rand_Isotropic', 'Rand_Dynamics', 'Zeros' (default is 'InCat', assuming that your spins are in your initial catalog.

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



