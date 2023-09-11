from AstroModel import AstroModel as AM
from astropy.cosmology import Planck15
import Parameters as param

#--------------------------------------TO BE FILLED------------------------------------------
# Load an prepare your astromodel:

astroModelParameters = {'name' :,    # (str) Name used to label all files related to this model
                        'pathToCosmorate' :,   # (str) Path to the output of cosmorate, catalogs and merger rate density files.
                        'redshiftRange' :,  # (float) maximal redshift considered in the study,
                                            # should be a multiple of deltaz (bin used in the *MRD* file, output of cosmorate)
                        'sizeCatalog' :,    
                        }

# Take the time to check that all your column names have a correspondance in the dictionnary input_parameters, in the file Parameter.py.
#--------------------------------------------------------------------------------------------

FirstModel = AM.AstroModel(name_model="FirstModel", observables= ['Mc', 'q', 'z'], Cosmology = "Planck15", path ='/home/perigois/Downloads/Rufolo_2021/output_sigma03/output_spinflag1/BBHs/', catsize=200 )
FirstModel.prepare_model()

# Create the merger rate file
FirstModel.create_merger_rate_file('MRD_spread_12Z_40_No_MandF2017_0.3_No_No_0.dat', range_z = 15, delimiter="\t")
FirstModel.create_catalog_file(delimiter="\t", input_catname_beg = 'BBHs_spin2020_',input_catname_end ='_50.dat')

# Selection of GW events :

observationSelection = {'p_astro' : 0, 'FAR' : 0, 'SNR' : 0}