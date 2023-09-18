import AstroModel as AM

def processAstroModel(model, _params)

# --------------------------------------------      User input       ---------------------------------------------------

num_header_cosmorate = 15  # number of variables in cosmorate header
del_cosrate = "\t"  # separator used in CosmoRate catalogs
range_z = 15.0  # range of redshift considered for the analysis
num_cat = 100000  # number of sources wanted in the catalog
del_cat = " "
# ---------------------------------------------      Main code       ---------------------------------------------------

# Make sure directories are created
if not os.path.exists("Astro_Models/"):
    os.mkdir("Astro_Models/")
if not os.path.exists("Astro_Models/Catalogs/"):
    os.mkdir("Astro_Models/Catalogs")
if not os.path.exists("Astro_Models/MergerRateDensity/"):
    os.mkdir("Astro_Models/MergerRateDensity")

FirstModel.prepare_model()


# Create the merger rate file
FirstModel.create_merger_rate_file('MRD_spread_12Z_40_No_MandF2017_0.3_No_No_0.dat', range_z = 15, delimiter="\t")
FirstModel.create_catalog_file(delimiter="\t", input_catname_beg = 'BBHs_spin2020_',input_catname_end ='_50.dat')
# Get all the parameters set in astro_model_param.py
dir_cosmo_rate, astro_param, co_param, mag_gen_param, name_spin_model = return_astro_param(sys.argv)

# CosmoRate processing
process_cosmorate(path_dir_cr=dir_cosmo_rate, num_var_header=num_header_cosmorate, del_cosrate=del_cosrate)

# Create the merger rate file
model_astro.create_merger_rate_file(dir_cosmorate=dir_cosmo_rate, range_z=range_z)

# Create the catalog
model_astro.create_catalog_file(dir_cosmorate=dir_cosmo_rate, num_cat=num_cat)