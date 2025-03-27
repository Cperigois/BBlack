import emcee
import json

param_path = 'Run/Params.json'
params = json.load(open(param_path, 'r'))


def initialization():
    for model in params['astro_model_list'].keys():
        astromodel = AstroModel(name=params['astro_model_list'][model]['name'],
                                path_to_MRD=params['astro_model_list'][model]['path_to_MRD'],
                                path_to_catalogs=params['astro_model_list'][model]['path_to_catalogs'])