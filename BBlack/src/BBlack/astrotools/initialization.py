import json
import importlib.resources

# Import parameter file
with importlib.resources.open_text("BBlack.Run", "Params.json") as f:
    params = json.load(f)


def initialization():
    for model in params['astro_model_list'].keys():
        astromodel = AstroModel(name=params['astro_model_list'][model]['name'],
                                path_to_MRD=params['astro_model_list'][model]['path_to_MRD'],
                                path_to_catalogs=params['astro_model_list'][model]['path_to_catalogs'])