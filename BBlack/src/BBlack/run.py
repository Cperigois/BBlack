import json
from BBlack.astrotools.astromodel import AstroModel
from BBlack.bayesiantools.bayesian_computation import compute_likelihood, multichannel_analysis
from BBlack.bayesiantools.process_bayes_model import process_bayes_model
from BBlack.Run.settings import Make_param_file, clean


if __name__ == '__main__':
    # (Re)Build the file params.json
    Make_param_file()
    params = json.load(open('Run/Params.json', 'r'))
    for m in params['astro_model_list'].keys():
        astromodel = AstroModel(name=m)
        astromodel.generate_samples()
        process_bayes_model(astromodel)
        compute_likelihood(astromodel)
    mc = params['compute_multi_channel']
    for key in mc.keys():
        multichannel_analysis(name=key)
    clean()