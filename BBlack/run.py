import json
import Run.getting_started as GS # Initiate the file Params.json
import Run.advanced_params as AP
import astrotools.AstroModel as AM
import bayesiantools.bayes_model as BA
import bayesiantools.bayesian_computation as BC


if __name__ == '__main__':
    params = json.load(open('Run/Params.json', 'r'))
    for m in params['astro_model_list'].keys():
        astromodel = AM.AstroModel(name=m)
        astromodel.generate_samples()
        BA.process_bayes_model(astromodel)
        BC.compute_likelihood(astromodel)
    mc = params['compute_multi_channel']
    for key in mc.keys():
        BC.multichannel_analysis(name=key)
    AP.clean()