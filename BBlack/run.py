import Run.getting_started
import Run.advanced_params as AP
import astrotools.AstroModel as AM
import bayesiantools.bayes_model as BA
import sys
import json

if __name__ == '__main__':
    params = json.load(open('Run/Params.json', 'r'))
    AM.initialization()
    for m in params['astro_model_list'].keys():
        astromodel = AM.AstroModel(name=m)
        astromodel.generate_samples()
        BA.process_bayes_model(astromodel, params)
        BA.compute_likelihood(astromodel, params)

    mc = params.computeMultiChannel
    for k in mc.keys :
        BA.MultichannelAnalysis(mc[k], params)
    AP.clean()