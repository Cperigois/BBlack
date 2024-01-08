import astrotools.AstroModel as AM
import bayesiantools.bayes_model as BA
import sys
import json
import Run.getting_started

if __name__ == '__main__':
    params = json.load(open('Run/Params.json', 'r'))
    for m in params.astroModelList:
        astromodel = AM.AstroModel(name=m)
        astromodel.generate_samples()
        BA.process_bayes_model(m, params)
        #BA.Compute_likelihood(m, params)

    mc = params.computeMultiChannel
    #for k in mc.keys :
        #BA.MultichannelAnalysis(mc[k], params)