import astrotools.AstroModel as AM
#import bayesiantools.process_bayes_model as BA
import sys
import json
import Run.getting_started


if __name__ == '__main__':
    print(sys.argv[1])
    params = json.load(open(sys.argv[1]+'/Params.json'))
    for m in params.astroModelList:
        m.process_astro_model(params)
        #BA.process_bayes_model(m, params)
        #BA.Compute_likelihood(m, params)

    mc = params.computeMultiChannel
    #for k in mc.keys :
        #BA.MultichannelAnalysis(mc[k], params)