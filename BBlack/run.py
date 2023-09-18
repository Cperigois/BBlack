import AstroModel.process_astro_model as AM
import BayesianAnalysis.process_bayes_model as BA
import GravitationalWave as GW

def importParams(path):


if __name__ == '__main__':
    print(sys.argv[1])
    params = json.load(open(sys.argv[1]+'/Params.json', 'r'))
    for m in params.astroModelList:
        AM.processAstroModel(m, params)
        BA.process_bayes_model(m, params)
        BA.Compute_likelihood(m, params)

    mc = params.computeMultiChannel
    for k in mc.keys :
        BA.MultichannelAnalysis(mc[k], params)