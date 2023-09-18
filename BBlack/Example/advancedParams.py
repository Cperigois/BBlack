import pandas

def set(_projectFolder, _paramDictionnary, _advParamDictionnary):
    output = merge(_paramDictionnary, _advParamDictionnary)
    print("Started writing dictionary to the file ", _projectFolder,'Param.json')
    with open(_projectFolder+'Param.json', "w") as file:
        json.dump(output, file)  # encode dict into JSON
    print("Done writing dict into Params.json file")



"""             *** ASTROMODEL ***           """

"""
Parameters of the user input catalogue. Please do not change the left column.
{<Input catalogue names> : <Names for output catalogues>}
"""

input_parameters = {
    "m1": "m1",   # mass of compact object 1
    "mrem1" : "m1",
    "Mass1" : "m1",

    "m2": "m2",   # mass of compact object 2
    "mrem2": "m2",
    "Mass2": "m2",

    "chi1": "chi1",   # spin magnitude of compact object 1
    "th1": "theta1",   # angle between angular momentum and spin for compact object 1
    "cos_nu1": "costheta1",   # cosine of tilt 1st supernova
    "cmu1" : "costheta1",   # cosine of tilt 1st supernova

    "chi2": "chi2",   # spin magnitude of compact object 2
    "th2": "theta2",   # angle between angular momentum and spin for compact object 2
    "cos_nu2": "costheta2",   # cosine of tilt 2nd supernova
    "cmu2" : "costheta2",   # cosine of tilt 2nd supernova

    "z_merg": "z",   # redshift at merger
    "z_form": "zForm",   # redshift at merger

    "ID": "id",   # Id of the binary
    "time_delay [yr]": "timeDelay",   # time delay
    "mZAMS1" : "mzams1",   # Zero Age Main Sequence mass of the 1st component
    "mZAMS2" : "mzams2",   # Zero Age Main Sequence mass of the 2nd component
    "M_SC" : "m_sc",   # mass of star cluster
    "Z progenitor" : "z_progenitor",
    "Channel": "Channel",
    "Ng" : "Ng",
    "tSN1" : "tsn1",
    "tSN2" : "tsn2"
    }

spin_model = 'InCat'

"""             *** Sampling parameters ***           """

"""
Parameters for the sampling of the catalogue.
"""

samplingNumberWalkers = 16  # number of MCMC walkers
samplingNumberChain = 500  # 50 #500  # length of MCMC chain
samplingBandwidthKDE = 0.075  # KDE bandwidth to use


"""             *** Bayes Model Processing ***           """

"""
Parameters for the computation of the match and the efficiency.
"""

bayesModelProcessingWaveformApproximant = "IMRPhenomPv2"  # waveform approximant the fastest beeing "IMRPhenomD" but not accounting for precession
bayesModelProcessingBandWidthKDE = 0.075  # KDE bandwidth to use

bayesOptionComputeLikelihood = "All"
bayesOptionMultiChannel = "NoRate"

advParams = {'input_parameters': input_parameters,
             'samplingNumberWalkers': samplingNumberWalkers,
             'samplingNumberChain': samplingNumberChain,
             'samplingBandwidthKDE': samplingBandwidthKDE,
             'bayesModelProcessingWaveformApproximant': bayesModelProcessingWaveformApproximant,
             'bayesModelProcessingBandWidthKDE': bayesModelProcessingBandWidthKDE,
             'spin_model':spin_model}