"""
This file contain all the parameters to be used for the analysis.
It's divides in three main parts respectively dedicated to the 3 main packages : \
            AstroModel, BaesianAnalysis, GravitationalWave
"""

"""             *** ASTROMODEL ***           """

"""
Parameters of the user input catalogue. Please do not change the right column.
{<Input catalogue names> : <Names for output catalogues>}
"""

input_params = {
    "m1": "m1",  # mass of compact object 1
    "m2": "m2",  # mass of compact object 2
    "chi1": "chi1",  # spin magnitude of compact object 1
    "chi2": "chi2",  # spin magnitude of compact object 2
    "cos_nu1": "costheta1",  # cosine of tilt 1st supernova
    "cos_nu2": "costheta2",  # cosine of tilt 2nd supernova
    "z_merg": "z",  # redshift at merger
    "ID": "id",  # Id of the binary
    "time_delay [yr]": "time_delay",  # time delay
    "th1": "theta1",  # angle between angular momentum and spin for compact object 1
    "th2": "theta2",  # angle between angular momentum and spin for compact object 2
    "Channel": "Channel"
    }

