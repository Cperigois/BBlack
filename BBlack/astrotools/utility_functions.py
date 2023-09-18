import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import maxwell
import pandas as pd
import os
from math import ceil
from math import pow


def mc_q_to_m1_m2(mc, q):
    """This function does the mapping (mc,q) --> (m1,m2)

    Parameters
    ----------
    mc : float or numpy array
        Chirp mass of the sources(s)
    q : float or numpy array
        Mass ratio of the source(s)

    Returns
    -------
    m1 : float or numpy array
        Mass of primary of the source(s)
    m2 : float or numpy array
        Mass of seconday of the source(s)
    """

    m1 = mc*np.power((1.0+q)/(q*q*q), 0.2)
    m2 = q*m1

    return m1, m2


def m1_m2_to_mc_q(m1, m2):
    """This function does the mapping (m1,m2) --> (mc,q)

    Parameters
    ----------
    m1 : float or numpy array
        Mass of primary of the source(s)
    m2 : float or numpy array
        Mass of seconday of the source(s)

    Returns
    -------
    mc : float or numpy array
        Chirp mass of the sources(s)
    q : float or numpy array
        Mass ratio of the source(s)
    """

    mc = np.power((m1*m2), 0.6) / (np.power(m1 + m2, 0.2))
    q = np.minimum(m2, m1) / np.maximum(m1,m2)

    return mc, q

def fmerg_f(m1, m2, xsi, zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm)
	eta = m1*m2/pow(m1+m2,2.)
	fmerg_mu0 = 1.-4.455*pow(1-xsi,0.217)+3.521*pow(1.-xsi,0.26)
	fmerg_y = 0.6437*eta -0.05822*eta*eta -7.092*eta*eta*eta +0.827*eta*xsi -0.706*eta*xsi*xsi -3.935*eta*eta*xsi
	return (fmerg_mu0+fmerg_y)/(math.pi*mtot)


def comp_chieff(m1, m2, chi1, chi2, cos_theta_1, cos_theta_2):
    """This function computes the chi effective for source(s).

    Parameters
    ----------
    m1 : float or numpy array
        Mass of the 1st binary component
    m2 : float or numpy array
        Mass of the 2nd binary component
    chi1 : float or numpy array
        Spin magnitude of the 1st binary component
    chi2 : float or numpy_array
        Spin magnitude of the 2nd binary component
    cos_theta_1 : float or numpy array
        Cosine angle betwen angular momentum and spin of 1st component
    cos_theta_2 : float or numpy array
        Cosine angle betwen angular momentum and spin of 2nd component

    Returns
    -------
    chieff : float or numpy array
        Chi effective
    """

    chieff = (chi1 * cos_theta_1 * m1 + chi2 * cos_theta_2 * m2) / (m1 + m2)

    return chieff

def comp_chip(m1, m2, chi1, chi2, cos_theta_1, cos_theta_2):
    """This function computes the chi effective for source(s).

    Parameters
    ----------
    m1 : float or numpy array
        Mass of the 1st binary component
    m2 : float or numpy array
        Mass of the 2nd binary component
    chi1 : float or numpy array
        Spin magnitude of the 1st binary component
    chi2 : float or numpy_array
        Spin magnitude of the 2nd binary component
    cos_theta_1 : float or numpy array
        Cosine angle betwen angular momentum and spin of 1st component
    cos_theta_2 : float or numpy array
        Cosine angle betwen angular momentum and spin of 2nd component

    Returns
    -------
    chieff : float or numpy array
        Chi precessing
    """

    chip1 = (2. + (3. * m2) / (2. * m1)) * chi1 * m1 * m1 * (1. - cos_theta_1 * cos_theta_1)**0.5
    chip2 = (2. + (3. * m1) / (2. * m2)) * chi2 * m2 * m2 * (1. - cos_theta_2 * cos_theta_2)**0.5
    chipmax = np.maximum(chip1, chip2)
    chip = chipmax / ((2. + (3. * m2) / (2. * m1)) * m1 * m1)
    return chip


def cos_theta_isolated_tilt(cos_nu1, cos_nu2):
    """This function computes the cosine between angular momentum and spin for the isolated case in order to allow
    non-aligned spins (see for instance arxiv 2102.12495).

    Parameters
    ----------
    cos_nu1 : float or numpy array
       cosine of the angle between the orbital angular momentumvector after and before a SN explosion for
       1st component of the binary
    cos_nu2 : float or numpy array
       cosine of the angle between the orbital angular momentumvector after and before a SN explosion for
       2nd component of the binary

    Returns
    -------
    cos_theta_1 : float or numpy array
        Cosine angle betwen angular momentum and spin of component (equal for both components)
    """

    # Check the length of inputs
    if len(cos_nu1) != len(cos_nu2):
        raise IndexError("cos_nu1 and cos_nu2 do not have the same length.")

    # See equation in arxiv 2102.12495
    phi = np.random.rand(len(cos_nu1)) * 2. * np.pi
    cos_phi = np.cos(phi)
    sin_nu1 = np.sin(np.arccos(cos_nu1))
    sin_nu2 = np.sin(np.arccos(cos_nu2))
    cos_theta = cos_nu1 * cos_nu2 + sin_nu1 * sin_nu2 * cos_phi

    return cos_theta


def trimmed_gen_spin(size, mag_gen):
    """This function generates the magnitudes of the spins such that the spins are inferior to 0.998

    Parameters
    ----------
    size : int
        size wanted for the output of the spins
    mag_gen : scipy stats function
        distribution used to generate the magnitude

    Returns
    -------
    chi1 : numpy array
        Spin magnitude of the 1st binary component
    chi2 : numpy_array
        Spin magnitude of the 2nd binary component
    """

    # Magnitude of 1st component
    chi1 = mag_gen.rvs(size)
    ind_out = np.where(chi1 > 0.998)
    size_out = len(ind_out[0])
    while size_out != 0:
        resamp = mag_gen.rvs(size_out)
        chi1[ind_out] = resamp
        ind_out = np.where(chi1 > 0.998)
        size_out = len(ind_out[0])

    # Magnitude of 2nd component
    chi2 = mag_gen.rvs(size)
    ind_out = np.where(chi2 > 0.998)
    size_out = len(ind_out[0])
    while size_out != 0:
        resamp = mag_gen.rvs(size_out)
        chi2[ind_out] = resamp
        ind_out = np.where(chi2 > 0.998)
        size_out = len(ind_out[0])

    return chi1, chi2


def gen_spin_bhns(size, mag_gen):
    """This function generates the magnitudes of the spins for BHNS systems. It assumes that the first component is
    the BH and the second component the BNS

    Parameters
    ----------
    size : int
        size wanted for the output of the spins
    mag_gen : scipy stats function
        distribution used to generate the magnitude

    Returns
    -------
    chi1 : numpy array
        Spin magnitude of the 1st binary component
    chi2 : numpy_array
        Spin magnitude of the 2nd binary component
    """

    # Magnitude of 1st component (BH)
    chi1 = mag_gen.rvs(size)
    ind_out = np.where(chi1 > 0.998)
    size_out = len(ind_out[0])
    while size_out != 0:
        resamp = mag_gen.rvs(size_out)
        chi1[ind_out] = resamp
        ind_out = np.where(chi1 > 0.998)
        size_out = len(ind_out[0])

    # Magnitude of 2nd component (NS), close to 0
    chi2 = mag_gen.rvs(size)
    ind_out = np.where(chi2 > 0.998)
    size_out = len(ind_out[0])
    distribution = maxwell(loc=0.0, scale=0.01)
    while size_out != 0:
        resamp = distribution(size_out)
        chi2[ind_out] = resamp
        ind_out = np.where(chi2 > 0.998)
        size_out = len(ind_out[0])

    return chi1, chi2


def hierarchical_spin(size, mag_gen, m1, m2, frac_hier):
    """This function computes the values of the spin magnitudes assuming a hierarchical model. For the
    hierarchical mergers (m1 and m2 superior to 60), the magnitude is drawn around 0.75

    Parameters
    ----------
    size : int
        size wanted for the output of the spins
    mag_gen : scipy stats function
        distribution used to generate the magnitude
    m1 : numpy array
        Values for the masses of the 1st component
    m2 : numpy array
        Values for the masses of the 2nd component
    frac_hier : float
        Fraction of hierarchical mergers

    Returns
    -------
    chi1 : numpy array
        Spin magnitude of the 1st binary component
    chi2 : numpy_array
        Spin magnitude of the 2nd binary component
    """

    # Generate trimmed magnitude of spins
    chi1, chi2 = trimmed_gen_spin(size, mag_gen)

    # If one of the mass is superior to 60, set the spin around 0.75
    ind_m1_massive = np.where(m1 >= 60.0)
    ind_m2_massive = np.where(m2 >= 60.0)
    chi1[ind_m1_massive] = np.random.normal(0.75, 0.025, len(ind_m1_massive[0]))
    chi2[ind_m2_massive] = np.random.normal(0.75, 0.025, len(ind_m2_massive[0]))

    # If both masses are inferior to 60, set a fraction to be hierarchical
    ind_no_massive_events = np.where((m1 < 60.0) & (m2 < 60.0))
    unif_dist = np.random.uniform(0.0, 1.0, len(ind_no_massive_events[0]))
    ind_hier = ind_no_massive_events[0][np.where(unif_dist < frac_hier)]
    chi1[ind_hier] = np.random.normal(0.75, 0.025, len(ind_hier))

    return chi1, chi2


def clean_path(path_dir):
    """This function ensure that a directory path (usually set in by user) finishes with "/"

    Parameters
    ----------
    path_dir : str
        Directory path

    Returns
    -------
    path_dir : str
        Properly set directory path
    """

    if path_dir[-1] != "/":
        path_dir += "/"
    return path_dir


def berti_pdet_fit(name_file="auxiliary_files/Pw_single.dat"):
    """This function returns a interp1d object computed from Emanuele Berti estimation of pdet.

    Parameters
    ----------
    name_file : str
        Path and name where to find the file with Emanuele Berti's data.

    Returns
    -------
    interpolate : interp1d object
        Interpolation from Berti's data
    """

    # Check that the file exists
    if not os.path.isfile(name_file):
        raise FileNotFoundError(f"Emanuele Berti's fit to p_det could not be found at {name_file}")

    # Read data and interpolate them
    data_fit = np.loadtxt(name_file)
    interpolate = interp1d(data_fit[:, 0], data_fit[:, 1])

    return interpolate


def detection_probability(pdet_fit, rho_opt, rho_thr):
    """This function computes the detection probability given SNR of a source and the interpolation of Berti.

    Parameters
    ----------
    pdet_fit : interp1d
        Interpolation from Emanuele Berti
    rho_opt : float or numpy array
        Optimal SNR of source(s)
    rho_thr : float
        Threshold SNR used

    Returns
    -------
    pdet : float or numpy array
        Probabillity of detection for source(s)
    """

    w = rho_thr / rho_opt
    if w > 1.0:
        pdet = 0.0
    else:
        pdet = pdet_fit(w)

    return pdet


def flatten_restrict_range_output_emcee(sampler, list_name_param, min_range, max_range):
    """Function that takes the outputs from a sampler, flattens it adn then only keep the points that are in the
    range specified by min_range anx max_rage

    Parameters
    ----------
    sampler : emcee Sampler
        Emcee sampler that was already ran for some iterations
    list_name_param : list of str
        List of the parameters name ran for the MCMC
    min_range : numpy array
        List of minimum for each parameter, needs to be in same order than list_name_param
    max_range : numpy array
        List of maximum for each parameter, needs to be in same order than list_name_param

    Returns
    -------
    samples : pandas dataframe
        Normalised samples
    """

    samples = pd.DataFrame(sampler.get_chain(flat=True), columns=list_name_param)
    for i, k in enumerate(list_name_param):
        samples = samples[(samples[k] > min_range[i]) & (samples[k] < max_range[i])]

    return samples


def parallel_array_range(length, n_cpu):
    """Function that creates a list of tuple that contain the ranges that will be used to divide an iterable
    over various CPUs for parallelization.

    Parameters
    ----------
    length : int
        Length of the iterable that will be divided among the CPUs
    n_cpu : int
        number of CPUs for the simulation

    Returns
    -------
    ranges_parallel : list of tuples
        List of length n_cpu where each element is a tuple with the 1st and 2nd elements correspond to the left
        and right ranges of the iterable for this CPU.
    """

    # Check that the length is not inferior to number of CPUs
    if length < n_cpu:
        raise IndexError(f"Length={length} inferior to n_cpu={n_cpu}. Data could not be divided.")

    # Set the ranges
    range_left = [i * ceil(length / n_cpu) for i in range(n_cpu)]
    range_right = [(i + 1) * ceil(length / n_cpu) for i in range(n_cpu - 1)] + [length]
    ranges_parallel = [(i, j) for i, j in zip(range_left, range_right)]

    return ranges_parallel


def check_inputlist_with_accessible_values(list_to_check, name_list_to_check, list_accessible,
                                           name_list_accessible):
    """This function is used to check if one list in input is indeed a list, and takes values from a list of
    accessible values given by another list.

    Parameters
    ----------
    list_to_check : list
        Input list that needs to be checked for formatting.
    name_list_to_check
        Name of the variable list_to_check for nicer error reports
    list_accessible
        List of available values form which list_to_check must have values from.
    name_list_accessible
        Name of the variable list_accessible for nicer error reports
    """

    if type(list_to_check) != list:
        raise TypeError(f"{name_list_to_check} must be a list.")
    if type(list_accessible) != list:
        raise TypeError(f"{name_list_accessible} must be a list.")

    for val in list_to_check:
        if val not in list_accessible:
            raise ValueError(f"Parameter {val} not available. Choose in the set {list_accessible}.")


def check_inputdict_with_accessible_values(dict_to_check, name_dict_to_check, dict_accessible,
                                           name_dict_accessible):
    """This function is used to check if one dictionary is a "sub-dictionary" of another. It checks that the keys
    are well set and that the values for each key is a sublist or sub-dictionary of a refernce dictionary.

    Parameters
    ----------
    dict_to_check : dict
        Input list that needs to be checked for formatting.
    name_dict_to_check : str
        Name of the variable list_to_check for nicer error reports
    dict_accessible : dict
        Dictionary with keys available available values form which list_to_check must have values from.
    name_dict_accessible : str
        Name of the variable list_accessible for nicer error reports
    """

    if type(dict_to_check) != dict:
        raise TypeError(f"{name_dict_to_check} must be a dictionary.")
    if type(dict_accessible) != dict:
        raise TypeError(f"{name_dict_accessible} must be a dictionary.")

    for key in dict_to_check:
        if key not in dict_accessible:
            raise KeyError(f"Key {key} not available. Choose in the set {list(dict_accessible.keys())}.")
        if type(dict_to_check[key]) not in [list, dict]:
            raise TypeError(f"Value for key {key} of {name_dict_to_check} must be a list or dictionary.")
        if type(dict_accessible[key]) not in [list, dict]:
            raise TypeError(f"Value for key {key} of {dict_accessible} must be a list or dictionary.")

        for val in dict_to_check[key]:
            if val not in dict_accessible[key]:
                raise ValueError(f"Value {val} not available for key {key}. Choose in {dict_accessible[key]}")


def compute_log_likelihood(bayes_opt, integral_match_model, n_obs, n_sources, detection_efficiency):
    """This function computes the log-likelihood according to the option selected for a number of observed
    that were observed during a given observing run

    Parameters
    ----------
    bayes_opt : str
        Choice for the form of the log-likelihood
    integral_match_model : float
        Sum of the logarithm of the integral matching model with distribution over all observed events
    n_obs : int
        Number of observed events.
    n_sources : int
        Number of sources that is predicted by the model
    detection_efficiency : float
        Value of the detection efficiency

    Returns
    -------
    log_likelihood : float of tuples
        Value of the log-likelihood
    """

    log_likelihood = None
    if bayes_opt == "MatchOnly":
        log_likelihood = integral_match_model
    elif bayes_opt == "NoRate":
        log_likelihood = integral_match_model - float(n_obs) * np.log(detection_efficiency)
    elif bayes_opt == "All":
        log_likelihood = integral_match_model - detection_efficiency * n_sources + float(n_obs) * np.log(n_sources)
    else:
        print("Choose a Bayesian option between All, NoRate and MatchOnly.")
        exit()

    return log_likelihood


def jump_mix_frac(current_mixing_frac, scale_jump):
    """This function defines a simple jump distribution when estimating mixing fraction in a MCMC chain

    Parameters
    ----------
    current_mixing_frac : numpy array
        Current values for the set of mixing fraction in the chain
    scale_jump : float
        Scale used for the gaussian jump

    Returns
    -------
    jump_mixing_frac : numpy array
        New position for the chain
    """

    flag = True
    jump_mixing_frac = None
    length = len(current_mixing_frac)
    while flag:
        jump_mixing_frac = 1.1*np.ones(length)

        sum_mix_frac = 0.0
        for channel in range(length-1):
            while jump_mixing_frac[channel] < 0.0 or jump_mixing_frac[channel] > 1.0:
                jump_mixing_frac[channel] = np.random.normal(current_mixing_frac[channel], scale_jump)
            sum_mix_frac += jump_mixing_frac[channel]

        jump_mixing_frac[-1] = 1.0 - sum_mix_frac
        if 0.0 <= jump_mixing_frac[-1] <= 1.0:
            flag = False

    return jump_mixing_frac


def indices_closest_neighbors_in_list(value, list_points):
    """This function returns the indices of the points in the storted list list_points that are
    enclosing value

    Parameters
    ----------
    value : float
        Value that we are interested in
    list_points : list
        Sorted list of values

    Returns
    -------
    i : int
        Left index
    """

    if value < list_points[0] or value > list_points[-1]:
        raise ValueError(f"Value {value} outside of possible range [{list_points[0]}, {list_points[-1]}]")

    for i in range(len(list_points)-1):
        if value < list_points[i+1]:
            return i


def linear_interp(x, x_1, x_2, y_1, y_2):
    a = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - a * x_1

    return a * x + b

class SchemaErrorVerbose(Exception):
    """This error class is raised when a schema is not valid
    """

    def __init__(self, name, schema):
        self.name = name
        self.schema = schema

    def __str__(self):
        error_string = """\n
        The format of {} is not respected.
        Make sure that the format is :\n
        {}""".format(self.name, self.schema)
        return error_string


# This function is not used currently but might in future
def compute_autocorrelation_chain(mcmc_chain, num_var, len_chain):

    # Initialisation
    means = []
    var_unscaled = []
    autocorr = []

    # Compute the means for the chains
    for i in range(num_var):
        means.append(np.mean(mcmc_chain[i]))

    # Compute the squared differences
    for i in range(num_var):
        sum_var = 0.0
        for j in range(len_chain):
            sum_var += (mcmc_chain[i][j] - means[i]) * (mcmc_chain[i][j] - means[i])
        var_unscaled.append(sum_var)

    # Compute the autocorrelation for a delay tau=1
    for i in range(num_var):
        val_autocorr = 0.0
        for j in range(len_chain - 1):
            val_autocorr += (mcmc_chain[i][j] - means[i]) * (mcmc_chain[i][j + 1] - means[i])
        autocorr.append(val_autocorr / var_unscaled[i])

    return autocorr
