import pandas as pd


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

def berti_pdet_fit(name_file="AuxiliaryFiles/Pw_single.dat"):
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

def f_merg(m1,m2,xsi,zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm)
	eta = m1*m2/np.power(m1+m2,2.)
	fmerg_mu0 = 1.-4.455*np.power(1-xsi,0.217)+3.521*np.power(1.-xsi,0.26)
	fmerg_y = 0.6437*eta -0.05822*eta*eta -7.092*eta*eta*eta +0.827*eta*xsi -0.2706*eta*xsi*xsi -3.935*eta*eta*xsi
	return (fmerg_mu0+fmerg_y)/(math.pi*mtot)

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

