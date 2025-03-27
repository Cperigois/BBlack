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
