import os
import re
import json
import BBlack.astrotools.utility_functions as UF
import BBlack.Example.Parameters as P

#----------------------------------------------------------------------------------

# This expression is the one corresponding to catalogs' filenames
#regex_catalog = r'^([A-Za-z0-9_]*)_\d+_(\d+).dat$'
regex_catalog = r'^([A-Za-z0-9_]*)_\d+().dat$'


def process_cosmorate(path_dir_cr, num_var_header, del_cosrate="\t", del_cat="\t"):
    """Main function called to process CosmoRate files in order to make then in appropriate format for the
    Bayesian analysis.

    Parameters
    ----------
    path_dir_cr : str
        name of the directory where CosmoRate files can be found
    num_var_header : int
        number of variables in CosmoRate catalog headers
    del_cosrate : str
        string delimiter used in the original cosmorate header (default = '\t')
    del_cat : str
        string delimiter used for the newly created catalog (default = '\t')
    """

    print("*******  START : COSMORATE PRE-PROCESS  *******")

    # Check that the input directory exists
    path_dir_cr = UF.clean_path(path_dir_cr)
    if not os.path.exists(path_dir_cr):
        raise FileNotFoundError(f"The directory {path_dir_cr} was not found !")

    # Create a custom log-file
    logfile = LogFileCR(path_dir_cr)

    # Prepare directory by separating catalog files
    prepare_directory(path_dir_cr, logfile)

    # Rewrite the header of the catalog files according to BayesBlack standard
    rewrite_header_cosmorate(path_dir_cr, logfile, num_var_header, delimiter_cr=del_cosrate, delimiter_new_cat=del_cat)
    print("*******  END : COSMORATE PRE-PROCESS  *******")


def prepare_directory(path_dir_cr, logfile):
    """Create a directory 'catalogs/' to put the catalogs of CosmoRate inside

    Parameters
    ----------
    path_dir_cr : str
        name of the directory where CosmoRate files can be found
    logfile : object LogFileCR
        logfile object that is constructed during CosmoRate process
    """

    # If log-file indicates that files were already moved, stop the routine
    if logfile.status["files_moved"]:
        print("Logfile indicates that files have already moved correctly. Interrupting routine.")
        return

    # Check/create a directory catalogs/ where catalog files will be put
    path_new_dir_cat = path_dir_cr + "/catalogs/"
    if not os.path.exists(path_new_dir_cat):
        os.mkdir(path_new_dir_cat)

    # List all the files
    cat_files = os.listdir(path_dir_cr)

    # Get the merger-rate related files
    other_files = []
    for f in cat_files:
        if re.match(r'^MRD_', f) is not None:
            other_files.append(f)
        if re.match(r'^Zperc_', f) is not None:
            other_files.append(f)
        if re.match(r'^Sampled_', f) is not None:
            other_files.append(f)
        if re.match(r'^catalogs$', f) is not None:
            other_files.append(f)
        if re.match(r'^log_file_cr.in$', f) is not None:
            other_files.append(f)
    for f in other_files:
        cat_files.remove(f)

    # Check that only catalog files are selected
    not_catalogs = []
    for f in cat_files:
        match_cat = re.search(regex_catalog, f)
        if match_cat is None:
            not_catalogs.append(f)
    if len(not_catalogs) > 0:
        raise ValueError(f"The files {not_catalogs} are not recognised as outputs from CosmoRate.")

    # Move the catalog files if they are found
    if len(cat_files) >= 1:
        for file in cat_files:
            os.replace(path_dir_cr + file, path_new_dir_cat + file)
        logfile.status["files_moved"] = True
        logfile.update()
    else:
        raise FileNotFoundError("No catalogs files found. Check the regex pattern in 'auxiliary_cosmorate.py'.")


def rewrite_header_cosmorate(path_dir_cr, logfile, num_var_header, delimiter_cr="\t", delimiter_new_cat="\t"):
    """Rewrite the header of each CosmoRate catalog files to be adequate with the header defined for the Bayesian
    analysis.

    Parameters
    ----------
    path_dir_cr : str
        name of the directory where CosmoRate files can be found
    logfile : object LogFileCR
        logfile object that is constructed during CosmoRate process
    num_var_header : int
        number of variables in CosmoRate catalog headers
    delimiter_cr : str
        string delimiter used in the original cosmorate header (default = '\t')
    delimiter_new_cat : str
        string delimiter used for the newly created catalog (default = '\t')
    """

    # If logfile indicates that header was already rewritten, stop the routine
    if logfile.status["header_rewritten"]:
        print("Logfile indicates that headers have already been rewritten for all files. Interrupting routine.\n")
        return

    # Path towards BayesBlack catalogs
    path_new_dir_cat = path_dir_cr + "catalogs/"

    # Check that files were moved. If files are found while logfile indicates no, print a warning
    if not logfile.status["files_moved"]:
        if not os.path.exists(path_new_dir_cat) or not os.listdir(path_new_dir_cat):
            raise FileNotFoundError("Catalogs files were not found in {}. Run function 'prepare_directory' from "
                                    "auxiliary_cosmorate.py first to move catalog files".format(path_new_dir_cat))
        else:
            print("Warning : according to log file, catalog files seem to have been moved manually. We advise to "
                  "carefully check that the files are indeed the good ones!\n For future uses, it is recommended "
                  "to use the function 'prepare_directory' to move files instead.\n")
            logfile.status["files_moved"] = True

    # For every file, rewrite the first header line
    for file in os.listdir(path_new_dir_cat):
        with open(path_new_dir_cat + file, "r") as filein:
            list_of_lines = filein.readlines()

            # Get header line and cleans the # in the header
            line_header = list_of_lines[0]
            line_header = ''.join(re.split('\\s*#\\s*', line_header)).replace("\n", "")

            # Check that the number of variables found is the same than the one in input
            if len(line_header.split(delimiter_cr)) != num_var_header:
                raise IndexError(f"Problems in header number of variables. Check that there are indeed "
                                 f"{num_var_header} and that 'del_cosrate' is correctly set.")

            # Try to apply the mapping for variable names. If the variable was already mapped towards BayesBlack
            # variables, do not map it, otherwise try the mapping and raise an exception if error
            try:
                mapped_param = list(map(lambda x: param.mapping_header_cosmoRate[x] if x not in
                                    param.mapping_header_cosmoRate.values() else x, line_header.split(delimiter_cr)))
            except Exception as e:
                raise MappingError(e, delimiter_cr)

            # Rewrite the file
            list_of_lines[0] = delimiter_new_cat.join(mapped_param)+"\n"
            #list_of_lines[1:] = [delimiter_new_cat.join(x.split(delimiter_cr)) for x in list_of_lines[1:]]
            # for massy cosmoRate header
            list_of_lines[1:] = [delimiter_new_cat.join(x.split(" ")) for x in list_of_lines[1:]]

        # Rewrite the files
        with open(path_new_dir_cat + file, "w") as filein:
            filein.writelines(list_of_lines)

    # Update the logfile
    logfile.status["header_rewritten"] = True
    logfile.update()


class LogFileCR:
    """Class that corresponds to a log-file associated with CosmoRate pre-processing.

    Attributes
    ----------
    path_file : str
        path towards the logfile
    status : dict
        dictionary containing status information on CosmoRate processing

    Methods
    ----------
    update()
        print the updated values for status in logfile
    """

    file_name = "log_file_cr.in"
    status = {
        "files_moved": False,
        "header_rewritten": False,
    }

    def __init__(self, path_dir):
        """Creates an instance of LogFileCR by either reading from existing log-file or printing new log-file
        with default values.

        Parameters
        ----------
        path_dir : str
            name of the directory where catalog files can be found
        """

        # Set attribute values
        self.path_file = path_dir + self.file_name
        self.status = self.status.copy()

        # If file already exists, read it, otherwise create it with default values for status
        if not os.path.exists(self.path_file):
            with open(self.path_file, "w") as fileout:
                json.dump(self.status, fileout)
        else:
            with open(self.path_file, "r") as fileout:
                self.status = json.load(fileout)

    def update(self):
        """Prints updated values of dictionary status to the log-file."""

        # Rewrite file with values of status
        with open(self.path_file, "w") as fileout:
            json.dump(self.status, fileout)


class MappingError(Exception):
    """Specific error to signal an error during mapping from CosmoRate header to BayesAnalysis header."""

    def __init__(self, key, delimiter):
        self.key = key
        self.delimiter = delimiter

    def __str__(self):
        error_string = """
        The key {0} was not found in the mapping dictionary in 'auxiliary_cosmorate.py':\n
            1) check that the selected delimiter {1} is the one used in CosmoRate source files
            2) check that {0} corresponds to one of the parameter in the header of CosmoRate source files
            3) update mapping dictionary 'mapping_header_cosmoRate' in 'auxiliary_cosmorate.py' with key {0}\n
        """.format(self.key, self.delimiter)
        return error_string