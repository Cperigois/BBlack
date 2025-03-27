import numpy as np
import math


def fmerg_f(m1, m2, xsi, zm) :
	mtot = (m1+m2)*4.9685e-6*(1+zm)
	eta = m1*m2/pow(m1+m2,2.)
	fmerg_mu0 = 1.-4.455*pow(1-xsi,0.217)+3.521*pow(1.-xsi,0.26)
	fmerg_y = 0.6437*eta -0.05822*eta*eta -7.092*eta*eta*eta +0.827*eta*xsi -0.706*eta*xsi*xsi -3.935*eta*eta*xsi
	return (fmerg_mu0+fmerg_y)/(math.pi*mtot)

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