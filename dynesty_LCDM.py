import numpy as np

#from dynesty_phen import H0_A_rd_mean

# We will set some flags to control what computations to perform
compute_minimum_chi2 = True # Compute the minimum chi squared contributions
perform_nested_sampling = True # Set to True to perform nested sampling
nested_sampling_data = "all" # Options: "all", "CMB_only", "no_desi", "desi_only"
compute_Universe_age = False # This requires the nested sampling chains
compute_BAO_predictions = False # Compute BAO predictions for best-fit parameters for figure 4. Needs minimum chi2 first.
compute_BAO_significance_from_CMB = False # Compute BAO significance from CMB only chains
compute_BAO_significance_from_noDESI = False # Compute BAO significance from no DESI chains
compute_BAO_significance_parameter_space_CMB = False # Compute BAO significance from parameter space volume reduction
compute_BAO_significance_parameter_space_noDESI = False # Compute BAO significance from parameter space volume reduction

# --- Physical Constants ---
Sixth = 1 / 6
G = 6.6743e-11
c = 299792458
h_Planck = 6.62607e-34
hbar = h_Planck / (2 * np.pi)
k_Boltzmann = 1.380649e-23
eV = 1.602176634e-19
AU = 149597870700
Radian = 180 / np.pi  # Degrees
Degree = np.pi / 180
Arcsec = Degree / 3600
pc = AU / Arcsec
Mpc = pc * 1e6
H_100 = 100e3 / Mpc

GM_Sun = 1.32712440041279419e20
Year = 2*np.pi*np.sqrt(AU*AU*AU/GM_Sun)
Gyr = Year*1e9

# --- Cosmological Parameters ---
N_eff = 3.044  # effective neutrino number
Sum_mnu = 0.06 * eV / (c * c)
R_nu_gamma = N_eff * 7/8 * (4/11)**(4/3)
f_nu = R_nu_gamma / (1 + R_nu_gamma)

T_CMB = 2.72548  # K
rho_crit_100 = 3 * H_100**2 / (8 * np.pi * G)
rho_gamma = np.pi**2 / 15 * (k_Boltzmann**4) / (hbar**3 * c**5) * T_CMB**4

# --- Photon and Neutrino Densities ---
zeta_3 = 1.202056903159594  # ζ(3)
n_gamma = 16 * np.pi * zeta_3 * ((k_Boltzmann * T_CMB / (h_Planck * c))**3)
n_nu = 3/11 * n_gamma * (N_eff / 3)**0.68
rho_nu = n_nu * Sum_mnu
rho_nu_ur = rho_gamma * R_nu_gamma
a_nr_sqinv = (rho_nu**2) / (rho_nu_ur**2) - 1
a_nr_sq = 1 / a_nr_sqinv

w_nu = rho_nu / rho_crit_100
    #print(f"w_nu = {w_nu:.7e}")
w_gamma = rho_gamma / rho_crit_100
w_nu_ur = rho_nu_ur / rho_crit_100

import numpy as np

# --- DESI DR2 BAO Data ---
z_DESI = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
Data_DESI = np.array([7.942, 13.588, 21.863, 17.351, 19.455, 21.576, 17.641, 27.601, 14.176, 30.512, 12.817, 38.988, 8.632])
C_blocks_DESI = [0.075**2, np.array([[0.167**2, -0.459*0.167*0.425], [-0.459*0.167*0.425, 0.425**2]]), 
                 np.array([[0.177**2, -0.404*0.177*0.330], [-0.404*0.177*0.330, 0.330**2]]),
                 np.array([[0.152**2, -0.416*0.152*0.193], [-0.416*0.152*0.193, 0.193**2]]), 
                 np.array([[0.318**2, -0.434*0.318*0.221], [-0.434*0.318*0.221, 0.221**2]]),
                 np.array([[0.760**2, -0.500*0.760*0.516], [-0.500*0.760*0.516, 0.516**2]]),
                 np.array([[0.531**2, -0.431*0.531*0.101], [-0.431*0.531*0.101, 0.101**2]])]

# Initialize 13x13 covariance matrix
Cov_DESI = np.zeros((13, 13))

# DV/rd is the first element
Cov_DESI[0, 0] = C_blocks_DESI[0]  # Variance for DV/rd

# Fill in the DM/DH blocks
for i, Ci in enumerate(C_blocks_DESI[1:]):
    row = 2*i + 1
    Cov_DESI[row:row+2, row:row+2] = Ci

C_inv_total = np.linalg.inv(Cov_DESI)
C_inv_total = np.array(C_inv_total)

z_CC, H_z_CC, sigma_H_z_CC = np.loadtxt('CC_data.txt', unpack=True)

from scipy.special import lambertw
from scipy.optimize import root_scalar

def reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq):
    # NOTE (paper-vs-code): be careful with closure here.
    # Omega_Lambda is defined using (Omega_bc, Omega_gamma, Omega_nu), but E(z)^2 below also includes
    # an additional neutrino transition term parameterized by Omega_nu_ur and a_nr_sq.
    # If Omega_nu_ur is not included consistently in the z=0 closure relation, then E(0) may deviate
    # from 1. This can propagate into distances (D_M) and the compressed CMB likelihood.
    Omega_Lambda = 1 - Omega_bc - Omega_gamma - Omega_nu
    E = np.sqrt(Omega_bc * (1 + z)**3 + Omega_gamma*(1+z)**4 + Omega_Lambda + Omega_nu_ur * np.sqrt(1+1/((1+z)**2*a_nr_sq)) * (1+z)**4 )
    return E

def integrand_rec(u, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0):
    z = np.sinh(u)
    E = reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq)
    return np.cosh(u) / E

def integrand_age_rec(u, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0):
    z = np.sinh(u)
    E = reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq)
    return np.cosh(u) / (E*(1 + z))

def integrand(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0):
    E = reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq)
    return 1 / E

def integrand_Einv_simple(z, Omega_M):
    E = np.sqrt(Omega_M * (1 + z)**3 + (1 - Omega_M))
    return 1 / E

from scipy.integrate import quad

def comoving_distance_arcsinh(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0, z_star):
    integral, _ = quad(integrand_rec, 0, np.arcsinh(z_star), args=(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0))
    return c / H0 * integral  # Return in metres.

def comoving_distance(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0, z):
    integral, _ = quad(integrand, 0, z, args=(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0))
    return c / H0 * integral  # Return in metres.

comoving_distance_vec = np.vectorize(comoving_distance, excluded=['Omega_bc', 'Omega_gamma', 'Omega_nu', 'Omega_nu_ur', 'a_nr_sq', 'H0'])

def model_predictions(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0, r_d):
    DM_array = comoving_distance_vec(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0, z)
    DH_array = c / (H0 * reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq))  # DH in Mpc
    DV = np.power((DM_array[0] ** 2) * DH_array[0] * z[0], 1.0 / 3.0)
    interleaved = np.stack([DM_array[1:], DH_array[1:]], axis=1).reshape(-1)
    vec = np.concatenate([np.array([DV]), interleaved]) / r_d
    return vec

mu_cmb = np.array([1.04161312, 0.02238286, 0.14247121]) # Canphuis et al. 2025

cov_cmb = 1.0e-9 * np.array([[ 54.39601311,   0.80698092, -20.64886707],
                            [  0.80698092,   8.54260314, -13.26477391],
                            [-20.64886707, -13.26477391, 692.15996597]])

inv_cov_cmb = np.linalg.inv(cov_cmb)

def theta_star(w_b, w_bc, h):
    rho_b = w_b * rho_crit_100
    rho_bc = w_bc * rho_crit_100

    R_b_gamma = rho_b / rho_gamma
    R_bc_gamma = rho_bc / rho_gamma
    R_0 = 3/4 * R_b_gamma  # Multiply by a for other epochs

    a_eq = (1 + R_nu_gamma) / R_bc_gamma
    z_eq = 1 / a_eq - 1

    # --- Recombination and Drag Epochs ---
    g_1 = 0.0783 * (w_b**-0.238) / (1 + 39.5 * (w_b**0.763))
    g_2 = 0.560 / (1 + 21.1 * (w_b**1.81))
    z_rec = 1045 * (1 + 0.00124 * (w_b**-0.738)) * (1 + g_1 * (w_bc**g_2)) #1048 in the paper
    a_rec = 1 / (1 + z_rec)

    b_1 = 0.313 * (w_bc**-0.419) * (1 + 0.607 * (w_bc**0.674))
    b_2 = 0.238 * (w_bc**0.223)
    z_drag = 1345 * (w_bc**0.251) / (1 + 0.659 * (w_bc**0.828)) * (1 + b_1 * (w_b**b_2)) * 0.997
    a_drag = 1 / (1 + z_drag)

    # --- Ratios ---
    R_eq = R_0 * a_eq
    R_rec = R_0 * a_rec
    R_drag = R_0 * a_drag

    # --- Sound Horizon ---
    u = (np.sqrt(1 + R_rec) + np.sqrt(R_rec + R_eq)) / (1 + np.sqrt(R_eq))
    r_rec = c / H_100 * 2/3 * np.sqrt(3 / w_bc * a_eq / R_eq) * np.log(u)

    DM_rec = comoving_distance_arcsinh(w_bc / h**2, w_gamma / h**2, w_nu / h**2, w_nu_ur / h**2, a_nr_sq, h * H_100, z_rec)

    theta_star_value = r_rec / DM_rec

    return theta_star_value

def chi2_cmb(w_m, w_b, H0):
    
    theta_star_value = 100*theta_star(w_b, w_m, H0/100)

    mu_model = np.array([theta_star_value, w_b, w_m])

    chi2_cmb = (mu_model - mu_cmb).T @ inv_cov_cmb @ (mu_model - mu_cmb)

    return chi2_cmb  # Return log likelihood

from scipy.stats import multivariate_normal, norm

def log_prior(theta):
    w_b, omega_m, H0 = theta
    if not (0.00 < omega_m < 0.50):
        return -np.inf
    if not (50.0 < H0 < 100.0):
        return -np.inf
    if not (0.020 < w_b < 0.025):
        return -np.inf
    return 0  # uniform priors for omega_m and H0, Gaussian for r_d

H0_SHOES = 73.17  # Hubble constant from SH0ES in km/s/Mpc
eH0_SHOES = 0.86  # Uncertainty in H0_SHOES in km/s/Mpc
H0_SNII = 74.9  # Hubble constant from SNII in km/s/Mpc
eH0_SNII = 2.7  # Uncertainty in H0_SNII
H0_SBF = 73.8  # Hubble constant from SBF in km/s/Mpc
eH0_SBF = 2.4  # Uncertainty in H0_SBF
H0_maser = 73.9  # Hubble constant from maser in km/s/Mpc
eH0_maser = 3.0  # Uncertainty in H0_maser

def log_likelihood(theta, use_desi=True, use_h0=True, use_cc=True, use_cmb=True, desi_only=False):
    if not desi_only:
        w_b, omega_m, H0 = theta

        h = H0 / 100.0
    
        w_bc = omega_m * h**2
    else: 
        omega_m, H0_rd = theta
    chi2_total = 0
    log_det_total = 0
    
    # CMB term (always computed if use_cmb=True)
    if use_cmb:
        chi2_CMB = chi2_cmb(w_bc, w_b, H0)
        chi2_total += chi2_CMB
        log_det_total += np.log(2 * np.pi * np.linalg.det(cov_cmb))
    
    # DESI term
    if use_desi:
        if not desi_only:
            r_d = 147.05 * Mpc * (w_b/0.02236)**-0.13 * (w_bc/0.1432)**-0.23 * (N_eff/3.04)**-0.1
            predictions = model_predictions(z_DESI, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, h*H_100, r_d)
        else: 
            predictions = model_predictions(z_DESI, omega_m, 0, 0, 0, a_nr_sq, 1000*H0_rd, 1)
        diff = predictions - Data_DESI
        chi2_total += diff @ C_inv_total @ diff
        log_det_total += np.log(2 * np.pi * np.linalg.det(Cov_DESI))
    
    # H0 measurements
    if use_h0:
        chi2_total += ((H0 - H0_SHOES) / eH0_SHOES)**2
        chi2_total += ((H0 - H0_SNII) / eH0_SNII)**2
        chi2_total += ((H0 - H0_SBF) / eH0_SBF)**2
        chi2_total += ((H0 - H0_maser) / eH0_maser)**2
        log_det_total += np.log(2 * np.pi * eH0_SHOES**2)
        log_det_total += np.log(2 * np.pi * eH0_SNII**2)
        log_det_total += np.log(2 * np.pi * eH0_SBF**2)
        log_det_total += np.log(2 * np.pi * eH0_maser**2)
    
    # Cosmic Chronometers
    if use_cc:
        H_z_predicted = H0 * reduced_hubble_factor(z_CC, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq)
        chi2_total += np.sum(((H_z_predicted - H_z_CC) / sigma_H_z_CC)**2)
        log_det_total += np.sum(np.log(2 * np.pi * sigma_H_z_CC**2))

    return -0.5 * (chi2_total + log_det_total)

def chi_squared(theta, print_output=False):
    w_b, omega_m, H0 = theta

    h = H0 / 100.0
    
    w_bc = omega_m * h**2
    r_d = 147.05 * Mpc * (w_b/0.02236)**-0.13 * (w_bc/0.1432)**-0.23 * (N_eff/3.04)**-0.1
    
    predictions = model_predictions(z_DESI, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, h*H_100, r_d)
    diff = predictions - Data_DESI
    chi2_DESI = diff @ C_inv_total @ diff
    
    chi2_H0 = ((H0 - H0_SHOES) / eH0_SHOES)**2
    chi2_CMB = chi2_cmb(w_bc, w_b, H0)
    # SNII H_0
    chi2_SNII = ((H0 - H0_SNII) / eH0_SNII)**2

    # SBF H_0
    chi2_SBF = ((H0 - H0_SBF) / eH0_SBF)**2

    # Maser H_0
    chi2_maser = ((H0 - H0_maser) / eH0_maser)**2

    # Cosmic Chronometers (CC) data
    H_z_CC_predicted = H0 * reduced_hubble_factor(z_CC, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq)

    chi2_CC = np.sum(((H_z_CC_predicted - H_z_CC)/sigma_H_z_CC)**2)

    chi2_total = chi2_DESI + chi2_H0 + chi2_CMB + chi2_SNII + chi2_SBF + chi2_maser + chi2_CC
    if print_output:
        print(f"Minimum chi squared contributions:")
        print(f"Chi2 DESI: {chi2_DESI:.3f}, Chi2 H0: {chi2_H0+chi2_maser+chi2_SBF+chi2_SNII:.3f}, Chi2 CMB: {chi2_CMB:.3f}, Chi2 CC: {chi2_CC:.3f}")
        print(f"Total Chi2: {chi2_total:.3f}")
    return chi2_total  # Return log likelihood

def minimize_chi2():
    from scipy.optimize import minimize

    initial_guess = [0.022, 0.3, 70.0]  # Initial guess for [w_b, omega_m, H0]
    result = minimize(chi_squared, initial_guess, method='Nelder-Mead')

    if result.success:
        fitted_params = result.x
        print("Fitted parameters:")
        print(f"w_b: {fitted_params[0]:.6f}, omega_m: {fitted_params[1]:.6f}, H0: {fitted_params[2]:.6f}")
        chi2_min = chi_squared(fitted_params, print_output=True)
    else:
        print("Optimization failed.")
    return result

if compute_minimum_chi2:
    results_min_chi2 = minimize_chi2()

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

import numpy as np
from dynesty import NestedSampler
from scipy.stats import norm

# --- Define the prior transform ---
def prior_transform(u):
    if not nested_sampling_data == "desi_only":
        # u: array of 3 numbers in [0, 1]
        w_b = 0.020 + 0.005 * u[0]        # [0.020, 0.025]
        omega_m = 0.01 + 0.48 * u[1]         # [0.01, 0.49]
        H0 = 50.0 + 50.0 * u[2]              # [50.0, 100.0]
        return [w_b, omega_m, H0]
    else:
        omega_m = 0.01 + 0.48 * u[0]         # [0.01, 0.49]
        H0_rd = 5000 + 8000 * u[1]         # [0.050, 0.100] in units of 1000 km/s/Mpc
        return [omega_m, H0_rd]

# --- Set up and run dynesty ---
if not nested_sampling_data == "desi_only":
    ndim = 3
else:
    ndim = 2
nprocs = 4  # or however many CPU cores you want to use

import multiprocessing

if perform_nested_sampling:
    # Define which datasets to use
    if nested_sampling_data == "CMB_only":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=False, use_h0=False, use_cc=False, use_cmb=True)
    elif nested_sampling_data == "all":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=True, use_h0=True, use_cc=True, use_cmb=True)
    elif nested_sampling_data == "no_desi":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=False, use_h0=True, use_cc=True, use_cmb=True)
    elif nested_sampling_data == "desi_only":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=True, use_h0=False, use_cc=False, use_cmb=False, desi_only=True)
    else:
        likelihood_func = log_likelihood
    
    with multiprocessing.Pool(processes=nprocs) as pool:
        sampler = NestedSampler(
            likelihood_func,
            prior_transform,
            ndim,
            nlive=500,
            bound='multi',
            sample='rwalk',
            pool=pool,
            queue_size=nprocs
        )
        sampler.run_nested(dlogz=0.01, print_progress=True)
        results = sampler.results

    # --- Extract evidence and samples ---
    logZ = results.logz[-1]
    logZerr = results.logzerr[-1]
    print(f"Log evidence: {logZ:.3f} ± {logZerr:.3f}")

    # Posterior samples (weighted)
    from dynesty import utils as dyfunc
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    mean = np.average(samples, axis=0, weights=weights)
    print("Posterior mean:", mean)

    if not nested_sampling_data == "desi_only":
        # If your log_posterior can accept a 2D array, you can do:
        log_posteriors = np.apply_along_axis(log_posterior, 1, samples)

        print("Log posteriors shape:", log_posteriors.shape)
    if nested_sampling_data == "CMB_only":
        np.savez("dynesty_results_LCDM_final_CMB_only.npz", samples=samples, weights=weights, logp_posteriors = log_posteriors)
    elif nested_sampling_data == "no_desi":
        np.savez("dynesty_results_LCDM_final_no_desi.npz", samples=samples, weights=weights, logp_posteriors = log_posteriors)
    elif nested_sampling_data == "desi_only":
        np.savez("dynesty_results_LCDM_final_desi_only.npz", samples=samples, weights=weights)
    else:
        np.savez("dynesty_results_LCDM_final.npz", samples=samples, weights=weights, logp_posteriors = log_posteriors)

def age_of_universe(Omega_bc, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0):
    h = H0 / 100.0
    def integrand(a):
        z = 1 / a - 1
        E = a * reduced_hubble_factor(z, Omega_bc, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq)
        return 1.0 / E

    integral, _ = quad(integrand, 0, 1)
    t_0 = integral / H0 # H0 in km/s/Mpc
    t_0 = t_0 * Mpc / (1e3)  # Convert to seconds
    t_0 = t_0 / Gyr  # Convert seconds to Gyr
    return t_0

#def reduced_hubble_factor_cpl(z, Omega_bc, w_gamma, w_nu, w_nu_ur, a_nr_sq, w0, wa):
#    Omega_Lambda = 1 - Omega_bc - w_gamma - w_nu
#    w_DE = w0 + wa * (z / (1 + z))
#    E = np.sqrt(Omega_bc * (1 + z)**3 + w_gamma*(1+z)**4 + Omega_Lambda * (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z)) + w_nu_ur * np.sqrt(1+1/((1+z)**2*a_nr_sq)) * (1+z)**4 )
#    return E

#def age_of_universe_cpl(Omega_bc, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0, w0, wa):
#    h = H0 / 100.0
#    def integrand(a):
#        z = 1 / a - 1
#        E = a * reduced_hubble_factor_cpl(z, Omega_bc, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, w0, wa)
#        return 1.0 / E

#    integral, _ = quad(integrand, 0, 1)
#    t_0 = integral / H0 # H0 in km/s/Mpc
#    t_0 = t_0 * Mpc / (1e3)  # Convert to seconds
#    t_0 = t_0 / Gyr  # Convert seconds to Gyr
#    return t_0

#Omega_bc_cpl = 0.353
#H0_cpl = 63.6
#w0_cpl = -0.42
#wa_cpl = -1.75

if compute_Universe_age:
    try:
        data = np.load("dynesty_results_LCDM_final.npz")
        samples = data['samples']
        weights = data['weights']
    except FileNotFoundError:
        raise FileNotFoundError("Nested sampling results file not found. Please run nested sampling first.")

    ages = np.array([age_of_universe(theta[1], w_gamma, w_nu, w_nu_ur, a_nr_sq, theta[2]) for theta in samples])
    mean_age = np.average(ages, weights=weights)
    variance_age = np.average((ages - mean_age)**2, weights=weights)
    std_age = np.sqrt(variance_age)

    print(f"Universe Age from Nested sampling chain: {mean_age:.3f} ± {std_age:.3f} Gyr")

    #print("Universe Age for CPL best-fit parameters:")
    #age_cpl = age_of_universe_cpl(Omega_bc_cpl, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0_cpl, w0_cpl, wa_cpl)
    #print(f"Age (CPL best-fit): {age_cpl:.3f} Gyr")

if compute_BAO_predictions:
    z_BAO = np.loadtxt("f_Q_gravity_bao_data.txt", usecols=0, unpack=True)

    def comoving_distance_Hubble_distance_from_z(Omega_m, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0, z):
        """
        Calculate the comoving distance and Hubble distances in Mpc from redshift z using the Hubble distance.
        """
        h = H0 / 100.0
        comoving_distance_ = comoving_distance_vec(Omega_m, w_gamma/(h**2), w_nu/(h**2), w_nu_ur/(h**2), a_nr_sq, h*H_100, z)  # in Mpc
        Hubble_distance_ = c / (h*H_100*reduced_hubble_factor(z, Omega_m, w_gamma/(h**2), w_nu/(h**2), w_nu_ur/(h**2), a_nr_sq))  # DH in Mpc
        return comoving_distance_, Hubble_distance_  # Return in Mpc

    w_b_mean, Omega_m_mean, H0_mean = results_min_chi2.x

    z_general = np.arange(0.0, 5.001, 0.001)

    comoving_distance_general, Hubble_distance_general = comoving_distance_Hubble_distance_from_z(
        Omega_m_mean, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0_mean, z_general
        )

    comoving_distance_BAO, Hubble_distance_BAO = comoving_distance_Hubble_distance_from_z(
        Omega_m_mean, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0_mean, z_BAO
        )
    
    w_bc_mean = Omega_m_mean * (H0_mean / 100.0)**2

    r_d = 147.05 * Mpc * (w_b_mean/0.02236)**-0.13 * (w_bc_mean/0.1432)**-0.23 * (N_eff/3.04)**-0.1 # Sound horizon at drag epoch in Mpc

    with open("output/fQ_gravity_LCDM_bao_predictions.txt", "w") as f:
        f.write("#BAO predictions for the LCDM model\n")
        f.write("# z, DM, DH, DV, DM/r_d, DH/r_d, DV/r_d\n")
        for z, DM, DH in zip(z_BAO, comoving_distance_BAO, Hubble_distance_BAO):
            DV = (DM**2 * DH * z)**(1/3)
            f.write(f"{z:.6f} {DM/Mpc:.6f} {DH/Mpc:.6f} {DV/Mpc:.6f} {DM/r_d:.6f} {DH/r_d:.6f} {DV/r_d:.6f}\n")

    with open("output/fQ_gravity_LCDM_general_predictions.txt", "w") as f:
        f.write("#General predictions for the LCDM model\n")
        f.write("# z, DM, DH, DV, DM/r_d, DH/r_d, DV/r_d\n")
        for z, DM, DH in zip(z_general, comoving_distance_general, Hubble_distance_general):
            DV = (DM**2 * DH * z)**(1/3)
            f.write(f"{z:.6f} {DM/Mpc:.6f} {DH/Mpc:.6f} {DV/Mpc:.6f} {DM/r_d:.6f} {DH/r_d:.6f} {DV/r_d:.6f}\n")

if compute_BAO_significance_from_CMB:
    print("Computing BAO significance from CMB-only chains...")
    try:
        data_cmb = np.load("dynesty_results_LCDM_final_CMB_only.npz")
        samples_cmb = data_cmb['samples']
        weights_cmb = data_cmb['weights']
    except FileNotFoundError:
        raise FileNotFoundError("CMB-only Nested sampling results file not found. Please run nested sampling with CMB-only data first.")
    
    BAO_predictions = np.zeros((samples_cmb.shape[0], Data_DESI.shape[0]))
    # Compute the BAO predictions for each sample in the CMB-only chains
    for i, theta in enumerate(samples_cmb):
        w_b, omega_m, H0 = theta
        h = H0 / 100.0
        w_m = omega_m * h**2
        r_d = 147.05 * Mpc * (w_b/0.02236)**-0.13 * (w_m/0.1432)**-0.23 * (N_eff/3.04)**-0.1
        BAO_predictions[i] = model_predictions(z_DESI, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, h*H_100, r_d)
    
    # Compute the mean and covariance of the BAO predictions
    mean_BAO_predictions = np.average(BAO_predictions, axis=0, weights=weights_cmb)
    cov_BAO_predictions = np.cov(BAO_predictions.T, aweights=weights_cmb)
    correlation_matrix = np.zeros_like(cov_BAO_predictions)
    
    # Now compute the chi-squared difference between the DESI data and the CMB-only BAO predictions
    diff_BAO = Data_DESI - mean_BAO_predictions
    total_cov = Cov_DESI + cov_BAO_predictions
    inv_total_cov = np.linalg.inv(total_cov)

    chi2_BAO_significance = diff_BAO @ inv_total_cov @ diff_BAO
    print(f"Chi-squared difference for BAO significance from CMB-only chains: {chi2_BAO_significance:.3f}")
    #Compute the tension in units of sigma
    dof = len(Data_DESI) 
    print(f"Degrees of freedom: {dof}")
    from scipy.stats import chi2, norm
    p_value = chi2.sf(chi2_BAO_significance, dof)
    sigma_tension = norm.isf(p_value/2)
    print(f"BAO significance from CMB-only chains in observation space: {sigma_tension:.3f} sigma")
    print(f"P-value: {p_value:.3e}")

    # Diagnostic: Compare the "size" of the covariance matrices
    #trace_data = np.trace(Cov_DESI)
    #trace_model = np.trace(cov_BAO_predictions)

    #print(f"Trace of Data Covariance: {trace_data:.2e}")
    #print(f"Trace of Model Covariance: {trace_model:.2e}")
    #print(f"Ratio (Model/Data): {trace_model/trace_data:.2%}")

if compute_BAO_significance_from_noDESI:
    print("Computing BAO significance from no-DESI chains...")
    try:
        data_no_desi = np.load("dynesty_results_LCDM_final_no_desi.npz")
        samples_no_desi = data_no_desi['samples']
        weights_no_desi = data_no_desi['weights']
    except FileNotFoundError:
        raise FileNotFoundError("No-DESI Nested sampling results file not found. Please run nested sampling with no-DESI data first.")
    
    BAO_predictions = np.zeros((samples_no_desi.shape[0], Data_DESI.shape[0]))
    # Compute the BAO predictions for each sample in the no-DESI chains
    for i, theta in enumerate(samples_no_desi):
        w_b, omega_m, H0 = theta
        h = H0 / 100.0
        w_m = omega_m * h**2
        r_d = 147.05 * Mpc * (w_b/0.02236)**-0.13 * (w_m/0.1432)**-0.23 * (N_eff/3.04)**-0.1
        BAO_predictions[i] = model_predictions(z_DESI, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, h*H_100, r_d)
    
    # Compute the mean and covariance of the BAO predictions
    mean_BAO_predictions = np.average(BAO_predictions, axis=0, weights=weights_no_desi)
    cov_BAO_predictions = np.cov(BAO_predictions.T, aweights=weights_no_desi)

    # Now compute the chi-squared difference between the DESI data and the no-DESI BAO predictions
    diff_BAO = Data_DESI - mean_BAO_predictions
    total_cov = Cov_DESI + cov_BAO_predictions
    inv_total_cov = np.linalg.inv(total_cov)

    chi2_BAO_significance = diff_BAO @ inv_total_cov @ diff_BAO
    print(f"Chi-squared difference for BAO significance from no-DESI chains: {chi2_BAO_significance:.3f}")
    #Compute the tension in units of sigma
    dof = len(Data_DESI) 
    print(f"Degrees of freedom: {dof}")
    from scipy.stats import chi2, norm
    p_value = chi2.sf(chi2_BAO_significance, dof)
    sigma_tension = norm.isf(p_value/2)
    print(f"BAO significance from no-DESI chains in observation space: {sigma_tension:.3f} sigma")
    print(f"P-value: {p_value:.3e}")

if compute_BAO_significance_parameter_space_CMB:
    print("Computing BAO significance from parameter space...")
    try:
        data_all = np.load("dynesty_results_LCDM_final_desi_only.npz")
        samples_all = data_all['samples']
        weights_all = data_all['weights']
        data_cmb = np.load("dynesty_results_LCDM_final_CMB_only.npz")
        samples_cmb = data_cmb['samples']
        weights_cmb = data_cmb['weights']
    except FileNotFoundError:
        raise FileNotFoundError("All-data Nested sampling results file not found. Please run nested sampling with all data first.")
    
    omega_m_values = samples_all[:,0]
    H0_rd_values = samples_all[:,1]
    omega_m_mean = np.average(omega_m_values, weights=weights_all)
    H0_rd_mean = np.average(H0_rd_values, weights=weights_all)
    covariance_omega_m_H0_rd_desi = np.cov(np.stack((omega_m_values, H0_rd_values)), aweights=weights_all)
    #covariance_omega_m_H0_rd_desi = np.diag(np.diag(covariance_omega_m_H0_rd_desi))

    wb_values = samples_cmb[:,0]
    Omega_bc_values = samples_cmb[:,1]
    h_values = samples_cmb[:,2] / 100.0
    h_sq_values = h_values * h_values

    rd = 147.05 * (wb_values/0.02236)**-0.13 * ((Omega_bc_values*h_sq_values)/0.1432)**-0.23 * (N_eff/3.04)**-0.1
    H0_rd_cmb = 100.0*rd*h_values
    omega_m_cmb = Omega_bc_values + w_nu/h_sq_values
    omega_m_cmb_mean = np.average(omega_m_cmb, weights=weights_cmb)
    #print(f"Omega_m CMB mean: {omega_m_cmb_mean:.5f}, DESI mean: {omega_m_mean:.5f}")
    H0_rd_cmb_mean = np.average(H0_rd_cmb, weights=weights_cmb)
    #print(f"H0*rd CMB mean: {H0_rd_cmb_mean:.3f}, DESI mean: {H0_rd_mean:.3f}")
    covariance_omega_m_H0_rd_cmb = np.cov(np.stack((omega_m_cmb, H0_rd_cmb)), aweights=weights_cmb)
    #covariance_omega_m_H0_rd_cmb = np.diag(np.diag(covariance_omega_m_H0_rd_cmb))

    mean_diff = np.array([omega_m_mean - omega_m_cmb_mean, H0_rd_mean - H0_rd_cmb_mean])
    total_covariance = covariance_omega_m_H0_rd_desi + covariance_omega_m_H0_rd_cmb
    mean_chi2_desi = mean_diff.T @ np.linalg.inv(total_covariance) @ mean_diff

    print(f"Mean Chi-squared for DESI from parameter space: {mean_chi2_desi:.3f}")
    from scipy.stats import chi2, norm
    dof = 2 
    p_value = chi2.sf(mean_chi2_desi, dof)
    sigma_tension = norm.isf(p_value/2)
    print(f"BAO significance from parameter space: {sigma_tension:.3f} sigma")
    print(f"P-value: {p_value:.3e}")

if compute_BAO_significance_parameter_space_noDESI:
    print("Computing BAO significance from parameter space (no-DESI)...")
    try:
        data_all = np.load("dynesty_results_LCDM_final_desi_only.npz")
        samples_all = data_all['samples']
        weights_all = data_all['weights']
        data_no_desi = np.load("dynesty_results_LCDM_final_no_desi.npz")
        samples_no_desi = data_no_desi['samples']
        weights_no_desi = data_no_desi['weights']
    except FileNotFoundError:
        raise FileNotFoundError("No-DESI Nested sampling results file not found. Please run nested sampling with no-DESI data first.")
    
    omega_m_values = samples_all[:,0]
    H0_rd_values = samples_all[:,1]
    omega_m_mean = np.average(omega_m_values, weights=weights_all)
    H0_rd_mean = np.average(H0_rd_values, weights=weights_all)
    covariance_omega_m_H0_rd_desi = np.cov(np.stack((omega_m_values, H0_rd_values)), aweights=weights_all)
    #covariance_omega_m_H0_rd_desi = np.diag(np.diag(covariance_omega_m_H0_rd_desi))

    wb_values = samples_no_desi[:,0]
    Omega_bc_values = samples_no_desi[:,1]
    h_values = samples_no_desi[:,2] / 100.0
    h_sq_values = h_values * h_values

    rd = 147.05 * (wb_values/0.02236)**-0.13 * ((Omega_bc_values*h_sq_values)/0.1432)**-0.23 * (N_eff/3.04)**-0.1
    H0_rd_noDESI = 100.0*rd*h_values
    omega_m_noDESI = Omega_bc_values + w_nu/h_sq_values
    omega_m_noDESI_mean = np.average(omega_m_noDESI, weights=weights_no_desi)
    #print(f"Omega_m CMB mean: {omega_m_cmb_mean:.5f}, DESI mean: {omega_m_mean:.5f}")
    H0_rd_noDESI_mean = np.average(H0_rd_noDESI, weights=weights_no_desi)
    #print(f"H0*rd CMB mean: {H0_rd_cmb_mean:.3f}, DESI mean: {H0_rd_mean:.3f}")
    covariance_omega_m_H0_rd_noDESI = np.cov(np.stack((omega_m_noDESI, H0_rd_noDESI)), aweights=weights_no_desi)

    mean_diff = np.array([omega_m_mean - omega_m_noDESI_mean, H0_rd_mean - H0_rd_noDESI_mean])
    total_covariance = covariance_omega_m_H0_rd_desi + covariance_omega_m_H0_rd_noDESI
    mean_chi2_desi = mean_diff.T @ np.linalg.inv(total_covariance) @ mean_diff
    print(f"Mean Chi-squared for DESI from parameter space (no-DESI): {mean_chi2_desi:.3f}")
    from scipy.stats import chi2, norm
    dof = 2 
    p_value = chi2.sf(mean_chi2_desi, dof)
    sigma_tension = norm.isf(p_value/2)
    print(f"BAO significance from parameter space (no-DESI): {sigma_tension:.3f} sigma")
    print(f"P-value: {p_value:.3e}")