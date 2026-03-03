Repository to produce the results in https://arxiv.org/abs/2510.20964 where we explore possible background solutions to the Hubble and BAO tensions. We explore phenomenological modified expansions to LCDM and f(Q) symmetric teleparallel gravity models.

Each file runs a different model. We include the files to run LCDM, the phenomenological models and the three f(Q) gravity models (Exp, Log, Tanh) along with their extensions includying the cosmological constant.

On top of each file, we have several flags to control what we want to compute. 

1. "compute_minimum_chi2": this computes the minimum chi squared contributions reported in table V.
2. "perform_nested_sampling": this does the nested sampling MCMC to compute the Bayesian evidence and save the samples to an npz file for using it later. These samples are required for all the following flags.
3. "nested_sampling_data": if running the nested sampling MCMC, this sets which data we are using. There is the option of running "all" (CMB+H0+CC+DESI BAO), "CMB_only", "no_desi" (CMB+H0+CC) and "desi_only".
4. "compute_Universe_age": compute the age of the Universe. It requires the MCMC chains run with "all".
5. "compute_BAO_predictions": data required for figures 5, 6, and 7.
6. "compute_BAO_significance_from_CMB": compute the data space tension between CMB and BAO. It requires the chains with "CMB_only". Reported in Table VI.
7. "compute_BAO_significance_from_noDESI": compute the data space tension between CMB+H0+CC and BAO. It requires the chains with "no_desi". Reported in Table VII.
8. "compute_BAO_significance_parameter_space_CMB": compute the parameter space tension between CMB and BAO. It requires the chains with "CMB_only" and "desi_only". Reported in Table VI.
9. "compute_BAO_significance_parameter_space_noDESI": compute the parameter space tension between CMB+H0+CC and BAO. It requires the chains with "no_desi" and "desi_only". Reported in Table VII.

For the phenomenological models, the code needs to be run with extra parsing arguments. For instance, to run the "Phen, exp" model with all the data, it needs "python dynesty_phen.py --data all --function exp".
