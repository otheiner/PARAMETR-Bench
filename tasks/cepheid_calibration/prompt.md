## Problem definition

The provided dataset contains results of an observational campaign aimed at calibrating the period-luminosity relation for Cepheid variables. The relation is assumed to take the form:

$$M = a [\log_{10}(P) - 1] + b.$$

The dataset is a database of observations stored as a CSV table. The data has been partly corrupted, so not all rows contain all data — discard incomplete rows and use only those where a galaxy spectrum, an estimated Cepheid period, and its mean apparent magnitude are all available.

Assume that the redshift of a Cepheid in a galaxy equals the redshift of the whole galaxy due to cosmological expansion. Assume the validity of Hubble's law in the form:

$$v_{\text{rec}} = H_0 \cdot d$$

across the whole provided dataset, with $H_0 = 67\\,\text{km}\\,\text{s}^{-1}\\,\text{Mpc}^{-1}$. Estimate galaxy redshifts (assumed to be same as the redshift of the Cepheid) using template cross-correlation in log-$\lambda$ space. Since the redshifts are small, assume the classical formula for the redshift. The laboratory reference spectrum is provided together with the dataset.

Use the provided data to estimate numeric values of parameters $a$ and $b$ in the formula relating period of Cepheid and its absolute magnitude. The estimated values have to be based on the data from the provided dataset. In your results include statistical uncertainties on your estimates. Also, for each Cepheid used in the analysis, state its measured redshift.

________________________________________
## Output format

The final output is used to grade your answer.

Structure your response according to the following template:

SUMMARY:
[high level summary of your approach, including key formulas and methods used to solve the task]

ESTIMATED REDSHIFTS
CEPHEID_ID , REDSHIFT
[ID1] , [z1]
[ID2] , [z2]
.....

PARAMETER a:
[a_value]
UNCERTAINTY ON PARAMETER a:
[stdev_a]

PARAMETER b:
[b_value]
UNCERTAINTY ON PARAMETER b:
[stdev_b]

________________________________________
## Files provided

Files provided to solve the task are:

- `cepheids_measurements.csv` — dataset containing information about Cepheid variables in some galaxies. 
- `spectral_lines_lab.png` — graphical representation of measured laboratory spectrum
- `observed_spectra/` — folder containing graphical representation of measured emission spectra of galaxies