# Problem definition

The provided dataset contains results of a simulated observational campaign aimed at calibrating the period-luminosity relation for Cepheid variables. The relation is assumed to take the form:

$$M = a [\log_{10}(P) - 1] + b.$$

The dataset is a database of observations stored as a PDF table. The data has been partly corrupted, so not all rows contain all data — discard incomplete rows and use only those where a galaxy spectrum, an estimated Cepheid period, and its mean apparent magnitude are all available.

Assume that the redshift of a Cepheid in a galaxy equals the redshift of the whole galaxy due to cosmological expansion. Assume the validity of Hubble's law in the form:

$$v_{\text{rec}} = H_0 \cdot d$$

across the whole provided dataset, with $H_0 = 67\\,\text{km}\\,\text{s}^{-1}\\,\text{Mpc}^{-1}$. Estimate galaxy redshifts using template cross-correlation in log-$\lambda$ space. Since the redshifts are small, assume the classical formula for the redshift. The laboratory reference spectrum is provided together with the dataset.

Produce the following outputs:

1. A PDF table with two columns — one listing Cepheids ordered by their ID and the other showing the estimated redshifts to 5 decimal places.
2. A plot of the data points used for the fit, including the best-fit straight line motivated by the period-luminosity relation.
3. A TXT file clearly stating the estimated fit parameters with uncertainties derived from the linear regression.
________________________________________

## Files provided

- `cepheides_measurements.pdf` — database of observations
- `spectral_lines_lab.png` — laboratory reference spectrum
- `spectra/` — folder containing observed galaxy spectra images