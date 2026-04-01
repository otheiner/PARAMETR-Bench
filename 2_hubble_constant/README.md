## Problem definition

The provided dataset contains results of a simulated observational campaign surveying galaxy spectra and Cepheid variable stars. For Cepheid variables, the period-luminosity relation is established as:

$$M = -2.43 \cdot (\log_{10}(P) - 1) - 4.05$$

The dataset contains measured spectra of several galaxies and photometric information about Cepheid variables in these galaxies, where available. There is at most one Cepheid per galaxy. Analysis of the measured spectra verified that the observed spectra contain only lines that were observed in the laboratory reference spectrum.

Use the provided dataset to determine Hubble's constant, incorporating all available data. Apply unweighted linear regression with the intercept fixed at the origin, consistent with the physical expectation that recession velocity vanishes at zero distance. Assume the relativistic Doppler formula for computing velocities from redshifts.

For redshift extraction, identify observed spectral lines with the lines in the laboratory reference spectrum. Estimate the redshift from the line with the largest laboratory wavelength present in the observed spectrum to minimize reading error. Assume that the brightness profile of each line can be modelled by a normal distribution, with the wavelength of the line corresponding to the mean of the distribution.

Produce the following outputs:

1. A PDF file containing a list of only those galaxies for which a spectroscopic measurement exists in the dataset and which also have a Cepheid variable allowing their distance to be determined. The list should state the redshift of each galaxy and be ordered by galaxy ID.
2. A scatter plot in a PNG file showing the recession velocity of each galaxy as a function of the distance estimated from its Cepheid variable. The plot should include a physically motivated fit of Hubble's law, with the fitted equation stated quantitatively.
3. A TXT file with a clearly stated value of Hubble's constant from this dataset, including the statistical uncertainty.
________________________________________

## Files provided

- `cepheides_measurements.csv` — database of observations of Cepheides in galaxies
- `spectral_lines_lab.png` — laboratory reference spectrum
- `observed_spectra/` — folder containing observed galaxy spectra images
________________________________________

## About the dataset

This dataset contains simulated emission spectra of multiple galaxies. I used python matplotlib library for creating images. Data are simulated by using tabulated data of some of the brightest spectral lines in the range of 400-750 nm. The brightness profile of these lines is approximated by normal distribution. These spectra were then shifted according to Doppler effect for each of the hypothetical galaxy. Redshifts of these galaxies come from the uniform distribution. The dataset also contains simulated apparent magnitudes and periods of hypothetical Cepheid variables in these galaxies. I modelled these numbers to match possibly measured values using Pogson's equation and Hubble's law, in order to get physically realistic result of the Hubble's constant. Finally, I smeared the data by normal distribution.
