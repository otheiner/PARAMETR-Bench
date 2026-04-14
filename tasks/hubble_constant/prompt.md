# Hubble Constant Estimation

The provided dataset contains results of a simulated observational campaign surveying galaxy spectra and Cepheid variable stars. For Cepheid variables, the period-luminosity relation is established as:

$$M = -2.43 \cdot (\log_{10}(P) - 1) - 4.05$$

The dataset contains measured spectra of several galaxies and photometric information about Cepheid variables in these galaxies, where available. There is at most one Cepheid per galaxy. Analysis of the measured spectra verified that the observed spectra contain only lines that were observed in the laboratory reference spectrum.

Use the provided dataset to determine Hubble's constant, incorporating all available data. Fit the data by physically well-motivated function. Assume the relativistic Doppler formula for computing velocities from redshifts. For redshift extraction, identify observed spectral lines with the lines in the laboratory reference spectrum. Assume that spectral lines in the observed spectrum are always subset of the presented lab spectrum.

State the result in km/s/Mpc rounded to one decimal place ± absolute statistical uncertainty. Also list computed redshifts of galaxies used for the analysis.
________________________________________
## Output format

Enclose your final results into formatting tags <final_result> </final_result> and structure your response exactly as follows:

<final_result> 
REDSHIFTS OF GALAXIES USED IN THE ANALYSIS:
<GID...>    <redshift value>
<GID...>    <redshift value>

HUBBLE CONSTANT ± STDEV:
<H_0> ± <STDEV> km/s/Mpc
</final_result>

________________________________________
## Files provided

Files provided to solve the task are:

- `<cepheids_measurements.csv>` — database of measured Cepheids
- `<spectral_lines_lab.png>` — laboratory emission spectrum
- `<observed_spectra/>` — folder containing observed emission spectra of some galaxies