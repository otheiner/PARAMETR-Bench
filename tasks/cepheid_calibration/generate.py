"""
Task: cepheid_calibration
Description: <FILL-THIS-IN>

Author: Ondrej Theiner
"""

from src.task import Task
import numpy as np
import random
import math
import string
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import correlate
from matplotlib.ticker import MultipleLocator, FuncFormatter
from collections import defaultdict


class CepheidCalibration(Task):
    """
    <FILL-THIS-IN> DESCRIBE WHAT THIS TASK SIMULATES AND WHAT THE MODEL MUST DO
    """

    # Helper function to roduces gaussian profile of the spectral peak
    @staticmethod
    def gaussian(x, mu, sigma, amplitude):
        return amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # ###########################################################
    # # Function to produce graphical spectra
    # ###########################################################
    def plot_spectrum(self, input_spectrum: dict[float, float],
                            lambda_min : float,
                            lambda_max : float,  
                            save_to: str,
                            title: str = '', 
                            name: str = '') -> None:

        wavelength = np.linspace(lambda_min, lambda_max, 3500)
        intensity = np.zeros_like(wavelength)
        for mu, amp in input_spectrum.items():
            intensity += self.gaussian(wavelength, float(mu), sigma=0.1, amplitude=amp)
        intensity /= intensity.max()

        fig, ax = plt.subplots(figsize=(14, 2), dpi=150)
        ax.plot(wavelength, intensity, color='white', linewidth=0.5)
        ax.fill_between(wavelength, intensity, alpha=0.3, color='cyan')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        ax.set_title(title, color='white')
        ax.set_xlabel("Wavelength (nm)", color='white')
        ax.set_ylabel("Relative Intensity", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis='x', which='major', length=8, colors='white')
        ax.tick_params(axis='x', which='minor', length=4, colors='white')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}' if x % 50 == 0 else ''))

        ax.set_xlim(400, 750)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        buf.seek(0)
        rgba = plt.imread(buf)
        gray = np.dot(rgba[:, :, :3], [0.299, 0.587, 0.114])
        plt.imsave(f"{save_to}/{name}.png", gray, cmap='gray', vmin=0, vmax=1)
    
    # ###########################################################
    # # Function simulating reading error to experimentally estimate
    # # z-log space error
    # ###########################################################
    @staticmethod
    def apply_reading_error(spectrum: dict, resolution: float) -> dict:
        """Round wavelengths to nearest multiple of `resolution` nm.
        If two lines collide, keep the brighter one (or sum them — see comment)."""
        rounded = defaultdict(float)
        for wl, amp in spectrum.items():
            key = round(float(wl) / resolution) * resolution
            rounded[key] = max(rounded[key], amp)   # or: rounded[key] += amp to sum
        return dict(rounded)

    # ###########################################################
    # # Function doing the spectral lines fit - cross-correlation 
    # # in log space. Also returns estimated error of the fit.
    # ###########################################################
    def estimate_z_logspace(self, observed_spectrum: dict, 
                                lab_spectrum: dict,
                                wav_min: float, 
                                wav_max: float,
                                reading_resolution : float,
                                n_pts=3500) -> tuple[float, float]:
        log_wav = np.linspace(np.log(wav_min), np.log(wav_max), n_pts)

        def make_intensity_log(spectrum):
            inten = np.zeros(n_pts)
            for mu, amp in spectrum.items():
                log_mu = np.log(float(mu))
                inten += amp * np.exp(-(log_wav - log_mu)**2 / (2 * 0.0003**2))
            return inten / inten.max()

        lab_inten = make_intensity_log(self.apply_reading_error(lab_spectrum, reading_resolution))
        obs_inten = make_intensity_log(self.apply_reading_error(observed_spectrum, reading_resolution))

        corr = correlate(obs_inten, lab_inten, mode='full')
        lags = np.arange(-(n_pts - 1), n_pts)
        best_idx = np.argmax(corr)
        best_lag = lags[best_idx]

        # Fit parabola around the peak using 5 neighbouring points
        half_w = 5
        idx_lo = max(best_idx - half_w, 0)
        idx_hi = min(best_idx + half_w + 1, len(corr))
        local_lags = lags[idx_lo:idx_hi]
        local_corr = corr[idx_lo:idx_hi]
        coeffs = np.polyfit(local_lags, local_corr, 2)   # a*x^2 + b*x + c
        # Peak of parabola at x = -b/2a, width ~ 1/sqrt(-a)
        a_coeff = coeffs[0]
        sigma_lag = 1.0 / np.sqrt(-a_coeff) if a_coeff < 0 else np.nan

        log_wav_per_sample = (np.log(wav_max) - np.log(wav_min)) / n_pts
        delta_log_wav       = best_lag * log_wav_per_sample
        sigma_log_wav       = sigma_lag * log_wav_per_sample

        z      = np.exp(delta_log_wav) - 1 # z = e^(Δlog λ) - 1
        sigma_z = np.exp(delta_log_wav) * sigma_log_wav   # error propagation

        return z, sigma_z

    # ############################################################
    # # Main method that needs to be implemented by user
    # ############################################################
    def _generate(self):
        """
        <FILL-THIS-IN> DESCRIBE HOW THE DATA IS GENERATED, WHAT THE GROUND TRUTH IS AND HOW THE
        TASK IS POPULATED
        """
        # ======= RANDOMNESS =======
        # Always use self.seed for reproducibility. Different generators can be used
        # but the seed has to be specified and it has to be self.seed. Seeds derived
        # deterministically from this seed are also fine. Use np.random.seed(self.seed),
        # or random.seed(self.seed). Doing something like np.random.seed(self.seed + 1)
        # is also fine if you need to generate different random numbers in different places.
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Create folder to store the observed spectra of galaxies
        spectra_folder = self.input_dir / 'observed_spectra'
        os.makedirs(spectra_folder)

        # ====== CONFIGURATION =======
        # Use config parameters as you define them in config.json. Avoid hardcoded values
        # in this method if the number might come from generating distributions configured
        # in the config.json. Use the configuration like this:
        # PARAMETER_1 = self.get_params()['PARAMETER_1']
        # PARAMETER_2 = self.get_params()['PARAMETER_2']
        # Variable parameters
        N_GALAXIES = self.get_params()["N_GALAXIES"]
        CEPHEID_GENERATE_PROBABILITY = self.get_params()["CEPHEID_GENERATE_PROBABILITY"]
        SPECTRUM_GENERATE_PROBABILITY = self.get_params()["SPECTRUM_GENERATE_PROBABILITY"]
        READING_PRECISION = self.get_params()["READING_PRECISION"]
        
        # Fixed parameters
        c = self.get_params()["c"]
        H0 = self.get_params()["H0"]
        Z_MAX = self.get_params()["Z_MAX"] 
        SIGMA_DISTANCE_CEPHEIDES = self.get_params()["SIGMA_DISTANCE_CEPHEIDES"] 
        LAMBDA_MIN = self.get_params()["LAMBDA_MIN"] #nm
        LAMBDA_MAX = self.get_params()["LAMBDA_MAX"] #nm
        LAB_SPECTRUM = self.get_params()["LAB_SPECTRUM"]

        # Cepheid calibration parameters for simulation
        # Roughly copies physics but this approach forces model to do the analysis properly
        A_MIN = self.get_params()["A_MIN"]
        A_MAX = self.get_params()["A_MAX"]
        B_MIN = self.get_params()["B_MIN"]
        B_MAX = self.get_params()["B_MAX"]
        a = np.random.uniform(A_MIN, A_MAX)
        b = np.random.uniform(B_MIN, B_MAX)

        RESULTS_A_B_TOLERANCE = self.get_params()["RESULTS_A_B_TOLERANCE"]

        # Rows accumulated per galaxy; concat once after the loop
        rows = []
        
        redshifts = pd.DataFrame({'cepheid_ID': pd.Series(dtype='str'),
                                  'z' : pd.Series(dtype='float') , 
                                  'z_allowed_interval' : pd.Series(dtype='float') ,
                                })
        
        excluded_cepheids = pd.DataFrame({'cepheid_ID': pd.Series(dtype='str')})
        
        results = pd.DataFrame({'a' : pd.Series(dtype='float') ,
                                'a_allowed_interval' : pd.Series(dtype='str') ,
                                'a_sigma' : pd.Series(dtype='float') ,
                                'a_sigma_allowed_interval' : pd.Series(dtype='str') ,
                                'b_allowed_interval' : pd.Series(dtype='str') ,
                                'b_sigma' : pd.Series(dtype='float') ,
                                'b_sigma_allowed_interval' : pd.Series(dtype='str') , 
                                })
        
        # Plot laboratory spectrum
        self.plot_spectrum(LAB_SPECTRUM, LAMBDA_MIN, LAMBDA_MAX,
                  save_to=self.input_dir,
                  title=f'Spectrum from the laboratory ({int(LAMBDA_MIN)} - {int(LAMBDA_MAX)} nm)',
                  name='spectral_lines_lab')

        # Main loop generating data
        for i in range(N_GALAXIES):
            random.seed(self.seed + i)
            np.random.seed(self.seed + i)
            index = i
            galaxy_ID = 'G_' + ''.join(random.choices(string.digits, k=8))

            if np.random.uniform(0,1) < SPECTRUM_GENERATE_PROBABILITY:
                z = random.uniform(0.2,1)*Z_MAX
                LINE_VISIBLE_PROBABILITY = 0.5
                spectrum = {}
                for line in LAB_SPECTRUM.keys():
                    shifted_line = float(line) * (z + 1)
                    if random.uniform(0,1) < LINE_VISIBLE_PROBABILITY:
                        spectrum[shifted_line] = LAB_SPECTRUM[line]
                self.plot_spectrum(spectrum, LAMBDA_MIN, LAMBDA_MAX,
                  save_to=spectra_folder,
                  title=f"Observed spectrum - {galaxy_ID}",
                  name=galaxy_ID)
            else:
                z = np.nan
                spectrum = {}

            true_distance = float(z*c/H0 * 10**6)
            if np.random.uniform(0,1) < CEPHEID_GENERATE_PROBABILITY:
                cepheid_ID = 'CEPH_' + ''.join(random.choices(string.digits, k=8))
                cepheid_distance = np.random.normal(loc=true_distance, 
                                                    scale=true_distance * SIGMA_DISTANCE_CEPHEIDES) #pc
                period = abs(np.random.normal(loc=60, scale=20))
                if not np.isnan(true_distance):
                    mean_mag = a*(math.log10(period) - 1) + b - 5 + 5*math.log10(cepheid_distance)
                else:
                    # This is dirty trick to confuse model when I merge some cells in the csv table  
                    # and gives some unrealistic values of magnitude in the table (while not being 
                    # so obvious), which can propagate to the result if the table is not parsed correctly. 
                    # These values are only for cepheids without redshift information, so they don't matter
                    # for the analysis. Targetted for .pdf conversion failure which is not valid anymore 
                    # but keeping it here for future reference as adversality in the task design.
                    mean_mag = -2.43*(math.log10(period * (1 + np.random.normal(loc=0, scale=0.3)) + 15) - 1) - 4.05 \
                                - 5 + 5*math.log10(cepheid_distance) + np.random.normal(loc=2, scale=2)
            else:
                cepheid_ID = np.nan
                cepheid_distance = np.nan
                period = np.nan
                mean_mag = np.nan

            if not np.isnan(z) and spectrum:
                estimated_z, sigma_z = self.estimate_z_logspace(spectrum, LAB_SPECTRUM, wav_min=LAMBDA_MIN,
                                                                wav_max=LAMBDA_MAX, reading_resolution=READING_PRECISION)
                distance_estimated = estimated_z * c / H0 * 10**6
                interval = f"[{z-0.0001:.5f}, {z+0.0001:.5f}]"
            else:
                sigma_z = np.nan
                distance_estimated = np.nan
                interval = np.nan

            if not np.isnan(mean_mag) and not np.isnan(distance_estimated) and distance_estimated > 0:
                absolute_mag_estimated = mean_mag + 5 - 5 * math.log10(distance_estimated)
            else:
                absolute_mag_estimated = np.nan

            rows.append({'index': index,
                         'galaxy_ID': galaxy_ID,
                         'z': z,
                         'spectrum': spectrum,
                         'true_distance [pc]': true_distance,
                         'cepheid_ID': cepheid_ID,
                         'cepheid_distance [pc]': cepheid_distance,
                         'mean_mag_cepheid': mean_mag,
                         'period [days]': period,
                         'sigma_z_estimated': sigma_z,
                         'z_allowed_interval': interval,
                         'distance_estimated [pc]': distance_estimated,
                         'absolute_mag_estimated': absolute_mag_estimated})

        generated_data = pd.DataFrame(rows)
        
        
        generated_data[['galaxy_ID', 'cepheid_ID', 'mean_mag_cepheid', 'period [days]']].round(
                    {'mean_mag_cepheid': 3, 'period [days]': 4}).to_csv(
                        self.input_dir / 'cepheids_measurements.csv',
                        index=False,
                        float_format='%.4f'
                        )

        # Plotting section
        # Collect fit data
        df_fit = generated_data[generated_data['period [days]'].notna() & generated_data['absolute_mag_estimated'].notna()].copy()
        df_fit['logP_m1'] = np.log10(df_fit['period [days]']) - 1

        x = df_fit['logP_m1'].values
        y = df_fit['absolute_mag_estimated'].values

        slope, intercept, r_value, _, _ = stats.linregress(x, y)

        # Uncertainty estimates
        n   = len(x)
        y_pred    = slope * x + intercept
        s2        = np.sum((y - y_pred)**2) / (n - 2)
        sxx       = np.sum((x - x.mean())**2)
        se_slope  = np.sqrt(s2 / sxx)
        se_inter  = np.sqrt(s2 * (1/n + x.mean()**2 / sxx))

        a_allowed_interval = f"[{slope-RESULTS_A_B_TOLERANCE:.2f}, {slope+RESULTS_A_B_TOLERANCE:.2f}]"
        a_sigma_allowed_interval = f"[{se_slope-RESULTS_A_B_TOLERANCE:.2f}, {se_slope+RESULTS_A_B_TOLERANCE:.2f}]"
        b_allowed_interval = f"[{intercept-RESULTS_A_B_TOLERANCE:.2f}, {intercept+RESULTS_A_B_TOLERANCE:.2f}]"
        b_sigma_allowed_interval = f"[{se_inter-RESULTS_A_B_TOLERANCE:.2f}, {se_inter+RESULTS_A_B_TOLERANCE:.2f}]"

        results_row = pd.DataFrame([{'a' : slope,
                                    'a_allowed_interval' : a_allowed_interval,
                                    'a_sigma' : se_slope,
                                    'a_sigma_allowed_interval' : a_sigma_allowed_interval,
                                    'b' : intercept,
                                    'b_allowed_interval' : b_allowed_interval, 
                                    'b_sigma' : se_inter,
                                    'b_sigma_allowed_interval' : b_sigma_allowed_interval}])
        results = pd.concat([results, results_row], ignore_index=True)

        # Plot
        x_line = np.linspace(x.min() - 0.05, x.max() + 0.05, 300)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Data points
        ax.scatter(x, y, color='steelblue', s=70, zorder=3,
                edgecolors='white', linewidths=0.6, label='Derived data')

        # Fitted line
        fit_label = (
            f'Fit:  $M = a[log_{{10}}(P) - 1] + b$\n'
            f'$a = {slope:.2f} ± {se_slope:.2f}$\n'
            f'$b = {intercept:.2f} ± {se_inter:.2f}$'
        )

        ax.plot(x_line, slope * x_line + intercept,
                color='steelblue', lw=2, label=fit_label)

        # Axes & styling
        ax.invert_yaxis()   # brighter = smaller M = top of plot
        ax.set_xlabel(r'$log_{10}(P (days)) - 1$', fontsize=12)
        ax.set_ylabel(r'Absolute Magnitude $M$', fontsize=12)
        ax.set_title('Cepheid Period–Luminosity Calibration', fontsize=13, fontweight='bold')

        ax.legend(fontsize=9, framealpha=0.9, loc='upper left')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.ground_truth_dir / 'cepheids_calibration.png', dpi=300,
                    bbox_inches='tight', facecolor='white')


        # Output estimated redshifts just for human reference
        generated_data.loc[generated_data[['cepheid_ID', 'mean_mag_cepheid', 'period [days]', 'z']].notna().all(axis=1),
                            ['cepheid_ID', 'z']].sort_values('cepheid_ID').reset_index(drop=True).assign(
                            z_estimated=lambda df: df['z'].map('{:.5f}'.format)).to_csv(self.ground_truth_dir / 'redshifts.csv', index=False)
        
        # Create ground truth dataframe for metarubrics
        redshifts = generated_data.loc[generated_data[['cepheid_ID', 'mean_mag_cepheid', 'period [days]', 'z']].notna().all(axis=1),
                            ['cepheid_ID', 'z', 'z_allowed_interval']].sort_values('cepheid_ID').reset_index(drop=True).assign(
                            z_estimated=lambda df: df['z'].map('{:.5f}'.format))
        # Create ground truth dataframe for metarubrics
        excluded_cepheids = generated_data.loc[generated_data[['cepheid_ID', 'mean_mag_cepheid', 
                                            'period [days]', 'z']].isna().any(axis=1), ['cepheid_ID']]

        # Output txt report solution for human reference
        a_str = f"a = {slope:.3f} ± {se_slope:.3f}"
        b_str = f"b = {intercept:.3f} ± {se_inter:.3f}"
        width = max(len(a_str), len(b_str)) + 4

        box = (
            f"┌{'─' * width}┐\n"
            f"│  {a_str:<{width-2}}│\n"
            f"│  {b_str:<{width-2}}│\n"
            f"└{'─' * width}┘"
        )

        with open(self.ground_truth_dir / 'cepheid_period-luminosity.txt', 'w') as f:
            print(
                f'We assumed the period-luminosity calibration for Cepheid variables\n'
                f'in form M = a[log(P) - 1] + b.\n\n'
                f'Parameters estimated by our analysis are:\n\n'
                f'{box}',
                file=f
            )

        # Store ground truth
        self.ground_truth['generated_data'] = generated_data 
        self.ground_truth['redshifts'] = redshifts
        self.ground_truth['excluded_cepheids'] = excluded_cepheids
        self.ground_truth['results'] = results
