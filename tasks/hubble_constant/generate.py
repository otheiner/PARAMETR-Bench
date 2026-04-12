"""
Task: Hubble Constant estimation
Description: This task recereates the famous observation done by Edwin Hubble in the 1920s, 
            which led to the discovery of the expansion of the universe.


Author: Ondrej Theiner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from src.task import Task
import random
import string
import math
import os
import shutil
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.stats import linregress


class HubbleConstant(Task):
    """
    Estimate the Hubble constant from observational data
    """   
    # ############################################################
    # # Function to produce graphical spectra
    # ############################################################
    @staticmethod
    def plot_spectrum(input_spectrum : dict[float, float], lambda_min: float, 
                      lambda_max : float, title: str, name : str, save_to: str) -> None:
        def gaussian(x, mu, sigma, amplitude):
            return amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        wavelength = np.linspace(lambda_min, lambda_max, 7000)
        intensity = np.zeros_like(wavelength)
        for mu, amp in input_spectrum.items():
            intensity += gaussian(wavelength, float(mu), sigma=0.1, amplitude=amp)
        intensity /= intensity.max()

        image = np.tile(intensity, (80, 1))

        fig, ax = plt.subplots(figsize=(14, 2))
        ax.imshow(image, cmap='gray', aspect='auto', extent=[wavelength[0], wavelength[-1], 0, 1])
        ax.set_title(f"{title}")
        ax.set_yticks([])
        ax.set_xlabel("Wavelength (nm)")

        # Major ticks every 10 nm (longer)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        # Minor ticks every 1 nm (shorter)
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        # Set tick lengths
        ax.tick_params(axis='x', which='major', length=8)
        ax.tick_params(axis='x', which='minor', length=4)

        # Labels only at multiples of 50 using FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}' if x % 50 == 0 else ''))

        plt.tight_layout()
        plt.savefig(f"{save_to}/{name}.png", dpi=150, bbox_inches="tight")
        plt.close()


    # ############################################################
    # # Function doing the fit of Hubble's law to generated data and
    # # producing ground truth plot for reference. 
    # ############################################################
    def fit_hubble(self, df: pd.DataFrame) -> tuple[float, float]:
        # Set LaTeX fonts for plot
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        mask = df['cepheid_distance [pc]'].notna() & df['z'].notna()
        x = df.loc[mask, 'cepheid_distance [pc]'] / 1e6
        y = self.get_params()['c'] * ((df.loc[mask, 'z'] + 1)**2 -1) \
                                    /((df.loc[mask, 'z'] + 1)**2 + 1) # Relativistic redshift

        # Fit through zero (because v = H_0*d) and its standard error
        H0        = np.sum(x * y) / np.sum(x ** 2)
        residuals = y - H0 * x
        H0_err    = np.sqrt(np.sum(residuals ** 2) / ((len(x) - 1) * np.sum(x ** 2)))

        x_fit = np.linspace(0, x.max() * 1.1, 200)
        y_fit = H0 * x_fit

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(x, y, s=20, alpha=0.7, color='crimson', label='Measured data')
        ax.plot(x_fit, y_fit, color='steelblue', lw=2,
                label=f'Fit: $v = H_0 \\cdot d$\n$H_0 = ({H0:.1f} \\pm {H0_err:.1f})$ km/s/Mpc')

        # Write the estimated Hubble's constant to a text file
        with open(self.ground_truth_dir / 'hubbles_constant.txt', 'w') as f:
            print(f"The Hubble's constant estimated from the presented dataset is {H0:.1f} ± \
                {H0_err:.1f} km/s/Mpc.", file=f)

        ax.set_xlabel('Cepheid Distance d [Mpc]', fontsize=12)
        ax.set_ylabel('Recession Velocity [km/s]', fontsize=12)
        ax.set_title('\\textbf{Hubble\'s Law}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        plt.tight_layout()
        plt.savefig(self.ground_truth_dir / 'hubble_law.png', dpi=150, bbox_inches='tight')
        
        return H0, H0_err

    # ############################################################
    # # Task generating method
    # ############################################################
    def generate_task(self):
        """
        Generate observed emission spectra of several galaxies, generate informations about 
        Cepheids in these galaxies and let the model estimate the Hubble constant based on these 
        data.
        """
        # Set random seed for reproducibility
        np.random.seed(self.seed) 

        # Physics constants
        c = self.get_params()['c']  # km/s
        H_DISTR_MEAN = self.get_params()['H_DISTR_MEAN']  # km/s/Mpc
        H_DISTR_STD = self.get_params()['H_DISTR_STD']  # km/s/Mpc
        H = np.random.normal(H_DISTR_MEAN,  H_DISTR_STD)

        #Simulation parameters
        N_GALAXIES = self.get_params()['N_GALAXIES'] #number of galaxies to simulate
        SIGMA_DISTANCE_CEPHEIDES = self.get_params()['SIGMA_DISTANCE_CEPHEIDES'] #Stdev of distance distribution of Cepheids from the Hubble's law
        Z_MAX = self.get_params()['Z_MAX'] #Maximum redshift of the simulated galaxies
        CEPHEID_GENERATE_PROBABILITY = self.get_params()['CEPHEID_GENERATE_PROBABILITY'] #Probability of generating a Cepheid for a given galaxy
        GALAXY_GENERATE_PROBABILITY = self.get_params()['GALAXY_GENERATE_PROBABILITY'] #Probability of generating measurement of observed emission spectrtum.

        # Laboratory spectrum
        LAMBDA_MIN = self.get_params()['LAMBDA_MIN']  # nm
        LAMBDA_MAX = self.get_params()['LAMBDA_MAX']  # nm
        LAB_SPECTRUM = self.get_params()['LAB_SPECTRUM']

        # Redshift estimation tolerance
        Z_TOLERANCE_BAND = self.get_params()["Z_TOLERANCE_BAND"]
        H_TOLERANCE_RELATIVE = self.get_params()["H_TOLERANCE_RELATIVE"]
        H_STDEV_TOLERANCE_RELATIVE = self.get_params()["H_STDEV_TOLERANCE_RELATIVE"]

        spectra_folder = self.input_dir / 'observed_spectra'
        os.makedirs(spectra_folder)

        # Define dataframes to store ground truth
        generated_data = pd.DataFrame({ 'index': pd.Series(dtype='int'),
                            'galaxy_ID': pd.Series(dtype='str'),
                            'z' : pd.Series(dtype='float'),
                            'spectrum' : pd.Series(dtype='object'),
                            'true_distance [pc]' : pd.Series(dtype='float'),
                            'cepheid_ID': pd.Series(dtype='str'),
                            'cepheid_distance [pc]' : pd.Series(dtype='float'),
                            'mean_mag_cepheid': pd.Series(dtype='float'),
                            'period [days]': pd.Series(dtype='float'),})
        
        analyzed_galaxies = pd.DataFrame({ 'index': pd.Series(dtype='int'),
                            'galaxy_ID': pd.Series(dtype='str'),
                            'z' : pd.Series(dtype='float'),
                            'z_min' : pd.Series(dtype='float'),
                            'z_max' : pd.Series(dtype='float'),
                            'spectrum' : pd.Series(dtype='object'),
                            'true_distance [pc]' : pd.Series(dtype='float'),
                            'cepheid_ID': pd.Series(dtype='str'),
                            'cepheid_distance [pc]' : pd.Series(dtype='float'),
                            'mean_mag_cepheid': pd.Series(dtype='float'),
                            'period [days]': pd.Series(dtype='float'),})

        results = pd.DataFrame({'hubble_constant' : pd.Series(dtype='float'),
                                'hubble_constant_min' : pd.Series(dtype='float'),
                                'hubble_constant_max' : pd.Series(dtype='float'),
                                'stdev' : pd.Series(dtype='float'),
                                'stdev_min' : pd.Series(dtype='float'),
                                'stdev_max' : pd.Series(dtype='float'),
                                'no_of_galaxies' : pd.Series(dtype='int')})

        # Generate laboratory spectrum
        self.plot_spectrum(input_spectrum = LAB_SPECTRUM, 
                           lambda_min = LAMBDA_MIN, 
                           lambda_max = LAMBDA_MAX, 
                           title = f'Spectrum from the laboratory ({int(LAMBDA_MIN)} - \
                                    {int(LAMBDA_MAX)} nm)', 
                            name= "spectral_lines_lab", 
                            save_to=self.input_dir)
        
        # Loop for the data generation
        for index in range(N_GALAXIES):
            galaxy_ID = 'GID' + ''.join(random.choices(string.digits, k=6))

            if np.random.uniform(0,1) < GALAXY_GENERATE_PROBABILITY:
                z = random.uniform(0.2,1) * Z_MAX
                LINE_VISIBLE_PROBABILITY = 0.5
                spectrum = {}

                for line in LAB_SPECTRUM.keys():
                    shifted_line = float(line) * (z + 1)
                    if random.uniform(0,1) < LINE_VISIBLE_PROBABILITY:
                        spectrum[shifted_line] = LAB_SPECTRUM[line]

                self.plot_spectrum(input_spectrum = spectrum, 
                                   lambda_min = LAMBDA_MIN,
                                   lambda_max = LAMBDA_MAX,
                                   title = f"Observed spectrum - {galaxy_ID}", 
                                   name = galaxy_ID, 
                                   save_to = spectra_folder)
            else:
                z = np.nan
                spectrum = {}

            true_distance = float(z * c / H * 10**6) # Distance in parsec
            if np.random.uniform(0,1) < CEPHEID_GENERATE_PROBABILITY and not np.isnan(z):
                cepheid_ID = 'HDD' + ''.join(random.choices(string.digits, k=6))
                cepheid_distance = np.random.normal(loc=true_distance, 
                                                    scale = true_distance * SIGMA_DISTANCE_CEPHEIDES) # parsec
                period = abs(np.random.normal(loc=60, scale=20))
                mean_mag = -2.43*(math.log10(period) - 1) - 4.05 - 5 + 5*math.log10(cepheid_distance)
            else:
                cepheid_ID = np.nan
                cepheid_distance = np.nan
                period = np.nan
                mean_mag = np.nan

            # Add generated data to ground truth dataframe
            new_row = pd.DataFrame([{'index' : index,
                                    'galaxy_ID' : galaxy_ID,
                                    'z' : z,
                                    'spectrum' : spectrum,
                                    'true_distance [pc]' : true_distance,
                                    'cepheid_ID' : cepheid_ID,
                                    'cepheid_distance [pc]' : cepheid_distance,
                                    'mean_mag_cepheid' : mean_mag,
                                    'period [days]' : period}])
            generated_data = pd.concat([generated_data, new_row], ignore_index=True)

        # Print input data about cepheides to .csv file
        df_out = generated_data.copy()
        df_out['mean_mag_cepheid'] = df_out['mean_mag_cepheid'].map('{:.3f}'.format)
        df_out['period [days]'] = df_out['period [days]'].map('{:.4f}'.format)
        df_out[['galaxy_ID' , 'cepheid_ID', 'mean_mag_cepheid', 'period [days]']].to_csv(\
                                self.input_dir / 'cepheides_mesurements.csv', index=False)
        
        # Print extracted redshifts to .csv file
        analyzed_galaxies = generated_data.loc[generated_data['z'].notna() & \
                                            generated_data['cepheid_ID'].notna()].copy()
        analyzed_galaxies['z'] = analyzed_galaxies['z'].round(5)
        analyzed_galaxies['z_min'] = round(analyzed_galaxies['z'] - Z_TOLERANCE_BAND, 5)
        analyzed_galaxies['z_max'] = round(analyzed_galaxies['z'] + Z_TOLERANCE_BAND, 5)
        analyzed_galaxies.sort_values(by='galaxy_ID', ascending=True, inplace=True)
        analyzed_galaxies[['galaxy_ID', 'z']].reset_index(drop=True).to_csv(\
                                        self.ground_truth_dir / 'redshifts.csv', index = False)
        
        # Fit Hubble's law
        df_est = generated_data.copy()
        df_est['z'] = df_est['z'].round(4)
        df_est['mean_mag_cepheid'] = df_est['mean_mag_cepheid'].round(3)
        df_est['period [days]'] = df_est['period [days]'].round(4)
        H0, H0_err = self.fit_hubble(df_est)

        # Save results to ground truth dataframe
        # Create results as a ONE-ROW DataFrame
        results = pd.DataFrame({
            'hubble_constant':      [round(H0, 1)],
            'hubble_constant_min':  [round(H0 * (1 - H_TOLERANCE_RELATIVE), 1)],
            'hubble_constant_max':  [round(H0 * (1 + H_TOLERANCE_RELATIVE), 1)],
            'stdev':                [round(H0_err, 1)],
            'stdev_min':            [round(H0_err * (1 - H_STDEV_TOLERANCE_RELATIVE), 1)],
            'stdev_max':            [round(H0_err * (1 + H_STDEV_TOLERANCE_RELATIVE), 1)],
            'no_of_galaxies':       [len(analyzed_galaxies)]
        })

        # Add dataframes to the ground_truth dictionary
        self.ground_truth['generated_data'] = generated_data
        self.ground_truth['analyzed_galaxies'] = analyzed_galaxies
        self.ground_truth['results'] = results
