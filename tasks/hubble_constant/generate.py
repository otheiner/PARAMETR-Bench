"""
Task: Hubble Constant estimation
Description: Estimate the Hubble constant from observational data

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
    This task recereates the famous observation done by Edwin Hubble in the 1920s, which led 
    to the discovery of the expansion of the universe.
    """   
    # ############################################################
    # # Function to produce graphical spectra
    # ############################################################
    def plot_spectrum(input_spectrum : dict[float, float], lambda_min: float, 
                      lambda_max : float, title: str, name : str, save_to: str) -> None:
        def gaussian(x, mu, sigma, amplitude):
            return amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        wavelength = np.linspace(lambda_min, lambda_max, 7000)
        intensity = np.zeros_like(wavelength)
        for mu, amp in input_spectrum.items():
            intensity += gaussian(wavelength, mu, sigma=0.1, amplitude=amp)
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
    # # Function doinf the fit of Hubble law to generated data and
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
        with open(self.ground_truth / 'hubbles_constant.txt', 'w') as f:
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
        # ====== CONFIGURATION =======
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

        spectra_folder = self.input_dir / 'observed_spectra'

        # ======= TASK GENERATION =======
        # This is the main code used to generate the task.
        # This code needs to:
        # 1) Generate input data and save it to self.input_dir
        # 2) Generate ground truth files and save it to self.ground_truth_dir
        # 3) Populate self.ground_truth dictionary with pandas DataFrames. These dataframes 
        #    can be arbitrary but they should contain all generated numbers, ground_truth and
        #    final answers. These dataframes are used as truth source for generating metarubrics,
        #    and can be used to study exact failure modes of LLMS. File ground_truth.json is 
        #    generated automaticall based on self.ground_truth during the task evaluation stage 
        #    when running evaluate.py.
        ############################################################
        ################ Dataframe format
        ############################################################

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

        result = pd.DataFrame({'hubble_constant' : pd.Series(dtype='float'),
                              'stdev' : pd.Series(dtype='flaot')})
        
        # Add dataframes to the ground_truth dictionary
        self.ground_truth['generated_data'] = generated_data
        self.ground_truth['result'] = result

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
                    shifted_line = line * (z + 1)
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
        df_solution = generated_data.loc[generated_data['z'].notna() & \
                                            generated_data['cepheid_ID'].notna()].copy()
        df_solution.sort_values(by='galaxy_ID', ascending=True, inplace=True)
        df_solution[['galaxy_ID', 'z']].reset_index(drop=True).to_csv(\
                    self.ground_truth_dir / 'redshifts.csv', index = False)
        
        # Fit Hubble's law
        df_est = generated_data.copy()
        df_est['z'] = generated_data['z'].round(4)
        df_est['mean_mag_cepheid'] = generated_data['mean_mag_cepheid'].round(3)
        df_est['period [days]'] = generated_data['period [days]'].round(4)
        H0, H0_err = self.fit_hubble(df_est)

        # Save H0 and its standard deviation to ground truth dataframe
        result['hubble_constant'] = H0
        result['stdev'] = H0_err

