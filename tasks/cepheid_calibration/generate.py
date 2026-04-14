"""
Task: cepheid_calibration
Description: <FILL-THIS-IN>

Author: Ondrej Theiner
"""

from src.task import Task
import numpy as np
import random


class CepheidCalibration(Task):
    """
    <FILL-THIS-IN> DESCRIBE WHAT THIS TASK SIMULATES AND WHAT THE MODEL MUST DO
    """

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

        # ====== CONFIGURATION =======
        # Use config parameters as you define them in config.json. Avoid hardcoded values
        # in this method if the number might come from generating distributions configured
        # in the config.json. Use the configuration like this:
        # PARAMETER_1 = self.get_params()['PARAMETER_1']
        # PARAMETER_2 = self.get_params()['PARAMETER_2']
        # Variable parameters
        N_GALAXIES = self.get_params()[N_GALAXIES]
        CEPHEID_GENERATE_PROBABILITY = self.get_params()[CEPHEID_GENERATE_PROBABILITY]
        SPECTRUM_GENERATE_PROBABILITY = self.get_params()[SPECTRUM_GENERATE_PROBABILITY]
        READING_PRECISION = self.get_params()[READING_PRECISION]
        
        # Fixed parameters
        c = self.get_params()[c]
        H0 = self.get_params()[H0]
        Z_MAX = self.get_params()[Z_MAX] 
        SIGMA_DISTANCE_CEPHEIDES = self.get_params()[SIGMA_DISTANCE_CEPHEIDES] 
        LAMBDA_MIN = self.get_params()[LAMBDA_MIN] #nm
        LAMBDA_MAX = self.get_params()[LAMBDA_MAX] #nm
        LAB_SPECTRUM = self.get_params()[LAB_SPECTRUM]

        # ======= TASK GENERATION =======
        # This is the main code used to generate the task.
        # This code needs to:
        # 1) Generate input data and save it to self.input_dir
        # 2) Generate ground truth files and save it to self.ground_truth_dir
        # 3) Populate self.ground_truth dictionary with pandas DataFrames. These dataframes
        #    can be arbitrary but they should contain all generated numbers, ground_truth and
        #    final answers. These dataframes are used as truth source for generating metarubrics,
        #    and can be used to study exact failure modes of LLMs. File ground_truth.json is
        #    generated automatically based on self.ground_truth during the task evaluation stage
        #    when running run.py.

        raise NotImplementedError("Implement data generation")