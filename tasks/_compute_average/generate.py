"""
Task: Compute average
Description: This is the test task to test the framework. The model needs 
             to compute the average of given numbers.

Author: Ondrej Theiner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from src.task import Task


class ComputeAverage(Task):
    """
    This task computes mean of provided list of numbers.
    """

    # ############################################################
    # # Main method that needs to be implemented by user
    # ############################################################
    def _generate(self):
        """
        This is test task to demonstrate the framework.
        """
        # Load parameters from config
        N_NUMBERS = self.get_params()['N_NUMBERS']
        OFFSET = self.get_params()['OFFSET']

        # Configure random seed for reproducibility
        np.random.seed(self.seed)

        # Generate numbers
        values = [np.random.rand() + OFFSET for _ in range(N_NUMBERS)]

        # Save generated numbers to ground truth dictionary
        self.ground_truth['numbers'] = pd.DataFrame({
            'input_number': values
        })

        # Compute average
        avg = self.ground_truth['numbers']['input_number'].mean()

        # Save final result to ground truth dictionary
        self.ground_truth['final_result'] = pd.DataFrame({
            'average': [avg]
        })

        # Write input file to the input_dir
        with open(self.input_dir / 'input_numbers.txt', 'w') as f:
            for num in self.ground_truth['numbers']['input_number']:
                f.write(f"{num}\n") 
        
        # Write ground truth files to the ground_truth_dir
        with open(self.ground_truth_dir / 'answer.txt', 'w') as f:
            f.write(f"Computed average is: {self.ground_truth['final_result']['average'].iloc[0]}\n")
