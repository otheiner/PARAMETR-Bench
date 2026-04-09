"""
Task: Test task
Description: <FILL-THIS-IN>

Author: <FILL-THIS-IN>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from src.task import Task


class TestTask(Task):
    """
    This test task computes mean of given numbers.
    """

    def generate_task(self):
        """
        This is test task
        """

        N_NUMBERS = self.get_params()['N_NUMBERS']
        FIXED_CONSTANT = self.get_params()['FIXED_CONSTANT']

        # Generate numbers properly
        values = [np.random.rand() + FIXED_CONSTANT for _ in range(N_NUMBERS)]

        self.ground_truth['numbers'] = pd.DataFrame({
            'input_number': values
        })

        # Compute average
        avg = self.ground_truth['numbers']['input_number'].mean()

        self.ground_truth['final_result'] = pd.DataFrame({
            'average': [avg]
        })

        with open(self.input_dir / 'input_numbers.txt', 'w') as f:
            for num in self.ground_truth['numbers']['input_number']:
                f.write(f"{num}\n") 
        
        with open(self.ground_truth_dir / 'answer.txt', 'w') as f:
            f.write(f"Computed average is: {self.ground_truth['final_result']['average'].iloc[0]}\n")
