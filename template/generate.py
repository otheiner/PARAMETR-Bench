"""
Task: [TASK NAME]
Description: [BRIEF DESCRIPTION]

Author: [YOUR NAME]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from src.task import Task


class TaskName(Task):
    """
    <FILL-THIS-IN> DESCRIBE WHAT THIS TASK SIMULATES AND WHAT THE MODEL MUST DO
    """

    def generate_task(self):
        """
        <FILL-THIS-IN> DESCRIBE HOW THE DATA IS GENERATED, WHAT THE GROUND TRUTH IS AND HOW THE 
        TASK IS POPULATED
        """
        # Always use self.seed for reproducibility. Different generators can be used
        # but the seed has to be specified and it has to be self.seed. Do something like:
        # np.random.seed(self.seed)

        # Use config parameters as you define them in config.json. Avoid hardcoded values
        # in this method if the number might come from generating distributions configured 
        # in the config.json. Use the configuration like this:
        # PARAMETER_1 = self.get_params()['PARAMETER_1']
        # PARAMETER_2 = self.get_params()['PARAMETER_2']

        # Here comes the main code used to generate the task.
        # This metod needs to:
        # 1) Generate input data and save it to self.input_dir
        # 2) Generate ground truth and save it to self.ground_truth_dir
        # 3) Populate self.ground_truth dictionary with pandas DataFrames. These dataframes 
        #    can be arbitrary but they should contain all generated numbers, ground_truth and
        #    final answers. These dataframes are used as truth source for generating metarubrics.
        #    File ground_truth.json is generated automaticall based on self.ground_truth during
        #    the task evaluation stage when running evaluate.py.
        
        raise NotImplementedError("Implement data generation")