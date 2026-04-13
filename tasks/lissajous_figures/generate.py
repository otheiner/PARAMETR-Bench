"""
Task: Lissajous figures
Description: <FILL-THIS-IN>

Author: Ondrej Theiner
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import string
import os
import json
from pathlib import Path
from src.task import Task


class LissajousFigures(Task):
    """
    <FILL-THIS-IN> DESCRIBE WHAT THIS TASK SIMULATES AND WHAT THE MODEL MUST DO
    """

    # ############################################################
    # # Task generating method
    # ############################################################
    def generate_task(self):
        """
        <FILL-THIS-IN> DESCRIBE HOW THE DATA IS GENERATED, WHAT THE GROUND TRUTH IS AND HOW THE 
        TASK IS POPULATED
        """
        # ======= RANOMNESS =======
        # Always use self.seed for reproducibility. Different generators can be used
        # but the seed has to be specified and it has to be self.seed. Seeds derived 
        # deterministically from this seed are also fine. Use np.random.seed(self.seed). 
        # Doing something like np.random.seed(self.seed + 1) is also fine if you need to 
        # generate different random numbers in different places. 
        np.random.seed(self.seed)
        random.seed(self.seed)

        image_folder = self.input_dir / 'oscilloscope_output'
        os.makedirs(image_folder)

        # ====== CONFIGURATION =======
        # Use config parameters as you define them in config.json. Avoid hardcoded values
        # in this method if the number might come from generating distributions configured 
        # in the config.json. Use the configuration like this:
        # PARAMETER_1 = self.get_params()['PARAMETER_1']
        # PARAMETER_2 = self.get_params()['PARAMETER_2']
        NUMBER_OF_MEASUREMENTS = self.get_params()["NUMBER_OF_MEASUREMENTS"]
        NUMBER_OF_UNITS = self.get_params()["NUMBER_OF_UNITS"]
        NUMBER_OF_DAYS = self.get_params()["NUMBER_OF_DAYS"]
        NUMBER_OF_MACHINES = self.get_params()["NUMBER_OF_MACHINES"]
        MACHINE_FAULTY_PROBABILITY = self.get_params()["MACHINE_FAULTY_PROBABILITY"]
        NUMBER_OF_BATCHES = self.get_params()["NUMBER_OF_BATCHES"]
        FREQUENCY_PHASESHIFTS = self.get_params()["FREQUENCY_PHASESHIFTS"]

        MACHINES_FAILURE = {}
        for machine in range(NUMBER_OF_MACHINES):
            machine_name = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
            probably_faulty = (np.random.uniform(0,1) <= MACHINE_FAULTY_PROBABILITY)
            if probably_faulty:
                failure_frequency = random.uniform(0.6, 1.0)
            else:
                failure_frequency = random.uniform(0.01, 0.1)
            MACHINES_FAILURE[machine_name] = failure_frequency

        BATCHES = list(string.ascii_lowercase[:NUMBER_OF_BATCHES])
            
            
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
        generated_data = pd.DataFrame({
                    'day' : pd.Series(dtype='int'),
                    'machine' : pd.Series(dtype='str'),
                    'supply_ID' : pd.Series(dtype='str'),
                    'failure' : pd.Series(dtype='int'),
                    'batch' : pd.Series(dtype='str'),
                    'frequency' : pd.Series(dtype='int'),
                    'reference_frequency' : pd.Series(dtype='int'),
                    'phase_shift' :pd.Series(dtype='float'),
                    'a1' :pd.Series(dtype='int'),
                    'a2' : pd.Series(dtype='int'),
                    'measurement_id' : pd.Series(dtype='str')
                    })
        
        qa_data = pd.DataFrame({
                    "supply_ID" : pd.Series(dtype='str'), 
                    "machine" : pd.Series(dtype='str'),
                    "batch" : pd.Series(dtype='str'), 
                    "frequency" : pd.Series(dtype='int')
                    })
        
        complement_qa_data = pd.DataFrame({
                    "supply_ID" : pd.Series(dtype='str'), 
                    "machine" : pd.Series(dtype='str'),
                    "batch" : pd.Series(dtype='str'), 
                    "frequency" : pd.Series(dtype='int')
                    })
        
        faulty_machines = pd.DataFrame({
                        "machine" : pd.Series(dtype='str')
                        })
        
        not_faulty_machines = pd.DataFrame({
                "machine" : pd.Series(dtype='str')
                })

        
        # Loop for the data generation
        for measurement in range(NUMBER_OF_UNITS):
            day = random.randint(1, NUMBER_OF_DAYS)
            reference_frequency = 100
            machine = random.choice(list(MACHINES_FAILURE.keys()))
            failure = (random.uniform(0,1) < MACHINES_FAILURE[machine])
            measured = (random.uniform(0,1) < NUMBER_OF_MEASUREMENTS/NUMBER_OF_UNITS)
            batch = random.choice(BATCHES)
            supply_id = ''.join(random.choices(string.digits, k=8))

            if failure:
                valid_keys = [k for k in FREQUENCY_PHASESHIFTS.keys() if int(k) > 61]
                frequency = int(random.choice(valid_keys))
            else:
                valid_keys = [k for k in FREQUENCY_PHASESHIFTS.keys() if int(k) < 61]
                frequency = int(random.choice(valid_keys))

            phase_shift = FREQUENCY_PHASESHIFTS[str(frequency)] # multiplicator of np.pi
            a1 = random.uniform(3, 5)
            a2 = random.uniform(3, 5)

            if measured:
                measurement_id = ''.join(random.choices(string.digits, k=12))
                # Time array
                t = np.linspace(0, 2 * np.pi, 10000)

                # Parametric equations
                x = a1 * np.sin(reference_frequency * t + np.pi * phase_shift)
                y = a2 * np.sin(frequency * t)

                # Plot
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_facecolor("black")
                fig.patch.set_facecolor("black")
                ax.plot(x, y, color="turquoise")
                ax.set_xticks(np.linspace(-6, 6, 9))
                ax.set_yticks(np.linspace(-6, 6, 9))
                ax.grid(True, color="white", linewidth=0.5)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

                plt.tight_layout()
                plt.savefig(image_folder / f"ID_{measurement_id}.png")
                plt.close()
            else:
                measurement_id = ""

            # Fill the dataframe with the generated data
            new_row = pd.DataFrame([{'day' : day,
                                    'supply_ID' : supply_id,
                                    'machine' : machine,
                                    'failure' : failure,
                                    'measured' : measured,
                                    'batch' : batch,
                                    'frequency' : frequency,
                                    'reference_frequency' : reference_frequency,
                                    'phase_shift' : phase_shift,
                                    'a1' : a1,
                                    'a2' : a2,
                                    'measurement_id' : measurement_id}])
            generated_data = pd.concat([generated_data, new_row], ignore_index=True)
            
        # Print the dataframe to a .csv file, only with the columns that are needed for the analysis, and sorted by day
        generated_data.sort_values("day")[['day', 'supply_ID', 'machine', 'batch', 'measurement_id']].to_csv(
                                            self.input_dir / "measurements.csv", index=False)
        
        # Produce ground truth histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_yticks(np.arange(0, 4, 1))
        df_filtered = generated_data[generated_data['measured'] == 1]
        bins = np.arange(0.5, NUMBER_OF_DAYS + 1.5, 1)

        ax.hist(df_filtered['day'], bins=bins, weights=df_filtered['failure'])
        ax.set_xticks(np.arange(1, NUMBER_OF_DAYS + 1, 5))
        ax.set_xlabel('Day of Testing Period', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Number of Faulty Power Supplies', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Faulty Power Supplies Detected per Day', fontsize=16, fontweight='bold', pad=20)
        ax.grid(False)

        fig.savefig(self.ground_truth_dir / "histogram.png")
        plt.close(fig)

        # Produce ground truth table for QA
        failure_rate = generated_data[generated_data["measured"] == 1].groupby('machine').agg(
            failures=('failure', 'sum'),
            total=('failure', 'count')
        ).reset_index()

        filtered_rows = generated_data[(generated_data["measured"] == 1) & 
                                        (generated_data["frequency"] > 60)]
        
        filtered_rows[["supply_ID", "machine" ,"batch", "frequency"]].to_csv(
                        self.ground_truth_dir / "table_for_QA.csv", index=False)
        
        qa_data = filtered_rows[["supply_ID", "machine" ,"batch", "frequency"]]

        complement_filtered_rows = generated_data[(generated_data["measured"] == 1) & 
                                (generated_data["frequency"] <= 60)]
        
        complement_qa_data = complement_filtered_rows[["supply_ID", "machine" ,"batch", "frequency"]]

        failure_rate['failure_rate'] = failure_rate['failures'] / failure_rate['total']

        # Filter machines with failure rate > 10%
        faulty_machines = failure_rate[failure_rate['failure_rate'] > 0.1][['machine']]
        faulty_machines.to_csv(self.ground_truth_dir / 
                                'faulty_machines_over_10percent.txt', 
                                index=False, header=False)
        not_faulty_machines = failure_rate[failure_rate['failure_rate'] <= 0.1][['machine']]

            # Add dataframes to the ground_truth dictionary
        self.ground_truth["generated_data"] = generated_data
        self.ground_truth["qa_data"] = qa_data
        self.ground_truth["complement_qa_data"] = complement_qa_data
        self.ground_truth["faulty_machines"] = faulty_machines
        self.ground_truth["not_faulty_machines"] = not_faulty_machines
            
