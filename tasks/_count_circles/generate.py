"""
Task: Count circles task
Description: Visual test task to test the framework with multimodal inputs.
             The model needs to count the number of circles in each image.

Author: Ondrej Theiner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.task import Task


RADIUS     = 12
IMAGE_SIZE = 512
MARGIN     = RADIUS * 2


class CountCircles(Task):
    """
    Visual test task — model must count circles in each image.
    """

    def _place_circles(self, n_circles: int) -> list[tuple]:
        """Place n_circles without overlap using rejection sampling."""
        positions    = []
        max_attempts = n_circles * 100
        attempts     = 0

        while len(positions) < n_circles and attempts < max_attempts:
            x = np.random.uniform(MARGIN, IMAGE_SIZE - MARGIN)
            y = np.random.uniform(MARGIN, IMAGE_SIZE - MARGIN)

            overlap = any(
                np.sqrt((x - px)**2 + (y - py)**2) < RADIUS * 2.5
                for px, py in positions
            )
            if not overlap:
                positions.append((x, y))
            attempts += 1

        return positions

    def _draw_image(self, positions: list[tuple], path):
        """Draw circles on white background and save to path."""
        fig, ax = plt.subplots(
            figsize = (IMAGE_SIZE / 100, IMAGE_SIZE / 100),
            dpi     = 100
        )
        ax.set_xlim(0, IMAGE_SIZE)
        ax.set_ylim(0, IMAGE_SIZE)
        ax.set_facecolor('white')
        ax.axis('off')

        for x, y in positions:
            ax.add_patch(patches.Circle((x, y), radius=RADIUS, color='black'))

        plt.tight_layout(pad=0)
        plt.savefig(path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    # ############################################################
    # # Main method that needs to be implemented by user
    # ############################################################
    def _generate(self):
        # Load configuration
        N_FILES   = self.get_params()['N_FILES']
        N_CIRCLES = self.get_params()['N_CIRCLES']

        # Initialize random seed for reproducibility
        np.random.seed(self.seed)

        # Main loop generating the task
        records = []
        for i in range(1, N_FILES + 1):
            positions = self._place_circles(np.random.randint(1, N_CIRCLES + 1))
            n_placed  = len(positions)
            self._draw_image(positions, self.input_dir / f'{i}.png')
            records.append({'filename': f'{i}.png', 'n_circles': n_placed})

        # Save dataframes with numbers used in the task to ground truth dictionary
        self.ground_truth['images'] = pd.DataFrame(records)

        self.ground_truth['result'] = pd.DataFrame({
            'average_circles': [np.mean([r['n_circles'] for r in records])]
        })
        