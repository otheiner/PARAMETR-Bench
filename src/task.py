"""
task.py — Abstract base class for benchmark tasks.

Contributors implementing new tasks should inherit from Task and implement:
    - _generate()
"""

import json
import mimetypes
from typing import Optional
import pandas as pd
import base64
from pathlib import Path
import shutil
from abc import ABC, abstractmethod

from src.metarubric import Metarubric


# ─────────────────────────────────────────────────────────────
# Task - abstract base class for benchmark tasks
# ─────────────────────────────────────────────────────────────
class Task(ABC):
    """
    Abstract base class for all benchmark tasks.

    Subclasses MUST implement:
        _generate()             — physics simulation, populates
                                  input_data/, ground_truth/,
                                  and metarubrics dataframe

    Subclasses should NOT override:
        generate_task()         - public entry point: clears dirs, then calls _generate()
        load_config()           - loads data generation parameters from config.json
        save_ground_truth()     - saves ground truth dataframes to ground_truth.json, should be
                                  called at the end of _generate() after populating self.ground_truth
        get_params()            - returns generating parameters for given difficulty level
        get_prompt()            - loads task prompt from prompt.md
        get_input_files()       - loads input data files and makes them ready to send to the LLM
        load_metarubrics()      - loads metarubric templates from metarubrics.json - helper
                                  function used in populate_metarubrics()
        populate_metarubrics()  — populates metarubrics dict with Metarubric objects where each row
                                  corresponds to data used to create one rubric item
        validate_metarubrics()  - validates the metarubrics (checks for mismatches in metarubrics.json and
                                  ground_truth.json)
        generate_rubrics()      - generates rubrics.json - data rows from Metarubric objects are
                                  unpacked to individual rubric criteria and saved to rubrics.json
    """

    def __init__(self, task_folder: str,
                difficulty: str = 'easy',
                seed: Optional[int] = None):

        self.folder     = Path(task_folder)
        self.difficulty = difficulty
        self.config     = self.load_config()
        self.seed       = seed

        # Populated by _generate() - used to store all generated/computed values
        self.ground_truth: dict[str, pd.DataFrame] = {}

        # Populated by populate_metarubrics()
        self.metarubrics: dict[str, Metarubric] = {}

        # Directory shortcuts
        self.input_dir        = self.folder / 'input_data'
        self.ground_truth_dir = self.folder / 'ground_truth'

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)


    # ─────────────────────────────────────────
    # Directory cleanup
    # ─────────────────────────────────────────
    def _clear_dirs(self):
        """Clear input_data/ and ground_truth/ contents, preserving .gitignore files."""
        for directory in (self.input_dir, self.ground_truth_dir):
            for path in directory.iterdir():
                if path.name == '.gitignore':
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


    # ─────────────────────────────────────────
    # Public entry point — do NOT override
    # ─────────────────────────────────────────
    def generate_task(self):
        """Clear working directories and generate a fresh task instance."""
        self._clear_dirs()
        self.ground_truth = {}
        self._generate()


    # ─────────────────────────────────────────
    # Abstract methods — MUST implement
    # ─────────────────────────────────────────
    @abstractmethod
    def _generate(self):
        """
        Generate input data and ground truth. Each task must implement this method.

        What needs to be done:

        ======= RANDOMNESS =======
        Always use self.seed for reproducibility. Different generators can be used
        but the seed has to be specified and it has to be self.seed. Seeds derived
        deterministically from this seed are also fine. Use np.random.seed(self.seed),
        or random.seed(self.seed). Doing something like np.random.seed(self.seed + 1)
        is also fine if you need to generate different random numbers in different places.

        ====== CONFIGURATION =======
        Use config parameters as you define them in config.json. Avoid hardcoded values
        in this method if the number might come from generating distributions configured
        in the config.json. Use the configuration like this:
        PARAMETER_1 = self.get_params()['PARAMETER_1']
        PARAMETER_2 = self.get_params()['PARAMETER_2']

        ======= TASK GENERATION =======
        This is the main code used to generate the task.
        This code needs to:
        1) Generate input data and save it to self.input_dir
        2) Generate ground truth files and save it to self.ground_truth_dir
        3) Populate self.ground_truth dictionary with pandas DataFrames. These dataframes
           can be arbitrary but they should contain all generated numbers, ground_truth and
           final answers. These dataframes are used as truth source for generating metarubrics,
           and can be used to study exact failure modes of LLMs. File ground_truth.json is
           generated automatically based on self.ground_truth during the task evaluation stage
           when running run.py.
        """
        raise NotImplementedError


    # ─────────────────────────────────────────
    # Loading parameters from config file
    # ─────────────────────────────────────────
    def load_config(self) -> dict:
        """Load config.json and return full config."""
        with open(self.folder / 'config.json') as f:
            return json.load(f)


    def get_params(self) -> dict:
        """
        Return parameters for current difficulty level.
        Merges fixed_parameters with difficulty-specific parameters.
        Difficulty-specific parameters take precedence.
        """
        fixed      = self.config.get('fixed_parameters', {})
        difficulty = self.config['difficulties'][self.difficulty]
        
        # Merge — difficulty params override fixed params if same key
        return {**fixed, **difficulty}


    # ─────────────────────────────────────────
    # Dumps ground truth to ground_truth/ground_truth.json
    # ─────────────────────────────────────────
    def save_ground_truth(self):
        """
        Persist self.ground_truth to ground_truth/ground_truth.json.
        
        Call at the end of generate_task() after populating self.ground_truth.
        self.ground_truth must be a dict of DataFrames.
        """
        if not self.ground_truth:
            raise ValueError(
                "self.ground_truth is empty. "
                "Populate it in generate_task() before calling save_ground_truth()."
            )
        
        gt_json = {
            key: df.to_dict(orient='records')
            for key, df in self.ground_truth.items()
        }
        
        gt_path = self.ground_truth_dir / 'ground_truth.json'
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(gt_json, f, indent=2, default=float)
        
        print(f"✓ Ground truth saved: {gt_path}")


    # ─────────────────────────────────────────
    # Loading prompt
    # ─────────────────────────────────────────
    def get_prompt(self) -> str:
        """Load task prompt from prompt.md."""
        return (self.folder / 'prompt.md').read_text()
    

    # ─────────────────────────────────────────
    # Loading input files and preparing them for LLM
    # ─────────────────────────────────────────
    def get_input_files(self, embed_data: bool = True) -> list[dict]:
        content = []

        for filepath in sorted(self.input_dir.rglob('*')):
            if not filepath.is_file():
                continue
            if filepath.name == '.gitignore':
                continue

            relative  = filepath.relative_to(self.input_dir)
            mime_type, _ = mimetypes.guess_type(str(filepath))

            # CSV / TXT / MD
            if filepath.suffix in ['.csv', '.txt', '.md']:
                if embed_data:
                    content.append({
                        'type': 'text',
                        'text': f'File: {relative}\n{filepath.read_text()}'
                    })
                else:
                    content.append({
                        'type': 'text',
                        'text': f'File: {relative}'
                    })

            # Images
            elif mime_type and mime_type.startswith('image/'):
                if embed_data:
                    data = base64.b64encode(filepath.read_bytes()).decode()
                    content.append({
                        'type': 'text',
                        'text': f'Image file: {relative}'
                    })
                    content.append({
                        'type':      'image_url',
                        'image_url': {'url': f'data:{mime_type};base64,{data}'}
                    })
                else:
                    content.append({
                        'type': 'text',
                        'text': f'Image file: {relative}'
                    })

            else:
                content.append({
                    'type': 'text',
                    'text': f'[Skipped: {relative} — unsupported type]'
                })

        return content
    

    # ─────────────────────────────────────────
    # Creating dictionary of metarubrics objects based on 
    # metarubrics.json
    # ─────────────────────────────────────────
    def load_metarubrics(self) -> dict[str, Metarubric]:
        """Load metarubric templates from metarubrics.json and return dict of MetaRubric objects."""
        with open(self.folder / 'metarubrics.json') as f:
            data = json.load(f)

        metarubrics = {}
        for metarubric in data.get('metarubrics', []):  # ensure key exists
            mr = Metarubric(
                key            = metarubric['key'],
                source         = metarubric['source'],
                dimension       = metarubric['dimension'],
                name           = metarubric['name'],
                description    = metarubric['description'],
                weight         = metarubric.get('weight', 1.0)
            )
            metarubrics[mr.key] = mr

        return metarubrics


    # ─────────────────────────────────────────
    # Validate rubrics - check that metarubrics.json
    # matches python code
    # ─────────────────────────────────────────
    def validate_metarubrics(self):
        errors = []
        
        for mr in self.metarubrics.values():
            if mr.dimension not in Metarubric.ALLOWED_DIMENSIONS:
                errors.append(
                    f"Metarubric '{mr.key}': "
                    f"invalid dimension '{mr.dimension}'. "
                    f"Allowed: {sorted(Metarubric.ALLOWED_DIMENSIONS)}"
                )

            if len(mr.dataframe) == 0 and mr.source != 'none':
                errors.append(
                    f"Metarubric '{mr.key}': "
                    f"dataframe is empty. "
                    f"Source was '{mr.source}' — Did populate_metarubrics() run?"
                    f"Does task.generate() populate ground_truth dictionary?"
                )
                continue
            
            missing = set(mr.columns) - set(mr.dataframe.columns)
            if missing:
                errors.append(
                    f"Metarubric '{mr.name}': "
                    f"missing columns {missing}"
                )
        
        if errors:
            raise ValueError(
                "Metarubric validation failed:\n" +
                '\n'.join(f"  ✗ {e}" for e in errors)
            )
        
        print(f"✓ Metarubrics validated: {self.folder.name}")


    # ─────────────────────────────────────────
    # Popoulate metarubrics with data from ground_truth.json
    # ─────────────────────────────────────────
    def populate_metarubrics(self):
        """
        Load metarubric templates and fill dataframes from ground_truth.
        Uses mr.source to look up the correct DataFrame in ground_truth.
        Uses mr.columns to select only the columns needed by the template.
        """
        self.metarubrics = self.load_metarubrics()
        
        for mr in self.metarubrics.values():
            if mr.source not in self.ground_truth and mr.source != 'none':
                raise ValueError(
                    f"Data for metarubric '{mr.key}' with source '{mr.source}' "
                    f"not found in ground_truth. "
                    f"Available: {list(self.ground_truth.keys())}"
                )
            
            if mr.source != 'none':
                df = self.ground_truth[mr.source]
                
                # Validate columns exist in source DataFrame
                missing = set(mr.columns) - set(df.columns)
                if missing:
                    raise ValueError(
                        f"Metarubric '{mr.key}': source '{mr.source}' "
                        f"missing columns {missing}. "
                        f"Template needs {mr.columns}, "
                        f"DataFrame has {list(df.columns)}"
                    )
                
                # Fill with only the columns needed by the template
                mr.dataframe = df[mr.columns].copy()

        print(f"✓ Metarubrics populated: {self.folder.name}")


    # ─────────────────────────────────────────
    # Generate rubrics - create individual rubric criteria
    # ─────────────────────────────────────────
    def generate_rubrics(self):
        """
        Unpack metarubric templates to individual rubric criteria
        and save to ground_truth/rubrics.json.

        Must be called after generate_task(), populate_metarubrics(),
        and validate_metarubrics().
        """
        rubrics_data = {
            'task':        self.folder.name,
            'difficulty':  self.difficulty,
            'seed':        self.seed,
            'metarubrics': [
                {
                    'key':    mr.key,
                    'name':   mr.name,
                    'dimension': mr.dimension,
                    'weight': mr.weight,
                    'total':  len(mr.dataframe) if mr.source != 'none' else 1,
                    'rubrics': [
                        {'id': i + 1, 'criterion': criterion}
                        for i, criterion in enumerate(mr.unpack())
                    ]
                }
                for mr in self.metarubrics.values()
            ]
        }

        rubrics_path = self.ground_truth_dir / 'rubrics.json'
        with open(rubrics_path, 'w') as f:
            json.dump(rubrics_data, f, indent=2)

        print(f"✓ Rubrics generated and saved: {rubrics_path}")