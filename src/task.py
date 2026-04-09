"""
task.py — Abstract base class for benchmark tasks, MetaRubric, and TaskResult.

Contributors implementing new tasks should inherit from Task and implement:
    - generate()
"""

import json
import re
import mimetypes
from typing import Optional
import numpy as np
import pandas as pd
import base64
from pathlib import Path

from dataclasses import dataclass, field
from statsmodels.stats.proportion import proportion_confint
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────
# Metarubric
# ─────────────────────────────────────────────────────────────
@dataclass
class Metarubric:
    """
    A template + dataframe that unpacks into individual rubric criteria.
    
    Attributes:
        key:                    - short snake_case key to identify the metarubric
        source:                 - name of the dataframe in ground_truth.json that contains the data for this metarubric
        name:                   - short human readable name
        description:            - f-string with {column_name} placeholders - this gets unpacked 
                                  to individual rubric criteria by unpacking metarubric
        weight:                 - overall weight of the metarubric item (when the metarubric 
                                  is unpacked to multiple rubric criteria, the weight is distributed 
                                  equally among them)
        columns:                - list of column names extracted from description placeholders
        dataframe:              - dataframe with columns corresponding to the placeholders in 
                                  description   
    """

    key :             str
    source:           str
    name:             str
    description:      str
    weight:           float = 1.0
    
    # Not passed in __init__ — computed from description
    columns:        list[str]    = field(init=False)
    dataframe:      pd.DataFrame = field(init=False)
    
    def __post_init__(self):
        """Called automatically after __init__."""
        self.columns   = re.findall(r'\{(\w+)[^}]*\}', self.description)
        self.dataframe = pd.DataFrame(columns=self.columns)
    
    def unpack(self) -> list[str]:
        """Expand description with each row of dataframe."""
        return [
            self.description.format(**row.to_dict())
            for _, row in self.dataframe.iterrows()
        ]
    

# ─────────────────────────────────────────────────────────────
# MetarubricResult
# ─────────────────────────────────────────────────────────────
@dataclass
class MetarubricResult:
    """
    Result of evaluating one Metarubric.
    
    Attributes:
        metarubric_name: name of the metarubric
        total:           total number of criteria
        passed:          number passing numerical check
        weight:          importance relative to other metarubrics
    """
    metarubric_name: str
    total:           int
    passed:          int
    weight:          float = 1.0

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def standard_error(self) -> float:
        """Standard deviation of success rate — Bernoulli."""
        p = self.success_rate
        return np.sqrt(p * (1 - p) / self.total) if self.total > 0 else 0.0

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Wilson score 95% CI — better than normal approximation for small N."""
        if self.total == 0:
            return (0.0, 0.0)
        lo, hi = proportion_confint(
            self.passed,
            self.total,
            alpha=0.05,
            method='wilson'
        )
        return (lo, hi)

    def __str__(self) -> str:
        lo, hi = self.confidence_interval
        return (
            f"{self.metarubric_name}: "
            f"      {self.passed}/{self.total} "
            f"      ({self.success_rate:.1%} ± {self.standard_error:.1%}, "
            f"      95% CI: [{lo:.1%}, {hi:.1%}])"
        )
    

# ─────────────────────────────────────────────────────────────
# Task - abstract base class for benchmark tasks
# ─────────────────────────────────────────────────────────────
class Task(ABC):
    """
    Abstract base class for all benchmark tasks.
    
    Subclasses MUST implement:
        generate_task()         — physics simulation, populates
                                  input_data/, ground_truth/,
                                  and metarubrics dataframe
    
    Subclasses should NOT override:
        load_config()          - loads data generation parameters from config.json
        save_ground_truth()     - saves ground truth dataframes to ground_truth.json, should be 
                                  called at the end of generate_task() after populating self.ground_truth
        get_params()            - returns generating parameters for given difficulty level
        get_prompt()            - loads task prompt from README.md
        get_input_files()       - loads input data files and makes them ready to send to the LLM
        load_metarubrics()     - loads metarubric templates from metarubrics.json - helper 
                                 function  used in populate_metarubrics() 
        populate_metarubrics()  — populates metarubrics dict with Metarubric objects where each row
                                  corresponds to data used to create one rubric item
        validate_metarubrics()  - validates the metarubrics (checks for mismatches in metarubrics.json and
                                  ground_truth.json)
        generate_rubrics()      - generates rubrics.json - data rows from Metarubric objects are 
                                  unpacked to individual rubric criteria and saved to rubrics.json
        evaluate()              - evaluates model response against rubrics.json
    """

    def __init__(self, task_folder: str, 
                difficulty: str = 'easy',
                seed: Optional[int] = None):
        
        self.folder     = Path(task_folder)
        self.difficulty = difficulty
        self.config     = self.load_config()
        self.seed       = seed

        # Populated by generate_task() - used to store all generated/computed values
        self.ground_truth: dict[str, pd.DataFrame] = {}

        # Populated by populate_metarubrics()
        self.metarubrics: dict[str, Metarubric] = {}

        # Directory shortcuts
        self.input_dir  = self.folder / 'input_data'
        self.ground_truth_dir     = self.folder / 'ground_truth'

        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)

   
    # ─────────────────────────────────────────
    # Abstract methods — MUST implement
    # ─────────────────────────────────────────
    @abstractmethod
    def generate_task(self):
        """
        Generate input data and ground truth.
        
        Must:
            #TODO
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
        """Load task prompt from README.md."""
        return (self.folder / 'README.md').read_text()
    

    # ─────────────────────────────────────────
    # Loading input files and preparing them for LLM
    # ─────────────────────────────────────────
    def get_input_files(self, model: str = '') -> list[dict]:
        content = []
        
        for filepath in sorted(self.input_dir.rglob('*')):
            if not filepath.is_file():
                continue
            
            relative  = filepath.relative_to(self.input_dir)
            mime_type, _ = mimetypes.guess_type(str(filepath))
            
            # CSV / TXT / MD
            if filepath.suffix in ['.csv', '.txt', '.md']:
                content.append({
                    'type': 'text',
                    'text': f'File: {relative}\n{filepath.read_text()}'
                })
            
            # Images
            elif mime_type and mime_type.startswith('image/'):
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
            if len(mr.dataframe) == 0:
                errors.append(
                    f"Metarubric '{mr.key}': "
                    f"dataframe is empty. "
                    f"Source was '{mr.source}' — did populate_rubrics() run?"
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
            if mr.source not in self.ground_truth:
                raise ValueError(
                    f"Data for metarubric '{mr.key}' with source '{mr.source}' "
                    f"not found in ground_truth. "
                    f"Available: {list(self.ground_truth.keys())}"
                )
            
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
                    'weight': mr.weight,
                    'total':  len(mr.dataframe),
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

        print(f"✓ Rubrics saved: {rubrics_path}")