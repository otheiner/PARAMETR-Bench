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
from scipy import stats
import shutil

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
        if self.source == 'none':
            return [self.description]

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

        # Make all names the same width
        aligned_name = (self.metarubric_name + ":").ljust(50)
        return (
            f"{aligned_name}"
            f"{self.passed}/{self.total} "
            f"({self.success_rate:.1%}, "
            f"95% CI: [{lo:.1%}, {hi:.1%}])"
        )

# ─────────────────────────────────────────────────────────────
# TaskResults
# ─────────────────────────────────────────────────────────────
@dataclass
class TaskResults:
    """
    Results of evaluating a task.
    
    Attributes:
        task_name:           name of the task
        seed:                seed used to generate the instance of the task
        difficulty:          difficulty of the task fixes value of parameters from config.json
        model:               model under test
        judge:               model used as a judge
        git_commit:          git commit hash to make it possible to trace back to the exact code used for evaluation
        timestamp:           timestamp of when the evaluation was run
        metarubric_results:  list of metarubric results
    """
    task_name:          str
    seed:               int
    difficulty:         str
    model:              str
    judge:              str
    git_commit:         str
    timestamp:          str
    metarubric_results: list[MetarubricResult]

    @property
    def weighted_success_rate(self) -> float:
        """Weighted average success rate across metarubrics."""
        total_weight = sum(mr.weight for mr in self.metarubric_results)
        weighted_sum = sum(mr.success_rate * mr.weight
                          for mr in self.metarubric_results)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Wilson score 95% CI aggregated across metarubrics using weights."""
        passed = sum(mr.passed for mr in self.metarubric_results)
        total  = sum(mr.total for mr in self.metarubric_results)
        if total == 0:
            return (0.0, 0.0)
        lo, hi = proportion_confint(
            passed,
            total,
            alpha=0.05,
            method='wilson'
        )
        return (lo, hi)

    def __str__(self) -> str:
        lo, hi = self.confidence_interval
        lines = [
            f"Task:       {self.task_name}",
            f"Model:      {self.model}",
            f"Judge:      {self.judge}",
            f"Difficulty: {self.difficulty}  |  Seed: {self.seed}",
            f"Commit:     {self.git_commit}  |  {self.timestamp}",
        ]
        for mr in self.metarubric_results:
            lines.append(f"      - {mr}")
        lines.append(f"      {'─' * 50}")
        lines.append(
            f"      Weighted total: {self.weighted_success_rate:.1%}"
        )
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        lo, hi = self.confidence_interval
        return {
            'task':       self.task_name,
            'seed':       self.seed,
            'difficulty': self.difficulty,
            'model':      self.model,
            'judge':      self.judge,
            'git_commit': self.git_commit,
            'timestamp':  self.timestamp,
            'metarubrics': [
                {
                    'name':         mr.metarubric_name,
                    'total':        mr.total,
                    'passed':       mr.passed,
                    'weight':       mr.weight,
                    'success_rate': mr.success_rate,
                    'ci_low':       mr.confidence_interval[0],
                    'ci_high':      mr.confidence_interval[1]
                }
                for mr in self.metarubric_results
            ],
            'aggregate': {
                'weighted_success_rate': self.weighted_success_rate,
                'ci_low':                lo,
                'ci_high':               hi
            }
        }

    def save(self, results_dir: str = 'results'):
        """
        Save results to results/{task_name}/{model}__{difficulty}__seed{seed}.json
        Creates directory if it doesn't exist.
        """
        # Sanitise model name for filename — replace / and : with -
        model_clean = self.model.replace('/', '-').replace(':', '-')

        task_dir = Path(results_dir) / self.task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{model_clean}__{self.difficulty}__seed{self.seed}.json"
        filepath = task_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"✓ Results saved: {filepath}")
        return filepath

# ─────────────────────────────────────────────────────────────
# BenchmarkResults
# ─────────────────────────────────────────────────────────────
@dataclass
class BenchmarkResults:
    """
    Results of evaluating a benchmark — collection of task results.

    Attributes:
        task_results:  list of TaskResults for each evaluated task
        models:        models evaluated - can be multiple
        judge:         judge model used for all runs
        difficulty:    difficulty level
        seeds:         random seeds used
        git_commit:    git commit hash for reproducibility
        timestamp:     timestamp of when benchmark run started
    """
    task_results:  list[TaskResults]
    models:        list[str]
    judge:         str
    difficulty:    str
    seeds:         list[int]
    git_commit:    str
    timestamp:     str

    @property
    def success_rate(self) -> float:
        """Success rate across all task results."""
        if not self.task_results:
            return 0.0
        return sum(tr.weighted_success_rate
                   for tr in self.task_results) / len(self.task_results)

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% CI across task success rates."""
        rates = [tr.weighted_success_rate for tr in self.task_results]
        if len(rates) < 2:
            return (0.0, 1.0)
        
        mean  = np.mean(rates)
        se    = np.std(rates, ddof=1) / np.sqrt(len(rates))
        z     = stats.norm.ppf(0.975)  # 95% confidence interval
        
        return (
            float(max(0.0, mean - z * se)),
            float(min(1.0, mean + z * se))
        )

    def results_by_model(self) -> dict[str, list[TaskResults]]:
        """Group task results by model for per-model analysis."""
        by_model = {}
        for tr in self.task_results:
            if tr.model not in by_model:
                by_model[tr.model] = []
            by_model[tr.model].append(tr)
        return by_model

    def __str__(self) -> str:
        lines = []

        # Run metadata 
        lines.append(f"Judge:      {self.judge}")
        lines.append(f"Difficulty: {self.difficulty}")
        lines.append(f"Seeds:      {self.seeds}")
        lines.append(f"Commit:     {self.git_commit}  |  {self.timestamp}")
        lines.append('-' * 50)

        # Per-model summary 
        for model, results in self.results_by_model().items():
            rates     = [tr.weighted_success_rate for tr in results]
            mean      = float(np.mean(rates))
            se        = float(np.std(rates, ddof=1) / np.sqrt(len(rates))) if len(rates) > 1 else 0.0
            z         = stats.norm.ppf(0.975)
            lo        = float(max(0.0, mean - z * se))
            hi        = float(min(1.0, mean + z * se))

            lines.append(f"Model: {model}")
            lines.append(f"{'─' * 50}")
            for tr in results:
                lines.append(str(tr))
                lines.append('')
            lines.append(
                f"  {model} total: {mean:.1%} "
                f"  95% CI: [{lo:.1%}, {hi:.1%}]  "
                f"({len(results)} runs)"
            )
            lines.append('')

        return '\n'.join(lines)

    def to_dict(self) -> dict:
        model_summaries = {}
        for model, results in self.results_by_model().items():
            rates        = [tr.weighted_success_rate for tr in results]
            mean         = float(np.mean(rates))
            se           = float(np.std(rates, ddof=1) / np.sqrt(len(rates))) if len(rates) > 1 else 0.0
            z            = stats.norm.ppf(0.975) # 95% confidence interval
            lo           = float(max(0.0, mean - z * se))
            hi           = float(min(1.0, mean + z * se))
            model_summaries[model] = {
                'success_rate': mean,
                'ci_low':       lo,
                'ci_high':      hi,
                'n_tasks':      len(results)
            }

        return {
            'run': {
                'models':     self.models,
                'judge':      self.judge,
                'difficulty': self.difficulty,
                'seeds':      self.seeds,
                'git_commit': self.git_commit,
                'timestamp':  self.timestamp
            },
            'by_model': model_summaries,
            'task_results': [tr.to_dict() for tr in self.task_results],
        }

    def save(self, results_dir: str = 'results'):
        """
        Save all results to results/benchmark_results_{timestamp}.json.
        """
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(results_dir) / f'benchmark_results_{self.timestamp}.json'

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"✓ Benchmark results saved: {filepath}")
        return filepath

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