import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from src.metarubric import Metarubric


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
        dimension:       metarubric dimension allowing to group items by skill type
    """
    metarubric_name: str
    total:           int
    passed:          int
    dimension:        str
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
        """95% CI on the weighted score via error propagation over metarubric binary outcomes.

        Uses Wilson-adjusted pass rates for variance so that 0/n and n/n rubrics
        contribute non-zero variance rather than collapsing the interval.
        """
        weight_sum = sum(mr.weight for mr in self.metarubric_results)
        if weight_sum == 0:
            return (0.0, 0.0)
        z = stats.norm.ppf(0.975)
        variance = 0.0
        for mr in self.metarubric_results:
            if mr.total == 0:
                continue
            p_adj = (mr.passed + z**2 / 2) / (mr.total + z**2)
            variance += (mr.weight / weight_sum) ** 2 * p_adj * (1 - p_adj) / mr.total
        se = np.sqrt(variance)
        return (
            float(max(0.0, self.weighted_success_rate - z * se)),
            float(min(1.0, self.weighted_success_rate + z * se)),
        )

    def __str__(self) -> str:
        lines = [
            f"Task:       {self.task_name}",
            f"Model:      {self.model}",
            f"Judge:      {self.judge}",
            f"Difficulty: {self.difficulty}  |  Seed: {self.seed}",
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
            'timestamp':  self.timestamp,
            'metarubrics': [
                {
                    'name':         mr.metarubric_name,
                    'dimension':     mr.dimension,
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

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskResults':
        """Reconstruct a TaskResults from a saved judge_response.json dict."""
        return cls(
            task_name          = d['task'],
            seed               = d['seed'],
            difficulty         = d['difficulty'],
            model              = d['model'],
            judge              = d['judge'],
            git_commit         = d.get('git_commit', ''),
            timestamp          = d['timestamp'],
            metarubric_results = [
                MetarubricResult(
                    metarubric_name = mr['name'],
                    dimension        = mr['dimension'],
                    total           = mr['total'],
                    passed          = mr['passed'],
                    weight          = mr['weight'],
                )
                for mr in d['metarubrics']
            ],
        )


# ─────────────────────────────────────────────────────────────
# BenchmarkResults
# ─────────────────────────────────────────────────────────────
@dataclass
class BenchmarkResults:
    """
    Results of evaluating a benchmark — collection of task results.

    Attributes:
        task_results:  list of TaskResults for each evaluated task
        model:         model evaluated
        judge:         judge model used for all runs
        difficulty:    difficulty level
        seeds:         random seeds used
        git_commit:    git commit hash for reproducibility
        timestamp:     timestamp of when benchmark run started
        agentic:       whether agentic evaluation was used
        max_turns:     maximum agentic turns (only relevant when agentic=True)
        partial:       whether this is a partial run (used to flag runs that have some failed tests)
    """
    task_results:  list[TaskResults]
    model:         str
    judge:         str
    difficulty:    str
    seeds:         list[int]
    git_commit:    str
    timestamp:     str
    agentic:       bool = False
    max_turns:     int  = 0
    partial:       bool = False

    @property
    def success_rate(self) -> float:
        """Weighted mean success rate across all metarubrics in all task results.

        Tasks with more or heavier metarubrics contribute proportionally more,
        making the score robust to tasks of different lengths.
        """
        weight_sum   = sum(mr.weight for tr in self.task_results for mr in tr.metarubric_results)
        weighted_sum = sum(mr.weight * mr.success_rate for tr in self.task_results for mr in tr.metarubric_results)
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% CI on the weighted score via error propagation over all metarubric binary outcomes.

        Uses Wilson-adjusted pass rates for variance so that 0/n and n/n rubrics
        contribute non-zero variance rather than collapsing the interval.
        Multiple seeds accumulate n_i across runs, naturally tightening the CI.
        """
        weight_sum = sum(
            mr.weight for tr in self.task_results for mr in tr.metarubric_results
        )
        if weight_sum == 0:
            return (0.0, 0.0)
        z = stats.norm.ppf(0.975)
        variance = 0.0
        for tr in self.task_results:
            for mr in tr.metarubric_results:
                if mr.total == 0:
                    continue
                p_adj = (mr.passed + z**2 / 2) / (mr.total + z**2)
                variance += (mr.weight / weight_sum) ** 2 * p_adj * (1 - p_adj) / mr.total
        se = np.sqrt(variance)
        return (
            float(max(0.0, self.success_rate - z * se)),
            float(min(1.0, self.success_rate + z * se)),
        )

    def results_by_task(self) -> dict[str, list[TaskResults]]:
        """Group task results by task name (one entry per seed)."""
        grouped: dict[str, list[TaskResults]] = {}
        for tr in self.task_results:
            grouped.setdefault(tr.task_name, []).append(tr)
        return grouped

    def results_by_dimension(self) -> dict[str, dict]:
        """Aggregate metarubric results by dimension using weighted success rates."""
        dims: dict[str, dict] = {}
        for tr in self.task_results:
            for mr in tr.metarubric_results:
                if mr.dimension not in dims:
                    dims[mr.dimension] = {'weight_sum': 0.0, 'weighted_sr_sum': 0.0}
                dims[mr.dimension]['weight_sum']     += mr.weight
                dims[mr.dimension]['weighted_sr_sum'] += mr.weight * mr.success_rate
        return {
            dim: {
                'weight':                vals['weight_sum'],
                'weighted_success_rate': (
                    vals['weighted_sr_sum'] / vals['weight_sum']
                    if vals['weight_sum'] > 0 else 0.0
                ),
            }
            for dim, vals in dims.items()
        }

    @staticmethod
    def _combine_metarubrics(results: list[TaskResults]) -> list[MetarubricResult]:
        """Sum passed/total per metarubric across TaskResults (e.g. multiple seeds).

        Raises ValueError if metarubric sets differ across results.
        """
        expected = [mr.metarubric_name for mr in results[0].metarubric_results]
        for tr in results[1:]:
            actual = [mr.metarubric_name for mr in tr.metarubric_results]
            if actual != expected:
                raise ValueError(
                    f"Metarubric mismatch for task '{results[0].task_name}': "
                    f"expected {expected}, got {actual} (seed {tr.seed})"
                )
        combined: dict[str, list] = {}
        for tr in results:
            for mr in tr.metarubric_results:
                if mr.metarubric_name not in combined:
                    combined[mr.metarubric_name] = [mr.dimension, mr.weight, 0, 0]
                combined[mr.metarubric_name][2] += mr.total
                combined[mr.metarubric_name][3] += mr.passed
        return [
            MetarubricResult(
                metarubric_name = name,
                dimension        = vals[0],
                weight          = vals[1],
                total           = vals[2],
                passed          = vals[3],
            )
            for name, vals in combined.items()
        ]

    def __str__(self) -> str:
        lines = []

        lines.append('=' * 50)
        lines.append("BENCHMARK RESULTS")
        lines.append('=' * 50)
        lines.append(f"Model:      {self.model}")
        lines.append(f"Judge:      {self.judge}")
        lines.append(f"Difficulty: {self.difficulty}")
        lines.append(f"Seeds:      {self.seeds}")
        lines.append(f"Commit:     {self.git_commit}  |  {self.timestamp}")
        if self.agentic:
            lines.append(f"Agentic:    YES  |  Max-turns: {self.max_turns}")
        else:
            lines.append(f"Agentic:    NO")
        
        lines.append('-' * 50)
        # Overall score
        if self.task_results:
            lo, hi = self.confidence_interval
            n_outcomes = sum(
                mr.total for tr in self.task_results for mr in tr.metarubric_results
            )
            lines.append(
                f"Overall score: {self.success_rate:.1%}  "
                f"95% CI: [{lo:.1%}, {hi:.1%}]  "
                f"({n_outcomes} binary outcomes across {len(self.task_results)} task runs)"
            )

        # Per-task summary combined across seeds
        lines.append('\n\n')
        lines.append('=' * 50)
        lines.append('TASK RESULTS AGGREGATED (combined across seeds)')
        lines.append('=' * 50)
        for task_name, task_results in self.results_by_task().items():
            combined_mrs = self._combine_metarubrics(task_results)
            total_weight = sum(mr.weight for mr in combined_mrs)
            weighted_sr  = (
                sum(mr.success_rate * mr.weight for mr in combined_mrs) / total_weight
                if total_weight > 0 else 0.0
            )
            seeds = [tr.seed for tr in task_results]
            lines.append(f"Task:       {task_name}")
            lines.append(f"Model:      {self.model}")
            lines.append(f"Judge:      {self.judge}")
            lines.append(f"Difficulty: {self.difficulty}  |  Seeds: {seeds}")
            for mr in combined_mrs:
                lines.append(f"      - {mr}  weight: {mr.weight}")
            lines.append(f"      {'─' * 50}")
            lines.append(f"      Weighted total: {weighted_sr:.1%}")

        # Individual task/seed results
        lines.append('\n\n')
        lines.append('=' * 50)
        lines.append('INDIVIDUAL TASK RESULTS')
        lines.append('=' * 50)
        for tr in self.task_results:
            lines.append(str(tr))
            lines.append('')

        # Per-dimension aggregation across all tasks and seeds
        lines.append('\n\n')
        lines.append('=' * 50)
        lines.append('BY DIMENSION (aggregated across all tasks and seeds)')
        lines.append('=' * 50)
        for dim, info in self.results_by_dimension().items():
            aligned = (dim + ":").ljust(30)
            lines.append(f"  {aligned} {info['weighted_success_rate']:.1%}")

        return '\n'.join(lines)

    def to_dict(self) -> dict:

        by_task = {}
        for task_name, task_results in self.results_by_task().items():
            combined_mrs = self._combine_metarubrics(task_results)
            total_weight = sum(mr.weight for mr in combined_mrs)
            weighted_sr  = (
                sum(mr.success_rate * mr.weight for mr in combined_mrs) / total_weight
                if total_weight > 0 else 0.0
            )
            by_task[task_name] = {
                'n_seeds':               len(task_results),
                'weighted_success_rate': weighted_sr,
                'metarubrics': [
                    {
                        'name':         mr.metarubric_name,
                        'dimension':     mr.dimension,
                        'total':        mr.total,
                        'passed':       mr.passed,
                        'weight':       mr.weight,
                        'success_rate': mr.success_rate,
                        'ci_low':       mr.confidence_interval[0],
                        'ci_high':      mr.confidence_interval[1],
                    }
                    for mr in combined_mrs
                ],
            }

        return {
            'run': {
                'model':      self.model,
                'judge':      self.judge,
                'difficulty': self.difficulty,
                'seeds':      self.seeds,
                'git_commit': self.git_commit,
                'timestamp':  self.timestamp,
                'agentic':    self.agentic,
                'max_turns':  self.max_turns if self.agentic else None,
                'partial':    self.partial,
            },
            'summary': {
                'success_rate': self.success_rate,
                'ci_low':       self.confidence_interval[0],
                'ci_high':      self.confidence_interval[1],
                'n_task_runs':  len(self.task_results),
            },
            'by_task':      by_task,
            'by_dimension': {
                dim: {
                    'weighted_success_rate': info['weighted_success_rate'],
                }
                for dim, info in self.results_by_dimension().items()
            },
            'task_results': [tr.to_dict() for tr in self.task_results],
        }

    def save(self, run_dir: Path, judge: str = ''):
        """Save aggregate results to run_dir/benchmark_results_judge_<judge>.json (or _partial-...)."""
        run_dir.mkdir(parents=True, exist_ok=True)
        judge_clean = judge.replace('/', '-').replace(':', '-') if judge else ''
        suffix = f'_judge_{judge_clean}' if judge_clean else ''
        prefix = '_partial-benchmark_results' if self.partial else 'benchmark_results'
        base = f'{prefix}{suffix}.json'
        filepath_candidate = run_dir / base
        if filepath_candidate.exists():
            idx = 1
            while (run_dir / f'{prefix}{suffix}_{idx}.json').exists():
                idx += 1
            base = f'{prefix}{suffix}_{idx}.json'
        filepath = run_dir / base

        if not self.partial:
            partial_base = f'_partial-benchmark_results{suffix}'
            for p in run_dir.iterdir():
                name = p.name
                if name == f'{partial_base}.json' or (name.startswith(f'{partial_base}_') and name.endswith('.json')):
                    p.unlink()
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        txt_path = filepath.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(str(self))

        print(f"✓ Benchmark results saved: {filepath}")
        return filepath
