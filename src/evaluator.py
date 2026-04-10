import json
import re
import subprocess
import litellm
from datetime import datetime
from pathlib import Path

from src.task import Task, TaskResults, MetarubricResult
from src.utils import get_git_hash

class Evaluator:

    # ─────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────

    def run(self, task: Task, model: str, judge: str) -> TaskResults:
        """Run full evaluation pipeline for one task/model/seed."""

        # Step 1 — send task to model
        model_output = self._send_to_model(task, model)

        # Step 2 — judge output against pre-generated rubrics
        mr_results = self._judge(task, model_output, judge)

        # Step 3 — build and return TaskResults
        return TaskResults(
            task_name          = task.folder.name,
            seed               = task.seed,
            difficulty         = task.difficulty,
            model              = model,
            judge              = judge,
            git_commit         = get_git_hash(),
            timestamp          = datetime.now().isoformat(),
            metarubric_results = mr_results
        )

    # ─────────────────────────────────────────
    # Send to model
    # ─────────────────────────────────────────

    def _send_to_model(self, task: Task, model: str) -> str:
        """Build message from task prompt + input files, call model, return response."""
        messages = [{
            'role':    'user',
            'content': [
                {'type': 'text', 'text': task.get_prompt()},
                *task.get_input_files(model)
            ]
        }]

        try:
            response = litellm.completion(
                model    = model,
                messages = messages,
                temperature = 0.0
            )
            model_output = response.choices[0].message.content
    
            print(f"\n{'─' * 50}")
            print(f"MODEL OUTPUT ({model}):")
            print(f"{'─' * 50}")
            print(model_output)
            print(f"{'─' * 50}\n")
            
            return model_output

        except litellm.AuthenticationError:
            print(f"✗ Authentication failed for '{model}' — check your API key")
            raise

        except Exception as e:
            print(f"✗ Model call failed: {e}")
            raise

    # ─────────────────────────────────────────
    # Judge
    # ─────────────────────────────────────────

    def _judge(self, task: Task,
               model_output: str,
               judge: str) -> list[MetarubricResult]:
        """Load pre-generated rubrics and judge model output against them."""

        with open(task.ground_truth_dir / 'rubrics.json') as f:
            rubrics_data = json.load(f)

        results = []
        for mr_data in rubrics_data['metarubrics']:
            rubrics = [r['criterion'] for r in mr_data['rubrics']]
            passed  = self._judge_metarubric(rubrics, model_output, judge)

            results.append(MetarubricResult(
                metarubric_name = mr_data['name'],
                total           = len(rubrics),
                passed          = passed,
                weight          = mr_data['weight']
            ))

        return results

    def _judge_metarubric(self, rubrics: list[str],
                           model_output: str,
                           judge: str) -> int:
        """Judge all criteria in one metarubric — one call per criterion."""
        passed = 0
        for rubric in rubrics:
            if self._judge_single(rubric, model_output, judge):
                passed += 1
        return passed

    def _judge_single(self, rubric: str,
                       model_output: str,
                       judge: str) -> bool:
        """One rubric, one YES/NO question — simplest possible judge call."""
        prompt = f"""Answer YES or NO only.

Criterion: {rubric}

Model response:
{model_output}

Answer (YES or NO):"""

        try:
            response = litellm.completion(
                model       = judge,
                messages    = [{'role': 'user', 'content': prompt}],
                temperature = 0.0
            )
            answer = response.choices[0].message.content.strip().upper()
            return answer.startswith('YES')

        except Exception as e:
            print(f"⚠  Judge call failed: {e} — counting as not passed")
            return False