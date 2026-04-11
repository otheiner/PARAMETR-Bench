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
    
            print(f"\n{'=' * 50}")
            print(f"MODEL OUTPUT ({model}):")
            print(f"{'=' * 50}")
            print(model_output)
            
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
        """
        Judge all criteria in one metarubric.
        Uses batch call for API judges, single calls for local models.
        """
        if judge.startswith('ollama/'):
            # Local models — one call per rubric, more reliable
            passed = 0
            for rubric in rubrics:
                if self._judge_single(rubric, model_output, judge):
                    passed += 1
            return passed
        else:
            # API models — batch all rubrics in one call
            return self._judge_batch(rubrics, model_output, judge)

    def _judge_single(self, rubric: str,
                       model_output: str,
                       judge: str) -> bool:
        """One rubric, one YES/NO question — simplest possible judge call."""
        prompt = f"""Model response:
                    {model_output}

                    Criterion:
                    {rubric}

                    Answer YES or NO only."""

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
        
    def _judge_batch(self, rubrics: list[str],
                  model_output: str,
                  judge: str) -> int:
        """
        Send all rubrics in one call — for capable API models.
        Returns number of criteria passed.
        """
        numbered = '\n'.join(
            f"{i+1}. {r}" for i, r in enumerate(rubrics)
        )
        
        prompt = f"""You are evaluating a scientific analysis response.

                    For each numbered criterion below, answer YES or NO.
                    Return ONLY a JSON array of booleans in the same order as the criteria.
                    No explanation. No markdown. No extra text.

                    Example for 3 criteria: [true, false, true]

                    Model response:
                    {model_output}

                    Criteria:
                    {numbered}

                    Answer JSON array only."""

        try:
            response = litellm.completion(
                model       = judge,
                messages    = [{'role': 'user', 'content': prompt}],
                temperature = 0.0
            )
            raw = response.choices[0].message.content.strip()
            
            # Strip markdown if present
            raw = re.sub(r'```json\s*', '', raw)
            raw = re.sub(r'```\s*',     '', raw)
            raw = raw.strip()
            
            verdicts = json.loads(raw)
            
            if len(verdicts) != len(rubrics):
                print(f"⚠  Judge returned {len(verdicts)} verdicts for {len(rubrics)} rubrics")
                return 0
            
            return sum(1 for v in verdicts if v)
        
        except json.JSONDecodeError:
            print(f"⚠  Judge parse failed — raw response: {raw[:200]}")
            return 0
        
        except Exception as e:
            print(f"⚠  Judge call failed: {e}")
            return 0