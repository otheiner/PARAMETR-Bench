import json
import re
from datetime import datetime
from pathlib import Path
import docker
import shutil
import tempfile
import time

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from src.task import Task, TaskResults, MetarubricResult
from src.utils import get_git_hash
from src.tools import TOOLS, load_agentic_prompt

import litellm
litellm.request_timeout = 300

class Evaluator:

    # ─────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────
    def run(self, task: Task, model: str, judge: str,
            agentic: bool = False,
            max_turns: int  = 10) -> TaskResults:
        """Run full evaluation pipeline for one task/model/seed."""

        # Step 1 — send task to model
        if agentic:
            model_output = self._send_to_model_agentic(task, model, max_turns)
        else:
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
    # Repeating attempts in case model API is unavailable
    # ─────────────────────────────────────────
    def _litellm_completion_with_retry(self, **kwargs):
        for attempt in range(3):
            try:
                return litellm.completion(**kwargs)
            except (litellm.ServiceUnavailableError,
                    litellm.RateLimitError) as e:
                if attempt == 2:
                    print(e)
                    print(f"✗ API unavailable after 3 attempts:")
                    raise
                wait = 30 * (attempt + 1)
                print(f"ERROR: {e}")
                print(f"⚠  API unavailable — retrying in {wait}s ({attempt+1}/3)")
                time.sleep(wait)

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
            response = self._litellm_completion_with_retry(
                model    = model,
                messages = messages,
                temperature = 0.0,
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
    # Send to model - multiple turns for agent
    # ─────────────────────────────────────────
    def _send_to_model_agentic(self, task: Task, model: str,
                            max_turns: int) -> str:
        """
        Agentic evaluation — model can write and execute Python scripts.
        Images are included in the first message and remain accessible
        throughout the conversation via the full history.
        A persistent workspace is shared across all turns so the agent
        can save and load intermediate files between tool calls.
        """
        # Create persistent workspace for this session
        session_dir = Path(tempfile.mkdtemp())

        try:
            # Copy input files once into session workspace
            shutil.copytree(task.input_dir, session_dir, dirs_exist_ok=True)

            messages = [{
                'role':    'user',
                'content': [
                    {'type': 'text', 
                     'text': task.get_prompt() + load_agentic_prompt(max_turns)},
                            *task.get_input_files(model)
                ]
            }]

            for turn in range(max_turns):
                response = self._litellm_completion_with_retry(
                    model       = model,
                    messages    = messages,
                    tools       = TOOLS,
                    temperature = 0.0
                )

                message = response.choices[0].message

                # No tool calls — model finished analysis, ask for summary
                if not message.tool_calls:
                    messages.append(message.model_dump())
                    messages.append({
                        'role':    'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': (
                                    'Analysis complete. Now state your final results '
                                    'explicitly following the output format specified '
                                    'in the task instructions. Do not use any tools — '
                                    'only summarise your findings in plain text.'
                                )
                            }
                        ]
                    })

                    summary = self._litellm_completion_with_retry(
                        model       = model,
                        messages    = messages,
                        temperature = 0.0
                    )

                    final_answer = summary.choices[0].message.content or ''

                    print(f"\n{'-' * 50}")
                    print("FINAL ANSWER:")
                    print('-' * 50)
                    print(final_answer)

                    return final_answer

                # Append assistant turn to history
                messages.append(message.model_dump())

                # Execute each tool call
                for tool_call in message.tool_calls:
                    code   = json.loads(tool_call.function.arguments)['code']
                    output = self._execute_python(code, task, session_dir)

                    print(f"\n{'-' * 50}")
                    print(f"TOOL CALL (turn {turn + 1}):")
                    print(f"\n{'-' * 50}")
                    print(code)
                    print(f"OUTPUT:")
                    print(output)

                    # Truncate long outputs to avoid context explosion
                    if len(output) > 5000:
                        output = output[:5000] + "\n...(truncated)"

                    messages.append({
                        'role':         'tool',
                        'tool_call_id': tool_call.id,
                        'name':         'execute_python',
                        'content':      output
                    })

            # Max turns reached — ask for summary of what was found
            messages.append({
                'role':    'user',
                'content': [
                    {
                        'type': 'text',
                        'text': (
                            'Maximum tool calls reached. State your final results '
                            'explicitly following the output format specified in the '
                            'task instructions based on what you have found so far.'
                        )
                    }
                ]
            })

            summary = self._litellm_completion_with_retry(
                model       = model,
                messages    = messages,
                temperature = 0.0
            )

            final_answer = summary.choices[0].message.content or "No summary produced."

            print(f"\n{'-' * 50}")
            print("FINAL ANSWER (max turns reached):")
            print('-' * 50)
            print(final_answer)

            return final_answer

        finally:
            # Clean up session workspace — always runs even if exception occurs
            shutil.rmtree(session_dir, ignore_errors=True)



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

    # ─────────────────────────────────────────
    # Judge metarubrics either as batch or single
    # ─────────────────────────────────────────
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

    # ─────────────────────────────────────────
    # Judge single rubric item
    # ─────────────────────────────────────────
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
            response = self._litellm_completion_with_retry(
                model       = judge,
                messages    = [{'role': 'user', 'content': prompt}],
                temperature = 0.0
            )
            answer = response.choices[0].message.content.strip().upper()
            return answer.startswith('YES')

        except Exception as e:
            print(f"⚠  Judge call failed: {e} — counting as not passed")
            return False
        
    # ─────────────────────────────────────────
    # Batch rubrics in one metarubric
    # ─────────────────────────────────────────
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
            response = self._litellm_completion_with_retry(
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

    # ─────────────────────────────────────────
    # Execute Python in Docker container sandbox
    # ─────────────────────────────────────────
    def _execute_python(self, code: str, task: Task,
                    session_dir: Path) -> str:
        """
        Execute model-generated Python code in an isolated Docker container.
        If session_dir is provided, it is mounted as read-write workspace
        allowing the agent to persist files between turns.
        Otherwise a fresh read-only workspace is created per call.
        """
        # Check Docker daemon is running before attempting anything
        try:
            client = docker.from_env()
            client.ping()
        except docker.errors.DockerException:
            raise RuntimeError(
                "⚠  Docker daemon is not running. "
                "Start Docker Desktop and try again."
            )

        # Use persistent session_dir
        tmpdir  = session_dir
        mode    = 'rw'      # read-write — agent can save files
        cleanup = False     # session_dir managed by caller

        try:
            # Write script into workspace
            (tmpdir / 'script.py').write_text(code)

            output = client.containers.run(
                image         = 'benchmark-sandbox',
                command       = 'python /home/agent/workspace/script.py',
                working_dir   = '/home/agent/workspace',
                user          = 'agent',
                network_mode  = 'none',
                mem_limit     = '512m',
                memswap_limit = '512m',
                cpu_quota     = 50000,
                volumes       = {
                    str(tmpdir): {
                        'bind': '/home/agent/workspace',
                        'mode': mode
                    }
                },
                tmpfs         = {'/tmp': ''},
                detach        = False,
                stdout        = True,
                stderr        = True,
                remove        = True,
            )

            return output.decode() if output else "(no output)"

        except docker.errors.ContainerError as e:
            return e.stderr.decode() if e.stderr else str(e)

        except Exception as e:
            return f"Execution error: {e}"

        finally:
            if cleanup:
                shutil.rmtree(tmpdir, ignore_errors=True)