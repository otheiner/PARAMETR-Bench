import json
import re
from datetime import datetime
from pathlib import Path
import docker
import tarfile
import io
import shutil
import tempfile

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
        model_output = self._send_to_model(task, model)
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
        
    # ─────────────────────────────────────────
    # Allow python execution for agentic evaluation
    # ─────────────────────────────────────────
    def _execute_python(self, code: str, task: Task) -> str:
        """
        This method calls _run_container and checks the execution time.
        If the runtime exceeds 30 seconds, it times out.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_container, code, task)
            try:
                return future.result(timeout=30)
            except TimeoutError:
                return "Execution timed out after 30 seconds"
        
    def _run_container(self, code: str, task: Task) -> str:
        """
        Execute model-generated Python code in an isolated Docker container.
        Input files available at /home/agent/workspace/.
        No network, memory capped, non-root user.
        """
        client = docker.from_env()

        # Build tar archive of input files + script in memory
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
            for filepath in task.input_dir.iterdir():
                if filepath.is_file():
                    tar.add(filepath, arcname=filepath.name)

            script_bytes = code.encode('utf-8')
            info         = tarfile.TarInfo(name='script.py')
            info.size    = len(script_bytes)
            tar.addfile(info, io.BytesIO(script_bytes))

        tar_buffer.seek(0)

        try:
            container = client.containers.create(
                image         = 'benchmark-sandbox',
                command       = 'python /home/agent/workspace/script.py',
                working_dir   = '/home/agent/workspace',
                user          = 'agent',
                network_mode  = 'none',
                mem_limit     = '512m',
                memswap_limit = '512m',
                cpu_quota     = 50000,
                read_only     = True,
                tmpfs         = {'/tmp': ''},
            )

            container.put_archive('/home/agent/workspace', tar_buffer)
            container.start()
            logs      = container.logs(stdout=True, stderr=True).decode()

            return logs if logs else "(no output)"

        except Exception as e:
            return f"Execution error: {e}"

        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

    # ─────────────────────────────────────────
    # Execute python in sandbox
    # ─────────────────────────────────────────
    def _execute_python(self, code: str, task: Task) -> str:
        client = docker.from_env()

        # Create temp directory with input files + script
        tmpdir = Path(tempfile.mkdtemp())

        try:
            # Copy input files
            for filepath in task.input_dir.iterdir():
                if filepath.is_file():
                    shutil.copytree(
                        task.input_dir,
                        tmpdir,
                        dirs_exist_ok = True
                    )

            # Write script
            (tmpdir / 'script.py').write_text(code)

            # Mount as read-only volume
            container = client.containers.run(
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
                        'mode': 'ro'        # read-only mount
                    }
                },
                tmpfs         = {'/tmp': ''},
                detach        = False,
                stdout        = True,
                stderr        = True,
                remove        = True,
            )

            return container.decode() if container else "(no output)"

        except docker.errors.ContainerError as e:
            return e.stderr.decode() if e.stderr else str(e)

        except Exception as e:
            return f"Execution error: {e}"

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ─────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────
    def _send_to_model_agentic(self, task: Task, model: str, 
                               max_turns: int) -> str:
        """
        Agentic evaluation — model can write and execute Python scripts.
        Loops until model returns final answer without tool calls
        or until max_turns is reached.
        """
        # Build initial message with full content including images
        initial_content = [
            {'type': 'text', 'text': task.get_prompt() + load_agentic_prompt(max_turns)},
            *task.get_input_files(model)
        ]

        messages = [{'role': 'user', 'content': initial_content}]

        for turn in range(max_turns):
            response = litellm.completion(
                model       = model,
                messages    = messages,
                tools       = TOOLS,
                temperature = 0.0
            )

            message = response.choices[0].message

            # No tool calls — model is done
            if not message.tool_calls:
                messages.append(message.model_dump())
                messages.append({
                    'role':    'user',
                    'content': (
                        'Analysis complete. Now state your final results '
                        'explicitly following the output format specified '
                        'in the task instructions. Do not use any tools — '
                        'only summarise your findings in plain text.'
                    )
                })

                summary = litellm.completion(
                    model       = model,
                    messages    = messages,
                    temperature = 0.0,
                    # No tools passed — forces text-only response
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
                output = self._execute_python(code, task)

                print(f"\n{'-' * 50}")
                print(f"TOOL CALL (turn {turn + 1}):")
                print(f"\n{'-' * 50}")
                print(code)
                print(f"OUTPUT:")
                print(output)

                messages.append({
                    'role':         'tool',
                    'tool_call_id': tool_call.id,
                    'name':         'execute_python',
                    'content':      output
                })

        return message.content or "Max turns reached."
    
    def _send_to_model_agentic(self, task: Task, model: str,
                            max_turns: int) -> str:
        """
        Agentic evaluation — model can write and execute Python scripts.
        Images are sent once in turn 1 for visual context.
        Subsequent turns reference files via execute_python only.
        Loops until model returns final answer without tool calls
        or until max_turns is reached.
        """
        messages = [{
            'role':    'user',
            'content': [
                {'type': 'text', 'text': task.get_prompt() + load_agentic_prompt(max_turns)},
                *task.get_input_files(model)
            ]
        }]

        for turn in range(max_turns):
            response = litellm.completion(
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
                    'content': (
                        'Analysis complete. Now state your final results '
                        'explicitly following the output format specified '
                        'in the task instructions. Do not use any tools — '
                        'only summarise your findings in plain text.'
                    )
                })

                summary = litellm.completion(
                    model       = model,
                    messages    = messages,
                    temperature = 0.0
                    # No tools — forces text-only response
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
                output = self._execute_python(code, task)

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

            # After turn 0 — strip base64 images, add file list
            if turn == 0:
                file_list = [
                    str(f.relative_to(task.input_dir))
                    for f in sorted(task.input_dir.rglob('*'))
                    if f.is_file() and f.name != '.gitignore'
                ]

                messages[0] = {
                    'role':    'user',
                    'content': [
                        {'type': 'text', 'text': task.get_prompt() + load_agentic_prompt(max_turns)},
                        {'type': 'text', 'text':
                        'Input files available in your working directory:\n' +
                        '\n'.join(f'  - {f}' for f in file_list) +
                        '\nAccess them via execute_python.'}
                    ]
                }

        # Max turns reached — ask for summary of what was found
        messages.append({
            'role':    'user',
            'content': (
                'Maximum tool calls reached. State your final results '
                'explicitly following the output format specified in the '
                'task instructions based on what you have found so far.'
            )
        })

        summary = litellm.completion(
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