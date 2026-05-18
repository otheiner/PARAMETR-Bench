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
from src.tools import TOOLS, _load_sandbox_libraries

import litellm
litellm.request_timeout = 300
#litellm._turn_on_debug()

class Evaluator:

    # ─────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────
    def get_model_output(self, task: Task, model: str,
                         agentic: bool, max_turns: int,
                         dest_dir: Path) -> str:
        """
        Run model on task, save model_response.json + rubrics.json to dest_dir.
        Returns the final answer string. Raises on any failure — nothing is saved
        unless the model call fully succeeds.
        """
        if agentic:
            model_output, messages = self._send_to_model_agentic(task, model, max_turns)
        else:
            model_output, messages = self._send_to_model(task, model)

        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / 'model_response.json', 'w') as f:
            json.dump({
                'task':     task.folder.name,
                'model':    model,
                'seed':     task.seed,
                'messages': messages,
            }, f, indent=2)

        shutil.copy(task.ground_truth_dir / 'rubrics.json', dest_dir / 'rubrics.json')
        print(f"✓ Model response saved: {dest_dir / 'model_response.json'}")
        return model_output

    def get_judge_results(self, task: Task, model: str, model_output: str,
                          rubrics_path: Path, judge: str,
                          dest_dir: Path) -> list[MetarubricResult]:
        """
        Judge model_output against rubrics_path, save judge_response.json to dest_dir.
        Raises on any failure — nothing is saved unless all metarubrics succeed.
        """
        with open(rubrics_path) as f:
            rubrics_data = json.load(f)

        mr_results, raw_data = self._judge(model_output, rubrics_data, judge)

        dest_dir.mkdir(parents=True, exist_ok=True)

        with open(dest_dir / 'task_results.json', 'w') as f:
            json.dump({
                'task':       task.folder.name,
                'model':      model,
                'judge':      judge,
                'seed':       task.seed,
                'difficulty': task.difficulty,
                'git_commit': get_git_hash(),
                'timestamp':  datetime.now().isoformat(),
                'metarubrics': [
                    {
                        'name':     mr.metarubric_name,
                        'category': mr.category,
                        'total':    mr.total,
                        'passed':   mr.passed,
                        'weight':   mr.weight,
                    }
                    for mr in mr_results
                ],
            }, f, indent=2)

        with open(dest_dir / 'judge_response.json', 'w') as f:
            json.dump({'metarubrics': raw_data}, f, indent=2)

        print(f"✓ Task results saved: {dest_dir / 'task_results.json'}")
        return mr_results

    # ─────────────────────────────────────────
    # Repeating attempts in case model API is unavailable
    # ─────────────────────────────────────────
    def _litellm_completion_with_retry(self, **kwargs):
        for attempt in range(3):
            try:
                return litellm.completion(**kwargs, request_timeout=300)
            except (litellm.ServiceUnavailableError,
                    litellm.RateLimitError,
                    litellm.InternalServerError,
                    litellm.Timeout,
                    litellm.APIConnectionError) as e:
                if attempt == 2:
                    print(e)
                    print(f"✗ API unavailable after 3 attempts:")
                    raise
                wait = 30 * (attempt + 1)
                print(f"ERROR: {e}")
                print(f"⚠  API unavailable — retrying in {wait}s ({attempt+1}/3)")
                time.sleep(wait)

    # ─────────────────────────────────────────
    # Print thinking blocks from a message dump
    # ─────────────────────────────────────────
    def _print_thinking(self, message_dump: dict):
        reasoning = message_dump.get('reasoning_content')
        if reasoning:
            print(f"\n[THINKING]\n{reasoning}\n[/THINKING]")

    # ─────────────────────────────────────────
    # Load judge prompt
    # ─────────────────────────────────────────
    def _load_judge_prompt(self, model_output: str, criteria: str) -> str:
        """Load judge prompt"""
        template = (Path(__file__).parent / 'judge_prompt.md').read_text()
        return template.format(model_output = model_output,
                               criteria = criteria)
    
    # ─────────────────────────────────────────
    # Load agentic prompt
    # ─────────────────────────────────────────
    def load_agentic_prompt(self, max_turns: int, vision: bool = True) -> str:
        """Load agentic prompt addition and fill in available libraries."""
        template = (Path(__file__).parent / 'agentic_prompt.md').read_text()
        view_image_note = ", and `view_image` to inspect image files" if vision else ""
        view_image_tool = (
            "- **view_image** — render an image file into your context so you can inspect it visually\n"
            if vision else ""
        )
        return template.format(
            libraries       = _load_sandbox_libraries(),
            max_turns       = max_turns,
            view_image_tool = view_image_tool,
            view_image_note = view_image_note,
        )

    # ─────────────────────────────────────────
    # Send to model
    # ─────────────────────────────────────────
    def _send_to_model(self, task: Task, model: str) -> tuple[str, list]:
        """Build message from task prompt + input files, call model, return response."""
        messages = [{
            'role':    'user',
            'content': [
                {'type': 'text', 'text': task.get_prompt()},
                *task.get_input_files(embed_data=True)
            ]
        }]

        try:
            response = self._litellm_completion_with_retry(
                model    = model,
                messages = messages,
                temperature = 0.0,
            )
            assistant_msg = response.choices[0].message.model_dump()
            messages.append(assistant_msg)
            model_output = response.choices[0].message.content

            self._print_thinking(assistant_msg)
            print(f"\n{'=' * 50}")
            print(f"MODEL OUTPUT ({model}):")
            print(f"{'=' * 50}")
            print(model_output)

            return model_output, messages

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

            # Providers that support multimodal content in tool result messages.
            # litellm.supports_vision() only checks the model, not whether the
            # provider accepts image blocks in tool results — Groq e.g. does not.
            _VISION_TOOL_PROVIDERS = {'anthropic', 'openai', 'azure', 'gemini', 'vertex_ai', 'bedrock', 'ollama'}
            try:
                _, provider, _, _ = litellm.get_llm_provider(model)
            except Exception:
                provider = ''
            vision   = provider in _VISION_TOOL_PROVIDERS and litellm.supports_vision(model=model)
            tools    = TOOLS if vision else [t for t in TOOLS if t['function']['name'] != 'view_image']

            # Include input files in the first message
            messages = [{
                'role':    'user',
                'content': [
                    {'type': 'text',
                     'text': task.get_prompt() + self.load_agentic_prompt(max_turns, vision)},
                            *task.get_input_files(embed_data=False)
                ]
            }]

            for turn in range(max_turns):
                response = self._litellm_completion_with_retry(
                    model       = model,
                    messages    = messages,
                    tools       = tools,
                    temperature = 0.0
                )

                if not response.choices:
                    raise RuntimeError(
                        f"Model {model} returned an empty response (no choices). "
                    )
                message = response.choices[0].message
                self._print_thinking(message.model_dump())

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

                    summary_msg  = summary.choices[0].message.model_dump()
                    messages.append(summary_msg)
                    self._print_thinking(summary_msg)
                    final_answer = summary.choices[0].message.content or ''

                    print(f"\n{'-' * 50}")
                    print("FINAL ANSWER:")
                    print('-' * 50)
                    print(final_answer)

                    return final_answer, messages

                # Append assistant turn to history
                messages.append(message.model_dump())

                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    print(f"\n{'-' * 50}")
                    print(f"TOOL CALL {name!r} (turn {turn + 1}):")
                    print(f"\n{'-' * 50}")

                    if name == 'execute_python':
                        output = self._execute_python(args['code'], session_dir)
                        print(args['code'])
                        print(f"OUTPUT:")
                        print(output)

                        if len(output) > 5000:
                            output = output[:5000] + "\n...(truncated)"

                        messages.append({
                            'role':         'tool',
                            'tool_call_id': tool_call.id,
                            'name':         'execute_python',
                            'content':      output
                        })

                    elif name == 'run_command':
                        command = args['command']
                        print(f"command: {command}")
                        content = self._run_command(command, session_dir)
                        print(content)
                        messages.append({
                            'role':         'tool',
                            'tool_call_id': tool_call.id,
                            'name':         'run_command',
                            'content':      content
                        })

                    elif name == 'write_file':
                        path, content = args['path'], args['content']
                        print(f"path: {path}")
                        result = self._write_file(path, content, session_dir)
                        print(result)
                        messages.append({
                            'role':         'tool',
                            'tool_call_id': tool_call.id,
                            'name':         'write_file',
                            'content':      result
                        })

                    elif name == 'read_file':
                        path = args['path']
                        print(f"path: {path}")
                        content = self._read_file(path, session_dir)
                        print(content)
                        messages.append({
                            'role':         'tool',
                            'tool_call_id': tool_call.id,
                            'name':         'read_file',
                            'content':      content
                        })

                    elif name == 'view_image':
                        path = args['path']
                        print(f"path: {path}")
                        content = self._view_image(path, session_dir)
                        messages.append({
                            'role':         'tool',
                            'tool_call_id': tool_call.id,
                            'name':         'view_image',
                            'content':      content
                        })

                    else:
                        print(f"WARNING: unknown tool '{name}' — returning error to model")
                        messages.append({
                            'role':         'tool',
                            'tool_call_id': tool_call.id,
                            'name':         name,
                            'content':      f"Error: unknown tool '{name}'."
                        })

                # Inform the model of remaining turns after each tool-call round
                remaining = max_turns - turn - 1
                messages.append({
                    'role':    'user',
                    'content': f"[Turn {turn + 1}/{max_turns} used. {remaining} turn{'s' if remaining != 1 else ''} remaining.]"
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

            summary_msg  = summary.choices[0].message.model_dump()
            messages.append(summary_msg)
            self._print_thinking(summary_msg)
            final_answer = summary.choices[0].message.content or "No summary produced."

            print(f"\n{'-' * 50}")
            print("FINAL ANSWER (max turns reached):")
            print('-' * 50)
            print(final_answer)

            return final_answer, messages

        finally:
            # Clean up session workspace — always runs even if exception occurs
            shutil.rmtree(session_dir, ignore_errors=True)



    # ─────────────────────────────────────────
    # Judge
    # ─────────────────────────────────────────
    def _judge(self, model_output: str,
               rubrics_data: dict,
               judge: str) -> tuple[list[MetarubricResult], list[dict]]:
        """
        Judge model output against pre-loaded rubrics. Raises on any failure.
        Returns (scores, raw_data) where raw_data drives judge_response.json.
        """
        results  = []
        raw_data = []
        for mr_data in rubrics_data['metarubrics']:
            rubrics = [r['criterion'] for r in mr_data['rubrics']]
            passed, rubric_verdicts = self._judge_metarubric(rubrics, model_output, judge)

            results.append(MetarubricResult(
                metarubric_name = mr_data['name'],
                category        = mr_data.get('category', ''),
                total           = len(rubrics),
                passed          = passed,
                weight          = mr_data['weight']
            ))
            raw_data.append({
                'name':     mr_data['name'],
                'category': mr_data.get('category', ''),
                'rubrics':  rubric_verdicts,
            })

        return results, raw_data

    # ─────────────────────────────────────────
    # Judge metarubrics either as batch or single
    # ─────────────────────────────────────────
    def _judge_metarubric(self, rubrics: list[str],
                           model_output: str,
                           judge: str) -> tuple[int, list[dict]]:
        """
        Returns (passed count, [{criterion, verdict}, ...]) where verdict is 'YES' or 'NO'.
        Uses batch call for API judges, single calls for local models.
        """
        if judge.startswith('ollama/'):
            rubric_verdicts = []
            for rubric in rubrics:
                verdict = self._judge_single(rubric, model_output, judge)
                rubric_verdicts.append({'criterion': rubric, 'verdict': verdict})
            return sum(1 for r in rubric_verdicts if r['verdict'] == 'YES'), rubric_verdicts
        else:
            return self._judge_batch(rubrics, model_output, judge)

    # ─────────────────────────────────────────
    # Judge single rubric item
    # ─────────────────────────────────────────
    def _judge_single(self, rubric: str,
                       model_output: str,
                       judge: str) -> str:
        """One rubric, one call. Returns 'YES' or 'NO'."""
        prompt = self._load_judge_prompt(model_output=model_output, criteria=rubric)
        response = self._litellm_completion_with_retry(
            model       = judge,
            messages    = [{'role': 'user', 'content': prompt}],
            temperature = 0.0
        )
        raw = response.choices[0].message.content.strip().upper()
        return 'YES' if raw.startswith('YES') else 'NO'

    # ─────────────────────────────────────────
    # Batch rubrics in one metarubric
    # ─────────────────────────────────────────
    def _judge_batch(self, rubrics: list[str],
                     model_output: str,
                     judge: str) -> tuple[int, list[dict]]:
        """
        All rubrics in one call — for capable API models.
        Returns (passed count, [{criterion, verdict}, ...]) where verdict is 'YES' or 'NO'.
        """
        numbered = '\n'.join(f"{i+1}. {r}" for i, r in enumerate(rubrics))
        prompt   = self._load_judge_prompt(model_output=model_output, criteria=numbered)

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
            raise ValueError(
                f"Judge returned {len(verdicts)} verdicts for {len(rubrics)} rubrics"
            )

        rubric_verdicts = [
            {'criterion': r, 'verdict': 'YES' if v else 'NO'}
            for r, v in zip(rubrics, verdicts)
        ]
        return sum(1 for v in verdicts if v), rubric_verdicts

    # ─────────────────────────────────────────
    # Run shell command in Docker sandbox
    # ─────────────────────────────────────────
    _ALLOWED_COMMANDS = {
        'grep', 'sed', 'awk', 'find', 'head', 'tail',
        'cat', 'wc', 'sort', 'uniq', 'cut', 'ls', 'file',
        'mkdir', 'touch', 'cp', 'cd',
    }

    def _run_command(self, command: str, session_dir: Path, max_chars: int = 5_000) -> str:
        # Validate every command segment (split on ||, &&, |, ;)
        for segment in re.split(r'\|\||&&|[|;]', command):
            tokens = segment.strip().split()
            if not tokens:
                continue
            if tokens[0] not in self._ALLOWED_COMMANDS:
                allowed = ', '.join(sorted(self._ALLOWED_COMMANDS))
                return f"Error: '{tokens[0]}' is not allowed. Allowed commands: {allowed}."

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
            output = client.containers.run(
                image         = 'benchmark-sandbox',
                command       = ['sh', '-c', command],
                working_dir   = '/home/agent/workspace',
                user          = 'agent',
                network_mode  = 'none',
                mem_limit     = '512m',
                memswap_limit = '512m',
                cpu_quota     = 50000,
                pids_limit    = 64,
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

            result = output.decode() if output else "(no output)"

        except docker.errors.ContainerError as e:
            result = e.stderr.decode() if e.stderr else str(e)

        except Exception as e:
            result = f"Execution error: {e}"

        finally:
            if cleanup:
                shutil.rmtree(tmpdir, ignore_errors=True)

        if len(result) > max_chars:
            result = result[:max_chars] + f"\n...(truncated at {max_chars} chars)"
        return result

    # ─────────────────────────────────────────
    # Write text file on agents request
    # ─────────────────────────────────────────
    def _write_file(self, path: str, content: str, session_dir: Path) -> str:
        file_path = session_dir / path
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return f"Written {len(content)} chars to '{path}'."
        except Exception as e:
            return f"Error writing '{path}': {e}"

    # ─────────────────────────────────────────
    # Read text/csv file on agents request
    # ─────────────────────────────────────────
    def _read_file(self, path: str, session_dir: Path, max_chars: int = 10_000) -> str:
        file_path = session_dir / path
        if not file_path.exists():
            return f"Error: '{path}' not found in workspace."
        try:
            content = file_path.read_text()
        except Exception as e:
            return f"Error reading '{path}': {e}"
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n...(truncated at {max_chars} chars)"
        return content

    # View image on agents request
    # ─────────────────────────────────────────
    def _view_image(self, path: str, session_dir: Path):
        import base64, mimetypes
        image_path = session_dir / path
        if not image_path.exists():
            return f"Error: '{path}' not found in workspace."
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type not in ('image/png', 'image/jpeg', 'image/gif', 'image/webp'):
            return f"Error: '{path}' is not a supported image type (png/jpeg/gif/webp)."
        data = base64.standard_b64encode(image_path.read_bytes()).decode()
        return [
            {'type': 'text',      'text': f"Image '{path}':"},
            {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{data}'}}
        ]

    # ─────────────────────────────────────────
    # Execute Python in Docker container sandbox
    # ─────────────────────────────────────────
    def _execute_python(self, code: str, session_dir: Path) -> str:
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