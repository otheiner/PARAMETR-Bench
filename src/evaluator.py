import json
import re
from datetime import datetime
from pathlib import Path
import docker
import shutil
import tempfile
import time

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from src.task import Task
from src.results import TaskResults, MetarubricResult
from src.utils import get_git_hash
from src.tools import TOOLS, _load_sandbox_libraries

import litellm
litellm.timeout = 300
litellm.modify_params = True   # inject dummy tool when history has tool-use but tools= is omitted
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

        If _partial_model_response.json exists in dest_dir, the agentic loop is
        resumed from the saved turn with the saved conversation history and
        session workspace restored from _partial_session/.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)

        if agentic:
            partial_path = dest_dir / '_partial_model_response.json'
            initial_messages, start_turn = None, 0
            if partial_path.exists():
                partial_data = json.loads(partial_path.read_text())
                initial_messages = partial_data['messages']
                start_turn       = partial_data['start_turn']
                print(f"↩  Resuming agentic loop from turn {start_turn + 1}/{max_turns} (partial state found)")
            model_output, messages = self._send_to_model_agentic(
                task, model, max_turns, dest_dir,
                initial_messages=initial_messages,
                start_turn=start_turn,
            )
        else:
            model_output, messages = self._send_to_model(task, model)

        with open(dest_dir / 'model_response.json', 'w') as f:
            json.dump({
                'task':     task.folder.name,
                'model':    model,
                'seed':     task.seed,
                'messages': messages,
            }, f, indent=2)

        print(f"✓ Model response saved: {dest_dir / 'model_response.json'}")
        return model_output

    def get_judge_results(self, task: Task, model: str, model_output: str,
                          rubrics_path: Path, judge: str,
                          dest_dir: Path) -> list[MetarubricResult]:
        """
        Judge model_output against rubrics_path, save judge_response.json to dest_dir.
        Raises on any failure — nothing is saved unless all metarubrics succeed.
        """
        model_output = model_output if isinstance(model_output, str) else ''

        with open(rubrics_path) as f:
            rubrics_data = json.load(f)

        mr_results, raw_data = self._judge(model_output, rubrics_data, judge)

        dest_dir.mkdir(parents=True, exist_ok=True)

        judge_clean = judge.replace('/', '-').replace(':', '-')
        base = dest_dir / f'judge_response_{judge_clean}.json'
        if base.exists():
            idx = 1
            while (dest_dir / f'judge_response_{judge_clean}_{idx}.json').exists():
                idx += 1
            judge_resp_path = dest_dir / f'judge_response_{judge_clean}_{idx}.json'
        else:
            judge_resp_path = base
        with open(judge_resp_path, 'w') as f:
            json.dump({'judge': judge, 'metarubrics': raw_data}, f, indent=2)

        print(f"✓ Judge response saved: {judge_resp_path}")
        return mr_results

    # ─────────────────────────────────────────
    # Repeating attempts in case model API is unavailable
    # ─────────────────────────────────────────
    def _litellm_completion_with_retry(self, **kwargs):
        for attempt in range(3):
            try:
                return litellm.completion(**kwargs, timeout=300)
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
    # Print data/cache usage blocks
    # ─────────────────────────────────────────
    def _print_cache_usage(self, response):
        usage = getattr(response, 'usage', None)
        if usage is None:
            return
        priv    = getattr(usage, '__pydantic_private__', {}) or {}
        created = priv.get('_cache_creation_input_tokens', 0) or 0
        read    = priv.get('_cache_read_input_tokens', 0) or 0
        if created or read:
            print(f"  [cache] write={created} read={read}")

    # ─────────────────────────────────────────
    # Applying caching for Anthropic models
    # ─────────────────────────────────────────
    def _apply_cache_breakpoint(self, messages: list, model: str) -> None:
        """Place Anthropic cache breakpoints on the first and last messages.

        Anthropic allows at most 4 cache_control blocks per request. We use 2:
        - messages[0]: stable cache for task instructions (cache hit every turn)
        - messages[-1]: rolling cache for the full conversation history

        Old markers are cleared before adding new ones so they never accumulate.
        Cache reads are content-based on Anthropic's side, so old cached prefixes
        are reused automatically even after their cache_control marker is removed.
        """
        provider = model.split('/')[0] if '/' in model else ''
        if not messages or not ('claude' in model or provider == 'anthropic'):
            return

        # Remove all existing markers left from previous turns
        for msg in messages:
            content = msg.get('content')
            if isinstance(content, list):
                for i, block in enumerate(content):
                    if isinstance(block, dict) and 'cache_control' in block:
                        content[i] = {k: v for k, v in block.items() if k != 'cache_control'}

        def mark_last_text_block(msg):
            content = msg.get('content')
            if isinstance(content, str):
                msg['content'] = [{'type': 'text', 'text': content,
                                    'cache_control': {'type': 'ephemeral'}}]
            elif isinstance(content, list):
                for i in range(len(content) - 1, -1, -1):
                    block = content[i]
                    if isinstance(block, dict) and block.get('type') in ('text', 'tool_result'):
                        content[i] = {**block, 'cache_control': {'type': 'ephemeral'}}
                        break

        mark_last_text_block(messages[0])
        if len(messages) > 1:
            mark_last_text_block(messages[-1])

    # ─────────────────────────────────────────
    # Strip base64 image data from old view_image 
    # tool results when context window is exceeded.
    # ─────────────────────────────────────────
    def _strip_old_images(self, messages: list) -> None:
        """Replace base64 image data in old view_image results with a placeholder.
        Called only when context window is exceeded; keeps images from the last turn intact."""
        # Find the start of the last turn: the last assistant message that had tool calls
        last_turn_start = 0
        for i, m in enumerate(messages):
            if m.get('role') == 'assistant' and m.get('tool_calls'):
                last_turn_start = i

        for i, m in enumerate(messages):
            if i >= last_turn_start:
                break
            if m.get('name') == 'view_image' and isinstance(m.get('content'), list):
                messages[i]['content'] = [
                    b if b.get('type') != 'image_url'
                    else {'type': 'text', 'text': '[image data removed to save context]'}
                    for b in messages[i]['content']
                ]

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
    def load_agentic_prompt(self, max_turns: int) -> str:
        """Load agentic prompt addition and fill in available libraries."""
        template = (Path(__file__).parent / 'agentic_prompt.md').read_text()
        return template.format(
            libraries = _load_sandbox_libraries(),
            max_turns = max_turns,
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
                            max_turns: int, dest_dir: Path,
                            initial_messages: list | None = None,
                            start_turn: int = 0) -> str:
        """
        Agentic evaluation — model can write and execute Python scripts.
        Images are included in the first message and remain accessible
        throughout the conversation via the full history.
        A persistent workspace is shared across all turns so the agent
        can save and load intermediate files between tool calls.

        If initial_messages / start_turn are provided the loop resumes
        mid-conversation; the session workspace is restored from
        dest_dir/_partial_session/ instead of being freshly populated.
        """
        # Create persistent workspace for this session
        session_dir = Path(tempfile.mkdtemp())

        try:
            partial_session = dest_dir / '_partial_session'

            if initial_messages is not None:
                # Resuming: restore saved workspace into fresh tmpdir
                if partial_session.exists():
                    shutil.copytree(partial_session, session_dir, dirs_exist_ok=True)
            else:
                # Fresh start: copy input files once into session workspace
                shutil.copytree(task.input_dir, session_dir, dirs_exist_ok=True)

            # Providers that support multimodal content in tool result messages.
            if initial_messages is not None:
                messages = initial_messages
            else:
                # Include input files in the first message
                content = [
                    {'type': 'text',
                     'text': task.get_prompt() + self.load_agentic_prompt(max_turns)},
                    *task.get_input_files(embed_data=False)
                ]
                messages = [{'role': 'user', 'content': content}]

            for turn in range(start_turn, max_turns):
                self._apply_cache_breakpoint(messages, model)
                try:
                    try:
                        response = self._litellm_completion_with_retry(
                            model       = model,
                            messages    = messages,
                            tools       = TOOLS,
                            temperature = 0.0,
                        )
                    except litellm.ContextWindowExceededError:
                        print("⚠  Context window exceeded — stripping old images and retrying")
                        self._strip_old_images(messages)
                        response = self._litellm_completion_with_retry(
                            model       = model,
                            messages    = messages,
                            tools       = TOOLS,
                            temperature = 0.0,
                        )
                except Exception:
                    # Save partial state so the run can be resumed with --continue-run
                    partial_path = dest_dir / '_partial_model_response.json'
                    with open(partial_path, 'w') as f:
                        json.dump({'start_turn': turn, 'messages': messages}, f, indent=2)
                    if partial_session.exists():
                        shutil.rmtree(partial_session)
                    shutil.copytree(session_dir, partial_session)
                    print(f"⚠  Partial state saved (turn {turn + 1}/{max_turns}) — resume with --continue-run")
                    raise

                if not response.choices:
                    raise RuntimeError(
                        f"Model {model} returned an empty response (no choices). "
                    )
                message = response.choices[0].message
                self._print_thinking(message.model_dump())
                #self._print_cache_usage(response)
                print(response.usage)

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
                    self._apply_cache_breakpoint(messages, model)
                    summary = self._litellm_completion_with_retry(
                        model       = model,
                        messages    = messages,
                        tools       = TOOLS,
                        tool_choice = 'none',
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
                            'Only the following message is what is is shown to the user.'
                        )
                    }
                ]
            })

            summary = self._litellm_completion_with_retry(
                model       = model,
                messages    = messages,
                tools       = TOOLS,
                tool_choice = 'none',
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
                dimension        = mr_data.get('dimension', ''),
                total           = len(rubrics),
                passed          = passed,
                weight          = mr_data['weight']
            ))
            raw_data.append({
                'name':     mr_data['name'],
                'dimension': mr_data.get('dimension', ''),
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
    # Batch rubrics in one metarubric and split into 
    # chunks if too many for the judge to handle reliably
    # ─────────────────────────────────────────

    def _judge_batch(self, rubrics: list[str],
                     model_output: str,
                     judge: str) -> tuple[int, list[dict]]:
        """
        Rubrics in chunked calls — for capable API models.
        Returns (passed count, [{criterion, verdict}, ...]) where verdict is 'YES' or 'NO'.
        """
        JUDGE_CHUNK_SIZE = 50  # max rubrics per API call to avoid judge miscounts

        all_verdicts: list[dict] = []
        for start in range(0, len(rubrics), JUDGE_CHUNK_SIZE):
            chunk = rubrics[start:start + JUDGE_CHUNK_SIZE]
            all_verdicts.extend(self._judge_chunk(chunk, model_output, judge))
        return sum(1 for v in all_verdicts if v['verdict'] == 'YES'), all_verdicts

    def _judge_chunk(self, rubrics: list[str],
                     model_output: str,
                     judge: str) -> list[dict]:
        """One API call for a chunk of rubrics. Returns [{criterion, verdict}, ...]."""
        numbered = '\n'.join(f"{i+1}. {r}" for i, r in enumerate(rubrics))
        prompt   = self._load_judge_prompt(model_output=model_output, criteria=numbered)

        response = self._litellm_completion_with_retry(
            model       = judge,
            messages    = [{'role': 'user', 'content': prompt}],
            temperature = 0.0
        )
        raw = response.choices[0].message.content or ''
        raw = raw.strip()

        # Strip markdown if present
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*',     '', raw)
        raw = raw.strip()

        try:
            verdicts = json.loads(raw)
        except json.JSONDecodeError:
            # Judge may have prefixed the JSON array with explanatory text — extract it.
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                verdicts = json.loads(m.group())
            else:
                print(f"⚠  Judge response is not valid JSON. Raw response:\n{raw!r}")
                raise

        if len(verdicts) != len(rubrics):
            rubric_lines  = '\n'.join(f"  {i+1}. {r}" for i, r in enumerate(rubrics))
            verdict_lines = '\n'.join(f"  {i+1}. {v}" for i, v in enumerate(verdicts))
            raise ValueError(
                f"Judge returned {len(verdicts)} verdicts for {len(rubrics)} rubrics\n"
                f"RUBRICS:\n{rubric_lines}\n"
                f"VERDICTS:\n{verdict_lines}"
            )

        return [
            {'criterion': r, 'verdict': 'YES' if v else 'NO'}
            for r, v in zip(rubrics, verdicts)
        ]

    # ─────────────────────────────────────────
    # Shared Docker sandbox runner
    # ─────────────────────────────────────────
    SANDBOX_TIMEOUT = 120  # seconds

    def _run_in_sandbox(self, command, session_dir: Path, cleanup: bool = False) -> str:
        try:
            client = docker.from_env()
            client.ping()
        except docker.errors.DockerException:
            raise RuntimeError(
                "⚠  Docker daemon is not running. "
                "Start Docker Desktop and try again."
            )
        container = None
        try:
            container = client.containers.run(
                image         = 'benchmark-sandbox',
                command       = command,
                working_dir   = '/home/agent/workspace',
                user          = 'agent',
                network_mode  = 'none',
                mem_limit     = '512m',
                memswap_limit = '512m',
                cpu_quota     = 50000,
                pids_limit    = 64,
                volumes       = {str(session_dir): {'bind': '/home/agent/workspace', 'mode': 'rw'}},
                tmpfs         = {'/tmp': ''},
                detach        = True,
                stdout        = True,
                stderr        = True,
            )
            timed_out = False
            try:
                container.wait(timeout=self.SANDBOX_TIMEOUT)
            except Exception as e:
                if 'ReadTimeout' in type(e).__name__:
                    container.stop()
                    timed_out = True
                else:
                    raise
            output = container.logs(stdout=True, stderr=True).decode()
            if timed_out:
                output += f"\n[killed: exceeded {self.SANDBOX_TIMEOUT}s time limit]"
            return output or "(no output)"
        except Exception as e:
            return f"Execution error: {e}"
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            if cleanup:
                shutil.rmtree(session_dir, ignore_errors=True)

    # ─────────────────────────────────────────
    # Run shell command in Docker sandbox
    # ─────────────────────────────────────────
    _ALLOWED_COMMANDS = {
        'grep', 'sed', 'awk', 'find', 'head', 'tail',
        'cat', 'wc', 'sort', 'uniq', 'cut', 'ls', 'file',
        'mkdir', 'touch', 'cp', 'cd',
    }

    def _run_command(self, command: str, session_dir: Path, max_chars: int = 5_000) -> str:
        for segment in re.split(r'\|\||&&|[|;]', command):
            tokens = segment.strip().split()
            if not tokens:
                continue
            if tokens[0] not in self._ALLOWED_COMMANDS:
                allowed = ', '.join(sorted(self._ALLOWED_COMMANDS))
                return f"Error: '{tokens[0]}' is not allowed. Allowed commands: {allowed}."
        result = self._run_in_sandbox(['sh', '-c', command], session_dir)
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
        """Execute model-generated Python code in an isolated Docker container.
        session_dir is mounted read-write so files persist between turns."""
        (session_dir / 'script.py').write_text(code)
        return self._run_in_sandbox('python /home/agent/workspace/script.py', session_dir)