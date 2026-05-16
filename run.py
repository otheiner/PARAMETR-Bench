# Load API keys from .env file
from dotenv import load_dotenv
load_dotenv()

import argparse
import importlib
import inspect
import random
import requests
import shutil
import string
import sys
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path
from src.task import Task, TaskResults, BenchmarkResults
from src.evaluator import Evaluator
from src.utils import get_git_hash

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LLMs on physics benchmark tasks.'
    )
    parser.add_argument('--task',          type = str,
                        help='Specific task folder names. Omit to run all.')
    parser.add_argument('--model',         type = str,
                        default='ollama/llama3.2',
                        help  = 'Model to evaluate.')
    parser.add_argument('--judge',         type = str,
                        default='ollama/llama3.2',
                        help  = 'Model used as judge.')
    parser.add_argument('--difficulty',    type = str,
                        default='easy',
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--seeds',         type = int, nargs = '+',
                        default=[0],
                        help  = 'Set of seeds used in evaluation.')
    parser.add_argument('--validate-only', action = 'store_true',
                        help='Setup only — no model calls.')
    parser.add_argument('--list',          action = 'store_true',
                        help='List discovered tasks and exit.')
    parser.add_argument('--agentic',       action = 'store_true',
                        default = False,
                        help    = 'Enable agentic evaluation with sandboxed Python execution.'
                                  'Requires Docker and benchmark-sandbox image.')
    parser.add_argument('--max-turns',     type = int,
                        default = 10,
                        help    = 'Maximum agentic turns per evaluation (default: 10).')
    parser.add_argument('--continue-run',  type = str,
                        default = None,
                        metavar = 'RUN_ID',
                        help    = 'Continue an existing run by its 6-character ID.')

    return parser.parse_args()


def generate_run_id(results_dir: Path) -> str:
    """Generate a unique 6-character alphanumeric run ID."""
    existing = {p.name for p in results_dir.iterdir() if p.is_dir()} \
               if results_dir.exists() else set()
    while True:
        rid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        if rid not in existing:
            return rid


def check_ollama_if_needed(model: str, judge: str):
    """Check Ollama server is reachable if the model or judge is an Ollama model."""
    needs_ollama = model.startswith('ollama/') or judge.startswith('ollama/')
    if not needs_ollama:
        return

    try:
        requests.get('http://localhost:11434', timeout=3)

    except Exception:
        print("✗ Could not reach Ollama server.")
        print("  Please check that Ollama is running: ollama serve")
        raise SystemExit(1)


def discover_tasks(tasks_dir='tasks') -> dict:
    """Autodiscover all Task subclasses in tasks/ folder."""
    discovered = {}

    for task_folder in sorted(Path(tasks_dir).iterdir()):
        if not task_folder.is_dir():
            continue
        if not (task_folder / 'generate.py').exists():
            continue

        module_path = f"tasks.{task_folder.name}.generate"
        module      = importlib.import_module(module_path)

        task_classes = [
            obj for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, Task)
            and obj is not Task
            and obj.__module__ == module_path
        ]

        if task_classes:
            discovered[task_folder.name] = task_classes[0]
            print(f"  ✓ Discovered: {task_folder.name} → {task_classes[0].__name__}")

    return discovered


def _aggregate_results(run_dir: Path, model: str, judge: str,
                        difficulty: str, seeds: list[int],
                        task_names: list[str]) -> BenchmarkResults:
    """Load all task_results.json files and produce a BenchmarkResults."""
    task_results = []
    incomplete   = []

    model_clean = model.replace('/', '-').replace(':', '-')
    for task_name in task_names:
        for seed in seeds:
            dest_dir          = run_dir / model_clean / task_name / str(seed)
            task_results_path = dest_dir / 'task_results.json'
            if task_results_path.exists():
                with open(task_results_path) as f:
                    task_results.append(TaskResults.from_dict(json.load(f)))
            else:
                incomplete.append({
                    'task': task_name, 'seed': seed
                })

    is_partial = bool(incomplete)
    if incomplete:
        incomplete_path = run_dir / 'incomplete_tasks.json'
        with open(incomplete_path, 'w') as f:
            json.dump({'incomplete': incomplete}, f, indent=2)
        print(f"\n⚠  {len(incomplete)} task(s) incomplete — "
              f"re-run with --continue-run to retry: {incomplete_path}")

    return BenchmarkResults(
        task_results = task_results,
        model        = model,
        judge        = judge,
        difficulty   = difficulty,
        seeds        = seeds,
        git_commit   = get_git_hash(),
        timestamp    = datetime.now().isoformat(),
        partial      = is_partial,
    )


def main():
    args       = parse_args()
    results_dir = Path('results')

    # ── Discover tasks ────────────────────────────────────────
    print("\nDiscovering tasks...")
    all_tasks = discover_tasks()

    if args.list:
        print(f"\nAvailable tasks ({len(all_tasks)}):")
        for name in all_tasks:
            print(f"  {name}")
        return

    # ── Determine run folder and parameters ───────────────────
    if args.continue_run:
        run_id  = args.continue_run
        run_dir = results_dir / run_id
        if not run_dir.exists():
            print(f"✗ Run '{run_id}' not found in {results_dir}/")
            sys.exit(1)
        with open(run_dir / 'run_params.json') as f:
            params = json.load(f)
        if get_git_hash() != params.get('git_commit', ''):
            print(f"✗ Repo has changed since run '{run_id}' was started.")
            print(f"  Checkout the original commit before continuing:")
            print(f"    git checkout {params.get('git_commit')}")
            sys.exit(1)
        model      = params['model']
        judge      = params['judge']
        difficulty = params['difficulty']
        seeds      = params['seeds']
        agentic    = params['agentic']
        max_turns  = params['max_turns']
        task_names = params['tasks']
        print(f"✓ Continuing run: {run_id}  ({run_dir})")
    else:
        run_id  = generate_run_id(results_dir)
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True)
        model      = args.model
        judge      = args.judge
        difficulty = args.difficulty
        seeds      = args.seeds
        agentic    = args.agentic
        max_turns  = args.max_turns

        if args.task:
            if args.task not in all_tasks:
                print(f"✗ Task '{args.task}' not found.")
                print(f"  Available: {list(all_tasks.keys())}")
                return
            task_names = [args.task]
        else:
            task_names = [n for n in all_tasks if not n.startswith('_')]

        if not args.validate_only:
            run_params = {
                'run_id':     run_id,
                'model':      model,
                'judge':      judge,
                'difficulty': difficulty,
                'seeds':      seeds,
                'agentic':    agentic,
                'max_turns':  max_turns,
                'tasks':      task_names,
                'git_commit': get_git_hash(),
                'timestamp':  datetime.now().isoformat(),
            }
            with open(run_dir / 'run_params.json', 'w') as f:
                json.dump(run_params, f, indent=2)
            print(f"✓ New run: {run_id}  ({run_dir})")

    if not args.validate_only:
        check_ollama_if_needed(model, judge)

        if agentic:
            try:
                import docker
                docker.from_env().ping()
            except Exception:
                print("✗ Docker daemon is not running. Start Docker Desktop and try again.")
                sys.exit(1)

        evaluator = Evaluator()

    failures: list[dict] = []

    # ── Main loop ─────────────────────────────────────────────
    for task_name in task_names:
        if task_name.startswith('_') and (not args.continue_run and args.task != task_name):
            print(f"✗ Skipping {task_name} — use --task {task_name} to run explicitly")
            continue

        task_class = all_tasks[task_name]

        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"TASK: {task_name}  |  SEED: {seed}")
            print(f"{'='*50}")

            task = task_class(
                task_folder = f'tasks/{task_name}',
                difficulty  = difficulty,
                seed        = seed
            )

            def _dest(m):
                return run_dir / m.replace('/', '-').replace(':', '-') / task_name / str(seed)

            skip_generation = not args.validate_only and (
                (_dest(model) / 'model_response.json').exists() and
                (_dest(model) / 'rubrics.json').exists()
            )

            if skip_generation:
                print(f"↩  Skipping task generation — model already produced response for seed {seed} and rubrics exist")
            else:
                task.generate_task()
                task.save_ground_truth()
                task.populate_metarubrics()
                task.validate_metarubrics()
                task.generate_rubrics()

            # ── Validate-only path ────────────────────────────
            if args.validate_only:
                gt_path  = task.ground_truth_dir / 'ground_truth.json'
                ref_fd, ref_path_str = tempfile.mkstemp(suffix='.json',
                                                        prefix=f'parametr_{task_name}_ref_')
                ref_path = Path(ref_path_str)
                try:
                    print(f"✓ Moving ground_truth.json to temporary file {ref_path}")
                    os.close(ref_fd)
                    shutil.copy(gt_path, ref_path)

                    print(f"✓ Generating {task_name} second time to check seed reproducibility")
                    task.generate_task()
                    task.save_ground_truth()

                    with open(gt_path,  'r') as f: gt  = json.load(f)
                    with open(ref_path, 'r') as f: ref = json.load(f)
                    if gt != ref:
                        print(f"✗ Validation failed: {task_name} — ground truth differs between runs with same seed!")
                        raise ValueError(f"Seed reproducibility check failed for {task_name}\n"
                                         "            Do all random number generators use self.seed, "
                                         "or seeds deterministically derived from it?")
                    else:
                        print(f"✓ Same seeds produces same results - correct")

                    print(f"✓ Generating {task_name} third time to check that different seeds produce different results")
                    task.seed = seed + 1
                    task.generate_task()
                    task.save_ground_truth()

                    with open(gt_path,  'r') as f: gt  = json.load(f)
                    with open(ref_path, 'r') as f: ref = json.load(f)
                    if gt == ref:
                        print(f"✗ Validation failed: {task_name} — different seeds produce same results!")
                        raise ValueError(f"Task is static and doesn't produce different results for different seeds.\n"
                                         "            Do you use stochastic processes in _generate()?, "
                                         "All random generators must use seed self.seed for reproducibility.")
                    else:
                        print(f"✓ Different seeds produce different results - correct")
                finally:
                    ref_path.unlink(missing_ok=True)

                print(f"✓ Validation {task_name} successfully finished — skipping model evaluation\n ")
                continue

            # ── Evaluation ───────────────────────────────────
            model_clean = model.replace('/', '-').replace(':', '-')
            dest_dir    = run_dir / model_clean / task_name / str(seed)

            model_resp_path   = dest_dir / 'model_response.json'
            rubrics_path      = dest_dir / 'rubrics.json'
            task_results_path = dest_dir / 'task_results.json'

            # ── Model step ───────────────────────────────────
            if model_resp_path.exists() and rubrics_path.exists():
                print(f"↩  Skipping model call — response exists")
                with open(model_resp_path) as f:
                    model_output = json.load(f)['messages'][-1].get('content', '')
            else:
                try:
                    model_output = evaluator.get_model_output(
                        task      = task,
                        model     = model,
                        agentic   = agentic,
                        max_turns = max_turns,
                        dest_dir  = dest_dir,
                    )
                except Exception as e:
                    print(f"✗ Model failed [{task_name} / seed {seed} / {model}]: {e}")
                    failures.append({'step': 'problem solving', 'task': task_name, 'seed': seed, 'model': model})
                    continue

            # ── Judge step ───────────────────────────────────
            if task_results_path.exists():
                print(f"↩  Skipping judge call — response exists")
            else:
                try:
                    evaluator.get_judge_results(
                        task         = task,
                        model        = model,
                        model_output = model_output,
                        rubrics_path = rubrics_path,
                        judge        = judge,
                        dest_dir     = dest_dir,
                    )
                except Exception as e:
                    print(f"✗ Judge failed [{task_name} / seed {seed} / {model}]: {e}")
                    failures.append({'step': 'judge', 'task': task_name, 'seed': seed, 'model': model})

    # ── Failure summary ───────────────────────────────────────
    if failures:
        print(f"\n{'=' * 50}")
        print(f"⚠  {len(failures)} task(s) failed:")
        for f in failures:
            print(f"   task: {f['task']}, step: {f['step']}, seed: {f['seed']}")

    # ── Aggregation ───────────────────────────────────────────
    if not args.validate_only:
        benchmark = _aggregate_results(
            run_dir    = run_dir,
            model      = model,
            judge      = judge,
            difficulty = difficulty,
            seeds      = seeds,
            task_names = task_names,
        )
        benchmark.save(run_dir)
        if not failures:
            print(f"\n{'=' * 50}")
            print("BENCHMARK RESULTS")
            print('=' * 50)
            print(benchmark)
            

if __name__ == '__main__':
    main()
