# Load API keys from .env file
from dotenv import load_dotenv
load_dotenv()

import argparse
import importlib
import inspect
import requests
import shutil
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path
from src.task import Task, BenchmarkResults
from src.evaluator import Evaluator
from src.utils import get_git_hash

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LLMs on physics benchmark tasks.'
    )
    parser.add_argument('--task',          type = str,
                        help='Specific task folder names. Omit to run all.')
    parser.add_argument('--models',        type = str, nargs = '+',
                        default=['ollama/llama3.2'],
                        help  = 'Set of models for evaluation.')
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
    
    return parser.parse_args()


def check_ollama_if_needed(models: list[str], judge: str):
    """Check Ollama server is reachable if any model is an Ollama model."""
    needs_ollama = any(
        m.startswith('ollama/') for m in models + [judge]
    )
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


def main():
    args = parse_args()
    
    print("\nDiscovering tasks...")
    all_tasks = discover_tasks()
    
    if args.list:
        print(f"\nAvailable tasks ({len(all_tasks)}):")
        for name in all_tasks:
            print(f"  {name}")
        return
    
    if args.task:
        if args.task not in all_tasks:
            print(f"✗ Task '{args.task}' not found.")
            print(f"  Available: {list(all_tasks.keys())}")
            return
        task_names = [args.task]
    else:
        task_names = list(all_tasks.keys())
    
    if not args.validate_only:
        check_ollama_if_needed(args.models, args.judge)
        evaluator = Evaluator()
        benchmark = BenchmarkResults(
            task_results = [],
            models       = args.models,
            judge        = args.judge,
            difficulty   = args.difficulty,
            seeds        = args.seeds,
            git_commit   = get_git_hash(),
            timestamp    = datetime.now().isoformat()
    )
    
    for task_name in task_names:
        # Exclude tasks starting with '_' from benchmark unless explicitly requested
        # Underscore prefix is a convention to hide tasks from the benchmark.
        if task_name.startswith('_') and args.task != task_name:
            print(f"✗ Skipping {task_name} — use --task {task_name} to run explicitly)")
            continue

        task_class = all_tasks[task_name]
        
        for seed in args.seeds:
            print(f"\n{'='*50}")
            print(f"TASK GENERATION")
            print(f"{'='*50}")
            print(f"Task:       {task_name}")
            
            # ── Setup phase ───────────────────────────────
            task = task_class(
                task_folder = f'tasks/{task_name}',
                difficulty  = args.difficulty,
                seed        = seed
            )

            task.generate_task()  # generate data + ground truth
            task.save_ground_truth() # dump ground truth to ground_truth.json
            task.populate_metarubrics()  # fill in metarubrics based on generated data
            task.validate_metarubrics()  # validate the metarubrics
            task.generate_rubrics()  # generate rubrics based on metarubrics and templates
            
            if args.validate_only:
                # Copy ground truth to system tmp before second generation
                # (second generate_task() clears ground_truth_dir)
                gt_path  = task.ground_truth_dir / 'ground_truth.json'
                ref_fd, ref_path_str = tempfile.mkstemp(suffix='.json', 
                                                        prefix=f'parametr_{task_name}_ref_')
                ref_path = Path(ref_path_str)

                try:
                    # Close the file descriptor — shutil.copy will write to it
                    print(f"✓ Moving ground_truth.json to temporary file {ref_path}")
                    os.close(ref_fd)
                    shutil.copy(gt_path, ref_path)

                    # Generate second time with same seed — clears ground_truth_dir
                    print(f"✓ Generating {task_name} second time to check seed " 
                          "reproducibility")
                    task.generate_task()
                    task.save_ground_truth()

                    # Compare
                    with open(gt_path,  'r') as f: gt  = json.load(f)
                    with open(ref_path, 'r') as f: ref = json.load(f)

                    if gt != ref:
                        print(f"✗ Validation failed: {task_name} — ground truth differs between runs with same seed!")
                        raise ValueError(f"Seed reproducibility check failed for {task_name}\n"
                                         "            Do all random number generators use self.seed, " \
                                         "or seeds deterministically derived from it?")
                    else:
                        print(f"✓ Same seeds produces same results - correct")


                    # Generate third time with different seed — clears ground_truth_dir
                    print(f"✓ Generating {task_name} third time to check that different seeds " 
                          "produce different results")
                    task.seed = seed + 1
                    task.generate_task()
                    task.save_ground_truth()

                    # Compare
                    with open(gt_path,  'r') as f: gt  = json.load(f)
                    with open(ref_path, 'r') as f: ref = json.load(f)

                    if gt == ref:
                        print(f"✗ Validation failed: {task_name} — different seeds produce same results!")
                        raise ValueError(f"Task is static and doesn't produce different results for different seeds.\n"
                                         "            Do you use stochastic processes in _generate()?, " \
                                         "All random generators must use seed self.seed for reproducibility.")
                    else:
                        print(f"✓ Different seeds produce different results - correct")

                finally:
                    # Always clean up tmp file
                    ref_path.unlink(missing_ok=True)

                print(f"✓ Validation {task_name} successfully finished — skipping model evaluation\n ")
                continue
            
            for tested_model in args.models:
                # ── Evaluation phase ──────────────────────────
                # print(f"Model:      {tested_model}")
                # print(f"Judge:      {args.judge}")
                
                result = evaluator.run(
                    task  = task,
                    model = tested_model,
                    judge = args.judge,
                    agentic = args.agentic,
                    max_turns = args.max_turns 
                )

                #print(result)
                result.save()
                benchmark.task_results.append(result)

    # ── Final summary ─────────────────────────────────────
    if not args.validate_only:
        print(f"\n{'=' * 50}")
        print("BENCHMARK RESULTS")
        print('=' * 50)
        print(benchmark)
        benchmark.save()


if __name__ == '__main__':
    main()