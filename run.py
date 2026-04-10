import argparse
import importlib
import inspect
from datetime import datetime
from pathlib import Path
from src.task import Task, BenchmarkResults
from src.evaluator import Evaluator
from src.utils import get_git_hash


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LLMs on physics benchmark tasks.'
    )
    parser.add_argument('--task',          type=str,
                        help='Specific task folder names. Omit to run all.')
    parser.add_argument('--models',        type=str, nargs='+',
                        default=['ollama/llama3.2'])
    parser.add_argument('--judge',         type=str,
                        default='ollama/llama3.2')
    parser.add_argument('--difficulty',    type=str,
                        default='medium',
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--seeds',         type=int, nargs='+',
                        default=[0])
    parser.add_argument('--validate-only', action='store_true',
                        help='Setup only — no model calls.')
    parser.add_argument('--list',          action='store_true',
                        help='List discovered tasks and exit.')
    return parser.parse_args()


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
        task_class = all_tasks[task_name]
        
        for seed in args.seeds:
            print(f"\n{'─'*50}")
            print(f"Task:       {task_name}")
            print(f"Difficulty: {args.difficulty}")
            print(f"Seed:       {seed}")
            
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
                print(f"✓ Validated {task_name} — skipping model evaluation")
                continue
            
            for tested_model in args.models:
                # ── Evaluation phase ──────────────────────────
                print(f"Model:      {tested_model}")
                print(f"Judge:      {args.judge}")
                
                result = evaluator.run(
                    task  = task,
                    model = tested_model,
                    judge = args.judge
                )

                print(result)
                result.save()
                benchmark.task_results.append(result)

    # ── Final summary ─────────────────────────────────────
    if not args.validate_only:
        print(f"\n{'═' * 50}")
        print(benchmark)
        benchmark.save()


if __name__ == '__main__':
    main()