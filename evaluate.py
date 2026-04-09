import argparse
import importlib
import inspect
from pathlib import Path
from src.task import Task
#from src.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LLMs on physics benchmark tasks.'
    )
    parser.add_argument('--task',          type=str,
                        help='Specific task folder name. Omit to run all.')
    parser.add_argument('--model',         type=str,
                        default='ollama/llama3.2')
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
    
    # Select tasks to run
    if args.task:
        if args.task not in all_tasks:
            print(f"✗ Task '{args.task}' not found.")
            print(f"  Available: {list(all_tasks.keys())}")
            return
        task_names = [args.task]
    else:
        task_names = list(all_tasks.keys())
    
    # evaluator = Evaluator()
    
    for task_name in task_names:
        task_cls = all_tasks[task_name]
        
        for seed in args.seeds:
            print(f"\n{'─'*50}")
            print(f"Task:       {task_name}")
            print(f"Difficulty: {args.difficulty}")
            print(f"Seed:       {seed}")
            
            # ── Setup phase ───────────────────────────────
            task = task_cls(
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
                print(f"✓ Validated — skipping model evaluation")
                continue
            
            # ── Evaluation phase ──────────────────────────
            print(f"Model:      {args.model}")
            print(f"Judge:      {args.judge}")
            
    #         results = evaluator.run(
    #             task  = task,
    #             model = args.model,
    #             judge = args.judge
    #         )
            
    #         # Print summary
    #         print(f"\nResults:")
    #         for r in results:
    #             print(f"  {r}")


if __name__ == '__main__':
    main()