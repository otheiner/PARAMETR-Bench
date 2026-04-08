import shutil
import argparse
from pathlib import Path
import re

def create_task(name: str, description: str):
    task_dir = Path('tasks') / name.replace(' ', '_')
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Essential for autodiscovery
    (task_dir / '__init__.py').touch()
    
    # Copy template files
    template_dir = Path('template')
    for f in template_dir.iterdir():
        if f.is_file():
            shutil.copy(f, task_dir / f.name)
    
    # Generate CamelCase class name from folder name
    clean      = re.sub(r'^[\d_\s]+', '', name)  # strip leading digits, underscores, spaces
    class_name = ''.join(
        word.capitalize() 
        for word in re.split(r'[\s_]+', clean)   # split on spaces OR underscores
        if word                                   # skip empty strings
    )
    
    # Replace placeholders in generate.py
    generate_py = task_dir / 'generate.py'
    content     = generate_py.read_text()
    content     = content.replace('TaskName',          class_name)
    content     = content.replace('[TASK NAME]',       name)
    content     = content.replace('[YOUR NAME]',       args.author)
    content     = content.replace('[BRIEF DESCRIPTION]', description)
    generate_py.write_text(content)
    
    print(f"✓ Created task: tasks/{name}/")
    print(f"  Class name: {class_name}Task")
    print(f"  Next steps:")
    print(f"  1. Implement generate_task() in tasks/{name}/generate.py")
    print(f"  2. Fill in metarubrics.json templates")
    print(f"  3. Fill in config.json difficulty parameters")
    print(f"  4. Implement populate_rubrics() in tasks/{name}/generate.py")
    print(f"  5. Run: python evaluate.py --task {name} --validate-only")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--author', default='<FILL-THIS-IN>')
    parser.add_argument('--description', default='<FILL-THIS-IN>')
    args = parser.parse_args()
    create_task(args.name, args.description)