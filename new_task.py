import shutil
import argparse
from pathlib import Path
import re

def create_task(name: str, author: str, description: str):
    task_dir = Path('tasks') / name.replace(' ', '_')
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Essential for autodiscovery
    (task_dir / '__init__.py').touch()
    
    # Copy template files
    template_dir = Path('template')
    for f in template_dir.iterdir():
        dest = task_dir / f.name
        if f.is_file():
            shutil.copy(f, dest)
        elif f.is_dir():
            shutil.copytree(f, dest, dirs_exist_ok=True)
    
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
    content     = content.replace('[YOUR NAME]',       author)
    content     = content.replace('[BRIEF DESCRIPTION]', description)
    generate_py.write_text(content)

    # Replace placeholders in config.json
    config_json = task_dir / 'config.json'
    content       = config_json.read_text()
    content       = content.replace('[TASK NAME]', name)
    config_json.write_text(content)
    
    print(f"✓ Created task: tasks/{name}/")
    print(f"  Class name: {class_name}Task")
    print(f"  Next steps:")
    print(f"  1. Implement generate_task() in tasks/{name}/generate.py")
    print(f"  2. Fill in metarubrics.json templates")
    print(f"  3. Fill in config.json difficulty parameters")
    print(f"  4. Run: python run.py --task {name} --validate-only")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--author', default='<FILL-THIS-IN>')
    parser.add_argument('--description', default='<FILL-THIS-IN>')
    args = parser.parse_args()
    create_task(args.name, args.author, args.description)