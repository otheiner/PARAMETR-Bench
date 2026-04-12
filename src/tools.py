from pathlib import Path

# Exctract allowed libraries from requirements.txt
def _load_sandbox_libraries() -> str:
    """Extract package names from sandbox/requirements.txt."""
    req_path = Path(__file__).parent.parent / 'sandbox' / 'requirements.txt'
    
    libraries = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            # Strip version pin — numpy==1.26.4 → numpy
            package = line.split('==')[0].split('>=')[0].split('<=')[0]
            libraries.append(package)
    
    return ', '.join(libraries)

# Addition to the task prompt when using agentic evaluation
def load_agentic_prompt(max_turns: int) -> str:
    """Load agentic prompt addition and fill in available libraries."""
    template = (Path(__file__).parent / 'agentic_prompt.md').read_text()
    return template.format(libraries=_load_sandbox_libraries(),
                           max_turns = max_turns)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute a Python script to analyse the task data. "
                "Input files are available in the current directory. "
                "Available libraries: {_load_sandbox_libraries()}. "
                "Print your results — stdout is returned to you."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python script to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }
]