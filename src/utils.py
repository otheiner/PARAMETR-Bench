import subprocess

def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode().strip()
    except FileNotFoundError:
        print("⚠  git not found — commit hash unavailable")
        return 'unknown'
    except subprocess.CalledProcessError:
        print("⚠  git command failed — are you in a git repository?")
        return 'unknown'

def is_working_tree_dirty() -> bool:
    """Return True if there are any uncommitted changes (staged or unstaged)."""
    try:
        subprocess.check_call(
            ['git', 'diff', 'HEAD', '--quiet'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return False
    except subprocess.CalledProcessError:
        return True
    except FileNotFoundError:
        return False