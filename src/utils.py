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