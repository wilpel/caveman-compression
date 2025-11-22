import os
import sys
from pathlib import Path

def find_project_root(marker_files=['.git', 'pyproject.toml', 'README.md']):
    """Find the project root by searching for marker files in parent directories."""
    current_path = Path.cwd()
    for _ in range(len(current_path.parts)):
        for marker in marker_files:
            if (current_path / marker).exists():
                return current_path
        if current_path.parent == current_path:
            break
        current_path = current_path.parent
    # Fallback to the directory of the script that is running.
    return Path(sys.argv[0]).parent.absolute()

def load_api_key():
    """Load OpenAI API key from environment or .env file in the project root."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return api_key

    project_root = find_project_root()
    env_path = project_root / '.env'

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith('OPENAI_API_KEY='):
                api_key = line.split('=', 1)[1].strip().strip('"')
                if api_key:
                    return api_key

    print("Error: OPENAI_API_KEY not found. Set it as an environment variable or in a .env file in the project root.", file=sys.stderr)
    sys.exit(1)
