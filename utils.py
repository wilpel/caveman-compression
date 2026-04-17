import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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


def load_env_value(key):
    """Load an env value from the environment first, then the project .env file."""
    value = os.environ.get(key)
    if value:
        return value

    project_root = find_project_root()
    env_path = project_root / '.env'

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith(f'{key}='):
                value = line.split('=', 1)[1].strip().strip('"')
                if value:
                    return value

    return None


def load_base_url():
    """Load an optional OpenAI-compatible base URL."""
    return load_env_value('OPENAI_BASE_URL')


def get_openai_client():
    """Build an OpenAI client, optionally targeting an OpenAI-compatible endpoint."""
    if OpenAI is None:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client_kwargs = {'api_key': load_api_key()}
    base_url = load_base_url()
    if base_url:
        client_kwargs['base_url'] = base_url

    return OpenAI(**client_kwargs)
