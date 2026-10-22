import os
from pathlib import Path


def _load_env_file():
    """Read KEY=value lines from .env in the project folder (works the same on Windows and Mac)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.split(" #")[0].strip())


_load_env_file()
