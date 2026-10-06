#!/usr/bin/env python3
"""Generic source-tree entrypoint for every OFDM experiment command."""
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from experiment_cli import main as package_main  # noqa: E402


def run(arguments=None):
    """Run the package CLI, optionally with an explicit argument list."""
    if arguments is None:
        package_main()
        return
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).name), *map(str, arguments)]
        package_main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    run()
