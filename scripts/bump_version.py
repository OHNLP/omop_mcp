"""Bump version in pyproject.toml based on semantic versioning."""

import re
import sys
from pathlib import Path


def get_current_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Could not find version in pyproject.toml")


def bump_version(current_version, bump_type):
    """Bump version based on type (major, minor, patch)."""
    parts = current_version.split(".")
    major, minor, patch = map(int, parts)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

    return f"{major}.{minor}.{patch}"


def update_version(new_version):
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    content = re.sub(r'version\s*=\s*"[^"]+"', f'version = "{new_version}"', content)
    pyproject_path.write_text(content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: bump_version.py [major|minor|patch]")
        sys.exit(1)

    bump_type = sys.argv[1]
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    update_version(new_version)
    print(f"Bumped version from {current_version} to {new_version}")
