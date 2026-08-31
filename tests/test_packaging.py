import re
import tomllib
from pathlib import Path


def test_conda_recipe_version_matches_project_metadata() -> None:
    """Keep all Pixi package outputs version-aligned with the Python package."""
    project_root = Path(__file__).parents[1]
    project = tomllib.loads((project_root / "pyproject.toml").read_text())
    recipe = (project_root / "recipe" / "recipe.yaml").read_text()

    recipe_version = re.search(r'^  version: "([^"]+)"$', recipe, re.MULTILINE)
    assert recipe_version is not None
    assert recipe_version.group(1) == project["project"]["version"]
