# Pydantic settings model holding general, cross-cutting configuration shared across the project.

# The module "GeneralSettings" adds:
# - The project's base working directory ("base_path"), used by other settings modules (e.g.
#   "PipelineSettings") to locate config files (like "model_settings.yaml") and to resolve a default
#   "output_path" when none is explicitly configured.

# The typical workflow is:
#   >>> from src.settings.general import general_settings
#   >>> general_settings.base_path

# Note:
#   As a "pydantic_settings.BaseSettings" subclass with no "model_config" override, "base_path" can
#   also be supplied via a matching environment variable (case-insensitive, e.g. "BASE_PATH"); it
#   only falls back to "Path.cwd()" when no such variable is set.

from pathlib import Path
from pydantic_settings import BaseSettings


class GeneralSettings(BaseSettings):
    # General, project-wide settings shared by other settings modules.

    # Fields:
    #   - base_path: Root directory for the project. Defaults to the current working directory at
    #     import time ("Path.cwd()") unless overridden via the environment.

    base_path: Path = Path.cwd()


# Module-level singleton.
general_settings = GeneralSettings()