"""
Pydantic settings model holding general, cross-cutting configuration shared across the project.

The module "GeneralSettings" adds:
    - The project's base working directory ("base_path"), used by other settings modules (e.g.
      "PipelineSettings") to locate config files (like "pipeline_settings.yaml") and to resolve a default
      "output_path" when none is explicitly configured.

Note:
    As a "pydantic_settings.BaseSettings" subclass with no "model_config" override, "base_path" can
    also be supplied via a matching environment variable (case-insensitive, e.g. "BASE_PATH"); it
    only falls back to "Path.cwd()" when no such variable is set.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class GeneralSettings(BaseSettings):
    """
    Base configuration for the project pipeline.
    """

    base_path: Path = Path.cwd()

general_settings = GeneralSettings()