'''
Pydantic settings model for the forecast pipeline, populated from "pipeline_settings.yaml" and
doubling as the "pipeline_settings" object passed into each Pipeline instance.

The module "PipelineSettings" adds:
    - Declarative configuration for which pipeline steps to run, which SKUs/targets to process.
    - Convenience expansion of shorthand pipeline steps ("ALL", "ETL") into their concrete step names.
    - Resolution of "files_path" to an absolute "Path", defaulting to "general_settings.base_path".

Note:
    "pipeline_settings" below is a module-level singleton constructed at import time by reading
    "{general_settings.base_path}/config/pipeline_settings.yaml". Importing this module therefore immediately
    reads and parses that file, raising "FileNotFoundError" if it's missing or a pydantic
    "ValidationError" if its contents are invalid.
'''

import yaml
import pandas as pd
import datetime as dt
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings
from src.settings.general import general_settings


class PipelineSettings(BaseSettings):
    """
    Full configuration for a single pipeline run: which steps to execute, 
    which SKUs/targets to process

    Fields:
        - pipeline_steps: List of step names to run (see "Pipeline.run"). May include the shorthands
          "ALL" and "ETL", which are expanded into their concrete steps by "treat_pipeline_steps".
    
        - valid_targets: Allowed forecast targets.
          Default = ("QT", "ASP").
    
        - targets: Forecast targets to actually process in this run. Must be a subset of
          "valid_targets" (enforced by "validate_targets").
          Defaults = all of "valid_targets".
    
        - start_date / end_date: Optional "YearMonth" (format = %Y/%m) bounds used to restrict the data used during
          featurization (forwarded as "ETL.featurize"'s "date_limits"). None leaves that bound unrestricted.
    
        - raw_table_name: Name of the source table/query used during extraction.

        - skus: SKUs to process.
    """
    
    pipeline_steps: list
    valid_targets: tuple = ('QT', 'ASP')
    targets: list = list(valid_targets)
    skus: list
    start_year_month: str | None = None
    end_year_month: str | None = None
    start_date: dt.date | None = None
    end_date: dt.date | None = None
    raw_table_name: str | None = None
    files_path: str | Path | None = None

    @model_validator(mode='after')
    def validate_targets(self):
        """ 
        Ensure "targets" only contains values present in "valid_targets".

        Raises:
            - ValueError: If "targets" is not a subset of "valid_targets".
        """

        if not set(self.targets).issubset(self.valid_targets):
            raise ValueError(f'Target list {self.targets} is not valid')
        return self

    @model_validator(mode='after')
    def treat_pipeline_steps(self):
        """
        Expand shorthand entries in "pipeline_steps" into their concrete step names.
            - "ALL" expands (via append) to "ETL", "HYPERPARAMETERIZE", "FIT", "FORECAST". Since "ETL" is
              then itself checked for presence within this same call, a user-supplied or "ALL"-added.

            - "ETL" is further expanded to "EXTRACT", "CLEAN", "FEATURIZE" — so "ALL" alone is enough to
              enable every step.

        Note:
            Expansion appends to "pipeline_steps" rather than replacing it, and does not deduplicate.
            Supplying "ALL" (or "ETL") alongside its own already-expanded step names, or more than
            once, results in duplicate entries. This is harmless for "Pipeline.run" (which only
            checks membership via "in"), but worth knowing if "pipeline_steps" is inspected directly.
        """

        if 'ALL' in self.pipeline_steps:
            self.pipeline_steps += ['ETL', 'HYPERPARAMETERIZE', 'FIT', 'FORECAST']

        if 'ETL' in self.pipeline_steps:
            self.pipeline_steps += ['EXTRACT', 'CLEAN', 'FEATURIZE']
        return self

    @model_validator(mode='after')
    def treat_files_path(self):
        """
        Resolve "files_path" to a concrete "Path", defaulting to "general_settings.base_path".

        Note:
          If "files_path" is already a "Path" instance, it is left untouched — only "None" and
          "str" values are handled/converted.

        """

        if self.files_path is None:
            self.files_path = general_settings.base_path
            return self

        if isinstance(self.files_path, str):
            self.files_path = Path(self.files_path)
        return self

    @model_validator(mode='after')
    def treat_dates(self):
        """ Resolve date strings to concrete "date" instances. """

        if self.start_year_month is not None:
            self.start_date = pd.to_datetime(self.start_year_month)

        if self.end_year_month is not None:
            self.end_date = pd.to_datetime(self.end_year_month)

        return self


with open(general_settings.base_path / 'config' / 'pipeline_settings.yaml', 'r') as file:
    settings_data = yaml.safe_load(file)

# Module-level singleton, see module header Note.
pipeline_settings = PipelineSettings.model_validate(settings_data)


if __name__ == '__main__':
    print(pipeline_settings)