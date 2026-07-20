# Pydantic settings model for the forecast pipeline, populated from "pipeline_settings.yaml" and
# doubling as the "model_settings" object passed into each "ForecastModel" instance.

# The module "PipelineSettings" adds:
# - Declarative configuration for which pipeline steps to run, which SKUs/targets to process, and
#   all Prophet/Optuna hyperparameter-search and forecast-horizon defaults.
# - Convenience expansion of shorthand pipeline steps ("ALL", "ETL") into their concrete step names.
# - Resolution of "output_path" to an absolute "Path", defaulting to "general_settings.base_path".

# The typical workflow is:
#   >>> from src.settings.pipeline import pipeline_settings
#   >>> pipeline: Pipeline = Pipeline(
#   ...     base_path=general_settings.base_path, database_settings=database_settings,
#   ...     pipeline_settings=pipeline_settings
#   ... )
#   >>> pipeline.run()

# Note:
#   "pipeline_settings" below is a module-level singleton constructed at import time by reading
#   "{general_settings.base_path}/pipeline_settings.yaml". Importing this module therefore immediately
#   reads and parses that file, raising "FileNotFoundError" if it's missing or a pydantic
#   "ValidationError" if its contents are invalid.

import yaml
import pandas as pd
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings
from src.settings.general import general_settings


class PipelineSettings(BaseSettings):
    # Full configuration for a single pipeline run: which steps to execute, which SKUs/targets to
    # process, and the hyperparameter-search/forecast defaults forwarded to each "ForecastModel".

    # Fields:
    #   - model_name: Name used to namespace persisted parameters/models/forecasts on disk.
    #   - pipeline_steps: List of step names to run (see "Pipeline.run"). May include the shorthands
    #     "ALL" and "ETL", which are expanded into their concrete steps by "treat_pipeline_steps".
    #   - valid_targets: Allowed forecast targets. Default = ("QT", "ASP").
    #   - targets: Forecast targets to actually process in this run. Must be a subset of
    #     "valid_targets" (enforced by "validate_targets"). Defaults to all of "valid_targets".
    #   - skus: SKUs/materials to process.
    #   - raw_table_name: Name of the source table/query used during extraction.
    #   - start_date / end_date: Optional "YearMonth" bounds used to restrict the data used during
    #     featurization (forwarded as "ETL.featurize"'s "date_limits"). None leaves that bound
    #     unrestricted.
    #   - warmup_days: Days after the first observation before the first cross-validation cutoff.
    #     Default = 366.
    #   - buffer_months: Months before the last observation after which no cross-validation cutoff is
    #     generated. Default = 3.
    #   - cutoff_freq: Pandas frequency string spacing consecutive cross-validation cutoffs. Default
    #     = "3MS" (quarterly, month start).
    #   - cross_validation_horizon_days: Forecast horizon, in days, used when scoring each
    #     cross-validation cutoff. Default = 90.
    #   - optimization_n_trials: Number of Optuna trials to run per SKU/target during hyperparameter
    #     search. Default = 15.
    #   - forecast_horizon_months: Number of future monthly periods to forecast. Default = 12.
    #   - include_history: Whether generated forecasts include in-sample (historical) dates.
    #     Default = True.
    #   - raw_data_file_name: Filename for the raw extracted CSV. Default = "raw_data.csv".
    #   - cleaned_data_file_name: Filename for the cleaned/aggregated CSV.
    #     Default = "FactSalesAct.csv".
    #   - output_path: Root directory for all pipeline I/O. Accepts a "str", "Path", or None; resolved
    #     to a "Path" by "treat_output_path" (defaulting to "general_settings.base_path" when None).

    model_name: str
    output_forecast_name: str | None = None
    pipeline_steps: list
    valid_targets: tuple = ('QT', 'ASP')
    targets: list = list(valid_targets)
    skus: list
    raw_table_name: str
    start_year_month: str | None = None
    end_year_month: str | None = None
    warmup_days: int = 366
    buffer_months: int = 3
    cutoff_freq: str = "3MS"
    cross_validation_horizon_days: int = 90
    optimization_n_trials: int = 15
    forecast_horizon_months: int = 12
    include_history: bool = True
    raw_data_file_name: str = 'raw_data.csv'
    cleaned_data_file_name: str = 'FactSalesAct.csv'
    output_path: str | Path | None = None


    @property
    def warmup_period(self) -> pd.Timedelta:
        # "warmup_days" expressed as a "pandas.Timedelta", for callers that need a Timedelta rather
        # than a raw day count.

        # Returns:
        #   pd.Timedelta equivalent to "self.warmup_days" days.

        return pd.Timedelta(days=self.warmup_days)

    @property
    def end_buffer(self) -> pd.DateOffset:
        # "buffer_months" expressed as a "pandas.DateOffset", for callers that need a DateOffset
        # rather than a raw month count.

        # Returns:
        #   pd.DateOffset equivalent to "self.buffer_months" months.

        return pd.DateOffset(months=self.buffer_months)

    @model_validator(mode='after')
    def validate_targets(self) -> PipelineSettings:
        # Ensure "targets" only contains values present in "valid_targets".

        # Raises:
        #   - ValueError: If "targets" is not a subset of "valid_targets".

        if not set(self.targets).issubset(self.valid_targets):
            raise ValueError(f'Target list {self.targets} is not valid')
        return self

    @model_validator(mode='after')
    def treat_pipeline_steps(self) -> PipelineSettings:
        # Expand shorthand entries in "pipeline_steps" into their concrete step names.

        # "ALL" expands (via append) to "ETL", "HYPERPARAMETERIZE", "FIT", "FORECAST". Since "ETL" is
        # then itself checked for presence within this same call, a user-supplied or "ALL"-added
        # "ETL" is further expanded to "EXTRACT", "CLEAN", "FEATURIZE" — so "ALL" alone is enough to
        # enable every step.

        # Note:
        #   Expansion appends to "pipeline_steps" rather than replacing it, and does not deduplicate.
        #   Supplying "ALL" (or "ETL") alongside its own already-expanded step names, or more than
        #   once, results in duplicate entries. This is harmless for "Pipeline.run" (which only
        #   checks membership via "in"), but worth knowing if "pipeline_steps" is inspected directly.

        if 'ALL' in self.pipeline_steps:
            self.pipeline_steps += ['ETL', 'HYPERPARAMETERIZE', 'FIT', 'FORECAST']

        if 'ETL' in self.pipeline_steps:
            self.pipeline_steps += ['EXTRACT', 'CLEAN', 'FEATURIZE']
        return self

    @model_validator(mode='after')
    def treat_output_path(self) -> PipelineSettings:
        # Resolve "output_path" to a concrete "Path", defaulting to "general_settings.base_path".

        # Note:
        #   If "output_path" is already a "Path" instance, it is left untouched — only "None" and
        #   "str" values are handled/converted.

        if self.output_path is None:
            self.output_path = general_settings.base_path
            return self

        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        return self

    @model_validator(mode='after')
    def treat_output_forecast_name(self) -> PipelineSettings:
        # Resolve output_forecast_name, if it is set in pipeline_settings.yaml

        if self.output_forecast_name is None:
            self.output_forecast_name = self.model_name

        return self


with open(general_settings.base_path / 'pipeline_settings.yaml', 'r') as file:
    settings_data = yaml.safe_load(file)

# Module-level singleton, see module header Note.
pipeline_settings = PipelineSettings.model_validate(settings_data)