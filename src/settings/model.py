'''
Pydantic settings model for the Forecast Model itself, populated from "model_settings.yaml" and
doubling as the "model_settings" object passed into each ForecastModel instance.

The module "ModelSettings" adds:
    - Declarative configuration for the Forecast Model's behavior and parameters.

Note:
    "model_settings" below is a module-level singleton constructed at import time by reading
    "{general_settings.base_path}/config/model_settings.yaml". Importing this module therefore immediately
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


class ModelSettings(BaseSettings):
    """
    Full configuration for a single ForecastModel instance.

    Fields:
        - model_name: Name of the model, used to locate the model's configuration file and to name.

        - output_forecast_name: Name of the output forecast file. If not specified, defaults to "model_name".

        - warmup_days: Number of days to use as the warmup period for the model.
          Default = 366.

        - buffer_months: Number of months to use as the buffer period for the model.
          Default = 3.

        - cutoff_freq: Frequency to use for the model's cutoff points.
          Default = "3MS".

        - cross_validation_horizon_days: Number of days to use for the model's cross-validation horizon.
          Default = 90.

        - optimization_n_trials: Number of trials to use for the model's hyperparameter optimization.
          Default = 15.

        - forecast_horizon_months: Number of months to use for the model's forecast future horizon.
          Default = 12.

        - include_history: Whether to include historical data in the model's forecast.
          Default = False.
    """
    
    model_name: str
    output_forecast_name: str | None = None
    warmup_days: int = 366
    buffer_months: int = 3
    cutoff_freq: str = "3MS"
    cross_validation_horizon_days: int = 90
    optimization_n_trials: int = 15
    forecast_horizon_months: int = 12
    include_history: bool = False

    @property
    def warmup_period(self) -> pd.Timedelta:
        """
        "warmup_days" expressed as a "pandas.Timedelta", for callers that need a Timedelta rather
        than a raw day count.

        Returns:
            pd.Timedelta equivalent to "self.warmup_days" days.
        """

        return pd.Timedelta(days=self.warmup_days)

    @property
    def end_buffer(self) -> pd.DateOffset:
        """
        "buffer_months" expressed as a "pandas.DateOffset", for callers that need a DateOffset
        rather than a raw month count.

        Returns:
            pd.DateOffset equivalent to "self.buffer_months" months.
        """

        return pd.DateOffset(months=self.buffer_months)

    @model_validator(mode='after')
    def treat_output_forecast_name(self):
        """Resolve output_forecast_name, if it is set in pipeline_settings.yaml"""

        if self.output_forecast_name is None:
            self.output_forecast_name = self.model_name

        return self

with open(general_settings.base_path / 'config' / 'model_settings.yaml', 'r') as file:
    settings_data = yaml.safe_load(file)

model_settings = ModelSettings.model_validate(settings_data)

if __name__ == '__main__':
    print(model_settings)