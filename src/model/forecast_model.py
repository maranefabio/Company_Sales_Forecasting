# Time-Series Forecast Model wrapper around Meta's Prophet algorithm, aiming to predict both and QT
# (Quantity / Sales Volume) and ASP (Average Selling Price) - which can be calculated as NetSales / QT - by SKU.

# The module "Forecast Model" adds:
#   - Optuna-based hyperparameter optimization with rolling-window cross-validation.
#   - JSON persistence for both tuned hyperparameters and fitted models.
#   - Consistent interface for two related but distinct targets, ASP and QT.

# The typical workflow is:
#   - With persistance:
#       >>> model: ForecastModel = ForecastModel(sku="SKU123", target="QT", model_settings=model_settings)
#       >>> model.hyperparameterize(data_df, store=True, base_path)
#       >>> model.fit(data-df, parameters_path=parameters_path, store=True)

# Raw data is required to be a sales time-series containing at least the columns:
#   - SKU [String]: Stock Keeping Unit / Unique identifier of the product;
#   - ds [Date / DateTime]: Time dimension;
#   - QT [Integer]: Quantity sold;
#   - ASP [Float]: Selling price


import json
import logging
import pandas as pd
import polars as pl
import optuna as ot
import numpy as np
from typing import Any
from pathlib import Path
from src.settings import PipelineSettings
from pandas import DatetimeIndex
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_from_json, model_to_json

# Instantiating logger for the file
logger = logging.getLogger(__name__)

# Accepted dataframe types across the module's public methods (pandas or polars)
type DataFrameType = pd.DataFrame | pl.DataFrame


# Model definition
class ForecastModel:
    # Model object is instantiated for each material and target time series, as its optimal parameters vary
    # between different SKUs

    def __init__(
        self,
        sku: str,
        target: str,
        model_settings: PipelineSettings,
    ) -> None:
        # Initialize model instance.

        # Args:
        # - sku: Material/SKU identifier
        # - target: Forecast target (ASP or QT)
        # - model_settings: Settings object bundling model configuration (model name, warmup/buffer
        #   windows, cross-validation horizon, forecast horizon, etc.). Useful for better organization
        #   and reproducibility of the persistent files.

        self.sku: str = sku
        self.target: str = target
        self.model_settings: PipelineSettings = model_settings
        self.model: Prophet | None = None
        self.parameters: dict[str, Any] | None = None
        self.best_mape: float | None = None
        self.best_rmse: float | None = None

        logger.debug(f'Model "{self.model_settings.model_name}" instantiated for {self.sku} - {self.target}')

    @property
    def is_fitted(self) -> bool:
        # Allows fitted state to be tracked
        return self.model is not None

    def load_model(self, output_path: Path) -> None:
        # Load a previously serialized model from disk.

        # Args:
        #   - output_path: directory where output files are stored. The JSON files containing the serialization
        #   of the models, produced by "write_model", are at output_path/models

        # Raises:
        #   - Exception: re-raises any error encountered while reading or deserializing.

        if self.model is not None:
            logger.warning(f'Model already loaded for {self.sku} - {self.target}. Overriding')

        model_path: Path = (
            output_path /
            'output' /
            'models' /
            self.model_settings.model_name /
            self.target
        )

        logger.debug(f'Loading model from disk for {self.sku} - {self.target}')

        try:
            with open(model_path / f'{self.sku}_model.json', 'r') as file:
                self.model = model_from_json(file.read())
        except Exception as e:
            logger.error(f'Error during model loading for SKU: {self.sku}, Target: {self.target}: {e}')
            raise

    def load_parameters(
        self,
        parameters: dict[str, Any] | None = None,
        output_path: Path | None = None
    ) -> None:
        # Loads parameters from disk or from dictionary passed as argument.

        # Args:
        #   - parameters: dictionary containing the parameters.
        #   - output_path: directory where output files are stored.

        # Raises:
        #   - ArgumentError: If both parameters and base_path are None.
        #   - Exception: Re-raises any error encountered while writing the file.

        if [parameters, output_path] == [None, None]:
            raise ValueError(f'Argument base_path cannot be None for parameters=None')

        if output_path is not None:
            if parameters is not None:
                logger.warning(f'Overriding parameters at runtime for {self.sku} - {self.target}')

                self.parameters = parameters

            else:
                parameters_path = (
                    output_path /
                    'output' /
                    'parameters' /
                    f'{self.model_settings.model_name}' /
                    f'{self.target}'
                )

                logger.debug(f'Loading parameters from disk for {self.sku} - {self.target}')

                try:
                    with open(parameters_path / f'{self.sku}_parameters.json', 'r') as file:
                        self.parameters = json.load(file)

                except Exception as e:
                    logger.error(f'Error opening {parameters_path}: {e}')
                    raise

    def hyperparameterize(
        self,
        df: DataFrameType
    ) -> dict[str, Any]:
        # Run an Optuna hyperparameter search using rolling-origin cross-validation.

        # Cutoffs are generated starting one warm-up period ("model_settings.warmup_days")
        # after the first observation and ending "model_settings.buffer_months" before the last observation,
        # spaced every "model_settings.cutoff_freq". Each trial fits a Prophet model
        # with sampled hyperparameters and scores it (RMSE, the optimization
        # objective; MAPE, tracked alongside for reporting) across all cutoffs
        # at a "model_settings.cross_validation_horizon_days"-length forecast.

        # Note:
        #   This method searches for the best hyperparameters and stores
        #   them, but does not leave self.model fit on those best
        #   parameters — the last trial's model is what remains in memory.

        #   Call "fit" with the resulting parameters file afterward to
        #   obtain a model trained on the best configuration.

        #   The Number of Optuna trials,the warmup period, cross-validation horizon and
        #   cutoff frequency are all taken from "self.model_settings" rather than passed explicitly.

        # Args:
        #   - df: Date ordered training data with "ds" (datetime) and "y" (numeric target) columns,
        #     as Prophet's convention. Accepts either a pandas or polars DataFrame; polars inputs are
        #     converted to pandas internally.



        # Returns:
        #   A dict with keys "parameters" (the best hyperparameter dict found), "rmse" (the corresponding
        #   mean RMSE, used as the optimization objective) and "mape" (the corresponding mean MAPE, tracked
        #   for reporting purposes).

        # Raises:
        #   - Exception: Re-raises any error encountered during cross-validation or
        #   the Optuna study.

        if isinstance(df, pl.DataFrame):
            try:
                df: pd.DataFrame = df.to_pandas()
            except Exception as e:
                logger.error(f'Unable to convert dataframe to pandas: {e}')
                raise

        warmup_days: int = self.model_settings.warmup_days
        buffer_months: int = self.model_settings.buffer_months

        start_date: np.datetime64 = (
                df['ds'].head(1).values[0].astype('datetime64[M]') + np.timedelta64(warmup_days, 'D')
        )
        end_date: np.datetime64 = (
                df['ds'].tail(1).values[0].astype('datetime64[M]') - np.timedelta64(buffer_months, 'M')
        )

        cutoffs: DatetimeIndex = pd.date_range(
            start=start_date,
            end=end_date,
            freq=self.model_settings.cutoff_freq,
        )

        logger.debug( f'Starting hyperparameter optimization for {self.sku} - {self.target}')

        def _objective(trial: ot.Trial) -> float:
            parameters: dict = {
                'seasonality_mode': trial.suggest_categorical(
                    'seasonality_mode',
                    ['additive', 'multiplicative']
                ),
                'changepoint_prior_scale': trial.suggest_float(
                    'changepoint_prior_scale', 0.001, 1.0, log=True
                ),
                'seasonality_prior_scale': trial.suggest_float(
                    'seasonality_prior_scale', 0.01, 10.0, log=True
                ),
                'changepoint_range': trial.suggest_float(
                    'changepoint_range', 0.8, 0.95
                ),
            }

            try:
                self.model = Prophet(**parameters, uncertainty_samples=0, n_changepoints=25)
                self.model.fit(df)
                df_cross_validation: pd.DataFrame = cross_validation(
                    self.model,
                    horizon=f'{self.model_settings.cross_validation_horizon_days} days',
                    cutoffs=cutoffs,
                    parallel='processes'
                )

            except Exception as e_in:
                logger.error(f'Error during cross validation for {self.sku} - {self.target}: {e_in}')
                raise

            df_performance: pd.DataFrame = performance_metrics(df_cross_validation)

            trial.set_user_attr('mape', df_performance['mape'].mean())

            return df_performance['rmse'].mean()

        try:
            study: ot.Study = ot.create_study(
                study_name=f'{self.model_settings.model_name}_{self.target}_{self.sku}',
                direction='minimize'
            )

            study.optimize(
                _objective,
                n_trials=self.model_settings.optimization_n_trials,
                show_progress_bar=True
            )

        except Exception as e_out:
            logger.error(f'Error during optimization for {self.sku} - {self.target}: {e_out}')
            raise

        best_mape: float = study.best_trial.user_attrs['mape']

        logger.debug(
            f'Best trial for {self.sku} - {self.target}'
            f': RMSE: {study.best_value:.4f}, MAPE: {best_mape:.4f}, params: {study.best_params}'
        )

        result: dict = {
            'parameters': study.best_params,
            'rmse': study.best_value,
            'mape': best_mape
        }

        self.parameters = result.get('parameters')
        self.best_rmse = result.get('rmse')
        self.best_mape = result.get('mape')

        self.model = None

        return result


    def fit(
        self,
        df: DataFrameType
    ) -> None:
        # Fit a Prophet model on "df" using parameters loaded in the object.

        # Args:
        #   - df: Sorted training data with "ds" (datetime) and "y" (numeric target)
        #     columns, as Prophet's convention. Accepts either a pandas or polars DataFrame; non-pandas
        #     inputs are converted to pandas internally.

        # Raises:
        #   - Exception: Re-raises any error encountered while reading the
        #     parameters file or fitting the model.

        if not isinstance(df, pd.DataFrame):
            try:
                df: pd.DataFrame = df.to_pandas()
            except Exception as e:
                logger.error(f'Unable to convert dataframe to pandas: {e}')
                raise

        model: Prophet = Prophet(**self.parameters, n_changepoints=25)

        try:
            self.model: Prophet = model.fit(df)
            logger.debug(f'Model fitted for {self.sku} - {self.target}')

        except Exception as e:
            logger.error(f'Error during model fit: {e}')
            raise

    def forecast(self) -> pd.DataFrame:
        # Generate a forecast beyond the training history.
        # The number of future monthly periods is taken from "self.model_settings.forecast_horizon_months",
        # and whether historical (in-sample) dates are included in the returned dataframe is controlled
        # by "self.model_settings.include_history". The future dataframe is built on a monthly start ("MS") frequency.
        #
        # Returns:
        #   The Prophet prediction DataFrame (includes historical fitted
        #   values - if required - plus future predictions, plus upper and lower boundaries)
        #
        # Raises:
        #   - Exception: If no model has been fit or loaded yet ("Model not loaded"), or re-raised
        #     from any error encountered while building the future dataframe or predicting.
        #     Note: unlike "write_model", this does not raise "ModelNotLoadedException" specifically —
        #     both the not-loaded case and any downstream errors surface as a plain "Exception".

        if self.model is None:
            raise ModelNotLoadedException(f'Model not loaded for {self.sku} - {self.target}')

        try:
            logger.debug(f'Forecasting for {self.sku} - {self.target}')

            future: pd.DataFrame = self.model.make_future_dataframe(
                periods=self.model_settings.forecast_horizon_months,
                freq='MS',
                include_history=self.model_settings.include_history
            )

            return self.model.predict(future)

        except Exception as e:
            logger.error(f'Error during model forecast for {self.sku} - {self.target}: {e}')
            raise


    def write_parameters(self, output_path: Path) -> None:
        # Store the current loaded parameters to disk. Allowed only if self.parameters is not None.

        # Args:
        #   - output_path: directory where output files are stored.

        # Raises:
        #   - ParametersNotLoadedException: If no parameters has been loaded yet.
        #   - Exception: Re-raises any error encountered while writing the file.

        if self.parameters is None:
            raise ParametersNotLoadedException(f'Parameters for {self.sku} - {self.target} not loaded')

        parameters_path = (
            output_path /
            'output' /
            'parameters' /
            f'{self.model_settings.model_name}' /
            f'{self.target}'
        )

        logger.debug(
            f'Writing parameters for {self.sku} - {self.target} at {parameters_path / f'{self.sku}_parameters.json'}'
        )

        Path(parameters_path).mkdir(parents=True, exist_ok=True)

        try:
            with open(parameters_path / f'{self.sku}_parameters.json', 'w+') as file:
                json.dump(self.parameters, file)
                logger.debug(f'Parameters saved at {parameters_path}')
        except Exception as e:
            logger.error(f'Error while saving parameters for {self.sku} - {self.target}: {e}')
            raise

    def write_model(self, output_path: Path) -> None:
        # Serialize and store the current loaded model to disk. Allowed only if model.is_fitted = True.

        # Args:
        #   - output_path: directory where output files are stored.

        # Raises:
        #   - ModelNotLoadedException: If no model has been fit or loaded yet.
        #   - Exception: Re-raises any error encountered while writing the file.

        if not self.is_fitted:
            raise ModelNotLoadedException(f'Model for {self.sku} - {self.target} not fitted')

        model_path: Path = (
            output_path /
            'output' /
            'models' /
            self.model_settings.model_name /
            self.target
        )

        logger.debug(
            f'Serializing model for {self.sku} - {self.target} at {model_path / f'{self.sku}_model.json'}'
        )

        Path.mkdir(model_path, parents=True, exist_ok=True)

        try:
            with open(model_path / f'{self.sku}_model.json', 'w+') as file:
                file.write(model_to_json(self.model))
        except Exception as e:
            logger.error(f'Error during model writing for {self.sku} - {self.target}: {e}')
            raise


class ModelNotLoadedException(RuntimeError):
    # Raised when an operation needs an unloaded model
    pass

class ParametersNotLoadedException(RuntimeError):
    # Raised when an operaton need an unloaded set of parameters
    pass