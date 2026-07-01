# Time-Series Forecast Model wrapper around Meta's Prophet algorithm, aiming to predict both ASP (Average Selling
# Price) and QT (Quantity / Sales Volume) by SKU.

# The module "Forecast Model" adds:
# - Optuna-based hyperparameter optimization with rolling-window cross-validation.
# - JSON persistence for both fitted models and tunned huperparameters.
# - Consistent interface for two related but distinct targets, ASP and QT.

# The typical workflow for a given SKU is:
#   >>> model: Forecast Model = ForecastModel(material="SKU123", target="QT", model_name="fc1_SKU123")
#   >>> model.hyperparameterize(df, n_trials=50, base_path="./files")
#   >>> model.fit(df, parameters_path="./files/parameters/fc1_SKU123/QT/SKU123_parameters.json")
#   >>> forecast_df = model.forecast(periods=12)
#   >>> model.write_model(base_path="./files")

#   where "df" is expected to follow the Prophet's convention, containing the target column "y"
#   and the datetime column "ds".


# Importing dependencies

import json
import logging
import os
import pandas as pd
import optuna as ot
import numpy as np

from pandas import DatetimeIndex
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_from_json, model_to_json

from src.config import config_model

# Instantiating logger for the file
logger = logging.getLogger(__name__)

# Model configuration constants
VALID_TARGETS = config_model.VALID_TARGETS
WARMUP_PERIOD = config_model.WARMUP_PERIOD
END_BUFFER = config_model.END_BUFFER
CUTOFF_FREQ = config_model.CUTOFF_FREQ
CROSS_VALID_HORIZON = config_model.CROSS_VALID_HORIZON


class ModelNotLoadedException(RuntimeError):
    # Raised when an operation needs an unloaded model
    pass


# Model definition
class ForecastModel:
    # Model object is instantiated for each material and target time series, as its optimal parameters vary
    # between different SKUs

    def __init__(
            self,
            material: str,
            target: str,
            model_name: str
    ) -> None:
        # Initialize model instance.

        # Args:
        # - material: Material identifier
        # - target: Forecast target (ASP or QT)
        # - model_name: Model chosen name. Useful for better organization and reproducibility of the persistence files

        # Raises:
        # - ValueError: If "target" is not in ('ASP', 'QT')

        if target not in VALID_TARGETS:
            raise ValueError(f'Undefined model target. Defined only for types in {VALID_TARGETS}"')

        self.material: str = material
        self.target: str = target
        self.model_name: str = model_name
        self.model: Prophet | None = None
        self.parameters: dict | None = None

        logger.info(f'Model "{self.model_name}" created for {self.material} - {self.target}')

    @property
    def is_fitted(self) -> bool:
        # Allows fitted state to be tracked
        return self.model is not None

    def load_model(self, model_path: str) -> None:
        # Load a previously serialized model from disk.

        # Args:
        #   - model_path: path to JSON file containing the serialization of a model, produced by "write_model".

        # Raises:
        #   - Exception: re-raises any error encountered while reading or deserializing.

        if self.model is not None:
            logger.warning('Model already loaded. Overriding ...')

        try:
            with open(model_path, 'r') as file:
                self.model = model_from_json(file.read())
        except Exception as e:
            raise Exception(f'Error during model loading for Material: {self.material}, Target: {self.target}: {e}')


    def hyperparameterize(
            self,
            df: pd.DataFrame,
            n_trials: int,
            base_path: str | None = None,
            store: bool = False
    ) -> dict:
        # Run an Optuna hyperparameter search using rolling-origin cross-validation.

        # Cutoffs are generated starting one warm-up period ("WARMUP_PERIOD")
        # after the first observation and ending "END_BUFFER" before the
        # last observation, spaced every "CUTOFF_FREQ". Each trial fits a
        # Prophet model with sampled hyperparameters and scores it by the
        # mean MAPE across all cutoffs at a CROSS_VALID_HORIZON-length forecast.

        # Note:
        #   This method searches for the best hyperparameters and stores
        #   them, but does not leave self.model fit on those best
        #   parameters — the last trial's model is what remains in memory.
        #   Call ``fit`` with the resulting parameters file afterward to
        #   obtain a model trained on the best configuration.

        # Args:
        #   - df: Date ordered training data with "ds" (datetime) and "Y" (numeric target) columns,
        #     as Prophet's convention.
        #   - n_trials: Number of Optuna trials to run.
        #   - base_path: Root directory under which "parameters/{model_name}/{target}/..." is created and
        #     the best-parameters JSON is saved.
        #   - store: whether to save the parameters in disk. Default = False

        # Returns:
        #   A dict with keys "parameters" (the best hyperparameter dict found) and "mape"
        #   (the corresponding mean MAPE).

        # Raises:
        #   - Exception: Re-raises any error encountered during cross-validation or
        #   the Optuna study.

        start_date: np.datetime64 = df['ds'].head(1).values[0].astype('datetime64[M]') + WARMUP_PERIOD
        end_date: np.datetime64 = df['ds'].tail(1).values[0].astype('datetime64[M]') - END_BUFFER

        cutoffs: DatetimeIndex = pd.date_range(
            start=start_date,
            end=end_date,
            freq=CUTOFF_FREQ
        )

        def objective(trial: ot.Trial) -> float:
            parameters: dict = {
                'seasonality_mode': trial.suggest_categorical(
                    'seasonality_mode',
                    ['additive', 'multiplicative']
                ),
                'n_changepoints': trial.suggest_int('n_changepoints', 1, 50),
                'changepoint_prior_scale': trial.suggest_float(
                    'changepoint_prior_scale', 0.001, 1.0, log=True
                ),
                'seasonality_prior_scale': trial.suggest_float(
                    'seasonality_prior_scale', 1, 20.0, log=True
                ),
            }

            try:
                self.model = Prophet(**parameters)
                self.model.fit(df)
                df_cross_validation: pd.DataFrame = cross_validation(
                    self.model,
                    horizon=CROSS_VALID_HORIZON,
                    cutoffs=cutoffs,
                    parallel='processes'
                )
            except Exception as e_in:
                raise Exception(
                    f'Error during cross valid. for Material: {self.material}, Target: {self.target}: {e_in}'
                )

            df_performance: pd.DataFrame = performance_metrics(df_cross_validation)

            return df_performance['mape'].mean()

        try:
            study: ot.Study = ot.create_study(
                study_name=f'{self.model_name}_{self.target}_{self.material}',
                direction='minimize'
            )

            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=True
            )
        except Exception as e_out:
            raise Exception(f'Error during optimization for Material: {self.material}, Target: {self.target}: {e_out}')

        logger.info(
            f'Best trial for {self.material} - {self.target}'
            f': MAPE: {study.best_value:.4f}, params: {study.best_params}'
        )

        result: dict = {
            'parameters': study.best_params,
            'mape': study.best_value
        }

        parameters_path = os.path.join(
            base_path,
            'parameters',
            f'{self.model_name}',
            f'{self.target}',
        )

        if store:
            if base_path is None:
                raise ValueError(f'Base path cannot be None for store=True')

            if not os.path.exists(parameters_path):
                os.makedirs(parameters_path)

            try:
                with open(os.path.join(parameters_path, f'{self.material}_parameters.json'), 'w+') as file:
                    json.dump(result, file)
                    logger.info(f'Parameters saved at {parameters_path}')
            except Exception as e:
                raise Exception(f'Error during saving parameters for Material: {self.material}, Target: {self.target}: {e}')

            self.model = None

        return result

    def fit(self, df: pd.DataFrame, parameters_path: str) -> None:
        # Fit a Prophet model on "df" using hyperparameters loaded from disk.

        # Args:
        #   - df: Sorted training data with "ds" (datetime) and "y" (numeric target)
        #     columns, as Prophet's convention.
        #   - parameters_path: Path to a JSON file (as produced by "hyperparameterize") containing a
        #     "parameters" key.

        # Raises:
        #   - Exception: Re-raises any error encountered while reading the
        #     parameters file or fitting the model.
        try:
            with open(parameters_path, 'r') as file:
                result_data: dict = json.load(file)
                self.parameters = result_data['parameters']
        except Exception as e:
            raise Exception(f'Error opening {parameters_path}: {e}')

        model: Prophet = Prophet(**self.parameters)
        try:
            self.model: Prophet = model.fit(df)
            logger.info(f'Model fitted for Material: {self.material}, Target: {self.target}.')
        except Exception as e:
            raise Exception(f'Error during model fit: {e}')

    def forecast(self, periods: int) -> None | pd.DataFrame:
        # Generate a forecast for "periods" months beyond the training history.
        #
        # Args:
        #   - periods: Number of future monthly periods to forecast.
        #
        # Returns:
        #   The Prophet prediction DataFrame (includes historical fitted
        #   values plus future predictions, plus upper and lower boundaries)
        #
        # Raises:
        #   - ModelNotLoadedError: If no model has been fit or loaded yet.
        #   - Exception: Re-raises any error encountered while building the
        #     future dataframe or predicting.
        #
        if self.model is None:
            raise Exception('Model not loaded')

        try:
            future: pd.DataFrame = self.model.make_future_dataframe(
                periods=periods,
                freq='MS',
                include_history=True
            )
            return self.model.predict(future)
        except Exception as e:
            raise Exception(f'Error during model forecast for Material: {self.material}, Target: {self.target}: {e}')

    def write_model(self, base_path: str) -> None:
        # Serialize and store the current loaded model to disk. Allowed only if model.is_fitted = True.

        # Args:
        #   - base_path: Root directory where local files are stored

        # Raises
        #   - ModelNotLoadedException: If no model has been fit or loaded yet
        #   - Exception: Re-raises any error encountered while writing the file

        if self.model.is_fitted:
            logger.error(f'Model for {self.material} - {self.target} not loaded.')

            raise ModelNotLoadedException(f'Model for {self.material} - {self.target} not fitted.')

        models_path: str = os.path.join(base_path, 'models', self.model_name, self.target)
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        try:
            with open(os.path.join(models_path, f'{self.material}_model.json'), 'w+') as file:
                file.write(model_to_json(self.model))
        except Exception as e:
            raise Exception(f'Error during model writing for Material: {self.material}, Target: {self.target}: {e}')
