"""
Orchestrator tying together "ETL" and "ForecastModel" into a single, configurable forecast
pipeline, driven by which steps are listed in "pipeline_settings.pipeline_steps".
The methods used are stored in the "Utils" class, which is a thin wrapper around the underlying "ETL" and
"ForecastModel" methods, to better control logging and error handling."

The module "Pipeline" adds:
    - A single entrypoint for running the full forecast pipeline end-to-end, executing only the steps
      listed in "pipeline_settings.pipeline_steps".

    - Logging and error handling for each step, including re-raising any exceptions encountered by
      the underlying "ETL" or "ForecastModel" calls.

The typical workflow is:
    >>> pipeline: Pipeline = Pipeline(
            base_path=base_path,
            database_settings=database_settings,
            pipeline_settings=pipeline_settings
        )
    >>> pipeline.run()

  where each settings object is loaded from its corresponding settings model (see "settings" directory).

Note:
    Each step reads/writes its inputs and outputs through local CSV layers on disk (via "ETL") rather
    than passing dataframes between steps in memory, except within a single step's own loops. This
    means steps can be run independently across separate "run()" calls, as long as the CSV layer(s) a
    given step depends on already exist from a prior run.
"""

import logging
import datetime as dt
import pandas as pd
import polars as pl
from pathlib import Path
from src.utils import Utils
from src.etl import ETL
from src.etl.etl import EmptyTableException
from src.settings import PipelineSettings, DatabaseSettings, ModelSettings
from src.model import ForecastModel


logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the full forecast pipeline, by running the steps listed in 
    "pipeline_settings.pipeline_steps" in a fixed order and configuring
    the objects used in each step via the provided settings objects.

    Arguments:
        - base_path (Path): Root directory used to locate query files for the "EXTRACT" step (forwarded
          to "ETL.extract").

        - database_settings (DatabaseSettings): Settings used to open the source database connection

        - pipeline_settings (PipelineSettings): Settings object bundling pipeline configuration — which steps to run
          ("pipeline_steps"), the local output root ("output_path"), the targets and SKUs to
          process.

        - model_settings (ModelSettings): Settings object used to configure the "ForecastModel" during the
          "HYPERPARAMETERIZE", "FIT" and "FORECAST" steps.


    Attributes:
        - base_path (Path).

        - database_settings (DatabaseSettings).

        - pipeline_settings (PipelineSettings).

        - model_settings (ModelSettings).
    """

    def __init__(
        self,
        base_path: Path,
        database_settings: DatabaseSettings,
        pipeline_settings: PipelineSettings,
        model_settings: ModelSettings
    ) -> None:
        self.base_path = base_path
        self.database_settings = database_settings
        self.pipeline_settings = pipeline_settings
        self.model_settings = model_settings

    def run(self) -> None:
        """
        Run the forecast pipeline end to end, executing only the steps listed in
        "self.pipeline_settings.pipeline_steps". 
        
        Recognized steps, run in this fixed order when present, are: 
            - EXTRACT,
            - CLEAN,
            - FEATURIZE,
            - HYPERPARAMETERIZE,
            - FIT,
            - FORECAST,

        with the shorthand "ALL" expanding to all of the above, and "ETL" expanding to the first three.

        Steps:
            - EXTRACT: Runs "raw_sales.sql" against the source database via "ETL.extract", storing
              the raw result as "data/raw/raw_data.csv".

            - CLEAN: Cleans, via "ETL.clean", the file "data/raw/raw_data.csv" to the expected schema,
              restricts it to the "[start_date, end_date]" window, and persists it, storing the result
              as "data/cleaned/FactSalesAct.csv".

            - FEATURIZE: Builds one Prophet-ready dataframe per entry in
              "self.pipeline_settings.targets" via "ETL.featurize", and
              persists each as "data/featurized/{target}.csv".

            - HYPERPARAMETERIZE: For each target and each SKU in "self.pipeline_settings.targets" /
              "self.pipeline_settings.skus", ingests that target's featurized CSV, filters it down to
              the SKU's rows, and runs "ForecastModel.hyperparameterize" followed by
              "ForecastModel.write_parameters" to persist the best hyperparameters found.
              
            - FIT: For each target and SKU, ingests the featurized CSV, filters to the SKU, loads the
              previously saved hyperparameters via "ForecastModel.load_parameters", fits the model,
              and persists it via "ForecastModel.write_model".

            - FORECAST: For each target and SKU, ingests the featurized CSV, filters to the SKU, loads
              the previously fitted model via "ForecastModel.load_model", and generates a forecast.
              All (SKU, target) forecasts are concatenated into a single polars DataFrame (columns
              "sku", "ds", "target", "yhat", "yhat_lower", "yhat_upper", "trend"), stamped with a
              "generated_at" timestamp (America/Sao_Paulo timezone), and written to
              "output/forecasts/{model_name}/FactForecast.csv".

        Raises:
            - EmptyTableException: Raised in the "HYPERPARAMETERIZE", "FIT" or "FORECAST" steps when,
              after filtering a target's featurized data down to a single SKU, the resulting
              dataframe is empty (e.g. the SKU has no rows for that target).
            - Exception: Re-raises any error encountered by the underlying "ETL" or "ForecastModel"
              calls within any enabled step.

        Returns:
            None
        """

        logger.info('Starting Forecast Pipeline')

        if 'EXTRACT' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Extraction step')
            try:
                Utils.extract(
                    base_path = self.base_path,
                    database_settings= self.database_settings,
                    query_file = 'raw_sales.sql',
                    store_raw_as = 'raw_data.csv',
                    files_path = self.pipeline_settings.files_path
                )
            except Exception as e:
                logger.error(f'Exception occured during extraction step: {e}')
                raise

            logger.info('Extraction complete')

        if 'CLEAN' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Cleaning step')

            try:
                Utils.clean(
                    files_path = self.pipeline_settings.files_path,
                    raw_file_name = 'raw_data.csv',
                    date_limits = [self.pipeline_settings.start_date, self.pipeline_settings.end_date]
                )
            except Exception as e:
                logger.error(f'Exception occured during cleaning step: {e}')
                raise

            logger.info('Cleaning complete')

        if 'FEATURIZE' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Featurization step')

            try:
                Utils.featurize(
                    targets = self.pipeline_settings.targets,
                    files_path = self.pipeline_settings.files_path,
                    file_name = 'FactSalesAct.csv'
                )
            except Exception as e:
                logger.error(f'Exception occured during featurization step: {e}')
                raise

            logger.info('Featurization complete')

        if 'HYPERPARAMETERIZE' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Hyperparameterization step')

            try:
                Utils.optimize_hyperparameters(
                    model_settings = self.model_settings,
                    skus = self.pipeline_settings.skus,
                    targets = self.pipeline_settings.targets,
                    files_path = self.pipeline_settings.files_path,
                )
            except Exception as e:
                logger.error(f'Exception occured during hyperparameterization step: {e}')
                raise


            logger.info('Hyperparameterization complete')

        if 'FIT' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Model Fitting step')

            try:
                Utils.fit(
                    model_settings = self.model_settings,
                    skus = self.pipeline_settings.skus,
                    targets = self.pipeline_settings.targets,
                    files_path = self.pipeline_settings.files_path,
                )
            except Exception as e:
                logger.error(f'Exception occured during fitting step: {e}')
                raise

            logger.info('Model Fitting complete')

        if 'FORECAST' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Forecast step')

            try:
                Utils.forecast(
                    model_settings = self.model_settings,
                    skus = self.pipeline_settings.skus,
                    targets = self.pipeline_settings.targets,
                    files_path = self.pipeline_settings.files_path,
                    output_forecast_name = self.model_settings.output_forecast_name
                )
            except Exception as e:
                logger.error(f'Exception occured during forecasting step: {e}')
                raise

            logger.info('Forecasting complete')
