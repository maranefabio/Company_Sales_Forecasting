# Orchestrator tying together "ETL" and "ForecastModel" into a single, configurable forecast
# pipeline, driven by which steps are listed in "pipeline_settings.pipeline_steps".

# The module "Pipeline" adds:
# - A single entry point ("run") that conditionally executes each pipeline step (EXTRACT, CLEAN,
#   FEATURIZE, HYPERPARAMETERIZE, FIT, FORECAST) based on "pipeline_settings.pipeline_steps".
# - Per-SKU, per-target orchestration of hyperparameter search, fitting and forecasting, using the
#   featurized CSV layer as the shared hand-off point between "ETL" and "ForecastModel".
# - A consolidated forecast output combining every (SKU, target) forecast into one timestamped CSV.

# The typical workflow is:
#   >>> pipeline: Pipeline = Pipeline(
#   ...     base_path=base_path, database_settings=database_settings, pipeline_settings=pipeline_settings
#   ... )
#   >>> pipeline.run()

#   where "pipeline_settings" is a "PipelineSettings" instance (itself used as the "model_settings"
#   passed into "ForecastModel") holding, among other things: "pipeline_steps" (which steps to run),
#   "output_path", "targets", "skus", "model_name", and "start_date"/"end_date".

# Note:
#   Each step reads/writes its inputs and outputs through local CSV layers on disk (via "ETL") rather
#   than passing dataframes between steps in memory, except within a single step's own loops. This
#   means steps can be run independently across separate "run()" calls, as long as the CSV layer(s) a
#   given step depends on already exist from a prior run.

import logging
import datetime as dt
import pandas as pd
import polars as pl
from pathlib import Path
from src.etl import ETL
from src.etl.etl import EmptyTableException
from src.settings import PipelineSettings, DatabaseSettings
from src.model import ForecastModel
from zoneinfo import ZoneInfo

# Instantiating logger for the file
logger = logging.getLogger(__name__)


class Pipeline:
    # Orchestrates the full forecast pipeline for a single configured database/output location,
    # running whichever steps are enabled in "pipeline_settings.pipeline_steps".

    def __init__(
        self,
        base_path: Path,
        database_settings: DatabaseSettings,
        pipeline_settings: PipelineSettings
    ) -> None:
        # Initialize pipeline instance.

        # Args:
        #   - base_path: Root directory used to locate query files for the "EXTRACT" step (forwarded
        #     to "ETL.extract").
        #   - database_settings: Settings used to open the source database connection during
        #     "EXTRACT".
        #   - pipeline_settings: Settings object bundling pipeline configuration — which steps to run
        #     ("pipeline_steps"), the local output root ("output_path"), the targets and SKUs to
        #     process, the model name, the date range for featurization, and any other
        #     "ForecastModel" hyperparameter/forecast settings (it is passed as-is as "model_settings"
        #     when constructing each "ForecastModel").

        self.base_path = base_path
        self.database_settings = database_settings
        self.pipeline_settings = pipeline_settings

    def run(self) -> None:
        # Run the forecast pipeline end to end, executing only the steps listed in
        # "self.pipeline_settings.pipeline_steps". Recognized steps, run in this fixed order when
        # present, are: "EXTRACT", "CLEAN", "FEATURIZE", "HYPERPARAMETERIZE", "FIT", "FORECAST".

        # Steps:
        #   - EXTRACT: Runs "raw_sales.sql" against the source database via "ETL.extract", storing
        #     the raw result as "data/raw/raw_data.csv".
        #   - CLEAN: Aggregates "data/raw/raw_data.csv" to monthly granularity via "ETL.clean",
        #     storing the result as "data/cleaned/FactSalesAct.csv".
        #   - FEATURIZE: Builds one Prophet-ready dataframe per entry in
        #     "self.pipeline_settings.targets" via "ETL.featurize", restricted to the
        #     "[start_date, end_date]" window, and persists each as "data/featurized/{target}.csv".
        #   - HYPERPARAMETERIZE: For each target and each SKU in "self.pipeline_settings.targets" /
        #     "self.pipeline_settings.skus", ingests that target's featurized CSV, filters it down to
        #     the SKU's rows, and runs "ForecastModel.hyperparameterize" followed by
        #     "ForecastModel.write_parameters" to persist the best hyperparameters found.
        #   - FIT: For each target and SKU, ingests the featurized CSV, filters to the SKU, loads the
        #     previously saved hyperparameters via "ForecastModel.load_parameters", fits the model,
        #     and persists it via "ForecastModel.write_model".
        #   - FORECAST: For each target and SKU, ingests the featurized CSV, filters to the SKU, loads
        #     the previously fitted model via "ForecastModel.load_model", and generates a forecast.
        #     All (SKU, target) forecasts are concatenated into a single polars DataFrame (columns
        #     "sku", "ds", "target", "yhat", "yhat_lower", "yhat_upper", "trend"), stamped with a
        #     "generated_at" timestamp (America/Sao_Paulo timezone), and written to
        #     "output/forecasts/{model_name}/FactForecast.csv".

        # Raises:
        #   - EmptyTableException: Raised in the "HYPERPARAMETERIZE", "FIT" or "FORECAST" steps when,
        #     after filtering a target's featurized data down to a single SKU, the resulting
        #     dataframe is empty (e.g. the SKU has no rows for that target).
        #   - Exception: Re-raises any error encountered by the underlying "ETL" or "ForecastModel"
        #     calls within any enabled step.

        logger.info('Starting Forecast Pipeline')

        featurized_df_schema: dict = {
            'SKU': pl.String,
            'ds': pl.Date,
            'y': pl.Float64
        }

        if 'EXTRACT' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Extraction step')

            ETL.extract(
                self.base_path,
                self.database_settings,
                'raw_sales.sql',
                'raw_data.csv',
                self.pipeline_settings.output_path
            )

            logger.info('Extraction complete')

        if 'CLEAN' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Cleaning step')

            raw_data_schema: dict = {
                'SKU': pl.String,
                'Dt': pl.Date,
                'QT': pl.Float64,
                'ASP': pl.Float64,
            }

            ETL.clean(
                None,
                self.pipeline_settings.output_path,
                'raw_data.csv',
                'FactSalesAct.csv',
                raw_data_schema,
            )

            logger.info('Cleaning complete')

        if 'FEATURIZE' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Featurization step')

            ETL.featurize(
                None,
                self.pipeline_settings.output_path,
                self.pipeline_settings.targets,
                True,
                date_limits=[self.pipeline_settings.start_date, self.pipeline_settings.end_date],
            )

            logger.info('Featurization complete')

        if 'HYPERPARAMETERIZE' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Hyperparameter Optimization step')

            for target in self.pipeline_settings.targets:
                df_in: pl.DataFrame = ETL.ingest(
                    output_path = self.pipeline_settings.output_path,
                    query = f'{target}.csv',
                    schema_overrides = featurized_df_schema,
                    layer = 'featurized'
                )

                for sku in self.pipeline_settings.skus:
                    logger.debug(
                        f'Optimizing hyperparameters for model {self.pipeline_settings.model_name} - {sku} - {target}'
                    )

                    group_df: pl.DataFrame = df_in.filter(pl.col('SKU') == sku).sort('ds')

                    if len(group_df) == 0:
                        raise EmptyTableException(f'Grouping gor {sku} - {target} returned empty dataframe')

                    model: ForecastModel = ForecastModel(
                        sku = sku,
                        target = target,
                        model_settings = self.pipeline_settings,
                    )

                    model.hyperparameterize(group_df)
                    model.write_parameters(output_path=self.pipeline_settings.output_path)

            logger.info('Hyperparameters Optimization complete')

        if 'FIT' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Model Fitting step')

            for target in self.pipeline_settings.targets:
                logger.debug(
                    f'Fitting for model {self.pipeline_settings.model_name} - {target}'
                )

                df_in: pl.DataFrame = ETL.ingest(
                    output_path = self.pipeline_settings.output_path,
                    query = f'{target}.csv',
                    schema_overrides = featurized_df_schema,
                    layer = 'featurized'
                )

                for sku in self.pipeline_settings.skus:
                    logger.debug(
                        f'Fitting model {self.pipeline_settings.model_name} for {sku} - {target}'
                    )

                    group_df: pl.DataFrame = df_in.filter(pl.col('SKU') == sku).sort('ds')

                    if len(group_df) == 0:
                        raise EmptyTableException(f'Grouping gor {sku} - {target} returned empty dataframe')

                    model: ForecastModel = ForecastModel(
                        sku = sku,
                        target = target,
                        model_settings = self.pipeline_settings,
                    )

                    model.load_parameters(output_path=self.pipeline_settings.output_path)
                    model.fit(group_df)
                    model.write_model(output_path=self.pipeline_settings.output_path)

            logger.info('Model Fitting complete')

        if 'FORECAST' in self.pipeline_settings.pipeline_steps:
            logger.info('Starting Forecast step')

            forecast_df_schema: dict = {
                'sku': pl.String,
                'ds': pl.Datetime,
                'target': pl.String,
                'yhat': pl.Float64,
                'yhat_lower': pl.Float64,
                'yhat_upper': pl.Float64,
                'trend': pl.Float64
            }

            final_forecast_df: pl.DataFrame = pl.DataFrame(schema=forecast_df_schema)

            for target in self.pipeline_settings.targets:
                logger.debug(
                    f'Forecasting for {target}'
                )

                df_in: pl.DataFrame = ETL.ingest(
                    output_path = self.pipeline_settings.output_path,
                    query = f'{target}.csv',
                    schema_overrides = featurized_df_schema,
                    layer = 'featurized'
                )

                target_forecast_df: pl.DataFrame = pl.DataFrame(schema=forecast_df_schema)

                for sku in self.pipeline_settings.skus:
                    logger.debug(
                        f'Forecasting {sku} - {target}'
                    )

                    group_df: pl.DataFrame = df_in.filter(pl.col('SKU') == sku).sort('ds')

                    if len(group_df) == 0:
                        raise EmptyTableException(f'Grouping gor {sku} - {target} returned empty dataframe')

                    model: ForecastModel = ForecastModel(
                        sku = sku,
                        target = target,
                        model_settings = self.pipeline_settings,
                    )

                    model.load_model(output_path=self.pipeline_settings.output_path)

                    sku_forecast_df: pd.DataFrame = model.forecast()
                    sku_forecast_df: pl.DataFrame = pl.from_pandas(sku_forecast_df).with_columns([
                        pl.lit(sku).alias('sku'),
                        pl.lit(target).alias('target'),
                    ]).select([
                        'sku', 'ds', 'target', 'yhat', 'yhat_lower', 'yhat_upper', 'trend'
                    ])

                    target_forecast_df = target_forecast_df.vstack(sku_forecast_df)

                final_forecast_df = final_forecast_df.vstack(target_forecast_df)

            final_forecast_df = final_forecast_df.with_columns([
                pl.col('ds').cast(pl.Date),
                pl.lit(dt.datetime.now(ZoneInfo('America/Sao_Paulo'))).alias('generated_at')
            ])

            forecasts_path: Path = (
                self.pipeline_settings.output_path /
                'output' /
                'forecasts' /
                self.pipeline_settings.model_name
            )

            Path.mkdir(forecasts_path, parents = True, exist_ok = True)
            final_forecast_df.write_csv(forecasts_path / f'FactForecast.csv', separator=';')

        logger.info('Forecasting complete')
