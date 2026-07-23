"""
Utility static methods for handling all pipeline tasks.

The typical workflow is:
    >>> Utils.extract(
            base_path,
            database_settings,
            query_file,
            store_raw_as,
            files_path
        )
    >>> Utils.clean(...)
    >>> Utils.featurize(...)
    >>> ...
"""

import logging
import polars as pl
import pandas as pd
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo
from src.model import ForecastModel
from src.etl import ETL
from src.settings import DatabaseSettings, ModelSettings
from src.etl.etl import EmptyTableException

logger = logging.getLogger(__name__)

class Utils:
    """
    Collection of utility static methods for handling all pipeline tasks.
    """

    @staticmethod
    def extract(
        base_path: Path,
        database_settings: DatabaseSettings,
        query_file: str,
        store_raw_as: str,
        files_path: Path
    ) -> None:
        """
        Run ETL.extract (See module ETL for more information).

        Arguments:
            - base_path (Path): Base path for the project, where query files are located under "base_path/src/files/queries/".

            - database_settings (DatabaseSettings): Database connection settings.

            - query_file (str): Name of the SQL query file to execute 
              (located under "base_path/src/files/queries/").

            - store_raw_as (str): Name of the raw CSV file to store the extracted data.

            - files_path (Path): Path where the raw CSV file will be stored.

        """
        ETL.extract(
            base_path = base_path,
            database_settings= database_settings,
            query_file = query_file,
            store_raw_as = store_raw_as,
            files_path = files_path
        )

    @staticmethod
    def clean(
        files_path: Path,
        raw_file_name: str,
        date_limits: list[dt.date]
    ) -> None:
        """
        Run ETL.clean (See module ETL for more information).

        Arguments:
            - files_path (Path): Path where the raw CSV file is located and where the cleaned CSV file will be stored.

            - raw_file_name (str): Name of the raw CSV file to clean.

            - date_limits (list[dt.date]): List containing two dates (start and end) to filter the data by date.
        """

        raw_data_schema: dict = {
            'SKU': pl.String,
            'Dt': pl.Date,
            'QT': pl.Float64,
            'ASP': pl.Float64,
        }

        ETL.clean(
            files_path = files_path,
            raw_file_name = raw_file_name,
            schema_overrides = raw_data_schema,
            date_limits = date_limits,
            store = True
        )

    @staticmethod
    def featurize(
        targets: list[str],
        files_path: Path,
        file_name: str,
    ) -> None:
        """
        Run ETL.featurize for each target passed (See module ETL for more information).

        Arguments:
            - targets (list[str]): List of target names to featurize.

            - files_path (Path): Path where the cleaned CSV file is located and where the
              featurized CSV files will be stored.

            - file_name (str): Name of the cleaned CSV file to featurize.
        """
        for target in targets:
            ETL.featurize(
                target = target,
                files_path = files_path,
                file_name=file_name,
                store = True,
            )

    @staticmethod
    def optimize_hyperparameters(
        model_settings: ModelSettings,
        skus: list[str],
        targets: list[str],
        files_path: Path,
    ) -> None:
        """
        Run ForecastModel.hyperparameterize for each combination of sku/target passed
        (See module ForecastModel for more information).

        Arguments:
            - model_settings (ModelSettings): Model settings to be used for model instantiation.

            - skus (list[str]): List of SKUs to optimize hyperparameters for.

            - targets (list[str]): List of target names to optimize hyperparameters for.

            - files_path (Path): Path where the featurized CSV files are located and where the
              optimized hyperparameters will be stored.
        """
        featurized_df_schema: dict = {
            'SKU': pl.String,
            'ds': pl.Date,
            'y': pl.Float64
        }

        for target in targets:
            df_in: pl.DataFrame = ETL.ingest(
                files_path = files_path,
                file_name = f'{target}.csv',
                schema_overrides = featurized_df_schema,
                layer = 'featurized'
            )

            for sku in skus:
                logger.debug(
                    f'Optimizing hyperparameters for model {model_settings.model_name} - {sku} - {target}'
                )

                group_df: pl.DataFrame = df_in.filter(pl.col('SKU') == sku).sort('ds')

                if len(group_df) == 0:
                    raise EmptyTableException(f'Grouping for {sku} - {target} returned empty dataframe')

                model: ForecastModel = ForecastModel(
                    model_settings = model_settings,
                    sku = sku,
                    target = target,
                )

                model.hyperparameterize(group_df)
                model.write_parameters(files_path=files_path)

    @staticmethod
    def fit(
        model_settings: ModelSettings,
        skus: list[str],
        targets: list[str],
        files_path: Path,
    ) -> None:
        """
        Run ForecastModel.fit for each combination of sku/target passed
        (See module ForecastModel for more information).

        Arguments:
            - model_settings (ModelSettings): Model settings to be used for model instantiation.

            - skus (list[str]): List of SKUs to fit models for.

            - targets (list[str]): List of target names to fit models for.

            - files_path (Path): Path where the featurized CSV files are located and where the
              fitted models will be stored.
        """

        for target in targets:
            logger.debug(
                f'Fitting model {model_settings.model_name} - {target}'
            )

            featurized_df_schema: dict = {
                'SKU': pl.String,
                'ds': pl.Date,
                'y': pl.Float64
            }

            df_in: pl.DataFrame = ETL.ingest(
                files_path = files_path,
                file_name = f'{target}.csv',
                schema_overrides = featurized_df_schema,
                layer = 'featurized'
            )

            for sku in skus:
                logger.debug(
                    f'Fitting model {model_settings.model_name} for {sku} - {target}'
                )

                group_df: pl.DataFrame = df_in.filter(pl.col('SKU') == sku).sort('ds')

                if len(group_df) == 0:
                    raise EmptyTableException(f'Grouping for {sku} - {target} returned empty dataframe')

                model: ForecastModel = ForecastModel(
                    model_settings = model_settings,
                    sku = sku,
                    target = target,
                )

                model.load_parameters(files_path=files_path)
                model.fit(group_df)
                model.write_model(files_path=files_path)

    @staticmethod
    def forecast(
        model_settings: ModelSettings,
        skus: list[str],
        targets: list[str],
        files_path: Path, 
        output_forecast_name: str
    ) -> None:
        """
        Run ForecastModel.forecast for each combination of sku/target passed
        (See module ForecastModel for more information).

        Arguments:
            - model_settings (ModelSettings): Model settings to be used for model instantiation.

            - skus (list[str]): List of SKUs to fit models for.

            - targets (list[str]): List of target names to fit models for.

            - files_path (Path): Path where the featurized CSV files are located and where the
              fitted models will be stored.

            - output_forecast_name (str): Name of the output forecast CSV file to be stored under
              "files_path/data/output/forecasts/{output_forecast_name}/FactForecast.csv".
        """

        featurized_df_schema: dict = {
            'SKU': pl.String,
            'ds': pl.Date,
            'y': pl.Float64
        }

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

        for target in targets:
            logger.debug(
                f'Forecasting for {target}'
            )

            df_in: pl.DataFrame = ETL.ingest(
                files_path = files_path,
                file_name = f'{target}.csv',
                schema_overrides = featurized_df_schema,
                layer = 'featurized'
            )

            target_forecast_df: pl.DataFrame = pl.DataFrame(schema=forecast_df_schema)

            for sku in skus:
                logger.debug(
                    f'Forecasting {sku} - {target}'
                )

                group_df: pl.DataFrame = df_in.filter(pl.col('SKU') == sku).sort('ds')

                if len(group_df) == 0:
                    raise EmptyTableException(f'Grouping gor {sku} - {target} returned empty dataframe')

                model: ForecastModel = ForecastModel(
                    model_settings = model_settings,
                    sku = sku,
                    target = target,
                )

                model.load_model(files_path=files_path)

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
            files_path /
            'data' /
            'output' /
            'forecasts' /
            output_forecast_name
        )

        Path.mkdir(forecasts_path, parents = True, exist_ok = True)

        logger.debug(f'Saving forecast at {forecasts_path / 'FactForecast.csv'}')
        final_forecast_df.write_csv(forecasts_path / f'FactForecast.csv', separator=';')

