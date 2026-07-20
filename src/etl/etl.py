# Local ETL pipeline extracting SKU-level ASP/QT fact data from SQL Server into CSV files, then
# cleaning and featurizing it into Prophet-ready dataframes.

# The module "ETL" adds:
# - A thin "extract" step running a SQL query file against the source database and persisting the
#   raw result as a semicolon-separated CSV.
# - An "ingest" step for (re)loading any previously persisted CSV layer back into a polars DataFrame.
# - A "clean" step aggregating raw fact-table rows to monthly granularity per SKU.
# - A "featurize" step turning monthly-cleaned data into one Prophet-convention ("SKU", "ds", "y")
#   dataframe per requested target (e.g. ASP, QT).

# The typical workflow is:
#   >>> ETL.extract(
#   ...     base_path=base_path, database_settings=database_settings,
#   ...     query_file='fact_sales.sql', store_raw_as='FactSalesAct.csv', output_path=output_path
#   ... )
#   >>> cleaned_df = ETL.clean(
#   ...     raw_data_df=None, output_path=output_path, raw_file_name='FactSalesAct.csv',
#   ...     store_as='FactSalesAct.csv', schema_overrides=None
#   ... )
#   >>> asp_df, qt_df = ETL.featurize(
#   ...     cleaned_data_df=cleaned_df, output_path=output_path, targets=['ASP', 'QT'], store=True
#   ... )

#   Each static method reads/writes its own CSV layer under "output_path/data/{raw,cleaned,featurized}",
#   so steps can also be run independently (e.g. "featurize" re-ingests the cleaned layer from disk
#   when "cleaned_data_df" is not supplied directly).

import logging
import polars as pl
import pandas as pd
from pathlib import Path
from src.etl import Database
from src.settings.database import DatabaseSettings
from polars.exceptions import ColumnNotFoundError

# Instantiating logger for the file
logger = logging.getLogger(__name__)


class EmptyTableException(RuntimeError):
    # Raised when a query, an ingested CSV file, or a per-target featurization step in "ETL"
    # returns/produces a table with zero rows.
    pass


class ETL:
    # Collection of static ETL steps (extract -> ingest -> clean -> featurize) that move data from
    # the source database into local CSV layers and, ultimately, into Prophet-ready dataframes.

    @staticmethod
    def extract(
        base_path: Path,
        database_settings: DatabaseSettings,
        query_file: str,
        store_raw_as: str,
        output_path: Path,
    ) -> None:
        # Run a single SQL query file against the source database and persist the result as a raw CSV.

        # Args:
        #   - base_path: Root directory used to locate the query file, expected at
        #     "base_path/src/files/queries/{query_file}".
        #   - database_settings: Settings used to open the database connection.
        #   - query_file: Filename of the SQL query file (under "base_path/src/files/queries/") to
        #     execute against the database.
        #   - store_raw_as: Filename under which the raw query result is saved as CSV.
        #   - output_path: Root directory under which "data/raw/{store_raw_as}" is written.

        # Raises:
        #   - EmptyTableException: Raised when the query returns an empty table (zero rows).
        #   - Exception: Re-raises any error encountered while connecting, reading the query file,
        #     querying, or writing the CSV.

        logger.debug(f'Extracting data from {database_settings.server}:{database_settings.database}')

        db: Database = Database(database_settings)

        with db.connect() as connection:
            try:
                with open(base_path / 'src' / 'files' / 'queries' / query_file, 'r') as file:
                    logger.debug(f'Processing query file {query_file}')

                    sql_query: str = file.read()

                    df: pl.DataFrame = pl.read_database(connection=connection, query=sql_query)

                    if len(df) == 0:
                        raise EmptyTableException(f'Query {query_file} returned empty table')

                    df.write_csv(output_path / 'data' / 'raw' / store_raw_as,  separator=';')

            except Exception as e:
                logger.error(f'Error updating local files: {e}')
                raise

    @staticmethod
    def ingest(
        output_path: Path,
        query: str,
        layer: str,
        schema_overrides: dict | None,
    ) -> pl.DataFrame:
        # Load a previously persisted CSV layer from local storage into a polars DataFrame.

        # Note:
        #   Despite its name, "query" is a CSV filename (e.g. "FactSalesAct.csv"), not a SQL query
        #   string — this method only reads from local disk, it does not hit the database.

        # Args:
        #   - output_path: Root directory containing the "data/{layer}/" folder to read from.
        #   - query: Filename (including extension) of the semicolon-separated CSV file to read,
        #     located under "output_path/data/{layer}/".
        #   - layer: Pipeline layer/subfolder to read from (e.g. "raw", "cleaned"). Required.
        #   - schema_overrides: Optional polars dtype override dict, forwarded to "pl.read_csv" to
        #     force specific column types.

        # Returns:
        #   The loaded polars DataFrame.

        # Raises:
        #   - Exception: Raised when "layer" is None, or re-raised from any error encountered while
        #     reading the CSV file.
        #   - EmptyTableException: Raised when the loaded CSV contains zero rows.

        if layer is None:
            raise Exception('Ingestion layer not specified')

        logger.debug(f'Ingesting data from data/{layer}/{query}')

        try:
            df: pl.DataFrame = pl.read_csv(
                output_path / 'data' / layer / f'{query}',
                schema_overrides=schema_overrides,
                separator=';'
            )

            if len(df) == 0:
                raise EmptyTableException(f'Ingestion returned empty table for data/{layer}/{query}')

            return df

        except Exception as e:
            logger.error(f'Error ingesting {query}: {e}')
            raise


    @staticmethod
    def clean(
        raw_data_df: pl.DataFrame | None,
        output_path: Path | None,
        raw_file_name: str | None,
        store_as: str | None,
        schema_overrides: dict | None,
    ) -> pd.DataFrame:
        # Aggregate raw fact-table rows to monthly granularity per SKU (summing QT, averaging ASP).

        # Args:
        #   - raw_data_df: Raw fact table dataframe (as produced by "extract"/"ingest"), expected to
        #     contain at least "Dt" (date), "SKU", "QT" and "ASP" columns. When None, it is loaded via
        #     "ETL.ingest" from "data/raw/{raw_file_name}".
        #   - output_path: Root directory used both to ingest the raw file (when "raw_data_df" is
        #     None) and to store the cleaned output (when "store_as" is provided).
        #   - raw_file_name: Filename of the raw CSV to ingest when "raw_data_df" is None. Also used
        #     for logging regardless.
        #   - store_as: Filename under which the cleaned data is saved as CSV under "data/cleaned/".
        #     When None, the cleaned data is only returned, not persisted.
        #   - schema_overrides: Optional polars dtype override dict, forwarded to "ETL.ingest" when
        #     "raw_data_df" is None.

        # Returns:
        #   A pandas DataFrame with one row per ("SKU", "YearMonth"), containing "SKU", "YearMonth",
        #   "Y" (year), "M" (month), the summed "QT" and the averaged "ASP".

        # Raises:
        #   - Exception: Re-raises any error encountered while deriving the date parts or aggregating
        #     the data (e.g. if "raw_data_df" is missing an expected column).

        logger.debug(f'Cleaning raw data from data/raw/{raw_file_name}')

        if raw_data_df is None:
            raw_data_df: pl.DataFrame = ETL.ingest(
                output_path = output_path,
                query = raw_file_name,
                schema_overrides = schema_overrides,
                layer = 'raw'
            )

        try:
            cleaned_data_df = raw_data_df.with_columns(
                pl.col('Dt').dt.year().alias('Y'),
                pl.col('Dt').dt.month().alias('M'),
                pl.col('Dt').dt.strftime("%Y/%m").alias('YearMonth')
            ).group_by([
                'SKU',
                'YearMonth',
                'Y',
                'M',
            ]).agg([
                pl.col('QT').sum().alias('QT'),
                pl.col('ASP').mean().alias('ASP'),
            ])

        except Exception as e:
            logger.error(f'Error cleaning raw data: {e}')
            raise

        if store_as is not None:
            logger.debug(f'Storing cleaned data in data/cleaned/{store_as}')

            cleaned_data_df.write_csv(
                output_path / 'data' / 'cleaned' / f'{store_as}',
                separator=';'
            )

        return cleaned_data_df.to_pandas()

    @staticmethod
    def featurize(
        cleaned_data_df: pl.DataFrame | None,
        output_path: Path,
        targets: list[str],
        store: bool = False,
        date_limits: list | None = None
    ) -> list[pd.DataFrame]:
        # Transformation layer. Builds one Prophet-convention ("SKU", "ds", "y") dataframe per
        # requested target out of monthly-cleaned data.

        # Note:
        #   "date_limits[0]"/"date_limits[1]" are accessed unconditionally, so despite its type hint
        #   and default, "date_limits" must be passed as a two-element list (either element may
        #   individually be None to leave that bound unrestricted) — passing "date_limits=None"
        #   itself will raise a TypeError.

        # Args:
        #   - cleaned_data_df: Monthly-cleaned dataframe (as produced by "clean"), expected to contain
        #     at least "SKU", "Y", "M", "YearMonth" and the columns named in "targets". When None, it
        #     is loaded via "ETL.ingest" from "data/cleaned/FactSalesAct.csv" using a fixed schema.
        #   - output_path: Root directory used both to ingest the cleaned file (when "cleaned_data_df"
        #     is None) and to store featurized output (when "store" is True).
        #   - targets: List of target column names (e.g. ["ASP", "QT"]) to featurize; one output
        #     dataframe is produced per target, in the same order.
        #   - store: Whether to persist each target's featurized dataframe as CSV under
        #     "data/featurized/{target}.csv". Requires "output_path" to be set. Default = False.
        #   - date_limits: Two-element list "[lower, upper]" of inclusive "YearMonth" bounds used to
        #     filter "cleaned_data_df" before featurizing. Either element may be None to leave that
        #     bound unrestricted.

        # Returns:
        #   A list of pandas DataFrames, one per entry in "targets" (same order), each with columns
        #   "SKU", "ds" (first-of-month datetime derived from "Y"/"M") and "y" (renamed from the
        #   target column), sorted by "ds" ascending.

        # Raises:
        #   - ColumnNotFoundError: If "cleaned_data_df" is missing any of "SKU", "Y", "M".
        #   - EmptyTableException: If, after filtering and sorting, a given target's dataframe ends up
        #     empty.
        #   - Exception: If "targets" is None, or wrapping any other error encountered while
        #     featurizing a given target (includes the target name and original error message).

        logger.debug(f'Featurizing {targets} data')

        if cleaned_data_df is None:
            cleaned_data_df = ETL.ingest(
                output_path = output_path,
                query = 'FactSalesAct.csv',
                schema_overrides = {
                    'SKU': pl.String,
                    'YearMonth': pl.String,
                    'Y': pl.Int16,
                    'M': pl.Int8,
                    'QT': pl.Float64,
                    'ASP': pl.Float64
               },
               layer = 'cleaned'
            )

        if date_limits[0] is not None:
            cleaned_data_df = cleaned_data_df.filter(
                pl.col('YearMonth') >= date_limits[0],
            )

        if date_limits[1] is not None:
            cleaned_data_df = cleaned_data_df.filter(
                pl.col('YearMonth') <= date_limits[1],
            )

        needed_columns: list[str] = ['SKU', "Y", "M"]
        if not pd.Index(needed_columns).isin(cleaned_data_df.columns).all():
            missing_columns: list[str] = list(set(needed_columns) - set(cleaned_data_df.columns))
            raise ColumnNotFoundError(f'Column(s) "{missing_columns}" missing')

        try:
            featurized_data_df = cleaned_data_df.select([
                'SKU',
                'Y',
                'M',
                'QT',
                'ASP',
            ])

            featurized_data_df: pd.DataFrame = featurized_data_df.to_pandas().reset_index()

            featurized_data_df['ds'] = pd.to_datetime({
                'year': featurized_data_df['Y'],
                'month': featurized_data_df['M'],
                'day': 1
            })

        except Exception as e:
            logger.error(f'Error while applying transformations in dataframe: {e}')
            raise

        if targets is None:
            logger.error('Failed to specify targets for featurization')
            raise

        featurized_data_df_list: list[pd.DataFrame] = []
        for target in targets:
            try:
                featurized_target_data_df: pd.DataFrame = featurized_data_df.rename(
                    columns={target: 'y'}
                ).sort_values(
                    'ds',
                    ascending=True
                )

                if len(featurized_target_data_df) == 0:
                    raise EmptyTableException(f'Featurization step returned empty table')

                featurized_target_data_df = featurized_target_data_df[['SKU', 'ds', 'y']]

                featurized_data_df_list.append(featurized_target_data_df)

                if store:
                    if output_path is None:
                        logger.error('Argument output_path must be specified for store=True')
                        raise

                    featurized_data_path: Path = output_path / 'data' / 'featurized'
                    Path.mkdir(featurized_data_path, parents=True, exist_ok=True)

                    logger.debug(f'Storing featurized data in data/featurized/{target}.csv')

                    featurized_target_data_df.to_csv(
                        featurized_data_path / f'{target}.csv',
                        sep=';',
                        index=False
                )
            except Exception as e:
                raise Exception(f'Unable to featurize for target: {target}: {e}')

        return featurized_data_df_list