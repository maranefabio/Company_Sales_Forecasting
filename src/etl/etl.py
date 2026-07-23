"""
Local ETL pipeline extracting SKU-level ASP/QT fact data from SQL Server into CSV files, then
cleaning and featurizing it into Prophet-ready dataframes.

The module "ETL" adds:
    - A thin "extract" step running a SQL query file against the source database and persisting the
      raw result as a semicolon-separated CSV.

    - An "ingest" step for (re)loading any previously persisted CSV layer back into a polars DataFrame.

    - A "clean" step aggregating raw fact-table rows to monthly granularity per SKU.

    - A "featurize" step turning monthly-cleaned data into one Prophet-convention ("SKU", "ds", "y")
      dataframe per requested target (e.g. ASP, QT).

The typical workflow is:
    >>> ETL.extract(
            base_path=base_path, database_settings=database_settings,
            query_file='fact_sales.sql', store_raw_as='FactSalesAct.csv', output_path=output_path
        )
    >>> cleaned_df = ETL.clean(
            raw_data_df=None, output_path=output_path, raw_file_name='FactSalesAct.csv',
            store_as='FactSalesAct.csv', schema_overrides=None
        )
    >>> asp_df, qt_df = ETL.featurize(
            cleaned_data_df=cleaned_df, output_path=output_path, targets=['ASP', 'QT'], store=True
        )

 Each static method reads/writes its own CSV layer under "output_path/data/{raw,cleaned,featurized}",
 so steps can also be run independently (e.g. "featurize" re-ingests the cleaned layer from disk
 when "cleaned_data_df" is not supplied directly).
"""

import logging
import polars as pl
import pandas as pd
import datetime as dt
from pathlib import Path
from src.etl import Database
from src.settings.database import DatabaseSettings
from polars.exceptions import ColumnNotFoundError

logger = logging.getLogger(__name__)

class ETL:
    """
    Collection of static ETL steps (extract -> ingest -> clean -> featurize) that move data from
    the source database into local CSV layers and, ultimately, into Prophet-ready dataframes.
    """

    @staticmethod
    def extract(
        base_path: Path,
        database_settings: DatabaseSettings,
        query_file: str,
        store_raw_as: str,
        files_path: Path,
    ) -> None:
        """
        Run a single SQL query file against the source database and persist the result as a raw CSV.

        Arguments:
            - base_path (Path): Root directory used to locate the query file, expected at
              "base_path/src/files/queries/{query_file}".

            - database_settings (DatabaseSettings): Settings used to open the database connection.

            - query_file (str): Filename of the SQL query file (under "base_path/src/files/queries/") to
              execute against the database.

            - store_raw_as (str): Filename under which the raw query result is saved as CSV.

            - files_path (Path): Root directory under which "data/raw/{store_raw_as}" is written.

        Raises:
            - EmptyTableException: Raised when the query returns an empty table (zero rows).

            - Exception: Re-raises any error encountered while connecting, reading the query file,
              querying, or writing the CSV.
        
        Returns:
            None
        """

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

                    df.write_csv(files_path / 'data' / 'raw' / store_raw_as,  separator=';')

            except Exception as e:
                logger.error(f'Error updating local files: {e}')
                raise

    @staticmethod
    def ingest(
        files_path: Path,
        file_name: str,
        layer: str,
        schema_overrides: dict[str, pl.DataType] | None,
    ) -> pl.DataFrame:
        """
        Load a previously persisted CSV layer from local storage into a polars DataFrame.

        Arguments:
            - files_path (Path): Root directory containing the "data/{layer}/" folder to read from.

            - file_name (str): Filename (including extension) of the semicolon-separated CSV file to read,
              located under "files_path/data/{layer}/".

            - layer (str): Pipeline layer/subfolder to read from (e.g. "raw", "cleaned"). Required.

            - schema_overrides (dict[str, pl.DataType]): Optional polars dtype override dict, forwarded to "pl.read_csv" to
              force specific column types.

        Returns:
            The loaded polars DataFrame.

        Raises:
            - Exception: Raised when "layer" is None, or re-raised from any error encountered while
              reading the CSV file.
            - EmptyTableException: Raised when the loaded CSV contains zero rows.
        """

        if layer is None:
            raise Exception('Ingestion layer not specified')

        logger.debug(f'Ingesting data from data/{layer}/{file_name}')

        try:
            df: pl.DataFrame = pl.read_csv(
                files_path / 'data' / layer / f'{file_name}',
                schema_overrides=schema_overrides,
                separator=';'
            )

            if len(df) == 0:
                raise EmptyTableException(f'Ingestion returned empty table for data/{layer}/{file_name}')

            return df

        except Exception as e:
            logger.error(f'Error ingesting {file_name}: {e}')
            raise


    @staticmethod
    def clean(
        raw_data_df: pl.DataFrame | None = None,
        files_path: Path | None = None,
        raw_file_name: str | None = None,
        store: bool = False,
        schema_overrides: dict[str, pl.DataType] | None = None,
        date_limits: list[dt.date] | None = None,
    ) -> pl.DataFrame:
        """
        Clean raw fact-table data to fit date boundaries and expected schema.

        Arguments:
            - raw_data_df (pl.DataFrame | None): Raw fact-table dataframe (as produced by "ETL.extract"). When None, it
              is loaded via "ETL.ingest" from "data/raw/{raw_file_name}" using the provided
              "schema_overrides".

            - files_path (Path | None): Root directory used both to ingest the raw file (when "raw_data_df" is None) and
              to store cleaned output (when "store" is True).

            - raw_file_name (str | None): Name of the file to ingest when "raw_data_df" is None.

            - store (bool): Whether to persist the cleaned dataframe as CSV under 
              "data/cleaned/FactSalesAct.csv". Requires "files_path" to be set.
              Default = False.

            - schema_overrides (dict[str, pl.DataType] | None): Optional polars dtype override dict, forwarded to "ETL.ingest" when
              "raw_data_df" is None.

            - date_limits (list[dt.date] | None): Two-element list "[lower, upper]" of inclusive "Dt" bounds used to filter.

        Returns:
            A polars DataFrame with cleaned data

        Raises:
            - Exception: Re-raises any error encountered while deriving the date parts or aggregating
              the data (e.g. if "raw_data_df" is missing an expected column).
        """
        
        logger.debug(f'Cleaning raw data from data/raw/{raw_file_name}')

        if raw_data_df is None:
            if (files_path is None) | (raw_file_name is None):
                raise ValueError('Arguments files_path and raw_file_name must be specified when raw_data_df is None')

            logger.debug(f'Cleaning step ingesting data')  
            raw_data_df: pl.DataFrame = ETL.ingest(
                files_path = files_path,
                file_name = raw_file_name,
                schema_overrides = schema_overrides,
                layer = 'raw'
            )

        try:
            cleaned_data_df: pl.DataFrame = raw_data_df
            if date_limits[0] is not None:
                cleaned_data_df = cleaned_data_df.filter(pl.col('Dt') >= date_limits[0])

            if date_limits[1] is not None:
                cleaned_data_df = cleaned_data_df.filter(pl.col('Dt') <= date_limits[1])

            cleaned_data_df = raw_data_df.with_columns(
                pl.col('Dt').dt.year().alias('Y'),
                pl.col('Dt').dt.month().alias('M'),
                pl.col('Dt').dt.strftime("%Y/%m").alias('YearMonth'),
            ).filter(
                pl.col('QT') >= 0,
                pl.col('ASP') >= 0,
                pl.col('Dt') <= dt.date.today()
            )

        except Exception as e:
            logger.error(f'Error cleaning raw data: {e}')
            raise

        if store:
            logger.debug(f'Storing cleaned data in data/cleaned/FactSalesAct.csv')
            cleaned_data_df.write_csv(
                files_path / 'data' / 'cleaned' / f'FactSalesAct.csv',
                separator=';'
            )

        return cleaned_data_df

    @staticmethod
    def featurize(
        target: str,
        cleaned_data_df: pl.DataFrame | None = None,
        files_path: Path | None = None,
        file_name: str | None = None,
        store: bool = False,
    ) -> pd.DataFrame:
        """
        Transformation layer. Builds a Prophet-convention ("SKU", "ds", "y") dataframe for the requested
        target out of cleaned data, aggregating it to monthly-granularity.

        Arguments:
            - target (str): Name of the target column to featurize.

            - cleaned_data_df (pl.DataFrame | None): Monthly-cleaned dataframe (as produced by "clean"), expected to contain
              at least "SKU", "Y", "M", "YearMonth" and the target column. When None, it
              is loaded via "ETL.ingest" from "data/cleaned/FactSalesAct.csv" using a fixed schema.

            - files_path (Path | None): Root directory used both to ingest the cleaned file (when "cleaned_data_df"
              is None) and to store featurized output (when "store" is True).

            - file_name (str | None): Name of the file to ingest when "cleaned_data_df" is None.

            - store (bool): Whether to persist each target's featurized dataframe as CSV under
              "data/featurized/{target}.csv". Requires "files_path" to be set. 
              Default = False.
        
        Returns:
            A pandas DataFrame with columns
            "SKU", "ds" (first-of-month datetime derived from "Y"/"M") and "y" (renamed from the
            target column), sorted by "ds" ascending.

        Raises:
            - ColumnNotFoundError: If "cleaned_data_df" is missing any of "SKU", "Y", "M".
            - EmptyTableException: If, after filtering and sorting, a given target's dataframe ends up
              empty.
            - Exception: If "target" is None, or wrapping any other error encountered while
              featurizing a given target (includes the target name and original error message).
        """
        
        logger.debug(f'Featurizing {target} data')

        if cleaned_data_df is None:
            if (files_path is None) | (file_name is None):
                raise ValueError('Arguments files_path and file_name must be specified for cleaned_data_df=None')
            cleaned_data_df = ETL.ingest(
                files_path = files_path,
                file_name = file_name,
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

        needed_columns: list[str] = ['SKU', 'Y', 'M']
        if not pd.Index(needed_columns).isin(cleaned_data_df.columns).all():
            missing_columns: list[str] = list(set(needed_columns) - set(cleaned_data_df.columns))
            raise ColumnNotFoundError(f'Column(s) "{missing_columns}" missing')

        try:
            aggregation_map: dict[str, pl.Expr] = {
                'QT': pl.col('QT').sum().alias('QT'),
                'ASP': pl.col('ASP').mean().alias('ASP')
            }

            featurized_data_df = cleaned_data_df.group_by(needed_columns).agg(
                aggregation_map[target]
            ).select([
                'SKU', 'Y', 'M', target
            ]).sort(
                by = ['Y', 'M'],
                descending = False
            ).rename({target: 'y'})

            featurized_data_df: pd.DataFrame = featurized_data_df.to_pandas().reset_index()

            featurized_data_df['ds'] = pd.to_datetime({
                'year': featurized_data_df['Y'],
                'month': featurized_data_df['M'],
                'day': 1
            })

        except Exception as e:
            logger.error(f'Error while applying transformations in dataframe: {e}')
            raise

        try:
            if len(featurized_data_df) == 0:
                raise EmptyTableException(f'Featurization step returned empty table')

            featurized_data_df = featurized_data_df[['SKU', 'ds', 'y']]

            if store:
                if files_path is None:
                    raise ValueError('Argument files_path must be specified when store=True')

                featurized_data_path: Path = files_path / 'data' / 'featurized'
                Path.mkdir(featurized_data_path, parents=True, exist_ok=True)

                logger.debug(f'Storing featurized data in data/featurized/{target}.csv')

                featurized_data_df.to_csv(
                    featurized_data_path / f'{target}.csv',
                    sep=';',
                    index=False
                )
            
        except Exception as e:
            raise Exception(f'Unable to featurize for target: {target}: {e}')

        return featurized_data_df


class EmptyTableException(RuntimeError):
    """
    Raised when a query, an ingested CSV file, or a per-target featurization step in "ETL"
    returns/produces a table with zero rows.
    """
    pass
