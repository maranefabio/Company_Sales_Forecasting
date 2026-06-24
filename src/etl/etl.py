# NOTE: This module is responsible for the ETL process, from the extraction of a SQL Server data to the manipulation
# needed to fit Prophet's tools. As this portfolio project uses mock data, the only part used will
# be the transformation. In production, this module serves as a complete ETL pipeline from the production database
# to the Forecast Model.

import os
import pyodbc
import logging
import polars as pl
import pandas as pd
import datetime as dt
from src.config.config import get_connection_string, get_config_dict
from src.config import config_model
from polars.exceptions import ColumnNotFoundError

logger = logging.getLogger(__name__)


VALID_TARGETS: tuple[str, ...] = config_model.VALID_TARGETS


class EmptyTableException(RuntimeError):
    # Raised when any query executed in ETL.update_local returns an empty table
    pass


class ETL:
    @staticmethod
    def _connect(connection_string: str) -> pyodbc.Connection:
        # Tries to connect to the database, using the connection string generated from the .env file data

        # Args:
        #   - connection_string: SQL Server connection string, generated from the .env file data

        # Returns:
        #   - pyobc.Connection, used to execute SQL commands

        # Raises:
        #   - Exception: Re-raises any error encountered while trying a connection with the given connection string

        config_dict: dict = get_config_dict()
        try:
            logger.info(f'Trying to connect to database {config_dict.get('SERVER')}: {config_dict.get("DATABASE")} ...')
            connection: pyodbc.Connection = pyodbc.connect(connection_string)
            logger.info('Sucessfully connected to database')
            return connection

        except Exception as e:
            raise Exception(f'Unable to connect to database: {e}')

    @staticmethod
    def _dataframe(query: str, query_name: str) -> pl.DataFrame:
        # Generate a Polars dataframe from a SQL query

        # Args:
        #   - query: SQL query loaded from a .sql file stored in "base_path/queries"
        #   - query_name: Name of the file (String before .sql)

        # Returns:
        #   - Polars dataframe generated from the SQL query

        # Raises:
        #   - Exception: Re-raises any error encountered while executing the query

        connection: pyodbc.Connection = ETL._connect(get_connection_string())
        try:
            df: pl.DataFrame = pl.read_database(connection=connection, query=query)
            return df
        except Exception as e:
            raise Exception(f'Error reading {query_name} from {get_config_dict().get("SERVER")}: {e}')

    @staticmethod
    def update_local(base_path: str) -> None:
        # Update the local CSV files containing the ASP and QT time series fact tables

        # Args:
        #   - base_path: Root directory where local files are stored

        # Raises:
        #   - EmptyTableException: Raised when any of the queries returns an empty table
        #   - Exception: Re-raises any error encountered while updating local files

        config_dict: dict = get_config_dict()
        query_dir: str = os.path.join(base_path, 'queries')
        logger.info('Extracting data from database ...')
        try:
            for query_file in os.listdir(query_dir):
                with open(os.path.join(query_dir, query_file), 'r') as file:
                    logger.info(
                        f'Processing query file: {query_file} from'
                        f' {config_dict.get("SERVER")}:{config_dict.get("DATABASE")}'
                    )
                    sql_query: str = file.read()
                    query_name: str = f'{query_file}'.replace('.sql', '')
                    df: pl.DataFrame = ETL._dataframe(sql_query, query_name)

                    if len(df) == 0:
                        raise EmptyTableException(f'Query file {query_file} is empty')


                    df.write_csv(os.path.join(base_path, 'csv', f'{query_name}.csv'), separator=';')
        except Exception as e:
            raise Exception(f'Error updating {query_name}: {e}')

    @staticmethod
    def prepare_dataframe(
            df: pd.DataFrame,
            x: str,
            y: str,
            date_cut: str | None = None,
            exclude_materials: list | None = None,
            single_material: str | None = None
    ) -> pd.DataFrame:
        # Transformation layer. Manipulates the Pandas (following Prophet compatibility) dataframes
        # loaded from the queries to better fit the Prophet conventions

        # Args:
        #   - df: Raw fact table dataframe containing the time series of
        #     QT or ASP, or both of them joined
        #   - y: Target column (ASP or QT)
        #   - date_cut: Date limit for model training. It will be fitted on a dateset where ds < date_cut
        #   - exclude_materials: Materials/SKUs to exclude from the model. Useful when the dataframe contains
        #     more materials than needed, or to apply the model to a limited set of them
        #   - single_material: A material identifier, when the user wants to apply the model to a single one.
        #     Overrides exclude_materials.

        # Returns:
        #   Treated Pandas dataframe containing the Prophet convention columns (ds, y). The day will
        #   be automatically set to 1 for each row date.

        # Raises:
        #   - Exception: Re-raises any error encountered while treating dataframe.

        if y not in VALID_TARGETS:
            raise ValueError(f'Invalid target "{y}" not in {VALID_TARGETS}')


        needed_columns: list[str] = [x, "Y", "M"]
        if not pd.Index(needed_columns).isin(df.columns).all():
            missing_columns: list[str] = list(set(needed_columns) - set(df.columns))
            raise ColumnNotFoundError(f'Column(s) "{missing_columns}" missing')

        if single_material is not None:
            df = df[df[x] == single_material]

        if single_material is None and exclude_materials is not None:
            df = df[~df[x].isin(exclude_materials)]

        if date_cut is not None:
            try:
                df = df[df['YearMonth'] < date_cut]
            except Exception as e:
                raise Exception(f'Error applying date limiting in dataframe: {e}')

        try:
            df = df.groupby([
                x,
                'Y',
                'M'
            ]).agg({
                f'{y}': 'sum',
            }).reset_index()

            df['ds'] = pd.to_datetime({
                'year': df['Y'],
                'month': df['M'],
                'day': 1
            })
        except Exception as e:
            raise Exception(f'Error while applying transformations in dataframe: {e}')

        final_df: pd.DataFrame = pd.DataFrame()
        if y == 'ASP':
            final_df = df
            final_df = final_df.rename(columns={y: 'y'}).sort_values('ds', ascending=True)
            final_df = final_df[[x, 'ds', 'y']]

        if y == 'QT':
            final_df: pd.DataFrame = df.rename(columns={y: 'y'}).sort_values('ds', ascending=True)
            final_df = final_df[[x, 'ds', 'y']]

        return final_df