"""
Thin wrapper around pyodbc providing retrying, context-managed database connections and cursors.

The module "Database" adds managed connections and cursors, with cursors automatically committing on success and
rolling back on error.

Since it is context-managed, the typical use require a "with" block.
    >>> db: Database = Database(database_settings)
    >>> with db.cursor() as cursor:
            cursor.execute(<QUERY>)

    >>> with db.connect() as connection:
            df = pl.read_sql(<QUERY>, connection)


where "database_settings" is a "DatabaseSettings" instance holding the ODBC connection string
plus the server and database names (used for logging only). "connect" can also be used directly
when only a raw connection (not a cursor) is needed.
"""

import pyodbc
import logging
import contextlib
from src.settings.database import DatabaseSettings
from collections.abc import Generator

logger = logging.getLogger(__name__)

class Database:
    """
    Wraps pyodbc connections for a single configured database, adding retry-on-failure logic and
    a startup connectivity check.

    Arguments:
        - database_settings (DatabaseSettings): Settings used to open the source database connection during
          "EXTRACT".

    Attributes:
        - database_settings (DatabaseSettings)
    """

    def __init__(self, database_settings: DatabaseSettings) -> None:
        self.database_settings = database_settings

    @contextlib.contextmanager
    def connect(self, max_retries: int=3) -> Generator[pyodbc.Connection, None]:
        """
        Context manager yielding a live, validated pyodbc connection, retrying on failure.

        On each attempt, a connection is opened using
        "database_settings.database_connection_string" with a 5-second timeout, then validated
        with a "SELECT 1" query (via a throwaway cursor) before being yielded to the caller. If
        "pyodbc.connect" or the validation query raises "pyodbc.Error", the attempt is logged and
        retried, up to "max_retries" additional times, before the error is re-raised.

        Note:
            - The connection is always closed in a "finally" clause on exit from each attempt,
              including the successful one — i.e. the connection yielded to the caller is closed as
              soon as the "with" block using it exits, so it should not be retained past that block.

        Arguments:
          - max_retries: Maximum number of additional connection attempts after the first failed
            one, before giving up and re-raising the underlying "pyodbc.Error". Default = 3.

        Yields:
          - pyodbc.Connection: an open connection that has passed validation check.

        Raises:
          - pyodbc.Error: Re-raised once "max_retries" has been exceeded.
        """

        connection: pyodbc.Connection | None = None
        connected: bool = False

        attempt: int = 1
        while True:
            logger.info(
                f'Attempting connection to database {self.database_settings.server}: '
                f'{self.database_settings.database}. Attempt nº{attempt}'
            )

            try:
                connection = pyodbc.connect(self.database_settings.database_connection_string, timeout=5)
                temp_cursor: pyodbc.Cursor | None = connection.cursor()

                temp_cursor.execute('SELECT 1')
                temp_cursor.close()

                logger.info('Sucessfully connected to database')
                connected = True

                yield connection

                break

            except pyodbc.Error as e:
                logger.error(f"Error connecting to database: {e}")
                attempt += 1

                if attempt > max_retries:
                    logger.critical(f'Unable to connect to database {e}')
                    raise

            finally:
                if connected:
                    logger.info('Closing connection to database')
                    connection.close()

    @contextlib.contextmanager
    def cursor(self) -> Generator[pyodbc.Cursor]:
        """
        Context manager yielding a cursor bound to a freshly opened connection (via "connect").

        On successful exit of the "with" block, the underlying connection is committed. If a
        "pyodbc.Error" is raised inside the "with" block, the connection is rolled back and the
        error is re-raised. The connection itself (opened via "connect") is closed automatically
        once this context manager exits, per "connect"'s behavior.

        Yields:
            - pyodbc.Cursor: a cursor for executing statements against the database.

        Raises:
            - pyodbc.Error: Re-raised, after rollback, if raised inside the "with" block.
        """

        with self.connect() as connection:
            try:
                cursor: pyodbc.Cursor = connection.cursor()
                yield cursor
                connection.commit()

            except pyodbc.Error as e:
                logger.error(f"Error generating cursor: {e}")
                connection.rollback()
                raise