# Pydantic settings model for the source SQL Server database connection, populated from environment
# variables (or a ".env" file) prefixed with "DB_".

# The module "DatabaseSettings" adds:
# - Typed, validated environment-based configuration for the database connection (server, database,
#   credentials, ODBC driver).
# - A ready-to-use ODBC connection string via the "database_connection_string" property, consumed by
#   "Database.connect" (see "src/database.py").

# The typical workflow is:
#   >>> from src.settings.database import database_settings
#   >>> db: Database = Database(database_settings)

# Note:
#   "database_settings" below is a module-level singleton constructed at import time. Importing this
#   module therefore immediately validates the environment/".env" file and raises a pydantic
#   "ValidationError" if any required "DB_*" variable is missing.

from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseSettings):
    # Connection settings for the source SQL Server database, loaded from environment variables
    # prefixed with "DB_" (or from a ".env" file). Unrecognized variables are ignored.

    # Fields:
    #   - server: SQL Server hostname/instance (env var "DB_SERVER").
    #   - database: Target database name (env var "DB_DATABASE").
    #   - uid: SQL Server login username (env var "DB_UID").
    #   - pwd: SQL Server login password (env var "DB_PWD").
    #   - driver: ODBC driver version to use, e.g. "17" for "ODBC Driver 17 for SQL Server"
    #     (env var "DB_DRIVER").

    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix = 'DB_',
        extra='ignore'
    )

    server: str
    database: str
    uid: str
    pwd: str
    driver: str

    @property
    def database_connection_string(self) -> str:
        # Build the ODBC connection string consumed by "pyodbc.connect" (see "Database.connect").

        # Returns:
        #   A formatted ODBC connection string built from this instance's "driver", "server",
        #   "database", "uid" and "pwd", with "TrustServerCertificate=yes" always set.

        return f'''
            DRIVER=ODBC Driver {self.driver} for SQL Server;
            SERVER={self.server};
            DATABASE={self.database};
            UID={self.uid};
            PWD={self.pwd};
            TrustServerCertificate=yes;
        '''

# Module-level singleton, validated against the environment/".env" file at import time.
database_settings = DatabaseSettings()