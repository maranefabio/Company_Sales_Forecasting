import os
from dotenv import load_dotenv

def get_config_dict() -> dict:
    load_dotenv()
    config: dict = {
        'DRIVER': os.getenv('DRIVER'),
        'SERVER': os.getenv('SERVER'),    
        'DATABASE': os.getenv('DATABASE'),
        'UID': os.getenv('UID'),
        'PWD': os.getenv('PWD')
    }

    return config

def get_connection_string() -> str:
    config: dict = get_config_dict()

    connection_string: str = f'''
        DRIVER={config.get('DRIVER')};
        SERVER={config.get('SERVER')};
        DATABASE={config.get('DATABASE')};
        UID={config.get('UID')};
        PWD={config.get('PWD')};
    '''

    return connection_string
