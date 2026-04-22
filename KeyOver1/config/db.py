# Configuración y fábrica de conexiones a PostgreSQL.
# Configurazione e factory di connessione a PostgreSQL.

import psycopg2
from psycopg2 import Error
from sqlalchemy import create_engine

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

_DB_URL = (
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**DB_CONFIG)
)


def get_connection():
    # Abre una conexión a PostgreSQL. Devuelve None si falla.
    # Apre una connessione a PostgreSQL. Restituisce None in caso di errore.
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"[DB] Errore di connessione a PostgreSQL: {e}")
        return None


def get_engine():
    return create_engine(_DB_URL)
