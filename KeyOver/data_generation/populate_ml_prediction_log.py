import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import Error
from psycopg2.extras import execute_batch

# Añadimos la raíz del proyecto al path para poder importar ml_model.py
# también cuando este archivo se ejecuta directamente desde data_generation/.
#
# Aggiungiamo la radice del progetto al path per poter importare ml_model.py
# anche quando questo file viene eseguito direttamente da data_generation/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MachineLearning.ml_model import load_model, predict_activity_with_model

"""
Este archivo realiza el backfill de la tabla ml_prediction_log.

Su función es recorrer todas las actividades ya almacenadas en activity_log,
aplicar el modelo de machine learning sobre cada una de ellas y guardar
el resultado de la predicción en la tabla ml_prediction_log.

En esta versión, el modelo trabaja con un enfoque de detección de anomalías
basado en el comportamiento habitual de cada usuario.

Questo file esegue il backfill della tabella ml_prediction_log.

La sua funzione è scorrere tutte le attività già memorizzate in activity_log,
applicare il modello di machine learning su ciascuna di esse e salvare
il risultato della previsione nella tabella ml_prediction_log.

In questa versione, il modello lavora con un approccio di rilevamento anomalie
basato sul comportamento abituale di ciascun utente.
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

MODEL_PATH = PROJECT_ROOT / "models" / "activity_model.pkl"


def get_connection():
    """
    Abre una conexión con PostgreSQL utilizando la configuración definida
    en DB_CONFIG. Si falla, devuelve None.

    Apre una connessione a PostgreSQL utilizzando la configurazione definita
    in DB_CONFIG. Se fallisce, restituisce None.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def load_activity_data() -> pd.DataFrame:
    """
    Carga todas las actividades almacenadas en activity_log.

    Devuelve un DataFrame con:
    - activity_log_id
    - user_id
    - element_id
    - entity_id
    - action_id
    - logged_at

    Carica tutte le attività memorizzate in activity_log.

    Restituisce un DataFrame con:
    - activity_log_id
    - user_id
    - element_id
    - entity_id
    - action_id
    - logged_at
    """
    connection = get_connection()
    if connection is None:
        return pd.DataFrame()

    query = """
        SELECT
            activity_log_id,
            user_id,
            element_id,
            entity_id,
            action_id,
            logged_at
        FROM activity_log
        ORDER BY activity_log_id
    """

    try:
        df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        print(f"Errore durante il caricamento di activity_log: {e}")
        return pd.DataFrame()
    finally:
        connection.close()


def build_prediction_rows(df: pd.DataFrame, model_bundle):
    """
    Recorre todas las actividades y construye las filas que se insertarán
    en ml_prediction_log.

    Para cada actividad:
    - llama al modelo correspondiente al usuario
    - obtiene la predicción
    - obtiene el score de anomalía
    - construye una tupla lista para insertar

    Scorre tutte le attività e costruisce le righe che verranno inserite
    in ml_prediction_log.

    Per ogni attività:
    - richiama il modello corrispondente all'utente
    - ottiene la previsione
    - ottiene lo score di anomalia
    - costruisce una tupla pronta per l'inserimento
    """
    rows = []

    for i, row in df.iterrows():
        try:
            prediction, probability = predict_activity_with_model(
                model_bundle=model_bundle,
                user_id=int(row["user_id"]),
                element_id=int(row["element_id"]),
                entity_id=int(row["entity_id"]),
                action_id=int(row["action_id"]),
                logged_at=row["logged_at"]
            )

            rows.append((
                int(row["activity_log_id"]),
                int(row["user_id"]),
                int(row["element_id"]),
                int(row["entity_id"]),
                int(row["action_id"]),
                bool(prediction),
                float(probability) if probability is not None else None
            ))

        except Exception as e:
            print(
                f"Errore nella previsione per activity_log_id={int(row['activity_log_id'])}: {e}"
            )

        if (i + 1) % 1000 == 0:
            print(f"Previsioni preparate: {i + 1}")

    return rows


def clear_ml_prediction_log(cursor):
    """
    Vacía completamente la tabla ml_prediction_log y reinicia su identidad.

    Svuota completamente la tabella ml_prediction_log e riavvia l'identità.
    """
    query = "TRUNCATE TABLE ml_prediction_log RESTART IDENTITY;"
    cursor.execute(query)


def insert_prediction_rows(cursor, rows: list[tuple]):
    """
    Inserta por lotes todas las predicciones generadas en ml_prediction_log.

    Inserisce in batch tutte le previsioni generate in ml_prediction_log.
    """
    query = """
        INSERT INTO ml_prediction_log (
            activity_log_id,
            user_id,
            element_id,
            entity_id,
            action_id,
            prediction,
            anomaly_probability,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    """
    execute_batch(cursor, query, rows, page_size=1000)


def main():
    """
    Flujo principal del backfill:
    1. Carga el bundle de modelos desde disco.
    2. Carga las actividades desde activity_log.
    3. Genera las predicciones.
    4. Vacía ml_prediction_log.
    5. Inserta las nuevas predicciones.

    Flusso principale del backfill:
    1. Carica il bundle dei modelli dal disco.
    2. Carica le attività da activity_log.
    3. Genera le previsioni.
    4. Svuota ml_prediction_log.
    5. Inserisce le nuove previsioni.
    """
    print("Caricamento del modello...")
    model_bundle = load_model(str(MODEL_PATH))

    print("Caricamento di activity_log...")
    df = load_activity_data()

    if df.empty:
        print("Non ci sono attività da elaborare.")
        return

    print(f"Attività caricate: {df.shape[0]}")

    print("Generazione delle previsioni...")
    prediction_rows = build_prediction_rows(df, model_bundle)

    if not prediction_rows:
        print("Nessuna previsione valida da inserire.")
        return

    connection = get_connection()
    if connection is None:
        return

    cursor = connection.cursor()

    try:
        print("Svuotamento di ml_prediction_log...")
        clear_ml_prediction_log(cursor)

        print("Inserimento delle previsioni in ml_prediction_log...")
        insert_prediction_rows(cursor, prediction_rows)

        connection.commit()
        print("Backfill completato correttamente.")

    except Exception as e:
        connection.rollback()
        print(f"Errore durante il backfill: {e}")

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()