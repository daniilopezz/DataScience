import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import Error
from psycopg2.extras import execute_batch

# Añadimos la raíz del proyecto al path para poder importar módulos
# también cuando este archivo se ejecuta directamente desde data_generation/.
#
# Aggiungiamo la radice del progetto al path per poter importare moduli
# anche quando questo file viene eseguito direttamente da data_generation/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MachineLearning.ml_model import load_model, predict_activity_with_model
from security.anomaly_guard import get_session_anomaly_threshold

"""
Este archivo realiza el backfill de la tabla ml_prediction_log.

Su función es:
- recorrer todas las actividades ya almacenadas en activity_log
- aplicar el modelo de machine learning sobre cada una de ellas
- guardar el resultado de la predicción en la tabla ml_prediction_log

En esta versión, el modelo trabaja con un enfoque de detección de anomalías
basado en el comportamiento habitual de cada usuario y genera una
anomaly_probability que se interpreta como coste de la operación.

Además, como la tabla ml_prediction_log ahora incluye información de sesión,
este script rellena también:
- login_log_id
- session_cumulative_cost
- session_threshold
- threshold_exceeded

Dado que se trata de un backfill histórico y no de una sesión en tiempo real,
no es posible reconstruir con fiabilidad el coste acumulado real por sesión
a partir de activity_log únicamente. Por eso:
- login_log_id se deja en NULL
- session_cumulative_cost se iguala al coste individual de la operación
- threshold_exceeded se calcula respecto a ese coste individual

Questo file esegue il backfill della tabella ml_prediction_log.

La sua funzione è:
- scorrere tutte le attività già memorizzate in activity_log
- applicare il modello di machine learning su ciascuna di esse
- salvare il risultato della previsione nella tabella ml_prediction_log

In questa versione, il modello lavora con un approccio di rilevamento anomalie
basato sul comportamento abituale di ciascun utente e genera una
anomaly_probability che viene interpretata come costo dell'operazione.

Inoltre, siccome la tabella ml_prediction_log ora include anche informazioni
di sessione, questo script compila anche:
- login_log_id
- session_cumulative_cost
- session_threshold
- threshold_exceeded

Poiché si tratta di un backfill storico e non di una sessione in tempo reale,
non è possibile ricostruire in modo affidabile il costo cumulato reale di
sessione partendo solo da activity_log. Per questo:
- login_log_id viene lasciato a NULL
- session_cumulative_cost viene impostato uguale al costo singolo
- threshold_exceeded viene calcolato rispetto a quel costo singolo
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
    - obtiene la anomaly_probability
    - rellena los campos de sesión con una aproximación coherente
    - construye una tupla lista para insertar

    Scorre tutte le attività e costruisce le righe che verranno inserite
    in ml_prediction_log.

    Per ogni attività:
    - richiama il modello corrispondente all'utente
    - ottiene la previsione
    - ottiene la anomaly_probability
    - compila i campi di sessione con un'approssimazione coerente
    - costruisce una tupla pronta per l'inserimento
    """
    rows = []
    session_threshold = float(get_session_anomaly_threshold())

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

            operation_cost = float(probability)
            session_cumulative_cost = operation_cost
            threshold_exceeded = session_cumulative_cost >= session_threshold

            rows.append((
                int(row["activity_log_id"]),   # activity_log_id
                None,                          # login_log_id -> no reconstruible de forma fiable aquí
                int(row["user_id"]),
                int(row["element_id"]),
                int(row["entity_id"]),
                int(row["action_id"]),
                bool(prediction),
                operation_cost,
                session_cumulative_cost,
                session_threshold,
                bool(threshold_exceeded)
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
            login_log_id,
            user_id,
            element_id,
            entity_id,
            action_id,
            prediction,
            anomaly_probability,
            session_cumulative_cost,
            session_threshold,
            threshold_exceeded,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
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