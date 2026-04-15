import pandas as pd
import psycopg2
from psycopg2 import Error
from psycopg2.extras import execute_batch

from ml_model import load_model, predict_activity_with_model

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

MODEL_PATH = "activity_model.pkl"


def get_connection():
    """
    Abre una conexión con la base de datos PostgreSQL utilizando la configuración
    definida en DB_CONFIG.
    Si la conexión falla, muestra un mensaje de error y devuelve None.

    Apre una connessione al database PostgreSQL utilizzando la configurazione
    definita in DB_CONFIG.
    Se la connessione fallisce, mostra un messaggio di errore e restituisce None.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def load_activity_data() -> pd.DataFrame:
    """
    Carga todas las actividades almacenadas en la tabla activity_log.
    La consulta recupera:
    - activity_log_id
    - user_id
    - element_id
    - entity_id
    - action_id
    - logged_at

    Los registros se devuelven ordenados por activity_log_id.
    Si ocurre un error o no se puede abrir la conexión, devuelve un DataFrame vacío.


    Carica tutte le attività memorizzate nella tabella activity_log.
    La query recupera:
    - activity_log_id
    - user_id
    - element_id
    - entity_id
    - action_id
    - logged_at

    I record vengono restituiti ordinati per activity_log_id.
    Se si verifica un errore o non è possibile aprire la connessione, restituisce un DataFrame vuoto.
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


def build_prediction_rows(df: pd.DataFrame, model):
    """
    Recorre todas las filas del DataFrame de actividades y genera la estructura
    de datos que posteriormente será insertada en la tabla ml_prediction_log.
    Para cada actividad:
    - llama al modelo de machine learning
    - obtiene la predicción de anomalía
    - obtiene la probabilidad asociada
    - construye una tupla con los valores necesarios para la inserción

    Además, cada 1000 registros procesados muestra un mensaje informativo
    por pantalla.

    Scorre tutte le righe del DataFrame delle attività e genera la struttura
    dei dati che successivamente verrà inserita nella tabella ml_prediction_log.
    Per ogni attività:
    - richiama il modello di machine learning
    - ottiene la previsione di anomalia
    - ottiene la probabilità associata
    - costruisce una tupla con i valori necessari per l'inserimento

    Inoltre, ogni 1000 record elaborati mostra un messaggio informativo
    a schermo.
    """
    rows = []

    for i, row in df.iterrows():
        prediction, probability = predict_activity_with_model(
            model=model,
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

        if (i + 1) % 1000 == 0:
            print(f"Previsioni preparate: {i + 1}")

    return rows


def clear_ml_prediction_log(cursor):
    """
    Vacía completamente la tabla ml_prediction_log y reinicia su contador
    de identidad.
    Esto permite reconstruir el contenido desde cero antes de insertar
    las nuevas predicciones generadas por el modelo.

    Svuota completamente la tabella ml_prediction_log e riavvia il relativo
    contatore di identità.
    Questo permette di ricostruire il contenuto da zero prima di inserire
    le nuove previsioni generate dal modello.
    """
    query = "TRUNCATE TABLE ml_prediction_log RESTART IDENTITY;"
    cursor.execute(query)


def insert_prediction_rows(cursor, rows):
    """
    Inserta por lotes todas las filas de predicción en la tabla ml_prediction_log.
    Cada fila contiene:
    - identificador de actividad
    - usuario
    - elemento
    - entidad
    - acción
    - predicción de anomalía
    - probabilidad de anomalía
    - fecha de creación generada automáticamente con CURRENT_TIMESTAMP

    Se utiliza execute_batch para mejorar el rendimiento en inserciones masivas.

    Inserisce in batch tutte le righe di previsione nella tabella ml_prediction_log.
    Ogni riga contiene:
    - identificativo dell'attività
    - utente
    - elemento
    - entità
    - azione
    - previsione di anomalia
    - probabilità di anomalia
    - data di creazione generata automaticamente con CURRENT_TIMESTAMP

    Viene utilizzato execute_batch per migliorare le prestazioni nelle inserzioni massive.
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
    Función principal del proceso de backfill de predicciones.
    Flujo de ejecución:
    1. Carga el modelo entrenado desde disco.
    2. Carga todas las actividades desde activity_log.
    3. Comprueba si existen actividades para procesar.
    4. Genera las predicciones con el modelo.
    5. Abre una conexión con la base de datos.
    6. Vacía la tabla ml_prediction_log.
    7. Inserta las nuevas predicciones generadas.
    8. Confirma la transacción si todo sale bien.
    9. En caso de error, realiza rollback.

    Funzione principale del processo di backfill delle previsioni.
    Flusso di esecuzione:
    1. Carica il modello addestrato dal disco.
    2. Carica tutte le attività da activity_log.
    3. Verifica se esistono attività da elaborare.
    4. Genera le previsioni con il modello.
    5. Apre una connessione con il database.
    6. Svuota la tabella ml_prediction_log.
    7. Inserisce le nuove previsioni generate.
    8. Conferma la transazione se tutto va bene.
    9. In caso di errore, esegue il rollback.
    """
    print("Caricamento del modello...")
    model = load_model(MODEL_PATH)

    print("Caricamento di activity_log...")
    df = load_activity_data()

    if df.empty:
        print("Non ci sono attività da elaborare.")
        return

    print(f"Attività caricate: {df.shape[0]}")

    print("Generazione delle previsioni...")
    prediction_rows = build_prediction_rows(df, model)

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