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
    Abre una conexión con PostgreSQL.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Error al conectar con PostgreSQL: {e}")
        return None


def load_activity_data() -> pd.DataFrame:
    """
    Carga todas las actividades desde activity_log.
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
        print(f"Error al cargar activity_log: {e}")
        return pd.DataFrame()
    finally:
        connection.close()


def build_prediction_rows(df: pd.DataFrame, model):
    """
    Recorre todas las actividades y construye las filas que se insertarán
    en ml_prediction_log.
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
            print(f"Predicciones preparadas: {i + 1}")

    return rows


def clear_ml_prediction_log(cursor):
    """
    Vacía la tabla ml_prediction_log y reinicia el contador.
    """
    query = "TRUNCATE TABLE ml_prediction_log RESTART IDENTITY;"
    cursor.execute(query)


def insert_prediction_rows(cursor, rows):
    """
    Inserta por lotes las predicciones generadas en ml_prediction_log.
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
    print("Cargando modelo...")
    model = load_model(MODEL_PATH)

    print("Cargando activity_log...")
    df = load_activity_data()

    if df.empty:
        print("No hay actividades para procesar.")
        return

    print(f"Actividades cargadas: {df.shape[0]}")

    print("Generando predicciones...")
    prediction_rows = build_prediction_rows(df, model)

    connection = get_connection()
    if connection is None:
        return

    cursor = connection.cursor()

    try:
        print("Vaciando ml_prediction_log...")
        clear_ml_prediction_log(cursor)

        print("Insertando predicciones en ml_prediction_log...")
        insert_prediction_rows(cursor, prediction_rows)

        connection.commit()
        print("Backfill completado correctamente.")

    except Exception as e:
        connection.rollback()
        print(f"Error durante el backfill: {e}")

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()