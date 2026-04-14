import pandas as pd
import psycopg2
from psycopg2 import Error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from rules import evaluate_activity_anomaly

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}


def get_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Error al conectar con PostgreSQL: {e}")
        return None


def load_activity_data() -> pd.DataFrame:
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


def prepare_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df["logged_at"] = pd.to_datetime(df["logged_at"])
    df["hour"] = df["logged_at"].dt.hour
    df["minute"] = df["logged_at"].dt.minute
    df["day_of_week"] = df["logged_at"].dt.dayofweek

    anomalies = []

    for _, row in df.iterrows():
        messages = evaluate_activity_anomaly(
            user_id=int(row["user_id"]),
            element_id=int(row["element_id"]),
            entity_id=int(row["entity_id"]),
            action_id=int(row["action_id"]),
            dt=row["logged_at"].to_pydatetime()
        )
        hard_anomaly = any(msg.startswith("Anomalía:") for msg in messages)
        anomalies.append(1 if hard_anomaly else 0)

    df["anomaly"] = anomalies
    return df


def train_activity_model(df: pd.DataFrame):
    if df.empty:
        print("No hay datos para entrenar.")
        return None, None, None, None, None

    features = [
        "user_id",
        "element_id",
        "entity_id",
        "action_id",
        "hour",
        "minute",
        "day_of_week"
    ]

    X = df[features]
    y = df["anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_features="sqrt"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== RESULTADOS DEL MODELO DE ACTIVITY ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test


def save_model(model, path: str = "activity_model.pkl"):
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def load_model(path: str = "activity_model.pkl"):
    return joblib.load(path)


def predict_activity_with_model(
    model,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at
):
    logged_at = pd.to_datetime(logged_at)

    row = pd.DataFrame([{
        "user_id": user_id,
        "element_id": element_id,
        "entity_id": entity_id,
        "action_id": action_id,
        "hour": logged_at.hour,
        "minute": logged_at.minute,
        "day_of_week": logged_at.dayofweek
    }])

    prediction = model.predict(row)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(row)[0][1]
    else:
        probability = None

    return prediction, probability


if __name__ == "__main__":
    df = load_activity_data()
    print("Shape original:", df.shape)

    df_prepared = prepare_activity_features(df)
    print("Shape preparado:", df_prepared.shape)

    if not df_prepared.empty:
        print("\nDistribución de anomalías:")
        print(df_prepared["anomaly"].value_counts())

    model, X_train, X_test, y_train, y_test = train_activity_model(df_prepared)

    if model is not None:
        save_model(model)