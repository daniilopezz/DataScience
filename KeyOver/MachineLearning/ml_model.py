from datetime import datetime
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import Error
from sklearn.ensemble import IsolationForest

"""
Este archivo se encarga de entrenar el modelo de actividad.

El modelo aprende el comportamiento habitual de cada usuario a nivel de acción,
teniendo en cuenta:
- elemento
- entidad
- acción
- hora
- minuto
- día de la semana
- segundos desde el inicio de sesión
- segundos desde la acción anterior

También almacena frecuencias históricas de combinaciones
(elemento, entidad, acción) para reforzar la noción de rareza.

En esta versión, la asignación de actividades a sesiones NO se hace con un JOIN
temporal pesado en SQL, sino de forma eficiente en Python por usuario.

Questo file si occupa di addestrare il modello di attività.

Il modello apprende il comportamento abituale di ciascun utente a livello di azione,
tenendo conto di:
- elemento
- entità
- azione
- ora
- minuto
- giorno della settimana
- secondi dall'inizio della sessione
- secondi dall'azione precedente

Inoltre memorizza frequenze storiche delle combinazioni
(elemento, entità, azione) per rafforzare la nozione di rarità.

In questa versione, l'assegnazione delle attività alle sessioni NON viene fatta con una JOIN
temporale pesante in SQL, ma in modo efficiente in Python per utente.
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "activity_model.pkl"
MIN_ANOMALY_PROBABILITY = 0.000001


def get_connection():
    """
    Abre una conexión con PostgreSQL.

    Apre una connessione a PostgreSQL.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def load_login_data() -> pd.DataFrame:
    """
    Carga login_log.

    Carica login_log.
    """
    connection = get_connection()
    if connection is None:
        return pd.DataFrame()

    query = """
        SELECT
            login_log_id,
            user_id,
            result,
            attempt,
            logged_at,
            logout_at
        FROM login_log
        ORDER BY user_id, logged_at, login_log_id
    """

    try:
        return pd.read_sql(query, connection)
    except Exception as e:
        print(f"Errore durante il caricamento di login_log: {e}")
        return pd.DataFrame()
    finally:
        connection.close()


def load_activity_data() -> pd.DataFrame:
    """
    Carga activity_log.

    Carica activity_log.
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
        ORDER BY user_id, logged_at, activity_log_id
    """

    try:
        return pd.read_sql(query, connection)
    except Exception as e:
        print(f"Errore durante il caricamento di activity_log: {e}")
        return pd.DataFrame()
    finally:
        connection.close()


def assign_activities_to_sessions(login_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna cada actividad a una sesión de login válida de forma eficiente por usuario.

    Regla:
    - mismo user_id
    - activity.logged_at entre login.logged_at y login.logout_at

    Si hay varias sesiones solapadas, se elige la más reciente cuyo inicio sea
    anterior o igual a la actividad.

    Assegna ogni attività a una sessione di login valida in modo efficiente per utente.

    Regola:
    - stesso user_id
    - activity.logged_at compreso tra login.logged_at e login.logout_at

    Se ci sono più sessioni sovrapposte, si sceglie la più recente il cui inizio sia
    precedente o uguale all'attività.
    """
    if login_df.empty or activity_df.empty:
        return pd.DataFrame()

    login_df = login_df.copy()
    activity_df = activity_df.copy()

    login_df["logged_at"] = pd.to_datetime(login_df["logged_at"], errors="coerce")
    login_df["logout_at"] = pd.to_datetime(login_df["logout_at"], errors="coerce")
    activity_df["logged_at"] = pd.to_datetime(activity_df["logged_at"], errors="coerce")

    login_df = login_df[
        (login_df["result"] == True) &
        (login_df["logged_at"].notna()) &
        (login_df["logout_at"].notna())
    ].copy()

    activity_df = activity_df[activity_df["logged_at"].notna()].copy()

    if login_df.empty or activity_df.empty:
        return pd.DataFrame()

    assigned_rows = []

    login_by_user = {
        int(user_id): user_df.sort_values(["logged_at", "logout_at", "login_log_id"]).reset_index(drop=True)
        for user_id, user_df in login_df.groupby("user_id")
    }

    activity_by_user = {
        int(user_id): user_df.sort_values(["logged_at", "activity_log_id"]).reset_index(drop=True)
        for user_id, user_df in activity_df.groupby("user_id")
    }

    common_users = sorted(set(login_by_user.keys()) & set(activity_by_user.keys()))

    for user_id in common_users:
        sessions = login_by_user[user_id]
        activities = activity_by_user[user_id]

        session_records = sessions.to_dict("records")
        activity_records = activities.to_dict("records")

        session_idx = 0
        active_candidates = []

        for activity in activity_records:
            activity_time = activity["logged_at"]

            while session_idx < len(session_records) and session_records[session_idx]["logged_at"] <= activity_time:
                active_candidates.append(session_records[session_idx])
                session_idx += 1

            active_candidates = [
                s for s in active_candidates
                if s["logout_at"] >= activity_time
            ]

            if not active_candidates:
                continue

            selected_session = max(active_candidates, key=lambda s: (s["logged_at"], s["login_log_id"]))

            assigned_rows.append({
                "login_log_id": int(selected_session["login_log_id"]),
                "user_id": int(user_id),
                "session_start": selected_session["logged_at"],
                "session_end": selected_session["logout_at"],
                "activity_log_id": int(activity["activity_log_id"]),
                "element_id": int(activity["element_id"]),
                "entity_id": int(activity["entity_id"]),
                "action_id": int(activity["action_id"]),
                "activity_time": activity_time
            })

    if not assigned_rows:
        return pd.DataFrame()

    assigned_df = pd.DataFrame(assigned_rows)
    assigned_df = assigned_df.sort_values(
        ["user_id", "login_log_id", "activity_time", "activity_log_id"]
    ).reset_index(drop=True)

    return assigned_df


def prepare_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara variables de actividad con contexto temporal de sesión.

    Prepara variabili di attività con contesto temporale di sessione.
    """
    if df.empty:
        return df

    df = df.copy()
    df["session_start"] = pd.to_datetime(df["session_start"], errors="coerce")
    df["session_end"] = pd.to_datetime(df["session_end"], errors="coerce")
    df["activity_time"] = pd.to_datetime(df["activity_time"], errors="coerce")
    df = df.dropna(subset=["session_start", "activity_time"])

    feature_rows = []

    for (user_id, login_log_id), group in df.groupby(["user_id", "login_log_id"], dropna=False):
        group = group.sort_values("activity_time").copy()
        prev_time = None

        for _, row in group.iterrows():
            activity_time = row["activity_time"]
            session_start = row["session_start"]

            if prev_time is None:
                seconds_since_prev_action = max((activity_time - session_start).total_seconds(), 0.0)
            else:
                seconds_since_prev_action = max((activity_time - prev_time).total_seconds(), 0.0)

            seconds_since_session_start = max((activity_time - session_start).total_seconds(), 0.0)

            feature_rows.append({
                "user_id": int(row["user_id"]),
                "login_log_id": int(row["login_log_id"]),
                "activity_log_id": int(row["activity_log_id"]),
                "element_id": int(row["element_id"]),
                "entity_id": int(row["entity_id"]),
                "action_id": int(row["action_id"]),
                "hour": int(activity_time.hour),
                "minute": int(activity_time.minute),
                "day_of_week": int(activity_time.dayofweek),
                "seconds_since_prev_action": float(seconds_since_prev_action),
                "seconds_since_session_start": float(seconds_since_session_start),
            })

            prev_time = activity_time

    return pd.DataFrame(feature_rows)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz de variables para el modelo de actividad.

    Costruisce la matrice delle variabili per il modello di attività.
    """
    if df.empty:
        return df

    feature_df = df[[
        "element_id",
        "entity_id",
        "action_id",
        "hour",
        "minute",
        "day_of_week",
        "seconds_since_prev_action",
        "seconds_since_session_start"
    ]].copy()

    feature_df["log_seconds_since_prev_action"] = np.log1p(feature_df["seconds_since_prev_action"])
    feature_df["log_seconds_since_session_start"] = np.log1p(feature_df["seconds_since_session_start"])

    feature_df = feature_df.drop(columns=["seconds_since_prev_action", "seconds_since_session_start"])

    categorical_columns = ["element_id", "entity_id", "action_id", "day_of_week"]

    feature_df = pd.get_dummies(
        feature_df,
        columns=categorical_columns,
        prefix=categorical_columns,
        dtype=int
    )

    return feature_df


def build_combo_frequency(df: pd.DataFrame) -> dict[str, float]:
    """
    Calcula la frecuencia histórica de cada combinación elemento-entidad-acción.

    Calcola la frequenza storica di ogni combinazione elemento-entità-azione.
    """
    if df.empty:
        return {}

    combo_series = (
        df["element_id"].astype(str)
        + "|"
        + df["entity_id"].astype(str)
        + "|"
        + df["action_id"].astype(str)
    )

    counts = combo_series.value_counts(normalize=True).to_dict()
    return {str(k): float(v) for k, v in counts.items()}


def train_activity_model(df: pd.DataFrame, contamination: float = 0.05):
    """
    Entrena un modelo de actividad por usuario.

    Addestra un modello di attività per utente.
    """
    if df.empty:
        print("Non ci sono dati per l'addestramento del modello di attività.")
        return None

    user_models = {}
    user_ids = sorted(df["user_id"].dropna().unique())

    for user_id in user_ids:
        user_df = df[df["user_id"] == user_id].copy()

        if len(user_df) < 20:
            print(f"Utente {user_id}: attività insufficienti per addestrare il modello.")
            continue

        X_user = build_feature_matrix(user_df)

        model = IsolationForest(
            n_estimators=250,
            contamination=contamination,
            random_state=42
        )

        model.fit(X_user)

        raw_scores = model.decision_function(X_user)
        anomaly_scores = np.maximum(0.0, -raw_scores)
        score_scale = float(np.percentile(anomaly_scores, 95)) if len(anomaly_scores) > 0 else 1.0
        score_scale = max(score_scale, 1e-6)

        combo_frequency = build_combo_frequency(user_df)

        user_models[int(user_id)] = {
            "model": model,
            "feature_columns": list(X_user.columns),
            "score_scale": score_scale,
            "combo_frequency": combo_frequency
        }

        print(f"Utente {int(user_id)} -> attività: {len(user_df)} | score_scale: {score_scale:.6f}")

    if not user_models:
        print("Nessun modello di attività è stato addestrato.")
        return None

    return user_models


def save_model(model_bundle, path: str = str(MODEL_PATH)):
    """
    Guarda en disco el conjunto de modelos de actividad entrenados.

    Salva su disco l'insieme dei modelli di attività addestrati.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path_obj)
    print(f"Modelli di attività salvati in: {path_obj}")


def load_model(path: str = str(MODEL_PATH)):
    """
    Carga desde disco el conjunto de modelos de actividad entrenados.

    Carica da disco l'insieme dei modelli di attività addestrati.
    """
    return joblib.load(path)


def build_single_row_features(
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at,
    session_started_at=None,
    previous_action_timestamps: list[datetime] | None = None
) -> pd.DataFrame:
    """
    Construye una fila de variables con el mismo formato usado durante el entrenamiento.

    Costruisce una riga di variabili con lo stesso formato usato durante l'addestramento.
    """
    logged_at = pd.to_datetime(logged_at)
    previous_action_timestamps = previous_action_timestamps or []

    if session_started_at is None:
        session_started_at = logged_at
    else:
        session_started_at = pd.to_datetime(session_started_at)

    if previous_action_timestamps:
        prev_time = pd.to_datetime(previous_action_timestamps[-1])
        seconds_since_prev_action = max((logged_at - prev_time).total_seconds(), 0.0)
    else:
        seconds_since_prev_action = max((logged_at - session_started_at).total_seconds(), 0.0)

    seconds_since_session_start = max((logged_at - session_started_at).total_seconds(), 0.0)

    row = pd.DataFrame([{
        "element_id": int(element_id),
        "entity_id": int(entity_id),
        "action_id": int(action_id),
        "hour": int(logged_at.hour),
        "minute": int(logged_at.minute),
        "day_of_week": int(logged_at.dayofweek),
        "seconds_since_prev_action": float(seconds_since_prev_action),
        "seconds_since_session_start": float(seconds_since_session_start)
    }])

    return build_feature_matrix(row)


def align_features_to_training(row_features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Alinea una fila de inferencia con las columnas exactas usadas durante el entrenamiento.

    Allinea una riga di inferenza con le colonne esatte usate durante l'addestramento.
    """
    aligned = row_features.copy()

    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0

    return aligned[feature_columns]


def predict_activity_with_model(
    model_bundle,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at,
    session_started_at=None,
    previous_action_timestamps: list | None = None
):
    """
    Realiza una predicción individual para una actividad concreta.

    Convención de salida:
    - prediction = 1  -> actividad anómala
    - prediction = 0  -> actividad normal
    - probability     -> score de anomalía entre 0 y 1, nunca igual a 0

    Esegue una previsione singola per una specifica attività.

    Convenzione di output:
    - prediction = 1  -> attività anomala
    - prediction = 0  -> attività normale
    - probability     -> score di anomalia tra 0 e 1, mai uguale a 0
    """
    if model_bundle is None:
        raise ValueError("Il bundle dei modelli di attività è vuoto.")

    user_id = int(user_id)

    if user_id not in model_bundle:
        raise ValueError(f"Nessun modello trovato per user_id={user_id}")

    user_model_data = model_bundle[user_id]
    model = user_model_data["model"]
    feature_columns = user_model_data["feature_columns"]
    score_scale = float(user_model_data["score_scale"])
    combo_frequency = user_model_data["combo_frequency"]

    row_features = build_single_row_features(
        element_id=element_id,
        entity_id=entity_id,
        action_id=action_id,
        logged_at=logged_at,
        session_started_at=session_started_at,
        previous_action_timestamps=previous_action_timestamps
    )

    row_aligned = align_features_to_training(row_features, feature_columns)

    raw_prediction = model.predict(row_aligned)[0]
    raw_score = float(model.decision_function(row_aligned)[0])

    prediction = 1 if raw_prediction == -1 else 0

    anomaly_score = max(0.0, -raw_score)
    model_probability = min(anomaly_score / score_scale, 1.0)

    combo_key = f"{int(element_id)}|{int(entity_id)}|{int(action_id)}"
    combo_freq = float(combo_frequency.get(combo_key, 0.0))
    combo_rarity = 1.0 - combo_freq

    probability = (0.75 * float(model_probability)) + (0.25 * float(combo_rarity))
    probability = max(float(probability), MIN_ANOMALY_PROBABILITY)

    return prediction, probability


if __name__ == "__main__":
    print("Caricamento login_log...")
    login_df = load_login_data()
    print("Shape login:", login_df.shape)

    print("Caricamento activity_log...")
    activity_df = load_activity_data()
    print("Shape activity:", activity_df.shape)

    print("Assegnazione attività a sessioni...")
    assigned_df = assign_activities_to_sessions(login_df, activity_df)
    print("Shape assigned:", assigned_df.shape)

    print("Preparazione feature di attività...")
    df_prepared = prepare_activity_features(assigned_df)
    print("Shape prepared:", df_prepared.shape)

    print("Addestramento modelli di attività...")
    model_bundle = train_activity_model(df_prepared, contamination=0.05)

    if model_bundle is not None:
        save_model(model_bundle)