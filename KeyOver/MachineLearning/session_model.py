from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import Error
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MachineLearning.ml_model import (
    load_model as load_activity_model,
    load_login_data,
    load_activity_data,
    assign_activities_to_sessions,
    prepare_activity_features,
    build_feature_matrix as build_activity_feature_matrix,
)

"""
Este archivo entrena el modelo de anomalías de sesión.

La diferencia clave es que ya no se entrena solo con sesiones completas,
sino con prefijos de sesión:
- acción 1
- acción 1-2
- acción 1-2-3
- etc.

Así, en producción, el modelo puede evaluar una sesión parcial con el mismo
tipo de datos con el que fue entrenado.

Las variables utilizadas incluyen:
- número de acciones
- variedad de elementos y acciones
- duración acumulada de la sesión en ese punto
- acciones por minuto
- media / mínimo / máximo de segundos entre acciones
- coste acumulado, coste medio y coste máximo de las acciones

En esta versión, las predicciones del modelo de actividad se calculan por lotes,
no fila por fila, para evitar bloqueos de rendimiento.

Questo file addestra il modello di anomalie di sessione.

La differenza chiave è che non viene più addestrato solo con sessioni complete,
ma con prefissi di sessione:
- azione 1
- azione 1-2
- azione 1-2-3
- ecc.

In questo modo, in produzione, il modello può valutare una sessione parziale con lo stesso
tipo di dati con cui è stato addestrato.

Le variabili utilizzate includono:
- numero di azioni
- varietà di elementi e azioni
- durata cumulata della sessione in quel punto
- azioni per minuto
- media / minimo / massimo dei secondi tra le azioni
- costo cumulato, costo medio e costo massimo delle azioni

In questa versione, le predizioni del modello di attività vengono calcolate in batch,
non riga per riga, per evitare blocchi di performance.
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

MODEL_PATH = PROJECT_ROOT / "models" / "session_model.pkl"
ACTIVITY_MODEL_PATH = PROJECT_ROOT / "models" / "activity_model.pkl"
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


def score_activities_in_batch(assigned_df: pd.DataFrame, activity_model_bundle) -> pd.DataFrame:
    """
    Calcula prediction y anomaly_probability para todas las actividades
    de forma vectorizada por usuario.

    Además conserva las columnas temporales de sesión necesarias para
    construir después los prefijos de sesión.

    Calcola prediction e anomaly_probability per tutte le attività
    in modo vettoriale per utente.

    Inoltre conserva le colonne temporali di sessione necessarie per
    costruire successivamente i prefissi di sessione.
    """
    if assigned_df.empty:
        return pd.DataFrame()

    base_df = assigned_df.copy()
    prepared_df = prepare_activity_features(base_df)

    if prepared_df.empty:
        return pd.DataFrame()

    # Conservamos columnas de contexto que prepare_activity_features no mantiene.
    # Conserviamo colonne di contesto che prepare_activity_features non mantiene.
    context_cols = [
        "activity_log_id",
        "login_log_id",
        "user_id",
        "session_start",
        "session_end",
        "activity_time",
        "element_id",
        "entity_id",
        "action_id",
    ]

    context_df = base_df[context_cols].copy()

    prepared_df = prepared_df.merge(
        context_df,
        on=["activity_log_id", "login_log_id", "user_id", "element_id", "entity_id", "action_id"],
        how="left"
    )

    scored_parts = []

    for user_id, user_df in prepared_df.groupby("user_id"):
        user_id = int(user_id)

        if user_id not in activity_model_bundle:
            continue

        model_data = activity_model_bundle[user_id]
        model = model_data["model"]
        feature_columns = model_data["feature_columns"]
        score_scale = float(model_data["score_scale"])
        combo_frequency = model_data["combo_frequency"]

        X_user = build_activity_feature_matrix(user_df).copy()

        for column in feature_columns:
            if column not in X_user.columns:
                X_user[column] = 0

        X_user = X_user[feature_columns]

        raw_predictions = model.predict(X_user)
        raw_scores = model.decision_function(X_user)

        anomaly_scores = np.maximum(0.0, -raw_scores)

        if score_scale <= 1e-6:
            model_probabilities = np.where(anomaly_scores > 0.0, 1.0, 0.0)
        else:
            model_probabilities = np.clip(anomaly_scores / score_scale, 0.0, 1.0)

        combo_keys = (
            user_df["element_id"].astype(str)
            + "|"
            + user_df["entity_id"].astype(str)
            + "|"
            + user_df["action_id"].astype(str)
        )

        combo_freq = combo_keys.map(lambda k: float(combo_frequency.get(k, 0.0))).astype(float)
        combo_rarity = 1.0 - combo_freq

        probabilities = (0.75 * model_probabilities) + (0.25 * combo_rarity.to_numpy())
        probabilities = np.maximum(probabilities, MIN_ANOMALY_PROBABILITY)

        scored_user_df = user_df.copy()
        scored_user_df["activity_prediction"] = np.where(raw_predictions == -1, 1, 0).astype(int)
        scored_user_df["activity_probability"] = probabilities.astype(float)

        scored_parts.append(scored_user_df)

    if not scored_parts:
        return pd.DataFrame()

    scored_df = pd.concat(scored_parts, ignore_index=True)

    scored_df["activity_time"] = pd.to_datetime(scored_df["activity_time"], errors="coerce")
    scored_df["session_start"] = pd.to_datetime(scored_df["session_start"], errors="coerce")
    scored_df["session_end"] = pd.to_datetime(scored_df["session_end"], errors="coerce")

    scored_df = scored_df.sort_values(
        ["user_id", "login_log_id", "activity_time", "activity_log_id"]
    ).reset_index(drop=True)

    return scored_df


def build_session_prefix_dataset(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un dataset de entrenamiento a nivel de prefijo de sesión.

    Por cada sesión y por cada acción de esa sesión se genera una fila que describe
    cómo iba la sesión en ese punto.

    Costruisce un dataset di addestramento a livello di prefisso di sessione.

    Per ogni sessione e per ogni azione di quella sessione viene generata una riga che descrive
    come stava andando la sessione in quel punto.
    """
    if scored_df.empty:
        return pd.DataFrame()

    df = scored_df.copy()
    df["activity_time"] = pd.to_datetime(df["activity_time"], errors="coerce")
    df["session_start"] = pd.to_datetime(df["session_start"], errors="coerce")
    df = df.dropna(subset=["activity_time", "session_start"])

    prefix_rows = []

    for (user_id, login_log_id), group in df.groupby(["user_id", "login_log_id"], dropna=False):
        group = group.sort_values("activity_time").copy()

        session_start = group["session_start"].iloc[0]

        cost_parts: list[float] = []
        previous_action_timestamps: list[pd.Timestamp] = []
        action_history: list[int] = []
        element_history: list[int] = []
        used_elements: set[int] = set()
        used_actions: set[int] = set()

        for _, row in group.iterrows():
            activity_time = row["activity_time"]

            current_cost = max(float(row["activity_probability"]), MIN_ANOMALY_PROBABILITY)
            current_prediction = int(row["activity_prediction"])
            current_action_id = int(row["action_id"])
            current_element_id = int(row["element_id"])

            cost_parts.append(current_cost)
            previous_action_timestamps.append(activity_time)
            action_history.append(current_action_id)
            element_history.append(current_element_id)
            used_elements.add(current_element_id)
            used_actions.add(current_action_id)

            action_count = len(cost_parts)
            cumulative_cost = float(sum(cost_parts))
            avg_cost = float(cumulative_cost / action_count)
            max_cost = float(max(cost_parts))

            session_elapsed_min = max((activity_time - session_start).total_seconds() / 60.0, 0.0001)
            actions_per_minute = float(action_count / session_elapsed_min)

            diffs_seconds = []
            if len(previous_action_timestamps) >= 2:
                for i in range(1, len(previous_action_timestamps)):
                    diff = (previous_action_timestamps[i] - previous_action_timestamps[i - 1]).total_seconds()
                    diffs_seconds.append(max(float(diff), 0.0))

            if diffs_seconds:
                avg_seconds_between_actions = float(sum(diffs_seconds) / len(diffs_seconds))
                min_seconds_between_actions = float(min(diffs_seconds))
                max_seconds_between_actions = float(max(diffs_seconds))
            else:
                fallback_seconds = session_elapsed_min * 60.0
                avg_seconds_between_actions = float(fallback_seconds)
                min_seconds_between_actions = float(fallback_seconds)
                max_seconds_between_actions = float(fallback_seconds)

            repeated_action_ratio = float(pd.Series(action_history).value_counts(normalize=True).max())
            repeated_element_ratio = float(pd.Series(element_history).value_counts(normalize=True).max())

            start_hour = (
                session_start.hour
                + session_start.minute / 60.0
                + session_start.second / 3600.0
            )
            day_of_week = int(session_start.weekday())

            prefix_rows.append({
                "user_id": int(user_id),
                "login_log_id": int(login_log_id),
                "action_count": action_count,
                "distinct_elements": len(used_elements),
                "distinct_actions": len(used_actions),
                "session_duration_min": float(session_elapsed_min),
                "start_hour": float(start_hour),
                "day_of_week": day_of_week,
                "actions_per_minute": actions_per_minute,
                "avg_seconds_between_actions": avg_seconds_between_actions,
                "min_seconds_between_actions": min_seconds_between_actions,
                "max_seconds_between_actions": max_seconds_between_actions,
                "cumulative_cost": cumulative_cost,
                "avg_cost": avg_cost,
                "max_cost": max_cost,
                "repeated_action_ratio": repeated_action_ratio,
                "repeated_element_ratio": repeated_element_ratio,
                "last_activity_prediction": current_prediction,
            })

    return pd.DataFrame(prefix_rows)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz de variables para el modelo de sesión.

    Costruisce la matrice delle variabili per il modello di sessione.
    """
    return df[[
        "action_count",
        "distinct_elements",
        "distinct_actions",
        "session_duration_min",
        "start_hour",
        "day_of_week",
        "actions_per_minute",
        "avg_seconds_between_actions",
        "min_seconds_between_actions",
        "max_seconds_between_actions",
        "cumulative_cost",
        "avg_cost",
        "max_cost",
        "repeated_action_ratio",
        "repeated_element_ratio",
    ]].copy()


def train_session_model(df: pd.DataFrame, contamination: float = 0.03):
    """
    Entrena un modelo de detección de anomalías de sesión por usuario.

    Addestra un modello di rilevamento anomalie di sessione per utente.
    """
    if df.empty:
        print("Non ci sono dati di sessione per l'addestramento.")
        return None

    user_models = {}
    user_ids = sorted(df["user_id"].dropna().unique())

    for user_id in user_ids:
        user_df = df[df["user_id"] == user_id].copy()

        if len(user_df) < 10:
            print(f"Utente {user_id}: dati insufficienti per addestrare il modello di sessione.")
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

        user_models[int(user_id)] = {
            "model": model,
            "feature_columns": list(X_user.columns),
            "score_scale": score_scale
        }

        print(f"Utente {int(user_id)} -> prefissi di sessione: {len(user_df)} | score_scale: {score_scale:.6f}")

    if not user_models:
        print("Nessun modello di sessione è stato addestrato.")
        return None

    return user_models


def save_model(model_bundle, path: str = str(MODEL_PATH)):
    """
    Guarda en disco el conjunto de modelos de sesión entrenados.

    Salva su disco l'insieme dei modelli di sessione addestrati.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path_obj)
    print(f"Modelli di sessione salvati in: {path_obj}")


def load_model(path: str = str(MODEL_PATH)):
    """
    Carga desde disco el conjunto de modelos de sesión entrenados.

    Carica da disco l'insieme dei modelli di sessione addestrati.
    """
    return joblib.load(path)


def build_single_session_features(
    action_count: int,
    cumulative_cost: float,
    avg_cost: float,
    max_cost: float,
    distinct_elements: int,
    distinct_actions: int,
    session_duration_min: float,
    start_hour: float,
    day_of_week: int,
    actions_per_minute: float,
    avg_seconds_between_actions: float,
    min_seconds_between_actions: float,
    max_seconds_between_actions: float,
    repeated_action_ratio: float,
    repeated_element_ratio: float
) -> pd.DataFrame:
    """
    Construye una fila de variables de sesión con el mismo formato usado
    durante el entrenamiento.

    Costruisce una riga di variabili di sessione con lo stesso formato usato
    durante l'addestramento.
    """
    return pd.DataFrame([{
        "action_count": float(action_count),
        "distinct_elements": float(distinct_elements),
        "distinct_actions": float(distinct_actions),
        "session_duration_min": float(session_duration_min),
        "start_hour": float(start_hour),
        "day_of_week": float(day_of_week),
        "actions_per_minute": float(actions_per_minute),
        "avg_seconds_between_actions": float(avg_seconds_between_actions),
        "min_seconds_between_actions": float(min_seconds_between_actions),
        "max_seconds_between_actions": float(max_seconds_between_actions),
        "cumulative_cost": float(cumulative_cost),
        "avg_cost": float(avg_cost),
        "max_cost": float(max_cost),
        "repeated_action_ratio": float(repeated_action_ratio),
        "repeated_element_ratio": float(repeated_element_ratio),
    }])


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


def predict_session_with_model(
    model_bundle,
    user_id: int,
    action_count: int,
    cumulative_cost: float,
    avg_cost: float,
    max_cost: float,
    distinct_elements: int,
    distinct_actions: int,
    session_duration_min: float,
    start_hour: float,
    day_of_week: int,
    actions_per_minute: float,
    avg_seconds_between_actions: float,
    min_seconds_between_actions: float,
    max_seconds_between_actions: float,
    repeated_action_ratio: float,
    repeated_element_ratio: float
):
    """
    Realiza una predicción individual para una sesión concreta.

    Convención de salida:
    - prediction = 1  -> sesión anómala
    - prediction = 0  -> sesión normal
    - probability     -> score de anomalía entre 0 y 1, nunca igual a 0

    Esegue una previsione singola per una sessione specifica.

    Convenzione di output:
    - prediction = 1  -> sessione anomala
    - prediction = 0  -> sessione normale
    - probability     -> score di anomalia tra 0 e 1, mai uguale a 0
    """
    if model_bundle is None:
        raise ValueError("Il bundle dei modelli di sessione è vuoto.")

    user_id = int(user_id)

    if user_id not in model_bundle:
        raise ValueError(f"Nessun modello di sessione trovato per user_id={user_id}")

    user_model_data = model_bundle[user_id]
    model = user_model_data["model"]
    feature_columns = user_model_data["feature_columns"]
    score_scale = float(user_model_data["score_scale"])

    row_features = build_single_session_features(
        action_count=action_count,
        cumulative_cost=cumulative_cost,
        avg_cost=avg_cost,
        max_cost=max_cost,
        distinct_elements=distinct_elements,
        distinct_actions=distinct_actions,
        session_duration_min=session_duration_min,
        start_hour=start_hour,
        day_of_week=day_of_week,
        actions_per_minute=actions_per_minute,
        avg_seconds_between_actions=avg_seconds_between_actions,
        min_seconds_between_actions=min_seconds_between_actions,
        max_seconds_between_actions=max_seconds_between_actions,
        repeated_action_ratio=repeated_action_ratio,
        repeated_element_ratio=repeated_element_ratio
    )

    row_aligned = align_features_to_training(row_features, feature_columns)

    raw_prediction = model.predict(row_aligned)[0]
    raw_score = float(model.decision_function(row_aligned)[0])

    prediction = 1 if raw_prediction == -1 else 0

    anomaly_score = max(0.0, -raw_score)
    probability = min(anomaly_score / score_scale, 1.0)
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
    base_df = assign_activities_to_sessions(login_df, activity_df)
    print("Shape base:", base_df.shape)

    print("Caricamento modello di attività...")
    activity_model_bundle = load_activity_model(str(ACTIVITY_MODEL_PATH))

    print("Calcolo score attività in batch...")
    scored_df = score_activities_in_batch(base_df, activity_model_bundle)
    print("Shape scored:", scored_df.shape)

    print("Costruzione dataset prefissi di sessione...")
    prefix_df = build_session_prefix_dataset(scored_df)
    print("Shape prefix:", prefix_df.shape)

    print("Addestramento modelli di sessione...")
    model_bundle = train_session_model(prefix_df, contamination=0.03)

    if model_bundle is not None:
        save_model(model_bundle)