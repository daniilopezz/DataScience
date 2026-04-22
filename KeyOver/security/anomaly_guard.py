from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MachineLearning.ml_model import (
    load_model as load_activity_model,
    predict_activity_with_model
)
from MachineLearning.session_model import (
    load_model as load_session_model,
    predict_session_with_model
)

"""
Este archivo centraliza la lógica de control de anomalías del sistema.

Su función es:
- cargar el modelo de actividad
- cargar el modelo de sesión
- evaluar si un login parece anómalo mediante un perfil histórico simple
- evaluar si una actividad parece anómala mediante el modelo de actividad
- evaluar si una sesión parece anómala mediante el modelo de sesión

Questo file centralizza la logica di controllo delle anomalie del sistema.

La sua funzione è:
- caricare il modello di attività
- caricare il modello di sessione
- valutare se un login sembra anomalo tramite un profilo storico semplice
- valutare se un'attività sembra anomala tramite il modello di attività
- valutare se una sessione sembra anomala tramite il modello di sessione
"""

ACTIVITY_MODEL_PATH = PROJECT_ROOT / "models" / "activity_model.pkl"
SESSION_MODEL_PATH = PROJECT_ROOT / "models" / "session_model.pkl"
MIN_ANOMALY_PROBABILITY = 0.000001


def get_activity_model_bundle():
    """
    Carga el bundle/modelo de actividad.

    Carica il bundle/modello di attività.
    """
    return load_activity_model(str(ACTIVITY_MODEL_PATH))


def get_session_model_bundle():
    """
    Carga el bundle/modelo de sesión.

    Carica il bundle/modello di sessione.
    """
    return load_session_model(str(SESSION_MODEL_PATH))


def get_session_anomaly_threshold() -> float:
    """
    Mantiene compatibilidad con el resto del proyecto.

    Ahora la decisión principal es ML, por lo que este valor se usa solo
    para registro y trazabilidad.

    Mantiene compatibilità con il resto del progetto.

    Ora la decisione principale è ML, quindi questo valore viene usato solo
    per registrazione e tracciabilità.
    """
    return 0.0


def build_login_profile(login_df: pd.DataFrame) -> dict[int, dict]:
    """
    Construye un perfil estadístico simple por usuario a partir de login_log.

    Para cada usuario calcula:
    - ventana horaria habitual basada en percentiles
    - días de la semana habituales

    Solo se usan logins correctos para definir el comportamiento normal.

    Costruisce un profilo statistico semplice per utente a partire da login_log.

    Per ogni utente calcola:
    - finestra oraria abituale basata sui percentili
    - giorni della settimana abituali

    Si usano solo i login corretti per definire il comportamento normale.
    """
    if login_df.empty:
        return {}

    df = login_df.copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce")
    df = df.dropna(subset=["logged_at"])

    if "result" in df.columns:
        df = df[df["result"] == True].copy()

    df["hour_float"] = (
        df["logged_at"].dt.hour
        + df["logged_at"].dt.minute / 60
        + df["logged_at"].dt.second / 3600
    )
    df["day_of_week"] = df["logged_at"].dt.dayofweek

    profiles = {}

    for user_id, user_data in df.groupby("user_id"):
        if user_data.empty:
            continue

        q10 = user_data["hour_float"].quantile(0.10)
        q90 = user_data["hour_float"].quantile(0.90)

        hour_min = max(0.0, float(q10) - 0.5)
        hour_max = min(23.99, float(q90) + 0.5)

        common_days = sorted(user_data["day_of_week"].unique().tolist())

        profiles[int(user_id)] = {
            "hour_min": hour_min,
            "hour_max": hour_max,
            "common_days": common_days
        }

    return profiles


def evaluate_login_with_profile(
    login_profiles: dict[int, dict],
    user_id: int,
    logged_at=None
) -> dict:
    """
    Evalúa un intento de login comparándolo con el patrón histórico del usuario.

    Valuta un tentativo di login confrontandolo con il pattern storico dell'utente.
    """
    if logged_at is None:
        logged_at = datetime.now()

    logged_at = pd.to_datetime(logged_at)

    if user_id not in login_profiles:
        return {
            "is_anomalous": False,
            "anomaly_score": 0.0,
            "message": ""
        }

    profile = login_profiles[user_id]

    current_hour = (
        logged_at.hour
        + logged_at.minute / 60
        + logged_at.second / 3600
    )
    current_day = int(logged_at.dayofweek)

    out_of_hours = current_hour < profile["hour_min"] or current_hour > profile["hour_max"]
    unusual_day = current_day not in profile["common_days"]

    anomaly_score = 0.0
    if out_of_hours:
        anomaly_score += 1.0
    if unusual_day:
        anomaly_score += 1.0

    is_anomalous = bool(out_of_hours or unusual_day)

    return {
        "is_anomalous": is_anomalous,
        "anomaly_score": anomaly_score,
        "message": "Accesso strano, attendi conferma." if is_anomalous else ""
    }


def evaluate_activity_with_model(
    model_bundle,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at=None,
    session_started_at=None,
    previous_action_timestamps=None
) -> dict:
    """
    Evalúa una actividad con el modelo y devuelve el coste de la operación.

    Valuta un'attività con il modello e restituisce il costo dell'operazione.
    """
    if logged_at is None:
        logged_at = datetime.now()

    try:
        prediction, anomaly_score = predict_activity_with_model(
            model_bundle=model_bundle,
            user_id=user_id,
            element_id=element_id,
            entity_id=entity_id,
            action_id=action_id,
            logged_at=logged_at,
            session_started_at=session_started_at,
            previous_action_timestamps=previous_action_timestamps
        )
    except Exception as e:
        return {
            "prediction": 0,
            "anomaly_score": MIN_ANOMALY_PROBABILITY,
            "message": f"Errore nel controllo ML dell'attività: {e}"
        }

    anomaly_score = max(float(anomaly_score), MIN_ANOMALY_PROBABILITY)

    return {
        "prediction": int(prediction),
        "anomaly_score": anomaly_score,
        "message": "Azione strana, attendi conferma." if int(prediction) == 1 else ""
    }


def evaluate_session_with_model(
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
) -> dict:
    """
    Evalúa una sesión parcial o completa con el modelo de sesión.

    Valuta una sessione parziale o completa con il modello di sessione.
    """
    try:
        prediction, anomaly_score = predict_session_with_model(
            model_bundle=model_bundle,
            user_id=user_id,
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
    except Exception as e:
        return {
            "prediction": 0,
            "anomaly_score": MIN_ANOMALY_PROBABILITY,
            "message": f"Errore nel controllo ML della sessione: {e}"
        }

    anomaly_score = max(float(anomaly_score), MIN_ANOMALY_PROBABILITY)

    return {
        "prediction": int(prediction),
        "anomaly_score": anomaly_score,
        "message": "Sessione anomala rilevata dal modello." if int(prediction) == 1 else ""
    }