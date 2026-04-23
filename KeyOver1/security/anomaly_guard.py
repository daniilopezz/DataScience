# Centraliza la lógica de detección de anomalías.
# Centralizza la logica di rilevamento delle anomalie.
#
# Expone tres tipos de evaluación:
# - login: perfil estadístico (hora habitual, días de la semana)
# - actividad: modelo de actividad (IsolationForest por usuario)
# - sesión: modelo de sesión (IsolationForest por usuario)
#
# Espone tre tipi di valutazione:
# - login: profilo statistico (ora abituale, giorni della settimana)
# - attività: modello di attività (IsolationForest per utente)
# - sessione: modello di sessione (IsolationForest per utente)

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MachineLearning.train_models import (
    load_combined_model,
    predict_activity,
    predict_session,
    COMBINED_MODEL_PATH,
)

MIN_PROB = 1e-6

# Singleton del modelo en memoria / Singleton del modello in memoria
_COMBINED_MODEL_CACHE: dict | None = None


def get_combined_model() -> dict | None:
    # Carga el modelo combinado una sola vez y lo reutiliza.
    # Carica il modello combinato una sola volta e lo riutilizza.
    global _COMBINED_MODEL_CACHE
    if _COMBINED_MODEL_CACHE is None:
        _COMBINED_MODEL_CACHE = load_combined_model(COMBINED_MODEL_PATH)
    return _COMBINED_MODEL_CACHE


# ─── Consultas al modelo combinado / Query al modello combinato ───────────────

def get_user_known_elements(combined_model: dict, user_id: int) -> set[int] | None:
    # Devuelve los element_ids que el modelo reconoce para este usuario (frecuencia >= 1%).
    # Restituisce gli element_id che il modello riconosce per questo utente (frequenza >= 1%).
    # Devuelve None si no hay modelo o el usuario no tiene datos.
    # Restituisce None se non c'è modello o l'utente non ha dati.
    if combined_model is None:
        return None
    activity = combined_model.get("activity", {})
    uid = int(user_id)
    if uid not in activity:
        return None
    known = activity[uid].get("known_elements")
    return set(known) if known else None


def get_model_session_threshold(combined_model: dict, user_id: int) -> float:
    # Devuelve el umbral de coste de sesión calculado durante el entrenamiento.
    # Restituisce la soglia di costo sessione calcolata durante l'addestramento.
    if combined_model is None:
        return float("inf")
    session = combined_model.get("session", {})
    uid = int(user_id)
    if uid not in session:
        return float("inf")
    return float(session[uid].get("session_cost_threshold", float("inf")))


# ─── Perfil de login / Profilo di login ───────────────────────────────────────

def build_login_profile(login_df: pd.DataFrame) -> dict:
    # Construye un perfil estadístico por usuario basado en logins correctos.
    # Costruisce un profilo statistico per utente basato sui login corretti.
    # Calcula: ventana horaria habitual (percentil 10-90) + días de la semana.
    # Calcola: finestra oraria abituale (percentile 10-90) + giorni della settimana.
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
    for uid, udata in df.groupby("user_id"):
        if udata.empty:
            continue
        q10 = float(udata["hour_float"].quantile(0.10))
        q90 = float(udata["hour_float"].quantile(0.90))
        profiles[int(uid)] = {
            "hour_min":    max(0.0, q10 - 0.5),
            "hour_max":    min(23.99, q90 + 0.5),
            "common_days": sorted(udata["day_of_week"].unique().tolist()),
        }

    return profiles


def evaluate_login(profiles: dict, user_id: int, logged_at=None) -> dict:
    # Detecta si un login cae fuera del horario/días habituales del usuario.
    # Rileva se un login cade fuori dall'orario/giorni abituali dell'utente.
    if logged_at is None:
        logged_at = datetime.now()
    logged_at = pd.to_datetime(logged_at)

    if user_id not in profiles:
        return {"is_anomalous": False, "message": ""}

    p = profiles[user_id]
    h = logged_at.hour + logged_at.minute / 60 + logged_at.second / 3600
    d = int(logged_at.dayofweek)

    out_hours = h < p["hour_min"] or h > p["hour_max"]
    out_days  = d not in p["common_days"]
    anomalous = bool(out_hours or out_days)

    return {
        "is_anomalous": anomalous,
        "message": "Accesso fuori orario o giorno insolito." if anomalous else "",
    }


# ─── Evaluación de actividad / Valutazione attività ───────────────────────────

def evaluate_activity(
    combined_model: dict,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at=None,
    session_started_at=None,
    previous_timestamps: list | None = None,
) -> dict:
    # Evalúa una acción individual con el modelo de actividad.
    # Valuta una singola azione con il modello di attività.
    if logged_at is None:
        logged_at = datetime.now()

    try:
        prediction, score = predict_activity(
            combined_model=combined_model,
            user_id=user_id,
            element_id=element_id,
            entity_id=entity_id,
            action_id=action_id,
            logged_at=logged_at,
            session_started_at=session_started_at,
            previous_timestamps=previous_timestamps,
        )
    except Exception as e:
        return {
            "prediction":    0,
            "anomaly_score": MIN_PROB,
            "message":       f"[ML-ACTIVITY] Errore: {e}",
        }

    return {
        "prediction":    int(prediction),
        "anomaly_score": max(float(score), MIN_PROB),
        "message":       "Azione anomala rilevata." if prediction == 1 else "",
    }


# ─── Evaluación de sesión / Valutazione sessione ──────────────────────────────

def evaluate_session(
    combined_model: dict,
    user_id: int,
    action_count: int,
    distinct_elements: int,
    distinct_actions: int,
    session_duration_min: float,
    start_hour: float,
    day_of_week: int,
    actions_per_minute: float,
    avg_seconds_between_actions: float,
    min_seconds_between_actions: float,
    max_seconds_between_actions: float,
    cumulative_cost: float,
    avg_cost: float,
    max_cost: float,
    repeated_action_ratio: float,
    repeated_element_ratio: float,
) -> dict:
    # Evalúa una sesión parcial o completa con el modelo de sesión.
    # Valuta una sessione parziale o completa con il modello di sessione.
    try:
        prediction, score = predict_session(
            combined_model=combined_model,
            user_id=user_id,
            action_count=action_count,
            distinct_elements=distinct_elements,
            distinct_actions=distinct_actions,
            session_duration_min=session_duration_min,
            start_hour=start_hour,
            day_of_week=day_of_week,
            actions_per_minute=actions_per_minute,
            avg_seconds_between_actions=avg_seconds_between_actions,
            min_seconds_between_actions=min_seconds_between_actions,
            max_seconds_between_actions=max_seconds_between_actions,
            cumulative_cost=cumulative_cost,
            avg_cost=avg_cost,
            max_cost=max_cost,
            repeated_action_ratio=repeated_action_ratio,
            repeated_element_ratio=repeated_element_ratio,
        )
    except Exception as e:
        return {
            "prediction":    0,
            "anomaly_score": MIN_PROB,
            "message":       f"[ML-SESSION] Errore: {e}",
        }

    return {
        "prediction":    int(prediction),
        "anomaly_score": max(float(score), MIN_PROB),
        "message":       "Sessione anomala rilevata." if prediction == 1 else "",
    }
