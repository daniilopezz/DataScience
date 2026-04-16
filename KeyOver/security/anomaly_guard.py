from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

# Añadimos la raíz del proyecto al path para poder importar ml_model.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from MachineLearning.ml_model import load_model, predict_activity_with_model

"""
Este archivo centraliza la lógica de control de anomalías del sistema.

Su función es:
- cargar el modelo de activity
- evaluar si un login parece anómalo mediante reglas estadísticas simples
- evaluar si una actividad parece anómala mediante el modelo de machine learning
- devolver una decisión clara para que app/login.py sepa si debe permitir,
  bloquear o cerrar la sesión

Questo file centralizza la logica di controllo delle anomalie del sistema.

La sua funzione è:
- caricare il modello di activity
- valutare se un login sembra anomalo tramite regole statistiche semplici
- valutare se un'attività sembra anomala tramite il modello di machine learning
- restituire una decisione chiara affinché app/login.py sappia se deve permettere,
  bloccare o chiudere la sessione
"""

ACTIVITY_MODEL_PATH = PROJECT_ROOT / "models" / "activity_model.pkl"

# Umbral para bloqueo de actividad.
ACTIVITY_ANOMALY_THRESHOLD = 0.002 

# Umbral horario para login:
# si el acceso cae demasiado lejos de la media habitual del usuario,
# se considera extraño.
LOGIN_HOUR_DISTANCE_THRESHOLD = 3.0


def get_activity_model_bundle():
    """
    Carga el bundle de modelos de actividad.

    Carica il bundle di modelli di activity.
    """
    return load_model(str(ACTIVITY_MODEL_PATH))


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

    # Nos quedamos solo con logins correctos.
    # Consideriamo solo i login corretti.
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

        # Añadimos un pequeño margen de 30 minutos.
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

    Criterios:
    - día de la semana no habitual
    - hora fuera de la ventana habitual del usuario

    Devuelve:
    - is_anomalous
    - anomaly_score
    - message

    Valuta un tentativo di login confrontandolo con il pattern storico dell'utente.

    Criteri:
    - giorno della settimana non abituale
    - orario fuori dalla finestra abituale dell'utente

    Restituisce:
    - is_anomalous
    - anomaly_score
    - message
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
    logged_at=None
) -> dict:
    """
    Evalúa una actividad con el modelo y decide si debe bloquearse.

    Devuelve:
    - is_anomalous
    - prediction
    - anomaly_score
    - message

    Valuta un'attività con il modello e decide se deve essere bloccata.

    Restituisce:
    - is_anomalous
    - prediction
    - anomaly_score
    - message
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
            logged_at=logged_at
        )
    except Exception as e:
        return {
            "is_anomalous": False,
            "prediction": 0,
            "anomaly_score": 0.0,
            "message": f"Errore nel controllo ML dell'attività: {e}"
        }

    is_anomalous = bool(
        prediction == 1 and anomaly_score >= ACTIVITY_ANOMALY_THRESHOLD
    )

    return {
        "is_anomalous": is_anomalous,
        "prediction": int(prediction),
        "anomaly_score": float(anomaly_score),
        "message": "Azione strana, attendi conferma." if is_anomalous else ""
    }