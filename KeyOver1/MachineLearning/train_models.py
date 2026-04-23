# Entrena los modelos de actividad y sesión y los exporta como UN único archivo.
# Addestra i modelli di attività e sessione e li esporta come UN unico file.
#
# Resultado: models/combined_model.pkl con la estructura:
# Risultato: models/combined_model.pkl con la struttura:
#   {
#     'activity': { user_id: { 'model', 'feature_columns', 'score_scale', 'combo_frequency' } },
#     'session':  { user_id: { 'model', 'feature_columns', 'df_p1', 'df_p5', 'df_p95', 'anom_range' } }
#   }

from datetime import datetime
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.db import get_connection

COMBINED_MODEL_PATH = PROJECT_ROOT / "models" / "combined_model.pkl"
MIN_PROB = 1e-6
ACTIVITY_CONTAMINATION = 0.05
SESSION_CONTAMINATION  = 0.03


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS / CARICAMENTO DATI
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Carga login_log y activity_log desde PostgreSQL.
    # Carica login_log e activity_log da PostgreSQL.
    conn = get_connection()
    if conn is None:
        raise RuntimeError("Impossibile connettersi al database.")

    try:
        login_df = pd.read_sql(
            """
            SELECT login_log_id, user_id, result, logged_at, logout_at
            FROM login_log
            ORDER BY user_id, logged_at, login_log_id
            """,
            conn
        )
        activity_df = pd.read_sql(
            """
            SELECT activity_log_id, user_id, element_id, entity_id, action_id, logged_at
            FROM activity_log
            ORDER BY user_id, logged_at, activity_log_id
            """,
            conn
        )
    finally:
        conn.close()

    return login_df, activity_df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ASIGNACIÓN ACTIVIDADES → SESIONES / ASSEGNAZIONE ATTIVITÀ → SESSIONI
# ═══════════════════════════════════════════════════════════════════════════════

def assign_to_sessions(login_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    # Asigna cada actividad a su sesión de login más reciente y válida.
    # Assegna ogni attività alla sua sessione di login più recente e valida.
    # Regla: activity.logged_at ∈ [login.logged_at, login.logout_at], mismo user_id.
    # Regola: activity.logged_at ∈ [login.logged_at, login.logout_at], stesso user_id.
    if login_df.empty or activity_df.empty:
        return pd.DataFrame()

    login_df = login_df.copy()
    activity_df = activity_df.copy()

    login_df["logged_at"] = pd.to_datetime(login_df["logged_at"], errors="coerce")
    login_df["logout_at"] = pd.to_datetime(login_df["logout_at"], errors="coerce")
    activity_df["logged_at"] = pd.to_datetime(activity_df["logged_at"], errors="coerce")

    login_df = login_df[
        (login_df["result"] == True) &
        login_df["logged_at"].notna() &
        login_df["logout_at"].notna()
    ].copy()

    activity_df = activity_df[activity_df["logged_at"].notna()].copy()

    if login_df.empty or activity_df.empty:
        return pd.DataFrame()

    login_by_user = {
        int(uid): g.sort_values(["logged_at", "login_log_id"]).reset_index(drop=True)
        for uid, g in login_df.groupby("user_id")
    }
    activity_by_user = {
        int(uid): g.sort_values(["logged_at", "activity_log_id"]).reset_index(drop=True)
        for uid, g in activity_df.groupby("user_id")
    }

    rows = []
    for uid in sorted(set(login_by_user) & set(activity_by_user)):
        sessions = login_by_user[uid].to_dict("records")
        activities = activity_by_user[uid].to_dict("records")

        sess_idx = 0
        active = []

        for act in activities:
            act_time = act["logged_at"]
            while sess_idx < len(sessions) and sessions[sess_idx]["logged_at"] <= act_time:
                active.append(sessions[sess_idx])
                sess_idx += 1

            active = [s for s in active if s["logout_at"] >= act_time]
            if not active:
                continue

            best = max(active, key=lambda s: (s["logged_at"], s["login_log_id"]))
            rows.append({
                "login_log_id":    int(best["login_log_id"]),
                "user_id":         uid,
                "session_start":   best["logged_at"],
                "session_end":     best["logout_at"],
                "activity_log_id": int(act["activity_log_id"]),
                "element_id":      int(act["element_id"]),
                "entity_id":       int(act["entity_id"]),
                "action_id":       int(act["action_id"]),
                "activity_time":   act_time,
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["user_id", "login_log_id", "activity_time", "activity_log_id"]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODELO DE ACTIVIDAD / MODELLO DI ATTIVITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def _prepare_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    # Calcula timing por acción dentro de cada sesión.
    # Calcola il timing per azione all'interno di ogni sessione.
    df = df.copy()
    for col in ["session_start", "activity_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["session_start", "activity_time"])

    rows = []
    for (uid, lid), grp in df.groupby(["user_id", "login_log_id"], dropna=False):
        grp = grp.sort_values("activity_time")
        prev_time = None
        for _, row in grp.iterrows():
            at = row["activity_time"]
            ss = row["session_start"]
            if prev_time is None:
                secs_prev = max((at - ss).total_seconds(), 0.0)
            else:
                secs_prev = max((at - prev_time).total_seconds(), 0.0)

            rows.append({
                "user_id":         int(row["user_id"]),
                "login_log_id":    int(row["login_log_id"]),
                "activity_log_id": int(row["activity_log_id"]),
                "element_id":      int(row["element_id"]),
                "entity_id":       int(row["entity_id"]),
                "action_id":       int(row["action_id"]),
                "hour":            int(at.hour),
                "minute":          int(at.minute),
                "day_of_week":     int(at.dayofweek),
                "seconds_since_prev_action":    float(secs_prev),
                "seconds_since_session_start":  float(max((at - ss).total_seconds(), 0.0)),
                "activity_time":   at,
                "session_start":   ss,
            })
            prev_time = at

    return pd.DataFrame(rows)


def _build_feature_matrix_activity(df: pd.DataFrame) -> pd.DataFrame:
    # Construye la matriz de features para el modelo de actividad.
    # Costruisce la matrice delle feature per il modello di attività.
    feat = df[[
        "element_id", "entity_id", "action_id",
        "hour", "minute", "day_of_week",
        "seconds_since_prev_action", "seconds_since_session_start",
    ]].copy()

    feat["log_sec_prev"]  = np.log1p(feat["seconds_since_prev_action"])
    feat["log_sec_start"] = np.log1p(feat["seconds_since_session_start"])
    feat = feat.drop(columns=["seconds_since_prev_action", "seconds_since_session_start"])

    feat = pd.get_dummies(
        feat,
        columns=["element_id", "entity_id", "action_id", "day_of_week"],
        prefix=["element_id", "entity_id", "action_id", "day_of_week"],
        dtype=int,
    )
    return feat


def _build_combo_frequency(df: pd.DataFrame) -> dict:
    # Frecuencia histórica de combinaciones (elemento, entidad, acción).
    # Frequenza storica delle combinazioni (elemento, entità, azione).
    if df.empty:
        return {}
    combo = (
        df["element_id"].astype(str) + "|" +
        df["entity_id"].astype(str)  + "|" +
        df["action_id"].astype(str)
    )
    return {str(k): float(v) for k, v in combo.value_counts(normalize=True).items()}


def train_activity_models(prepared_df: pd.DataFrame) -> dict:
    # Entrena un IsolationForest por usuario para anomalías de actividad.
    # Addestra un IsolationForest per utente per anomalie di attività.
    user_models = {}
    for uid in sorted(prepared_df["user_id"].dropna().unique()):
        udf = prepared_df[prepared_df["user_id"] == uid].copy()
        if len(udf) < 20:
            print(f"  [activity] user {uid}: dati insufficienti ({len(udf)}), skip.")
            continue

        X = _build_feature_matrix_activity(udf)
        model = IsolationForest(n_estimators=250, contamination=ACTIVITY_CONTAMINATION, random_state=42)
        model.fit(X)

        raw = model.decision_function(X)
        scores = np.maximum(0.0, -raw)
        scale = float(np.percentile(scores, 95)) if len(scores) else 1.0
        scale = max(scale, MIN_PROB)

        element_freq = udf["element_id"].value_counts(normalize=True)
        known_elements = sorted([int(eid) for eid, freq in element_freq.items() if freq >= 0.01])

        user_models[int(uid)] = {
            "model":           model,
            "feature_columns": list(X.columns),
            "score_scale":     scale,
            "combo_frequency": _build_combo_frequency(udf),
            "known_elements":  known_elements,
        }
        print(f"  [activity] user {uid}: {len(udf):,} actividades | scale={scale:.6f} | elementos conocidos: {known_elements}")

    return user_models


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PUNTUAR ACTIVIDADES EN BATCH / PUNTEGGIO ATTIVITÀ IN BATCH
# ═══════════════════════════════════════════════════════════════════════════════

def score_activities_batch(prepared_df: pd.DataFrame, activity_models: dict) -> pd.DataFrame:
    # Aplica el modelo de actividad a todas las filas de forma vectorizada.
    # Applica il modello di attività a tutte le righe in modo vettoriale.
    parts = []
    for uid in sorted(prepared_df["user_id"].dropna().unique()):
        uid = int(uid)
        if uid not in activity_models:
            continue

        udf = prepared_df[prepared_df["user_id"] == uid].copy()
        md = activity_models[uid]
        model       = md["model"]
        feat_cols   = md["feature_columns"]
        scale       = md["score_scale"]
        combo_freq  = md["combo_frequency"]

        X = _build_feature_matrix_activity(udf)
        for c in feat_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[feat_cols]

        raw_pred  = model.predict(X)
        raw_score = model.decision_function(X)
        a_scores  = np.maximum(0.0, -raw_score)
        m_prob    = np.clip(a_scores / scale, 0.0, 1.0)

        combo_keys = (
            udf["element_id"].astype(str) + "|" +
            udf["entity_id"].astype(str)  + "|" +
            udf["action_id"].astype(str)
        )
        rarity = 1.0 - combo_keys.map(lambda k: float(combo_freq.get(k, 0.0))).to_numpy()
        probs  = np.maximum(0.75 * m_prob + 0.25 * rarity, MIN_PROB)

        udf = udf.copy()
        udf["activity_prediction"]  = np.where(raw_pred == -1, 1, 0).astype(int)
        udf["activity_probability"] = probs.astype(float)
        parts.append(udf)

    if not parts:
        return pd.DataFrame()

    scored = pd.concat(parts, ignore_index=True)
    scored["activity_time"]  = pd.to_datetime(scored["activity_time"],  errors="coerce")
    scored["session_start"]  = pd.to_datetime(scored["session_start"],  errors="coerce")
    return scored.sort_values(
        ["user_id", "login_log_id", "activity_time", "activity_log_id"]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODELO DE SESIÓN / MODELLO DI SESSIONE
# ═══════════════════════════════════════════════════════════════════════════════

def build_session_prefix_dataset(scored_df: pd.DataFrame) -> pd.DataFrame:
    # Genera una fila por cada punto acumulado dentro de cada sesión (prefijo).
    # Genera una riga per ogni punto accumulato all'interno di ogni sessione (prefisso).
    if scored_df.empty:
        return pd.DataFrame()

    df = scored_df.copy()
    df["activity_time"] = pd.to_datetime(df["activity_time"], errors="coerce")
    df["session_start"] = pd.to_datetime(df["session_start"], errors="coerce")
    df = df.dropna(subset=["activity_time", "session_start"])

    prefix_rows = []
    for (uid, lid), grp in df.groupby(["user_id", "login_log_id"], dropna=False):
        grp = grp.sort_values("activity_time")
        session_start = grp["session_start"].iloc[0]

        cost_parts:   list[float]         = []
        timestamps:   list[pd.Timestamp]  = []
        act_history:  list[int]           = []
        elem_history: list[int]           = []
        used_elems:   set[int]            = set()
        used_acts:    set[int]            = set()

        for _, row in grp.iterrows():
            at  = row["activity_time"]
            cost = max(float(row["activity_probability"]), MIN_PROB)
            pred = int(row["activity_prediction"])
            aid  = int(row["action_id"])
            eid  = int(row["element_id"])

            cost_parts.append(cost)
            timestamps.append(at)
            act_history.append(aid)
            elem_history.append(eid)
            used_elems.add(eid)
            used_acts.add(aid)

            n = len(cost_parts)
            cum_cost = float(sum(cost_parts))
            avg_cost = cum_cost / n
            max_cost = float(max(cost_parts))

            elapsed_min = max((at - session_start).total_seconds() / 60.0, 0.0001)
            apm = float(n / elapsed_min)

            diffs = [
                max((timestamps[i] - timestamps[i - 1]).total_seconds(), 0.0)
                for i in range(1, len(timestamps))
            ]
            if diffs:
                avg_sec = float(np.mean(diffs))
                min_sec = float(min(diffs))
                max_sec = float(max(diffs))
            else:
                fb = elapsed_min * 60.0
                avg_sec = min_sec = max_sec = float(fb)

            rep_act  = float(pd.Series(act_history).value_counts(normalize=True).max())
            rep_elem = float(pd.Series(elem_history).value_counts(normalize=True).max())
            start_hour = session_start.hour + session_start.minute / 60.0 + session_start.second / 3600.0

            prefix_rows.append({
                "user_id":                       int(uid),
                "login_log_id":                  int(lid),
                "action_count":                  n,
                "distinct_elements":             len(used_elems),
                "distinct_actions":              len(used_acts),
                "session_duration_min":          elapsed_min,
                "start_hour":                    float(start_hour),
                "day_of_week":                   int(session_start.weekday()),
                "actions_per_minute":            apm,
                "avg_seconds_between_actions":   avg_sec,
                "min_seconds_between_actions":   min_sec,
                "max_seconds_between_actions":   max_sec,
                "cumulative_cost":               cum_cost,
                "avg_cost":                      avg_cost,
                "max_cost":                      max_cost,
                "repeated_action_ratio":         rep_act,
                "repeated_element_ratio":        rep_elem,
                "last_activity_prediction":      pred,
            })

    return pd.DataFrame(prefix_rows)


_SESSION_FEATURE_COLS = [
    "action_count", "distinct_elements", "distinct_actions",
    "session_duration_min", "start_hour", "day_of_week",
    "actions_per_minute", "avg_seconds_between_actions",
    "min_seconds_between_actions", "max_seconds_between_actions",
    "cumulative_cost", "avg_cost", "max_cost",
    "repeated_action_ratio", "repeated_element_ratio",
]


def _build_feature_matrix_session(df: pd.DataFrame) -> pd.DataFrame:
    return df[_SESSION_FEATURE_COLS].copy()


def train_session_models(prefix_df: pd.DataFrame) -> dict:
    # Entrena un IsolationForest por usuario para anomalías de sesión.
    # Addestra un IsolationForest per utente per anomalie di sessione.
    user_models = {}
    for uid in sorted(prefix_df["user_id"].dropna().unique()):
        udf = prefix_df[prefix_df["user_id"] == uid].copy()
        if len(udf) < 10:
            print(f"  [session] user {uid}: dati insufficienti ({len(udf)}), skip.")
            continue

        X = _build_feature_matrix_session(udf)
        model = IsolationForest(n_estimators=250, contamination=SESSION_CONTAMINATION, random_state=42)
        model.fit(X)

        df_scores  = model.decision_function(X)
        df_p1      = float(np.percentile(df_scores, 1))
        df_p95     = float(np.percentile(df_scores, 95))
        anom_range = max(-df_p1, MIN_PROB)

        session_max_costs = udf.groupby("login_log_id")["cumulative_cost"].max()
        if len(session_max_costs) >= 5:
            cost_threshold = float(np.percentile(session_max_costs.values, 95))
        else:
            cost_threshold = float("inf")

        user_models[int(uid)] = {
            "model":                   model,
            "feature_columns":         list(X.columns),
            "df_p1":                   df_p1,
            "df_p95":                  df_p95,
            "anom_range":              anom_range,
            "session_cost_threshold":  cost_threshold,
        }
        threshold_display = f"{cost_threshold:.4f}" if cost_threshold != float("inf") else "∞"
        print(f"  [session] user {uid}: {len(udf):,} prefijos | df=[{df_p1:.4f}, {df_p95:.4f}] | anom_range={anom_range:.4f} | umbral_coste={threshold_display}")

    return user_models


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INFERENCIA (usada por anomaly_guard) / INFERENZA
# ═══════════════════════════════════════════════════════════════════════════════

def _align(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    # Alinea columnas de inferencia con las del entrenamiento.
    # Allinea le colonne di inferenza con quelle dell'addestramento.
    aligned = df.copy()
    for c in feature_cols:
        if c not in aligned.columns:
            aligned[c] = 0
    return aligned[feature_cols]


def predict_activity(
    combined_model: dict,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at: datetime,
    session_started_at: datetime | None = None,
    previous_timestamps: list | None = None,
) -> tuple[int, float]:
    # Predice si una acción individual es anómala. Devuelve (prediction, probability).
    # Predice se una singola azione è anomala. Restituisce (prediction, probability).
    bundle = combined_model.get("activity", {})
    uid = int(user_id)
    if uid not in bundle:
        raise ValueError(f"Nessun modello di attività per user_id={uid}")

    md = bundle[uid]
    previous_timestamps = previous_timestamps or []
    logged_at = pd.to_datetime(logged_at)
    session_started_at = pd.to_datetime(session_started_at or logged_at)

    if previous_timestamps:
        prev = pd.to_datetime(previous_timestamps[-1])
        sec_prev = max((logged_at - prev).total_seconds(), 0.0)
    else:
        sec_prev = max((logged_at - session_started_at).total_seconds(), 0.0)
    sec_start = max((logged_at - session_started_at).total_seconds(), 0.0)

    row = pd.DataFrame([{
        "element_id":  int(element_id),
        "entity_id":   int(entity_id),
        "action_id":   int(action_id),
        "hour":        int(logged_at.hour),
        "minute":      int(logged_at.minute),
        "day_of_week": int(logged_at.dayofweek),
        "seconds_since_prev_action":   float(sec_prev),
        "seconds_since_session_start": float(sec_start),
    }])

    X = _build_feature_matrix_activity(row)
    X = _align(X, md["feature_columns"])

    raw_pred  = md["model"].predict(X)[0]
    raw_score = float(md["model"].decision_function(X)[0])

    combo_key  = f"{int(element_id)}|{int(entity_id)}|{int(action_id)}"
    combo_freq = float(md["combo_frequency"].get(combo_key, 0.0))
    rarity     = 1.0 - combo_freq

    prediction  = 1 if raw_pred == -1 else 0
    a_score     = max(0.0, -raw_score)
    m_prob      = min(a_score / md["score_scale"], 1.0)
    probability = max(0.75 * m_prob + 0.25 * rarity, MIN_PROB)

    return prediction, probability


def predict_session(
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
) -> tuple[int, float]:
    # Predice si una sesión (parcial o completa) es anómala. Devuelve (prediction, score).
    # Predice se una sessione (parziale o completa) è anomala. Restituisce (prediction, score).
    bundle = combined_model.get("session", {})
    uid = int(user_id)
    if uid not in bundle:
        raise ValueError(f"Nessun modello di sessione per user_id={uid}")

    md = bundle[uid]

    row = pd.DataFrame([{
        "action_count":                  float(action_count),
        "distinct_elements":             float(distinct_elements),
        "distinct_actions":              float(distinct_actions),
        "session_duration_min":          float(session_duration_min),
        "start_hour":                    float(start_hour),
        "day_of_week":                   float(day_of_week),
        "actions_per_minute":            float(actions_per_minute),
        "avg_seconds_between_actions":   float(avg_seconds_between_actions),
        "min_seconds_between_actions":   float(min_seconds_between_actions),
        "max_seconds_between_actions":   float(max_seconds_between_actions),
        "cumulative_cost":               float(cumulative_cost),
        "avg_cost":                      float(avg_cost),
        "max_cost":                      float(max_cost),
        "repeated_action_ratio":         float(repeated_action_ratio),
        "repeated_element_ratio":        float(repeated_element_ratio),
    }])

    X = _align(row, md["feature_columns"])

    raw_pred  = md["model"].predict(X)[0]
    raw_score = float(md["model"].decision_function(X)[0])

    # Predicción binaria: -1 (sklearn) → 1 (anomalous), 1 → 0 (normal)
    # Previsione binaria: -1 (sklearn) → 1 (anomalo), 1 → 0 (normale)
    prediction = 1 if raw_pred == -1 else 0

    # Score continuo: umbral en 0 (igual que actividad). df>=0 normal→score~0; df<0 anómalo→score>0.
    # Punteggio continuo: soglia a 0 (come attività). df>=0 normale→score~0; df<0 anomalo→score>0.
    # anom_range = -df_p1 (valor máximo típico de anomalía en training) normaliza a [0, 1].
    # anom_range = -df_p1 (valore massimo tipico di anomalia nel training) normalizza a [0, 1].
    a_score = max(0.0, -raw_score)
    score   = float(np.clip(a_score / md["anom_range"], 0.0, 1.0))
    score   = max(score, MIN_PROB)

    return prediction, score


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GUARDAR / CARGAR MODELO / SALVATAGGIO / CARICAMENTO MODELLO
# ═══════════════════════════════════════════════════════════════════════════════

def save_combined_model(bundle: dict, path: Path = COMBINED_MODEL_PATH):
    # Guarda el modelo combinado en disco.
    # Salva il modello combinato su disco.
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"\nModello combinato salvato in: {path}")


def load_combined_model(path: Path = COMBINED_MODEL_PATH) -> dict:
    # Carga el modelo combinado desde disco.
    # Carica il modello combinato da disco.
    return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN / PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=== Addestramento modelli combinati (attività + sessione) ===\n")

    print("1/7 Caricamento dati da PostgreSQL...")
    login_df, activity_df = load_data()
    print(f"     login_log: {len(login_df):,} righe | activity_log: {len(activity_df):,} righe")

    print("2/7 Assegnazione attività a sessioni...")
    assigned_df = assign_to_sessions(login_df, activity_df)
    print(f"     {len(assigned_df):,} attività assegnate")

    if assigned_df.empty:
        print("[ERRORE] Nessun dato assegnato. Controlla login_log e activity_log.")
        return

    print("3/7 Preparazione feature di attività...")
    prepared_df = _prepare_activity_features(assigned_df)
    print(f"     {len(prepared_df):,} righe preparate")

    print("4/7 Addestramento modelli di attività...")
    activity_models = train_activity_models(prepared_df)
    if not activity_models:
        print("[ERRORE] Nessun modello di attività addestrato.")
        return

    print("5/7 Calcolo punteggi attività in batch...")
    scored_df = score_activities_batch(prepared_df, activity_models)
    print(f"     {len(scored_df):,} attività punteggiate")

    print("6/7 Costruzione dataset prefissi di sessione...")
    prefix_df = build_session_prefix_dataset(scored_df)
    print(f"     {len(prefix_df):,} prefissi generati")

    print("7/7 Addestramento modelli di sessione...")
    session_models = train_session_models(prefix_df)
    if not session_models:
        print("[ERRORE] Nessun modello di sessione addestrato.")
        return

    combined = {"activity": activity_models, "session": session_models}
    save_combined_model(combined)
    print("\nAddestramento completato con successo.")


if __name__ == "__main__":
    main()
