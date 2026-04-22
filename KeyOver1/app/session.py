# Gestiona la sesión activa del usuario: menú de elementos, menú de acciones,
# evaluación ML en tiempo real y logout automático por anomalía.
#
# Gestisce la sessione attiva dell'utente: menu elementi, menu azioni,
# valutazione ML in tempo reale e logout automatico per anomalia.

from datetime import datetime
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from security.anomaly_guard import evaluate_activity, evaluate_session

# ─── Mapeo de acciones / Mappatura azioni ─────────────────────────────────────
# (opción en menú) → (action_id en BD, etiqueta display)
# (opzione nel menu) → (action_id in DB, etichetta display)
ACTIONS: dict[str, tuple[int, str]] = {
    "1": (1000000, "Visualize"),
    "2": (1000001, "Create"),
    "3": (1000002, "Edit"),
    "4": (1000003, "Delete"),
    "5": (1000004, "Copy"),
    "6": (1000005, "Share"),
}

DEFAULT_ENTITY_ID = 1


# ─── Utilidades de BD / Utilità DB ────────────────────────────────────────────

def _save_activity_log(cursor, user_id: int, element_id: int, entity_id: int, action_id: int) -> int:
    # Guarda la acción en activity_log y devuelve activity_log_id.
    # Salva l'azione in activity_log e restituisce activity_log_id.
    cursor.execute(
        """
        INSERT INTO activity_log (user_id, element_id, entity_id, action_id, logged_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING activity_log_id
        """,
        (user_id, element_id, entity_id, action_id),
    )
    return cursor.fetchone()[0]


def _save_ml_log(
    cursor,
    activity_log_id: int,
    login_log_id: int,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    prediction: bool,
    anomaly_probability: float,
    session_cumulative_cost: float,
    session_threshold: float,
    threshold_exceeded: bool,
):
    # Guarda la predicción ML y el contexto de sesión en ml_prediction_log.
    # Salva la previsione ML e il contesto di sessione in ml_prediction_log.
    cursor.execute(
        """
        INSERT INTO ml_prediction_log (
            activity_log_id, login_log_id, user_id,
            element_id, entity_id, action_id,
            prediction, anomaly_probability,
            session_cumulative_cost, session_threshold, threshold_exceeded,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """,
        (
            activity_log_id, login_log_id, user_id,
            element_id, entity_id, action_id,
            bool(prediction), float(anomaly_probability),
            float(session_cumulative_cost), float(session_threshold), bool(threshold_exceeded),
        ),
    )


def _update_logout(cursor, login_log_id: int):
    # Registra logout_at en login_log.
    # Registra logout_at in login_log.
    cursor.execute(
        "UPDATE login_log SET logout_at = CURRENT_TIMESTAMP WHERE login_log_id = %s",
        (login_log_id,),
    )


def _get_elements(cursor) -> list[tuple[int, str]]:
    # Recupera la lista de elementos disponibles.
    # Recupera la lista degli elementi disponibili.
    cursor.execute("SELECT element_id, name FROM element ORDER BY element_id")
    return cursor.fetchall()


# ─── Features de sesión / Feature di sessione ─────────────────────────────────

def _build_session_features(
    session_started_at: datetime,
    cost_parts: list[float],
    action_timestamps: list[datetime],
    elements_used: set[int],
    action_ids_used: set[int],
    action_history: list[int],
    element_history: list[int],
) -> dict:
    # Calcula las 15 variables de sesión necesarias para el modelo de sesión.
    # Calcola le 15 variabili di sessione necessarie per il modello di sessione.
    now = datetime.now()
    n   = len(cost_parts)

    cum_cost = float(sum(cost_parts))
    avg_cost = float(cum_cost / n) if n else 0.0
    max_cost = float(max(cost_parts)) if cost_parts else 0.0

    elapsed_min = max((now - session_started_at).total_seconds() / 60.0, 0.0001)
    apm = float(n / elapsed_min)

    start_hour = (
        session_started_at.hour
        + session_started_at.minute / 60.0
        + session_started_at.second / 3600.0
    )

    diffs = []
    if len(action_timestamps) >= 2:
        for i in range(1, len(action_timestamps)):
            diffs.append(max((action_timestamps[i] - action_timestamps[i - 1]).total_seconds(), 0.0))

    if diffs:
        avg_sec = float(np.mean(diffs))
        min_sec = float(min(diffs))
        max_sec = float(max(diffs))
    else:
        fb = elapsed_min * 60.0
        avg_sec = min_sec = max_sec = float(fb)

    if n > 0 and action_history:
        import pandas as pd
        rep_act  = float(pd.Series(action_history).value_counts(normalize=True).max())
        rep_elem = float(pd.Series(element_history).value_counts(normalize=True).max())
    else:
        rep_act = rep_elem = 1.0

    return {
        "action_count":                  n,
        "distinct_elements":             len(elements_used),
        "distinct_actions":              len(action_ids_used),
        "session_duration_min":          elapsed_min,
        "start_hour":                    float(start_hour),
        "day_of_week":                   int(session_started_at.weekday()),
        "actions_per_minute":            apm,
        "avg_seconds_between_actions":   avg_sec,
        "min_seconds_between_actions":   min_sec,
        "max_seconds_between_actions":   max_sec,
        "cumulative_cost":               cum_cost,
        "avg_cost":                      avg_cost,
        "max_cost":                      max_cost,
        "repeated_action_ratio":         rep_act,
        "repeated_element_ratio":        rep_elem,
    }


# ─── Procesado de cada acción / Elaborazione di ogni azione ───────────────────

def _process_action(
    cursor,
    conn,
    combined_model,
    login_log_id: int,
    user_id: int,
    element_id: int,
    action_option: str,
    session_started_at: datetime,
    cost_parts: list[float],
    action_timestamps: list[datetime],
    elements_used: set[int],
    action_ids_used: set[int],
    action_history: list[int],
    element_history: list[int],
    session_cost_threshold: float = float("inf"),
) -> tuple[bool, list, list, set, set, list, list]:
    # Evalúa la acción con ML, guarda logs, detecta anomalía y decide logout.
    # Valuta l'azione con ML, salva i log, rileva anomalia e decide il logout.

    action_id, label = ACTIONS[action_option]
    now = datetime.now()

    if combined_model is None:
        # Sin modelo, solo registramos la actividad.
        # Senza modello, registriamo solo l'attività.
        act_id = _save_activity_log(cursor, user_id, element_id, DEFAULT_ENTITY_ID, action_id)
        conn.commit()
        print(f"\nAzione '{label}' registrata (ID: {act_id}). [Nessun modello ML]")
        new_ts = action_timestamps + [now]
        new_cp = cost_parts + [1e-6]
        new_eu = set(elements_used) | {element_id}
        new_au = set(action_ids_used) | {action_id}
        new_ah = action_history + [action_id]
        new_eh = element_history + [element_id]
        return True, new_cp, new_ts, new_eu, new_au, new_ah, new_eh

    try:
        # Evaluar actividad / Valutare attività
        act_result = evaluate_activity(
            combined_model=combined_model,
            user_id=user_id,
            element_id=element_id,
            entity_id=DEFAULT_ENTITY_ID,
            action_id=action_id,
            logged_at=now,
            session_started_at=session_started_at,
            previous_timestamps=action_timestamps,
        )

        op_cost     = float(act_result["anomaly_score"])
        new_cp      = cost_parts + [op_cost]
        new_ts      = action_timestamps + [now]
        new_eu      = set(elements_used) | {element_id}
        new_au      = set(action_ids_used) | {action_id}
        new_ah      = action_history + [action_id]
        new_eh      = element_history + [element_id]

        # Evaluar sesión / Valutare sessione
        sf = _build_session_features(
            session_started_at=session_started_at,
            cost_parts=new_cp,
            action_timestamps=new_ts,
            elements_used=new_eu,
            action_ids_used=new_au,
            action_history=new_ah,
            element_history=new_eh,
        )

        sess_result = evaluate_session(combined_model=combined_model, user_id=user_id, **sf)

        threshold_exceeded = sf["cumulative_cost"] >= session_cost_threshold
        final_pred = bool(
            act_result["prediction"] == 1
            or sess_result["prediction"] == 1
            or threshold_exceeded
        )

        act_log_id = _save_activity_log(cursor, user_id, element_id, DEFAULT_ENTITY_ID, action_id)
        _save_ml_log(
            cursor=cursor,
            activity_log_id=act_log_id,
            login_log_id=login_log_id,
            user_id=user_id,
            element_id=element_id,
            entity_id=DEFAULT_ENTITY_ID,
            action_id=action_id,
            prediction=final_pred,
            anomaly_probability=op_cost,
            session_cumulative_cost=sf["cumulative_cost"],
            session_threshold=session_cost_threshold if session_cost_threshold != float("inf") else 0.0,
            threshold_exceeded=threshold_exceeded,
        )
        conn.commit()

        # Mostrar estado ML / Mostrare stato ML
        cost_sum_str = " + ".join(f"{c:.4f}" for c in new_cp)
        threshold_display = f"{session_cost_threshold:.4f}" if session_cost_threshold != float("inf") else "∞"
        print(f"\nAzione eseguita: {label}")
        print(
            f"  [ML-ACTIVITY] {'ANOMALA' if act_result['prediction'] == 1 else 'normale'}"
            f" | costo azione: {op_cost:.4f}"
        )
        print(
            f"  [ML-SESSION]  {'ANOMALA' if sess_result['prediction'] == 1 else 'normale'}"
            f" | score={sess_result['anomaly_score']:.6f}"
            f" | apm={sf['actions_per_minute']:.3f}"
            f" | avg_sec={sf['avg_seconds_between_actions']:.1f}"
        )
        print(f"  costo sessione: {cost_sum_str} = {sf['cumulative_cost']:.4f}  [soglia: {threshold_display}]")
        print(f"  Attività registrata (ID: {act_log_id})")

        if act_result["prediction"] == 1:
            print("\n⚠  Attività anomala rilevata dal modello → logout automatico.")
            _update_logout(cursor, login_log_id)
            conn.commit()
            return False, new_cp, new_ts, new_eu, new_au, new_ah, new_eh

        if sess_result["prediction"] == 1:
            print("\n⚠  Sessione anomala rilevata dal modello → logout automatico.")
            _update_logout(cursor, login_log_id)
            conn.commit()
            return False, new_cp, new_ts, new_eu, new_au, new_ah, new_eh

        if threshold_exceeded:
            print(
                f"\n⚠  Soglia costo sessione superata"
                f" ({sf['cumulative_cost']:.4f} >= {session_cost_threshold:.4f}) → logout automatico."
            )
            _update_logout(cursor, login_log_id)
            conn.commit()
            return False, new_cp, new_ts, new_eu, new_au, new_ah, new_eh

        return True, new_cp, new_ts, new_eu, new_au, new_ah, new_eh

    except Exception as e:
        conn.rollback()
        print(f"[ERRORE] Elaborazione azione fallita: {e}")
        return False, cost_parts, action_timestamps, elements_used, action_ids_used, action_history, element_history


# ─── Menús / Menu ─────────────────────────────────────────────────────────────

def _show_element_menu(elements: list[tuple[int, str]]):
    print("\n=== MENU ELEMENTI ===")
    for eid, name in elements:
        print(f"  {eid} - {name}")
    print("  0 - Logout")


def _show_action_menu(element_name: str):
    print(f"\n=== AZIONI | {element_name} ===")
    print("  0 - Torna ai menu elementi")
    for opt, (_, label) in ACTIONS.items():
        print(f"  {opt} - {label}")


# ─── Bucle principal de sesión / Loop principale di sessione ──────────────────

def run_session(
    conn,
    cursor,
    user_id: int,
    login_log_id: int,
    combined_model,
    session_cost_threshold: float = float("inf"),
):
    # Loop interactivo principal tras el login exitoso.
    # Loop interattivo principale dopo il login avvenuto con successo.
    session_started_at = datetime.now()
    cost_parts:      list[float]    = []
    action_timestamps: list[datetime] = []
    elements_used:   set[int]       = set()
    action_ids_used: set[int]       = set()
    action_history:  list[int]      = []
    element_history: list[int]      = []

    while True:
        elements = _get_elements(cursor)
        element_map = {str(eid): (eid, name) for eid, name in elements}

        _show_element_menu(elements)
        choice = input("Seleziona un elemento: ").strip()

        if choice == "0":
            _update_logout(cursor, login_log_id)
            conn.commit()
            print("Logout registrato. Arrivederci!")
            break

        if choice not in element_map:
            print("Opzione non valida.")
            continue

        element_id, element_name = element_map[choice]
        print(f"\nElemento selezionato: {element_name}")

        # Sub-menú de acciones / Sotto-menu azioni
        while True:
            _show_action_menu(element_name)
            action_choice = input("Scegli un'azione: ").strip()

            if action_choice == "0":
                break

            if action_choice not in ACTIONS:
                print("Opzione non valida.")
                continue

            result = _process_action(
                cursor=cursor,
                conn=conn,
                combined_model=combined_model,
                login_log_id=login_log_id,
                user_id=user_id,
                element_id=element_id,
                action_option=action_choice,
                session_started_at=session_started_at,
                cost_parts=cost_parts,
                action_timestamps=action_timestamps,
                elements_used=elements_used,
                action_ids_used=action_ids_used,
                action_history=action_history,
                element_history=element_history,
                session_cost_threshold=session_cost_threshold,
            )

            still_active, cost_parts, action_timestamps, elements_used, action_ids_used, action_history, element_history = result

            if not still_active:
                return
