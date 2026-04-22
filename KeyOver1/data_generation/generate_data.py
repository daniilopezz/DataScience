# Genera datos sintéticos realistas para login_log y activity_log.
# Genera dati sintetici realistici per login_log e activity_log.
#
# Estrategia:
# - Cada usuario tiene un horario, elementos, entidades y acciones preferidas.
# - Se generan sesiones normales y anómalas con tasas configurables.
# - Las anomalías incluyen: fuera de horario, acceso a elementos no habituales,
#   acciones raras, ritmo tipo bot/ráfaga.
#
# Strategia:
# - Ogni utente ha un orario, elementi, entità e azioni preferite.
# - Vengono generate sessioni normali e anomale con tassi configurabili.
# - Le anomalie includono: fuori orario, accesso a elementi non abituali,
#   azioni rare, ritmo tipo bot/raffica.

import random
from datetime import datetime, timedelta, time
from pathlib import Path
import sys

from psycopg2.extras import execute_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.db import get_connection

# ─── Parámetros globales / Parametri globali ───────────────────────────────────
NUM_LOGINS = 100_000
NUM_ACTIVITIES = 100_000

LOGIN_ANOMALY_RATE = 0.05
ACTIVITY_ANOMALY_RATE = 0.05
LOGIN_SUCCESS_RATE = 0.92

# ─── Perfiles de usuario / Profili utente ─────────────────────────────────────
# IDs de elementos: 1=FVG, 2=AMCO, 3=VETTING, 4=RHODENSE, 5=PAPARDO, 6=PULEJO
# IDs de acciones:  1000000=Visualize, 1000001=Create, 1000002=Edit,
#                   1000003=Delete, 1000004=Copy, 1000005=Share
# IDs de entidades: 1=Website, 2=Database, 3=Server

USER_PROFILES = {
    1: {
        # Matteo: horario matutino estricto, pocas acciones, elementos FVG y AMCO
        # Matteo: orario mattutino stretto, poche azioni, elementi FVG e AMCO
        "name": "Matteo",
        "schedule_start": time(9, 0),
        "schedule_end": time(13, 0),
        "tolerance_minutes": 15,
        "allowed_elements": [1, 2],
        "element_weights": [0.90, 0.10],
        "allowed_entities": [1],
        "entity_weights": [1.0],
        "allowed_actions": [1000000, 1000004, 1000005],
        "action_weights": [0.72, 0.18, 0.10],
        "normal_session_action_choices": [1, 2, 3, 4, 5, 6],
        "normal_session_action_weights": [0.08, 0.22, 0.30, 0.22, 0.12, 0.06],
        "anomalous_session_action_choices": [5, 6, 7, 8, 9, 10],
        "anomalous_session_action_weights": [0.12, 0.18, 0.22, 0.20, 0.16, 0.12],
        "normal_session_minutes": (5, 35),
    },
    2: {
        # Diego: jornada completa, muchas acciones, solo elemento VETTING
        # Diego: giornata completa, molte azioni, solo elemento VETTING
        "name": "Diego",
        "schedule_start": time(9, 0),
        "schedule_end": time(17, 0),
        "tolerance_minutes": 20,
        "allowed_elements": [3],
        "element_weights": [1.0],
        "allowed_entities": [1, 2],
        "entity_weights": [0.82, 0.18],
        "allowed_actions": [1000000, 1000001, 1000002, 1000005],
        "action_weights": [0.40, 0.26, 0.22, 0.12],
        "normal_session_action_choices": [4, 5, 6, 7, 8, 9, 10, 11, 12],
        "normal_session_action_weights": [0.04, 0.08, 0.12, 0.16, 0.18, 0.16, 0.12, 0.08, 0.06],
        "anomalous_session_action_choices": [10, 11, 12, 13, 14, 15, 16],
        "anomalous_session_action_weights": [0.08, 0.12, 0.18, 0.20, 0.18, 0.14, 0.10],
        "normal_session_minutes": (15, 90),
    },
    3: {
        # Emilio: tarde, sesiones medianas, varios elementos
        # Emilio: pomeriggio, sessioni medie, vari elementi
        "name": "Emilio",
        "schedule_start": time(10, 0),
        "schedule_end": time(18, 0),
        "tolerance_minutes": 20,
        "allowed_elements": [1, 4, 5, 6],
        "element_weights": [0.58, 0.20, 0.14, 0.08],
        "allowed_entities": [1, 3],
        "entity_weights": [0.76, 0.24],
        "allowed_actions": [1000000, 1000002, 1000004, 1000005],
        "action_weights": [0.48, 0.22, 0.20, 0.10],
        "normal_session_action_choices": [2, 3, 4, 5, 6, 7, 8],
        "normal_session_action_weights": [0.08, 0.14, 0.22, 0.22, 0.16, 0.10, 0.08],
        "anomalous_session_action_choices": [7, 8, 9, 10, 11, 12],
        "anomalous_session_action_weights": [0.12, 0.20, 0.22, 0.20, 0.16, 0.10],
        "normal_session_minutes": (10, 60),
    },
}

ALL_ELEMENTS = [1, 2, 3, 4, 5, 6]
ALL_ENTITIES = [1, 2, 3]
ALL_ACTIONS = [1000000, 1000001, 1000002, 1000003, 1000004, 1000005]


# ─── Utilidades de tiempo / Utilità di tempo ──────────────────────────────────

def _random_dt_in_window(days_back: int = 120) -> datetime:
    now = datetime.now()
    start = now - timedelta(days=days_back)
    delta = int((now - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, delta))


def _weekday_dt(days_back: int = 120) -> datetime:
    while True:
        dt = _random_dt_in_window(days_back)
        if dt.weekday() < 5:
            return dt


def _weekend_dt(days_back: int = 120) -> datetime:
    while True:
        dt = _random_dt_in_window(days_back)
        if dt.weekday() >= 5:
            return dt


def _in_schedule(base_dt: datetime, start_t: time, end_t: time) -> datetime:
    # Combina fecha base con hora aleatoria dentro del horario.
    # Combina data base con ora casuale all'interno dell'orario.
    start_min = start_t.hour * 60 + start_t.minute
    end_min = end_t.hour * 60 + end_t.minute
    chosen = random.randint(start_min, max(start_min, end_min - 1))
    return datetime.combine(base_dt.date(), time(chosen // 60, chosen % 60, random.randint(0, 59)))


def _outside_schedule(base_dt: datetime, start_t: time, end_t: time, tolerance: int) -> datetime:
    # Genera hora claramente fuera de la ventana permitida.
    # Genera un'ora chiaramente al di fuori della finestra consentita.
    start_min = start_t.hour * 60 + start_t.minute
    end_min = end_t.hour * 60 + end_t.minute
    allowed_start = start_min - tolerance
    allowed_end = end_min + tolerance
    candidates = [m for m in range(24 * 60) if m < allowed_start or m > allowed_end]
    chosen = random.choice(candidates)
    return datetime.combine(base_dt.date(), time(chosen // 60, chosen % 60, random.randint(0, 59)))


def _weighted(values: list, weights: list):
    return random.choices(values, weights=weights, k=1)[0]


# ─── Generación de logins / Generazione di login ──────────────────────────────

def _normal_login_ts(profile: dict) -> datetime:
    base = _weekday_dt()
    return _in_schedule(base, profile["schedule_start"], profile["schedule_end"])


def _anomalous_login_ts(profile: dict) -> datetime:
    if random.random() < 0.5:
        base = _weekend_dt()
    else:
        base = _weekday_dt()
    return _outside_schedule(base, profile["schedule_start"], profile["schedule_end"], profile["tolerance_minutes"])


def generate_login_rows(num_logins: int):
    # Genera filas sintéticas para login_log.
    # Genera righe sintetiche per login_log.
    login_rows = []
    successful_sessions = []

    user_ids = list(USER_PROFILES.keys())
    user_weights = [0.45, 0.30, 0.25]

    for i in range(num_logins):
        user_id = _weighted(user_ids, user_weights)
        profile = USER_PROFILES[user_id]

        is_anomaly = random.random() < LOGIN_ANOMALY_RATE
        result = random.random() < LOGIN_SUCCESS_RATE
        attempt = 1 if result else random.choice([1, 2, 3])

        logged_at = _anomalous_login_ts(profile) if is_anomaly else _normal_login_ts(profile)

        if result:
            min_m, max_m = profile["normal_session_minutes"]
            session_min = random.randint(min_m, max_m)
            if is_anomaly and random.random() < 0.35:
                session_min = max(2, int(session_min * random.uniform(0.15, 0.45)))
            logout_at = logged_at + timedelta(minutes=session_min)
            successful_sessions.append({
                "user_id": user_id,
                "start": logged_at,
                "end": logout_at,
                "is_anomaly": is_anomaly,
            })
        else:
            logout_at = None

        login_rows.append((user_id, result, attempt, logged_at, logout_at))

        if (i + 1) % 10_000 == 0:
            print(f"  Login generati: {i + 1:,}")

    return login_rows, successful_sessions


# ─── Generación de actividades / Generazione di attività ──────────────────────

def _normal_activity(profile: dict) -> tuple:
    element_id = _weighted(profile["allowed_elements"], profile["element_weights"])
    entity_id = _weighted(profile["allowed_entities"], profile["entity_weights"])
    action_id = _weighted(profile["allowed_actions"], profile["action_weights"])
    return element_id, entity_id, action_id


def _anomalous_activity(profile: dict) -> tuple:
    # Rompe una sola regla: elemento, entidad o acción no habitual.
    # Viola una sola regola: elemento, entità o azione non abituali.
    anomaly_type = random.choices(["element", "entity", "action"], weights=[3, 2, 3], k=1)[0]
    element_id, entity_id, action_id = _normal_activity(profile)

    if anomaly_type == "element":
        invalid = [e for e in ALL_ELEMENTS if e not in profile["allowed_elements"]]
        if invalid:
            element_id = random.choice(invalid)
    elif anomaly_type == "entity":
        invalid = [e for e in ALL_ENTITIES if e not in profile["allowed_entities"]]
        if invalid:
            entity_id = random.choice(invalid)
    else:
        invalid = [a for a in ALL_ACTIONS if a not in profile["allowed_actions"]]
        if invalid:
            action_id = random.choice(invalid)

    return element_id, entity_id, action_id


def _session_timestamps(session: dict, action_count: int) -> list:
    # Genera timestamps dentro de la sesión. Las sesiones anómalas pueden tener ráfagas.
    # Genera timestamp all'interno della sessione. Le sessioni anomale possono avere raffiche.
    #
    # Sesiones normales: mínimo 45 seg entre acciones + idle inicial de 30-120 seg.
    # Sessioni normali: minimo 45 sec tra azioni + idle iniziale di 30-120 sec.
    #
    # Ráfaga anómala: acciones en 3-20 seg tras un breve idle inicial.
    # Raffica anomala: azioni in 3-20 sec dopo un breve idle iniziale.
    start = session["start"]
    end = session["end"]
    total_secs = max(int((end - start).total_seconds()), 120)

    if session["is_anomaly"] and action_count >= 4 and random.random() < 0.50:
        # Burst: todas las acciones en una ventana muy corta (3-20 seg)
        idle_secs = random.randint(20, min(90, total_secs // 3))
        burst_window = random.randint(3, 20)
        offsets = sorted(
            idle_secs + random.randint(0, burst_window)
            for _ in range(action_count)
        )
    else:
        # Normal: mínimo 45 seg entre acciones, primer idle 30-120 seg
        min_gap = 45
        first_idle = random.randint(30, min(120, total_secs // 4))
        offsets = [first_idle]
        for _ in range(action_count - 1):
            avg_remaining = (total_secs - offsets[-1]) // max(action_count - len(offsets), 1)
            gap = random.randint(min_gap, max(min_gap + 1, avg_remaining))
            offsets.append(min(offsets[-1] + gap, total_secs - 1))

    return [start + timedelta(seconds=off) for off in offsets]


def _session_action_count(session: dict, profile: dict) -> int:
    if session["is_anomaly"]:
        return _weighted(profile["anomalous_session_action_choices"], profile["anomalous_session_action_weights"])
    return _weighted(profile["normal_session_action_choices"], profile["normal_session_action_weights"])


def generate_activity_rows(num_activities: int, successful_sessions: list) -> list:
    # Genera filas sintéticas para activity_log agrupadas por sesión.
    # Genera righe sintetiche per activity_log raggruppate per sessione.
    activity_rows = []
    sessions = successful_sessions.copy()
    random.shuffle(sessions)

    for session in sessions:
        if len(activity_rows) >= num_activities:
            break

        profile = USER_PROFILES[session["user_id"]]
        action_count = _session_action_count(session, profile)
        timestamps = _session_timestamps(session, action_count)

        for ts in timestamps:
            if len(activity_rows) >= num_activities:
                break

            row_is_anomaly = (
                random.random() < 0.28 if session["is_anomaly"] else random.random() < 0.02
            )

            if row_is_anomaly:
                element_id, entity_id, action_id = _anomalous_activity(profile)
            else:
                element_id, entity_id, action_id = _normal_activity(profile)

            activity_rows.append((session["user_id"], element_id, entity_id, action_id, ts))

        if len(activity_rows) % 10_000 == 0 and len(activity_rows) > 0:
            print(f"  Attività generate: {len(activity_rows):,}")

    # Relleno si faltan actividades / Completamento se mancano attività
    while len(activity_rows) < num_activities:
        session = random.choice(successful_sessions)
        profile = USER_PROFILES[session["user_id"]]
        ts = random.choice(_session_timestamps(session, 3))
        row_is_anomaly = random.random() < ACTIVITY_ANOMALY_RATE
        if row_is_anomaly:
            element_id, entity_id, action_id = _anomalous_activity(profile)
        else:
            element_id, entity_id, action_id = _normal_activity(profile)
        activity_rows.append((session["user_id"], element_id, entity_id, action_id, ts))

    return activity_rows[:num_activities]


# ─── Inserción en BD / Inserimento in DB ──────────────────────────────────────

def _insert_logins(cursor, rows: list):
    sql = """
        INSERT INTO login_log (user_id, result, attempt, logged_at, logout_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, sql, rows, page_size=1000)


def _insert_activities(cursor, rows: list):
    sql = """
        INSERT INTO activity_log (user_id, element_id, entity_id, action_id, logged_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, sql, rows, page_size=1000)


# ─── Main / Principale ────────────────────────────────────────────────────────

def main():
    print("=== Generazione dati sintetici ===\n")

    print(f"Generazione di {NUM_LOGINS:,} login...")
    login_rows, successful_sessions = generate_login_rows(NUM_LOGINS)
    print(f"  → {len(login_rows):,} login generati ({len(successful_sessions):,} sessioni valide)\n")

    print(f"Generazione di {NUM_ACTIVITIES:,} attività...")
    activity_rows = generate_activity_rows(NUM_ACTIVITIES, successful_sessions)
    print(f"  → {len(activity_rows):,} attività generate\n")

    connection = get_connection()
    if connection is None:
        print("[ERRORE] Impossibile connettersi al database.")
        return

    cursor = connection.cursor()
    try:
        print("Inserimento login_log nel database...")
        _insert_logins(cursor, login_rows)

        print("Inserimento activity_log nel database...")
        _insert_activities(cursor, activity_rows)

        connection.commit()
        print("\nDati inseriti correttamente nel database.")

    except Exception as e:
        connection.rollback()
        print(f"[ERRORE] Inserimento fallito: {e}")

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
