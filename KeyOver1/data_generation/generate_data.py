# Genera datos sintéticos ultra-realistas para login_log y activity_log.
#
# Estrategia:
# - Generación día a día por usuario: 1-3 sesiones por jornada laboral.
# - Sesiones no solapadas garantizadas dentro del mismo usuario y día.
# - Conteos de acciones realistas: Matteo 5-6, Diego 7-8, Emilio 10-12.
# - Anomalías claras: acceso rápido (1-2 acciones) o ráfaga bot (20-55 acciones).

import random
from datetime import datetime, timedelta, time
from pathlib import Path
import sys

from psycopg2.extras import execute_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.db import get_connection

# ─── Parámetros globales ──────────────────────────────────────────────────────
DAYS_BACK = 120
LOGIN_ANOMALY_RATE = 0.04   # 4% de días laborables son anómalos por usuario
ACTIVITY_ANOMALY_RATE = 0.02
LOGIN_SUCCESS_RATE = 0.92

# ─── Perfiles de usuario ──────────────────────────────────────────────────────
# IDs de elementos: 1=FVG, 2=AMCO, 3=VETTING, 4=RHODENSE, 5=PAPARDO, 6=PULEJO
# IDs de acciones:  1000000=Visualize, 1000001=Create, 1000002=Edit,
#                   1000003=Delete, 1000004=Copy, 1000005=Share
# IDs de entidades: 1=Website, 2=Database, 3=Server

USER_PROFILES = {
    1: {
        # Matteo: horario matutino estricto, usuario ligero (5-6 acciones/sesión)
        "name": "Matteo",
        "schedule_start": time(9, 0),
        "schedule_end": time(13, 0),
        "tolerance_minutes": 15,
        "sessions_per_day": (1, 2),
        "allowed_elements": [1, 2],
        "element_weights": [0.90, 0.10],
        "allowed_entities": [1],
        "entity_weights": [1.0],
        "allowed_actions": [1000000, 1000004, 1000005],
        "action_weights": [0.72, 0.18, 0.10],
        "normal_session_action_choices": [5, 6],
        "normal_session_action_weights": [0.55, 0.45],
        # Anomalías: acceso ultra-rápido (grab-and-go) o ráfaga bot
        "anomalous_session_action_choices": [1, 2, 20, 25, 30],
        "anomalous_session_action_weights": [0.20, 0.25, 0.25, 0.20, 0.10],
        "normal_session_minutes": (8, 25),
    },
    2: {
        # Diego: jornada completa, usuario moderado (7-8 acciones/sesión)
        "name": "Diego",
        "schedule_start": time(9, 0),
        "schedule_end": time(17, 0),
        "tolerance_minutes": 20,
        "sessions_per_day": (2, 3),
        "allowed_elements": [3],
        "element_weights": [1.0],
        "allowed_entities": [1, 2],
        "entity_weights": [0.82, 0.18],
        "allowed_actions": [1000000, 1000001, 1000002, 1000005],
        "action_weights": [0.40, 0.26, 0.22, 0.12],
        "normal_session_action_choices": [7, 8],
        "normal_session_action_weights": [0.50, 0.50],
        "anomalous_session_action_choices": [1, 2, 30, 40, 50],
        "anomalous_session_action_weights": [0.15, 0.20, 0.25, 0.25, 0.15],
        "normal_session_minutes": (12, 45),
    },
    3: {
        # Emilio: tarde, usuario operativo intensivo (10-12 acciones/sesión)
        "name": "Emilio",
        "schedule_start": time(10, 0),
        "schedule_end": time(18, 0),
        "tolerance_minutes": 20,
        "sessions_per_day": (1, 2),
        "allowed_elements": [1, 4, 5, 6],
        "element_weights": [0.58, 0.20, 0.14, 0.08],
        "allowed_entities": [1, 3],
        "entity_weights": [0.76, 0.24],
        "allowed_actions": [1000000, 1000002, 1000004, 1000005],
        "action_weights": [0.48, 0.22, 0.20, 0.10],
        "normal_session_action_choices": [10, 11, 12],
        "normal_session_action_weights": [0.33, 0.34, 0.33],
        "anomalous_session_action_choices": [1, 2, 35, 45, 55],
        "anomalous_session_action_weights": [0.15, 0.20, 0.25, 0.25, 0.15],
        "normal_session_minutes": (15, 55),
    },
}

ALL_ELEMENTS = [1, 2, 3, 4, 5, 6]
ALL_ENTITIES = [1, 2, 3]
ALL_ACTIONS = [1000000, 1000001, 1000002, 1000003, 1000004, 1000005]


# ─── Utilidades de tiempo ─────────────────────────────────────────────────────

def _in_schedule(base_dt: datetime, start_t: time, end_t: time) -> datetime:
    start_min = start_t.hour * 60 + start_t.minute
    end_min = end_t.hour * 60 + end_t.minute
    chosen = random.randint(start_min, max(start_min, end_min - 1))
    return datetime.combine(base_dt.date(), time(chosen // 60, chosen % 60, random.randint(0, 59)))


def _outside_schedule(base_dt: datetime, start_t: time, end_t: time, tolerance: int) -> datetime:
    start_min = start_t.hour * 60 + start_t.minute
    end_min = end_t.hour * 60 + end_t.minute
    allowed_start = start_min - tolerance
    allowed_end = end_min + tolerance
    candidates = [m for m in range(24 * 60) if m < allowed_start or m > allowed_end]
    chosen = random.choice(candidates)
    return datetime.combine(base_dt.date(), time(chosen // 60, chosen % 60, random.randint(0, 59)))


def _weighted(values: list, weights: list):
    return random.choices(values, weights=weights, k=1)[0]


# ─── Generación de logins ─────────────────────────────────────────────────────

def generate_login_rows():
    """
    Genera logins por usuario por día laborable con sesiones no solapadas.
    Cada usuario tiene 1-3 sesiones/día espaciadas en su ventana horaria.
    """
    login_rows = []
    successful_sessions = []

    now = datetime.now()
    start_date = (now - timedelta(days=DAYS_BACK)).date()
    end_date = now.date()

    for user_id, profile in USER_PROFILES.items():
        current = start_date
        while current <= end_date:
            if current.weekday() >= 5:   # saltar fines de semana
                current += timedelta(days=1)
                continue

            is_anomalous_day = random.random() < LOGIN_ANOMALY_RATE
            min_s, max_s = profile["sessions_per_day"]
            n_sessions = random.randint(min_s, max_s)
            last_end_min = None  # minutos desde medianoche del fin de la última sesión

            for _ in range(n_sessions):
                base = datetime.combine(current, time(12, 0))

                if is_anomalous_day and random.random() < 0.60:
                    dt = _outside_schedule(base, profile["schedule_start"],
                                           profile["schedule_end"], profile["tolerance_minutes"])
                else:
                    dt = _in_schedule(base, profile["schedule_start"], profile["schedule_end"])

                dt_min = dt.hour * 60 + dt.minute

                # Garantizar no solapamiento: empujar hacia adelante si es necesario
                if last_end_min is not None and dt_min < last_end_min + 5:
                    dt_min = last_end_min + random.randint(5, 20)
                    if dt_min >= 23 * 60:
                        break
                    dt = datetime.combine(current, time(dt_min // 60, dt_min % 60,
                                                        random.randint(0, 59)))

                result = random.random() < LOGIN_SUCCESS_RATE
                attempt = 1 if result else random.choice([1, 2, 3])

                if result:
                    min_m, max_m = profile["normal_session_minutes"]
                    session_min = random.randint(min_m, max_m)
                    if is_anomalous_day and random.random() < 0.35:
                        # Sesión anómala muy corta: acceso sospechoso
                        session_min = max(2, int(session_min * random.uniform(0.15, 0.45)))
                    logout_at = dt + timedelta(minutes=session_min)
                    last_end_min = logout_at.hour * 60 + logout_at.minute
                    successful_sessions.append({
                        "user_id": user_id,
                        "start": dt,
                        "end": logout_at,
                        "is_anomaly": is_anomalous_day,
                    })
                else:
                    logout_at = None

                login_rows.append((user_id, result, attempt, dt, logout_at))

            # ~5% de días: intento fallido extra (contraseña incorrecta)
            if random.random() < 0.05:
                base = datetime.combine(current, time(12, 0))
                fail_dt = _in_schedule(base, profile["schedule_start"], profile["schedule_end"])
                login_rows.append((user_id, False, random.choice([1, 2, 3]), fail_dt, None))

            current += timedelta(days=1)

    return login_rows, successful_sessions


# ─── Generación de actividades ────────────────────────────────────────────────

def _normal_activity(profile: dict) -> tuple:
    element_id = _weighted(profile["allowed_elements"], profile["element_weights"])
    entity_id = _weighted(profile["allowed_entities"], profile["entity_weights"])
    action_id = _weighted(profile["allowed_actions"], profile["action_weights"])
    return element_id, entity_id, action_id


def _anomalous_activity(profile: dict) -> tuple:
    # Viola una sola regla: elemento, entidad o acción no habitual
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
    start = session["start"]
    end = session["end"]
    total_secs = max(int((end - start).total_seconds()), 120)

    # Sesiones anómalas con muchas acciones → ráfaga bot (todas en pocos segundos)
    if session["is_anomaly"] and action_count >= 10 and random.random() < 0.50:
        idle_secs = random.randint(20, min(90, total_secs // 3))
        burst_window = random.randint(3, 20)
        offsets = sorted(
            idle_secs + random.randint(0, burst_window)
            for _ in range(action_count)
        )
    else:
        # Normal: mínimo 45 seg entre acciones, idle inicial de 30-90 seg
        min_gap = 45
        first_idle = random.randint(30, min(90, total_secs // 4))
        offsets = [first_idle]
        for _ in range(action_count - 1):
            avg_remaining = (total_secs - offsets[-1]) // max(action_count - len(offsets), 1)
            gap = random.randint(min_gap, max(min_gap + 1, avg_remaining))
            offsets.append(min(offsets[-1] + gap, total_secs - 1))

    return [start + timedelta(seconds=off) for off in offsets]


def _session_action_count(session: dict, profile: dict) -> int:
    if session["is_anomaly"]:
        return _weighted(profile["anomalous_session_action_choices"],
                         profile["anomalous_session_action_weights"])
    return _weighted(profile["normal_session_action_choices"],
                     profile["normal_session_action_weights"])


def generate_activity_rows(successful_sessions: list) -> list:
    """
    Genera actividades directamente desde sesiones exitosas.
    Sin cap artificial: cada sesión produce su conteo natural de acciones.
    """
    activity_rows = []
    for session in successful_sessions:
        profile = USER_PROFILES[session["user_id"]]
        action_count = _session_action_count(session, profile)
        timestamps = _session_timestamps(session, action_count)

        for ts in timestamps:
            row_is_anomaly = (
                random.random() < 0.28 if session["is_anomaly"] else random.random() < 0.02
            )
            if row_is_anomaly:
                element_id, entity_id, action_id = _anomalous_activity(profile)
            else:
                element_id, entity_id, action_id = _normal_activity(profile)

            activity_rows.append((session["user_id"], element_id, entity_id, action_id, ts))

    return activity_rows


# ─── Inserción en BD ──────────────────────────────────────────────────────────

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Generación de datos sintéticos (ultra-realistas) ===\n")

    print(f"Generando logins para los últimos {DAYS_BACK} días...")
    login_rows, successful_sessions = generate_login_rows()
    print(f"  → {len(login_rows):,} logins ({len(successful_sessions):,} sesiones válidas)\n")

    print("Generando actividades desde sesiones...")
    activity_rows = generate_activity_rows(successful_sessions)
    print(f"  → {len(activity_rows):,} actividades generadas\n")

    print("Resumen por usuario:")
    for uid, profile in USER_PROFILES.items():
        user_sessions = [s for s in successful_sessions if s["user_id"] == uid]
        if not user_sessions:
            continue
        normal_s = [s for s in user_sessions if not s["is_anomaly"]]
        anomaly_s = [s for s in user_sessions if s["is_anomaly"]]
        user_acts = sum(1 for r in activity_rows if r[0] == uid)
        avg_acts = user_acts / len(user_sessions)
        print(f"  {profile['name']:8s}: {len(user_sessions):3d} sesiones "
              f"({len(normal_s)} normales / {len(anomaly_s)} anómalas) | "
              f"{user_acts:4d} acciones | ~{avg_acts:.1f} acc/sesión")
    print()

    connection = get_connection()
    if connection is None:
        print("[ERROR] No se pudo conectar a la base de datos.")
        return

    cursor = connection.cursor()
    try:
        print("Insertando login_log en la base de datos...")
        _insert_logins(cursor, login_rows)

        print("Insertando activity_log en la base de datos...")
        _insert_activities(cursor, activity_rows)

        connection.commit()
        print("\nDatos insertados correctamente en la base de datos.")

    except Exception as e:
        connection.rollback()
        print(f"[ERROR] Inserción fallida: {e}")

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
