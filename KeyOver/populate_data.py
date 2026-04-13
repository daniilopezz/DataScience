import random
from datetime import datetime, timedelta, time
import psycopg2
from psycopg2.extras import execute_batch

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

NUM_LOGINS = 10_000
NUM_ACTIVITIES = 10_000
ANOMALY_RATE = 0.05  # 5% anomalías

# Reglas actuales de tu proyecto
USER_PROFILES = {
    1: {  # Matteo
        "name": "Matteo",
        "login_window": (9, 12),
        "allowed_elements": [1, 2],
        "allowed_actions": [1000000, 1000004, 1000005]  # Visualize, Copy, Share
    },
    2: {  # Diego
        "name": "Diego",
        "login_window": (12, 15),
        "allowed_elements": [3],
        "allowed_actions": [1000000, 1000004, 1000005]
    },
    3: {  # Emilio
        "name": "Emilio",
        "login_window": (15, 18),
        "allowed_elements": [4, 1, 5, 6],
        "allowed_actions": [1000000, 1000004, 1000005]
    }
}

ALL_ELEMENTS = [1, 2, 3, 4, 5, 6]
ALL_ACTIONS = [1000000, 1000001, 1000002, 1000003, 1000004, 1000005]
ENTITY_ID = 1


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def random_datetime_last_days(days_back=120):
    """Genera una fecha aleatoria dentro de los últimos X días."""
    now = datetime.now()
    start = now - timedelta(days=days_back)
    delta = now - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def random_datetime_in_window(base_date, start_hour, end_hour):
    """Genera una fecha en una franja horaria concreta."""
    hour = random.randint(start_hour, max(start_hour, end_hour - 1))
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime.combine(base_date.date(), time(hour, minute, second))


def random_datetime_outside_window(base_date, start_hour, end_hour):
    """Genera una fecha fuera de la franja horaria permitida."""
    possible_hours = [h for h in range(24) if h < start_hour or h >= end_hour]
    hour = random.choice(possible_hours)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime.combine(base_date.date(), time(hour, minute, second))


def generate_login_rows(num_logins):
    """
    Genera filas sintéticas para login_log.
    Devuelve:
    - login_rows: datos para insertar en login_log
    - successful_sessions: sesiones válidas para poder generar activity_log realista
    """
    login_rows = []
    successful_sessions = []

    user_ids = list(USER_PROFILES.keys())
    user_weights = [0.45, 0.30, 0.25]  # Matteo más frecuente, luego Diego, luego Emilio

    for i in range(num_logins):
        user_id = random.choices(user_ids, weights=user_weights, k=1)[0]
        profile = USER_PROFILES[user_id]

        base_dt = random_datetime_last_days(120)
        is_anomaly = random.random() < ANOMALY_RATE

        # 90% de los logins correctos
        result = random.random() < 0.90

        # attempt realista
        if result:
            attempt = 1
        else:
            attempt = random.choice([1, 2, 3])

        start_hour, end_hour = profile["login_window"]

        if is_anomaly:
            logged_at = random_datetime_outside_window(base_dt, start_hour, end_hour)
        else:
            logged_at = random_datetime_in_window(base_dt, start_hour, end_hour)

        if result:
            session_minutes = random.randint(5, 90)
            logout_at = logged_at + timedelta(minutes=session_minutes)

            successful_sessions.append({
                "user_id": user_id,
                "start": logged_at,
                "end": logout_at
            })
        else:
            logout_at = None

        login_rows.append((
            user_id,
            result,
            attempt,
            logged_at,
            logout_at
        ))

        if (i + 1) % 10000 == 0:
            print(f"Logins generados: {i + 1}")

    return login_rows, successful_sessions


def generate_activity_rows(num_activities, successful_sessions):
    """
    Genera activity_log usando sesiones válidas para que el comportamiento
    sea más realista.
    """
    activity_rows = []

    if not successful_sessions:
        raise ValueError("No hay sesiones correctas para generar actividades.")

    for i in range(num_activities):
        session = random.choice(successful_sessions)
        user_id = session["user_id"]
        profile = USER_PROFILES[user_id]

        is_anomaly = random.random() < ANOMALY_RATE

        # timestamp dentro de la sesión
        session_seconds = int((session["end"] - session["start"]).total_seconds())
        offset_seconds = random.randint(0, max(1, session_seconds))
        logged_at = session["start"] + timedelta(seconds=offset_seconds)

        # normal o anómalo
        if is_anomaly:
            anomaly_type = random.choice(["element", "action", "time"])

            if anomaly_type == "element":
                invalid_elements = [e for e in ALL_ELEMENTS if e not in profile["allowed_elements"]]
                element_id = random.choice(invalid_elements) if invalid_elements else random.choice(ALL_ELEMENTS)
                action_id = random.choice(profile["allowed_actions"])

            elif anomaly_type == "action":
                invalid_actions = [a for a in ALL_ACTIONS if a not in profile["allowed_actions"]]
                action_id = random.choice(invalid_actions) if invalid_actions else random.choice(ALL_ACTIONS)
                element_id = random.choice(profile["allowed_elements"])

            else:  # time anomaly
                element_id = random.choice(profile["allowed_elements"])
                action_id = random.choice(profile["allowed_actions"])
                start_hour, end_hour = profile["login_window"]
                logged_at = random_datetime_outside_window(logged_at, start_hour, end_hour)

        else:
            element_id = random.choice(profile["allowed_elements"])
            action_id = random.choice(profile["allowed_actions"])

        activity_rows.append((
            user_id,
            element_id,
            ENTITY_ID,
            action_id,
            logged_at
        ))

        if (i + 1) % 10000 == 0:
            print(f"Activities generadas: {i + 1}")

    return activity_rows


def insert_login_rows(cursor, login_rows):
    query = """
        INSERT INTO login_log (user_id, result, attempt, logged_at, logout_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, query, login_rows, page_size=1000)


def insert_activity_rows(cursor, activity_rows):
    query = """
        INSERT INTO activity_log (user_id, element_id, entity_id, action_id, logged_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, query, activity_rows, page_size=1000)


def main():
    print("Generando login_log...")
    login_rows, successful_sessions = generate_login_rows(NUM_LOGINS)

    print("Generando activity_log...")
    activity_rows = generate_activity_rows(NUM_ACTIVITIES, successful_sessions)

    connection = get_connection()
    cursor = connection.cursor()

    try:
        print("Insertando login_log en base de datos...")
        insert_login_rows(cursor, login_rows)
        connection.commit()
        print("login_log insertado correctamente.")

        print("Insertando activity_log en base de datos...")
        insert_activity_rows(cursor, activity_rows)
        connection.commit()
        print("activity_log insertado correctamente.")

    except Exception as e:
        connection.rollback()
        print(f"Error durante la inserción: {e}")

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()