import random
from datetime import datetime, timedelta, time
import psycopg2
from psycopg2.extras import execute_batch

# =========================
# CONFIGURACIÓN DE LA BASE DE DATOS
# =========================
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

# =========================
# CANTIDAD DE DATOS A GENERAR
# =========================
NUM_LOGINS = 100_000
NUM_ACTIVITIES = 100_000

# =========================
# PORCENTAJES DE ANOMALÍAS
# =========================
# Los separamos para tener más control.
# Con estos valores, el resultado real suele quedar
# aproximadamente en el rango 7%-10% si el resto está bien alineado.
LOGIN_ANOMALY_RATE = 0.05
ACTIVITY_ANOMALY_RATE = 0.05

# =========================
# PROBABILIDAD DE LOGIN CORRECTO
# =========================
LOGIN_SUCCESS_RATE = 0.92

# =========================
# REGLAS ALINEADAS CON rules.py
# =========================
USER_PROFILES = {
    1: {  # Matteo
        "name": "Matteo",
        "schedule_start": time(9, 0),
        "schedule_end": time(13, 0),
        "tolerance_minutes": 15,
        "allowed_elements": [1, 2],
        "allowed_entities": [1],
        "allowed_actions": [1000000, 1000004, 1000005]  # Visualize, Copy, Share
    },
    2: {  # Diego
        "name": "Diego",
        "schedule_start": time(9, 0),
        "schedule_end": time(17, 0),
        "tolerance_minutes": 20,
        "allowed_elements": [3],
        "allowed_entities": [1, 2],
        "allowed_actions": [1000000, 1000001, 1000002, 1000005]  # Visualize, Create, Edit, Share
    },
    3: {  # Emilio
        "name": "Emilio",
        "schedule_start": time(10, 0),
        "schedule_end": time(18, 0),
        "tolerance_minutes": 20,
        "allowed_elements": [1, 4, 5, 6],
        "allowed_entities": [1, 3],
        "allowed_actions": [1000000, 1000002, 1000004, 1000005]  # Visualize, Edit, Copy, Share
    }
}

ALL_ELEMENTS = [1, 2, 3, 4, 5, 6]
ALL_ENTITIES = [1, 2, 3]
ALL_ACTIONS = [1000000, 1000001, 1000002, 1000003, 1000004, 1000005]


def get_connection():
    """
    Abre una conexión con PostgreSQL.
    """
    return psycopg2.connect(**DB_CONFIG)


def random_datetime_last_days(days_back=120):
    """
    Genera una fecha aleatoria dentro de los últimos X días.
    Puede caer en cualquier día de la semana.
    """
    now = datetime.now()
    start = now - timedelta(days=days_back)
    delta = now - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def random_weekday_datetime_last_days(days_back=120):
    """
    Genera una fecha aleatoria en los últimos X días,
    asegurando que caiga entre lunes y viernes.
    """
    while True:
        dt = random_datetime_last_days(days_back)
        if dt.weekday() < 5:
            return dt


def random_datetime_in_schedule(base_date: datetime, start_t: time, end_t: time):
    """
    Genera una fecha aleatoria dentro del horario principal del usuario.
    No incluye tolerancia: es un caso claramente normal.
    """
    start_minutes = start_t.hour * 60 + start_t.minute
    end_minutes = end_t.hour * 60 + end_t.minute

    current_minutes = random.randint(start_minutes, max(start_minutes, end_minutes - 1))
    hour = current_minutes // 60
    minute = current_minutes % 60
    second = random.randint(0, 59)

    return datetime.combine(base_date.date(), time(hour, minute, second))


def random_datetime_outside_tolerance(base_date: datetime, start_t: time, end_t: time, tolerance_minutes: int):
    """
    Genera una fecha claramente fuera del horario permitido,
    incluyendo la tolerancia.
    """
    start_minutes = start_t.hour * 60 + start_t.minute
    end_minutes = end_t.hour * 60 + end_t.minute

    allowed_start = start_minutes - tolerance_minutes
    allowed_end = end_minutes + tolerance_minutes

    possible_minutes = [
        m for m in range(24 * 60)
        if m < allowed_start or m > allowed_end
    ]

    selected_minutes = random.choice(possible_minutes)
    hour = selected_minutes // 60
    minute = selected_minutes % 60
    second = random.randint(0, 59)

    return datetime.combine(base_date.date(), time(hour, minute, second))


def random_weekend_datetime_last_days(days_back=120):
    """
    Genera una fecha aleatoria en fin de semana.
    """
    while True:
        dt = random_datetime_last_days(days_back)
        if dt.weekday() >= 5:
            return dt


def build_normal_login_timestamp(profile):
    """
    Genera un timestamp de login normal:
    - entre semana
    - dentro del horario principal
    """
    base_dt = random_weekday_datetime_last_days(120)
    return random_datetime_in_schedule(
        base_dt,
        profile["schedule_start"],
        profile["schedule_end"]
    )


def build_anomalous_login_timestamp(profile):
    """
    Genera un timestamp de login anómalo.
    Tipos posibles:
    - fin de semana
    - muy fuera de horario
    """
    anomaly_type = random.choice(["weekend", "outside_schedule"])

    if anomaly_type == "weekend":
        base_dt = random_weekend_datetime_last_days(120)
        return random_datetime_outside_tolerance(
            base_dt,
            profile["schedule_start"],
            profile["schedule_end"],
            profile["tolerance_minutes"]
        )

    base_dt = random_weekday_datetime_last_days(120)
    return random_datetime_outside_tolerance(
        base_dt,
        profile["schedule_start"],
        profile["schedule_end"],
        profile["tolerance_minutes"]
    )


def generate_login_rows(num_logins):
    """
    Genera filas sintéticas para login_log.

    Devuelve:
    - login_rows: lista de registros para insertar en login_log
    - successful_sessions_normal: sesiones correctas y normales
    - successful_sessions_all: todas las sesiones correctas
    """
    login_rows = []
    successful_sessions_normal = []
    successful_sessions_all = []

    user_ids = list(USER_PROFILES.keys())
    user_weights = [0.45, 0.30, 0.25]

    for i in range(num_logins):
        user_id = random.choices(user_ids, weights=user_weights, k=1)[0]
        profile = USER_PROFILES[user_id]

        is_anomaly = random.random() < LOGIN_ANOMALY_RATE
        result = random.random() < LOGIN_SUCCESS_RATE

        if result:
            attempt = 1
        else:
            attempt = random.choice([1, 2, 3])

        if is_anomaly:
            logged_at = build_anomalous_login_timestamp(profile)
        else:
            logged_at = build_normal_login_timestamp(profile)

        if result:
            session_minutes = random.randint(5, 90)
            logout_at = logged_at + timedelta(minutes=session_minutes)

            session_data = {
                "user_id": user_id,
                "start": logged_at,
                "end": logout_at,
                "is_anomaly": is_anomaly
            }

            successful_sessions_all.append(session_data)

            if not is_anomaly:
                successful_sessions_normal.append(session_data)
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

    return login_rows, successful_sessions_normal, successful_sessions_all


def build_normal_activity_row(session, profile):
    """
    Construye una actividad claramente normal:
    - dentro de una sesión normal
    - entre semana
    - dentro del horario principal
    - elemento, entidad y acción permitidos
    """
    session_seconds = int((session["end"] - session["start"]).total_seconds())
    offset_seconds = random.randint(0, max(1, session_seconds))
    logged_at = session["start"] + timedelta(seconds=offset_seconds)

    element_id = random.choice(profile["allowed_elements"])
    entity_id = random.choice(profile["allowed_entities"])
    action_id = random.choice(profile["allowed_actions"])

    return (
        session["user_id"],
        element_id,
        entity_id,
        action_id,
        logged_at
    )


def build_anomalous_activity_row(session, profile):
    """
    Construye una actividad anómala rompiendo una sola regla fuerte.
    Esto ayuda a que el dataset sea más realista y no se dispare el número
    total de anomalías por múltiples incumplimientos a la vez.
    """
    session_seconds = int((session["end"] - session["start"]).total_seconds())
    offset_seconds = random.randint(0, max(1, session_seconds))
    base_logged_at = session["start"] + timedelta(seconds=offset_seconds)

    anomaly_type = random.choice(["element", "entity", "action", "time", "weekend"])

    # Partimos siempre de un caso válido y rompemos una sola regla
    element_id = random.choice(profile["allowed_elements"])
    entity_id = random.choice(profile["allowed_entities"])
    action_id = random.choice(profile["allowed_actions"])
    logged_at = base_logged_at

    if anomaly_type == "element":
        invalid_elements = [e for e in ALL_ELEMENTS if e not in profile["allowed_elements"]]
        if invalid_elements:
            element_id = random.choice(invalid_elements)

    elif anomaly_type == "entity":
        invalid_entities = [e for e in ALL_ENTITIES if e not in profile["allowed_entities"]]
        if invalid_entities:
            entity_id = random.choice(invalid_entities)

    elif anomaly_type == "action":
        invalid_actions = [a for a in ALL_ACTIONS if a not in profile["allowed_actions"]]
        if invalid_actions:
            action_id = random.choice(invalid_actions)

    elif anomaly_type == "time":
        weekday_dt = random_weekday_datetime_last_days(120)
        logged_at = random_datetime_outside_tolerance(
            weekday_dt,
            profile["schedule_start"],
            profile["schedule_end"],
            profile["tolerance_minutes"]
        )

    else:  # weekend
        weekend_dt = random_weekend_datetime_last_days(120)
        logged_at = random_datetime_in_schedule(
            weekend_dt,
            profile["schedule_start"],
            profile["schedule_end"]
        )

    return (
        session["user_id"],
        element_id,
        entity_id,
        action_id,
        logged_at
    )


def generate_activity_rows(num_activities, successful_sessions_normal, successful_sessions_all):
    """
    Genera activity_log.

    Para los casos normales usa solo sesiones normales.
    Para los casos anómalos puede usar cualquier sesión correcta.
    """
    activity_rows = []

    if not successful_sessions_normal:
        raise ValueError("No hay sesiones normales suficientes para generar actividades normales.")

    if not successful_sessions_all:
        raise ValueError("No hay sesiones correctas para generar actividades.")

    for i in range(num_activities):
        is_anomaly = random.random() < ACTIVITY_ANOMALY_RATE

        if is_anomaly:
            session = random.choice(successful_sessions_all)
            profile = USER_PROFILES[session["user_id"]]
            row = build_anomalous_activity_row(session, profile)
        else:
            session = random.choice(successful_sessions_normal)
            profile = USER_PROFILES[session["user_id"]]
            row = build_normal_activity_row(session, profile)

        activity_rows.append(row)

        if (i + 1) % 10000 == 0:
            print(f"Activities generadas: {i + 1}")

    return activity_rows


def insert_login_rows(cursor, login_rows):
    """
    Inserta por lotes las filas generadas para login_log.
    """
    query = """
        INSERT INTO login_log (user_id, result, attempt, logged_at, logout_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, query, login_rows, page_size=1000)


def insert_activity_rows(cursor, activity_rows):
    """
    Inserta por lotes las filas generadas para activity_log.
    """
    query = """
        INSERT INTO activity_log (user_id, element_id, entity_id, action_id, logged_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, query, activity_rows, page_size=1000)


def main():
    """
    Flujo principal:
    1. Genera logins.
    2. Genera actividades.
    3. Inserta los datos en PostgreSQL.
    """
    print("Generando login_log...")
    login_rows, successful_sessions_normal, successful_sessions_all = generate_login_rows(NUM_LOGINS)

    print("Generando activity_log...")
    activity_rows = generate_activity_rows(
        NUM_ACTIVITIES,
        successful_sessions_normal,
        successful_sessions_all
    )

    connection = get_connection()
    cursor = connection.cursor()

    try:
        print("Insertando login_log en base de datos...")
        insert_login_rows(cursor, login_rows)

        print("Insertando activity_log en base de datos...")
        insert_activity_rows(cursor, activity_rows)

        connection.commit()
        print("Datos insertados correctamente.")

    except Exception as e:
        connection.rollback()
        print(f"Error durante la inserción: {e}")

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()