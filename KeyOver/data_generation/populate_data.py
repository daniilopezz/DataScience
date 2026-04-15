import random
from datetime import datetime, timedelta, time
from typing import Any

import psycopg2
from psycopg2 import Error
from psycopg2.extras import execute_batch

"""
Este archivo genera datos sintéticos para las tablas login_log y activity_log,
siguiendo patrones de comportamiento definidos para cada usuario.

El objetivo es construir un dataset coherente con los hábitos normales de uso,
introduciendo también un pequeño porcentaje de casos anómalos para que el modelo
pueda entrenarse y probarse sobre ejemplos más realistas.

En este caso, las reglas no se utilizan para detectar anomalías en tiempo real,
sino como referencia para construir datos sintéticos con sentido.

Questo file genera dati sintetici per le tabelle login_log e activity_log,
seguendo pattern di comportamento definiti per ciascun utente.

L'obiettivo è costruire un dataset coerente con le abitudini normali di utilizzo,
introducendo anche una piccola percentuale di casi anomali affinché il modello
possa essere addestrato e testato su esempi più realistici.

In questo caso, le regole non vengono utilizzate per rilevare anomalie in tempo reale,
ma come riferimento per costruire dati sintetici sensati.
"""

# =========================
# CONFIGURACIÓN DE LA BASE DE DATOS
# CONFIGURAZIONE DEL DATABASE
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
# QUANTITÀ DI DATI DA GENERARE
# =========================
NUM_LOGINS = 100_000
NUM_ACTIVITIES = 100_000

# =========================
# PORCENTAJES DE DATOS ANÓMALOS
# PERCENTUALI DI DATI ANOMALI
# =========================
LOGIN_ANOMALY_RATE = 0.05
ACTIVITY_ANOMALY_RATE = 0.05

# =========================
# PROBABILIDAD DE LOGIN CORRECTO
# PROBABILITÀ DI LOGIN CORRETTO
# =========================
LOGIN_SUCCESS_RATE = 0.92

# =========================
# PERFILES DE COMPORTAMIENTO
# PROFILI DI COMPORTAMENTO
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
    Abre una conexión con PostgreSQL usando la configuración definida en DB_CONFIG.

    Apre una connessione a PostgreSQL usando la configurazione definita in DB_CONFIG.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def random_datetime_last_days(days_back: int = 120) -> datetime:
    """
    Genera una fecha y hora aleatoria dentro de los últimos X días.

    Genera una data e un'ora casuale negli ultimi X giorni.
    """
    now = datetime.now()
    start = now - timedelta(days=days_back)
    delta_seconds = int((now - start).total_seconds())
    random_seconds = random.randint(0, delta_seconds)
    return start + timedelta(seconds=random_seconds)


def random_weekday_datetime_last_days(days_back: int = 120) -> datetime:
    """
    Genera una fecha aleatoria en día laborable.

    Genera una data casuale in un giorno lavorativo.
    """
    while True:
        dt = random_datetime_last_days(days_back)
        if dt.weekday() < 5:
            return dt


def random_weekend_datetime_last_days(days_back: int = 120) -> datetime:
    """
    Genera una fecha aleatoria en fin de semana.

    Genera una data casuale nel fine settimana.
    """
    while True:
        dt = random_datetime_last_days(days_back)
        if dt.weekday() >= 5:
            return dt


def random_datetime_in_schedule(base_date: datetime, start_t: time, end_t: time) -> datetime:
    """
    Genera una fecha y hora aleatoria dentro del horario principal del usuario.

    Genera una data e un'ora casuale all'interno dell'orario principale dell'utente.
    """
    start_minutes = start_t.hour * 60 + start_t.minute
    end_minutes = end_t.hour * 60 + end_t.minute

    selected_minutes = random.randint(start_minutes, max(start_minutes, end_minutes - 1))
    hour = selected_minutes // 60
    minute = selected_minutes % 60
    second = random.randint(0, 59)

    return datetime.combine(base_date.date(), time(hour, minute, second))


def random_datetime_outside_tolerance(
    base_date: datetime,
    start_t: time,
    end_t: time,
    tolerance_minutes: int
) -> datetime:
    """
    Genera una fecha y hora claramente fuera del horario permitido,
    incluyendo la tolerancia.

    Genera una data e un'ora chiaramente fuori dall'orario consentito,
    includendo la tolleranza.
    """
    start_minutes = start_t.hour * 60 + start_t.minute
    end_minutes = end_t.hour * 60 + end_t.minute

    allowed_start = start_minutes - tolerance_minutes
    allowed_end = end_minutes + tolerance_minutes

    possible_minutes = [
        minute_value
        for minute_value in range(24 * 60)
        if minute_value < allowed_start or minute_value > allowed_end
    ]

    selected_minutes = random.choice(possible_minutes)
    hour = selected_minutes // 60
    minute = selected_minutes % 60
    second = random.randint(0, 59)

    return datetime.combine(base_date.date(), time(hour, minute, second))


def build_normal_login_timestamp(profile: dict[str, Any]) -> datetime:
    """
    Genera un timestamp de login normal:
    - entre semana
    - dentro del horario principal del usuario

    Genera un timestamp di login normale:
    - nei giorni lavorativi
    - all'interno dell'orario principale dell'utente
    """
    base_dt = random_weekday_datetime_last_days(120)
    return random_datetime_in_schedule(
        base_dt,
        profile["schedule_start"],
        profile["schedule_end"]
    )


def build_anomalous_login_timestamp(profile: dict[str, Any]) -> datetime:
    """
    Genera un timestamp de login anómalo.

    Casos posibles:
    - fin de semana
    - muy fuera de horario

    Genera un timestamp di login anomalo.

    Casi possibili:
    - fine settimana
    - molto fuori orario
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


def generate_login_rows(num_logins: int):
    """
    Genera filas sintéticas para login_log.

    Devuelve:
    - login_rows
    - successful_sessions_normal
    - successful_sessions_all

    Genera righe sintetiche per login_log.

    Restituisce:
    - login_rows
    - successful_sessions_normal
    - successful_sessions_all
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

        attempt = 1 if result else random.choice([1, 2, 3])

        logged_at = (
            build_anomalous_login_timestamp(profile)
            if is_anomaly
            else build_normal_login_timestamp(profile)
        )

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
            print(f"Login generati: {i + 1}")

    return login_rows, successful_sessions_normal, successful_sessions_all


def build_normal_activity_row(session: dict[str, Any], profile: dict[str, Any]):
    """
    Construye una actividad claramente normal:
    - dentro de una sesión normal
    - en horario habitual
    - con elemento, entidad y acción permitidos

    Costruisce un'attività chiaramente normale:
    - all'interno di una sessione normale
    - nell'orario abituale
    - con elemento, entità e azione consentiti
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


def build_anomalous_activity_row(session: dict[str, Any], profile: dict[str, Any]):
    """
    Construye una actividad anómala rompiendo una sola regla fuerte,
    para evitar generar casos artificialmente exagerados.

    Costruisce un'attività anomala violando una sola regola forte,
    per evitare di generare casi artificialmente eccessivi.
    """
    session_seconds = int((session["end"] - session["start"]).total_seconds())
    offset_seconds = random.randint(0, max(1, session_seconds))
    base_logged_at = session["start"] + timedelta(seconds=offset_seconds)

    anomaly_type = random.choice(["element", "entity", "action", "time", "weekend"])

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


def generate_activity_rows(
    num_activities: int,
    successful_sessions_normal: list[dict[str, Any]],
    successful_sessions_all: list[dict[str, Any]]
):
    """
    Genera filas sintéticas para activity_log.

    Los casos normales usan solo sesiones normales.
    Los casos anómalos pueden usar cualquier sesión correcta.

    Genera righe sintetiche per activity_log.

    I casi normali usano solo sessioni normali.
    I casi anomali possono usare qualsiasi sessione corretta.
    """
    activity_rows = []

    if not successful_sessions_normal:
        raise ValueError("Non ci sono sessioni normali sufficienti per generare attività normali.")

    if not successful_sessions_all:
        raise ValueError("Non ci sono sessioni corrette per generare attività.")

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
            print(f"Attività generate: {i + 1}")

    return activity_rows


def insert_login_rows(cursor, login_rows: list[tuple]):
    """
    Inserta por lotes las filas generadas para login_log.

    Inserisce in batch le righe generate per login_log.
    """
    query = """
        INSERT INTO login_log (user_id, result, attempt, logged_at, logout_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, query, login_rows, page_size=1000)


def insert_activity_rows(cursor, activity_rows: list[tuple]):
    """
    Inserta por lotes las filas generadas para activity_log.

    Inserisce in batch le righe generate per activity_log.
    """
    query = """
        INSERT INTO activity_log (user_id, element_id, entity_id, action_id, logged_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, query, activity_rows, page_size=1000)


def main():
    """
    Flujo principal:
    1. Genera logins sintéticos.
    2. Genera actividades sintéticas.
    3. Inserta los datos en PostgreSQL.

    Flusso principale:
    1. Genera login sintetici.
    2. Genera attività sintetiche.
    3. Inserisce i dati in PostgreSQL.
    """
    print("Generazione di login_log...")
    login_rows, successful_sessions_normal, successful_sessions_all = generate_login_rows(NUM_LOGINS)

    print("Generazione di activity_log...")
    activity_rows = generate_activity_rows(
        NUM_ACTIVITIES,
        successful_sessions_normal,
        successful_sessions_all
    )

    connection = get_connection()
    if connection is None:
        return

    cursor = connection.cursor()

    try:
        print("Inserimento di login_log nel database...")
        insert_login_rows(cursor, login_rows)

        print("Inserimento di activity_log nel database...")
        insert_activity_rows(cursor, activity_rows)

        connection.commit()
        print("Dati inseriti correttamente.")

    except Exception as e:
        connection.rollback()
        print(f"Errore durante l'inserimento: {e}")

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()