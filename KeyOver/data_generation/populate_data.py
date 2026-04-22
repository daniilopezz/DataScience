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
introduciendo también un pequeño porcentaje de casos anómalos para que los modelos
puedan entrenarse y probarse sobre ejemplos más realistas.

En esta versión:
- se respetan horarios, elementos, entidades y acciones permitidas
- se incorporan pesos de frecuencia para reflejar hábitos reales
- se generan sesiones con volúmenes distintos según cada usuario
- se generan también sesiones con ritmo anómalo, por ejemplo tipo bot o ráfaga

Questo file genera dati sintetici per le tabelle login_log e activity_log,
seguendo pattern di comportamento definiti per ciascun utente.

L'obiettivo è costruire un dataset coerente con le abitudini normali di utilizzo,
introducendo anche una piccola percentuale di casi anomali affinché i modelli
possano essere addestrati e testati su esempi più realistici.

In questa versione:
- vengono rispettati orari, elementi, entità e azioni consentite
- vengono introdotti pesi di frequenza per riflettere abitudini reali
- vengono generate sessioni con volumi diversi per ciascun utente
- vengono generate anche sessioni con ritmo anomalo, per esempio tipo bot o raffica
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

NUM_LOGINS = 100_000
NUM_ACTIVITIES = 100_000

LOGIN_ANOMALY_RATE = 0.05
ACTIVITY_ANOMALY_RATE = 0.05
LOGIN_SUCCESS_RATE = 0.92

USER_PROFILES = {
    1: {
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

        "normal_session_minutes": (5, 35)
    },
    2: {
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

        "normal_session_minutes": (15, 90)
    },
    3: {
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

        "normal_session_minutes": (10, 60)
    }
}

ALL_ELEMENTS = [1, 2, 3, 4, 5, 6]
ALL_ENTITIES = [1, 2, 3]
ALL_ACTIONS = [1000000, 1000001, 1000002, 1000003, 1000004, 1000005]


def get_connection():
    """
    Abre una conexión con PostgreSQL usando la configuración definida.

    Apre una connessione a PostgreSQL usando la configurazione definita.
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


def weighted_choice(values: list[int], weights: list[float]) -> int:
    """
    Selecciona un valor usando una distribución ponderada.

    Seleziona un valore usando una distribuzione ponderata.
    """
    return random.choices(values, weights=weights, k=1)[0]


def build_normal_login_timestamp(profile: dict[str, Any]) -> datetime:
    """
    Genera un timestamp de login normal.

    Genera un timestamp di login normale.
    """
    base_dt = random_weekday_datetime_last_days(120)
    return random_datetime_in_schedule(base_dt, profile["schedule_start"], profile["schedule_end"])


def build_anomalous_login_timestamp(profile: dict[str, Any]) -> datetime:
    """
    Genera un timestamp de login anómalo.

    Genera un timestamp di login anomalo.
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


def choose_session_action_count(session: dict[str, Any], profile: dict[str, Any]) -> int:
    """
    Elige cuántas acciones tendrá una sesión según el usuario y si es normal o anómala.

    Sceglie quante azioni avrà una sessione in base all'utente e al fatto che sia normale o anomala.
    """
    if session["is_anomaly"]:
        return weighted_choice(
            profile["anomalous_session_action_choices"],
            profile["anomalous_session_action_weights"]
        )

    return weighted_choice(
        profile["normal_session_action_choices"],
        profile["normal_session_action_weights"]
    )


def generate_login_rows(num_logins: int):
    """
    Genera filas sintéticas para login_log.

    Devuelve:
    - login_rows
    - successful_sessions_all

    Genera righe sintetiche per login_log.

    Restituisce:
    - login_rows
    - successful_sessions_all
    """
    login_rows = []
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
            min_minutes, max_minutes = profile["normal_session_minutes"]
            session_minutes = random.randint(min_minutes, max_minutes)

            if is_anomaly and random.random() < 0.35:
                session_minutes = max(2, int(session_minutes * random.uniform(0.15, 0.45)))

            logout_at = logged_at + timedelta(minutes=session_minutes)

            successful_sessions_all.append({
                "user_id": user_id,
                "start": logged_at,
                "end": logout_at,
                "is_anomaly": is_anomaly
            })
        else:
            logout_at = None

        login_rows.append((user_id, result, attempt, logged_at, logout_at))

        if (i + 1) % 10000 == 0:
            print(f"Login generati: {i + 1}")

    return login_rows, successful_sessions_all


def build_normal_activity_values(profile: dict[str, Any]) -> tuple[int, int, int]:
    """
    Construye una combinación normal de elemento, entidad y acción.

    Costruisce una combinazione normale di elemento, entità e azione.
    """
    element_id = weighted_choice(profile["allowed_elements"], profile["element_weights"])
    entity_id = weighted_choice(profile["allowed_entities"], profile["entity_weights"])
    action_id = weighted_choice(profile["allowed_actions"], profile["action_weights"])
    return element_id, entity_id, action_id


def build_anomalous_activity_values(profile: dict[str, Any]) -> tuple[int, int, int]:
    """
    Construye una combinación anómala rompiendo una sola regla fuerte.

    Costruisce una combinazione anomala violando una sola regola forte.
    """
    anomaly_type = random.choices(
        ["element", "entity", "action"],
        weights=[3, 2, 3],
        k=1
    )[0]

    element_id, entity_id, action_id = build_normal_activity_values(profile)

    if anomaly_type == "element":
        invalid_elements = [e for e in ALL_ELEMENTS if e not in profile["allowed_elements"]]
        if invalid_elements:
            element_id = random.choice(invalid_elements)

    elif anomaly_type == "entity":
        invalid_entities = [e for e in ALL_ENTITIES if e not in profile["allowed_entities"]]
        if invalid_entities:
            entity_id = random.choice(invalid_entities)

    else:
        invalid_actions = [a for a in ALL_ACTIONS if a not in profile["allowed_actions"]]
        if invalid_actions:
            action_id = random.choice(invalid_actions)

    return element_id, entity_id, action_id


def build_session_action_timestamps(session: dict[str, Any], action_count: int) -> list[datetime]:
    """
    Genera timestamps crecientes para las acciones de una sesión.

    En sesiones normales:
    - las acciones se reparten de forma natural a lo largo de la sesión

    En algunas sesiones anómalas:
    - se generan ráfagas muy rápidas para simular comportamiento tipo bot

    Genera timestamp crescenti per le azioni di una sessione.

    Nelle sessioni normali:
    - le azioni vengono distribuite in modo naturale lungo la sessione

    In alcune sessioni anomale:
    - vengono generate raffiche molto rapide per simulare comportamento tipo bot
    """
    session_start = session["start"]
    session_end = session["end"]
    session_seconds = max(int((session_end - session_start).total_seconds()), 1)

    # Sesión anómala tipo bot / burst.
    # Sessione anomala tipo bot / burst.
    if session["is_anomaly"] and action_count >= 4 and random.random() < 0.45:
        burst_window = min(session_seconds, random.randint(4, 20))
        offsets = sorted(random.randint(0, max(0, burst_window)) for _ in range(action_count))
        return [session_start + timedelta(seconds=offset) for offset in offsets]

    # Sesión normal o anómala no burst.
    # Sessione normale o anomala non burst.
    offsets = sorted(random.randint(0, session_seconds) for _ in range(action_count))
    return [session_start + timedelta(seconds=offset) for offset in offsets]


def generate_activity_rows(num_activities: int, successful_sessions_all: list[dict[str, Any]]):
    """
    Genera filas sintéticas para activity_log de forma más realista.

    Estrategia:
    - cada sesión recibe un número de acciones coherente con el usuario
    - se generan secuencias de acciones dentro de la misma sesión
    - algunas sesiones anómalas presentan ráfagas rápidas tipo bot
    - las acciones anómalas siguen siendo minoría global

    Genera righe sintetiche per activity_log in modo più realistico.

    Strategia:
    - ogni sessione riceve un numero di azioni coerente con l'utente
    - vengono generate sequenze di azioni all'interno della stessa sessione
    - alcune sessioni anomale presentano raffiche rapide tipo bot
    - le azioni anomale restano una minoranza globale
    """
    activity_rows: list[tuple] = []

    if not successful_sessions_all:
        raise ValueError("Non ci sono sessioni corrette per generare attività.")

    sessions = successful_sessions_all.copy()
    random.shuffle(sessions)

    for session in sessions:
        if len(activity_rows) >= num_activities:
            break

        profile = USER_PROFILES[session["user_id"]]
        session_action_count = choose_session_action_count(session, profile)
        timestamps = build_session_action_timestamps(session, session_action_count)

        for ts in timestamps:
            if len(activity_rows) >= num_activities:
                break

            if session["is_anomaly"]:
                row_is_anomaly = random.random() < 0.28
            else:
                row_is_anomaly = random.random() < 0.02

            if row_is_anomaly:
                element_id, entity_id, action_id = build_anomalous_activity_values(profile)
            else:
                element_id, entity_id, action_id = build_normal_activity_values(profile)

            activity_rows.append((
                session["user_id"],
                element_id,
                entity_id,
                action_id,
                ts
            ))

        if len(activity_rows) % 10000 == 0 and len(activity_rows) > 0:
            print(f"Attività generate: {len(activity_rows)}")

    # Si faltan actividades, rellenamos usando sesiones ya existentes.
    # Se mancano attività, completiamo usando sessioni già esistenti.
    while len(activity_rows) < num_activities:
        session = random.choice(successful_sessions_all)
        profile = USER_PROFILES[session["user_id"]]
        ts = random.choice(build_session_action_timestamps(session, 3))

        row_is_anomaly = random.random() < ACTIVITY_ANOMALY_RATE

        if row_is_anomaly:
            element_id, entity_id, action_id = build_anomalous_activity_values(profile)
        else:
            element_id, entity_id, action_id = build_normal_activity_values(profile)

        activity_rows.append((
            session["user_id"],
            element_id,
            entity_id,
            action_id,
            ts
        ))

        if len(activity_rows) % 10000 == 0:
            print(f"Attività generate: {len(activity_rows)}")

    return activity_rows[:num_activities]


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
    login_rows, successful_sessions_all = generate_login_rows(NUM_LOGINS)

    print("Generazione di activity_log...")
    activity_rows = generate_activity_rows(NUM_ACTIVITIES, successful_sessions_all)

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