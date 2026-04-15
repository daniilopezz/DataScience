import random
from datetime import datetime, timedelta, time
import psycopg2
from psycopg2.extras import execute_batch

"""
Este archivo genera datos sintéticos para las tablas login_log y activity_log,
siguiendo unas reglas de negocio alineadas con el archivo rules.py.
El objetivo es simular tanto comportamientos normales como anómalos, manteniendo
un reparto controlado de anomalías para que el dataset resulte útil en pruebas,
análisis y entrenamiento de modelos.

Questo file genera dati sintetici per le tabelle login_log e activity_log,
seguendo regole di business allineate al file rules.py.
L'obiettivo è simulare sia comportamenti normali sia anomali, mantenendo
una distribuzione controllata delle anomalie in modo che il dataset risulti
utile per test, analisi e addestramento di modelli.
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
# PORCENTAJES DE ANOMALÍAS
# PERCENTUALI DI ANOMALIE
# =========================
# Los separamos para tener más control.
# Con estos valores, el resultado real suele quedar
# aproximadamente en el rango 7%-10% si el resto está bien alineado.
#
# Li separiamo per avere più controllo.
# Con questi valori, il risultato reale di solito si colloca
# approssimativamente nell'intervallo 7%-10% se il resto è ben allineato.
LOGIN_ANOMALY_RATE = 0.05
ACTIVITY_ANOMALY_RATE = 0.05

# =========================
# PROBABILIDAD DE LOGIN CORRECTO
# PROBABILITÀ DI LOGIN CORRETTO
# =========================
LOGIN_SUCCESS_RATE = 0.92

# =========================
# REGLAS ALINEADAS CON rules.py
# REGOLE ALLINEATE CON rules.py
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

# Conjuntos globales de elementos, entidades y acciones posibles.
# Se utilizan para elegir valores válidos o inválidos según el caso.
#
# Insiemi globali di elementi, entità e azioni possibili.
# Vengono utilizzati per scegliere valori validi o non validi a seconda del caso.
ALL_ELEMENTS = [1, 2, 3, 4, 5, 6]
ALL_ENTITIES = [1, 2, 3]
ALL_ACTIONS = [1000000, 1000001, 1000002, 1000003, 1000004, 1000005]


def get_connection():
    """
    Abre una conexión con PostgreSQL utilizando la configuración definida
    en DB_CONFIG.

    Apre una connessione a PostgreSQL utilizzando la configurazione definita
    in DB_CONFIG.
    """
    return psycopg2.connect(**DB_CONFIG)


def random_datetime_last_days(days_back=120):
    """
    Genera una fecha y hora aleatoria dentro de los últimos X días.
    Puede caer en cualquier día de la semana y en cualquier momento del día.

    Genera una data e un'ora casuale negli ultimi X giorni.
    Può cadere in qualsiasi giorno della settimana e in qualsiasi momento della giornata.
    """
    now = datetime.now()
    start = now - timedelta(days=days_back)
    delta = now - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def random_weekday_datetime_last_days(days_back=120):
    """
    Genera una fecha aleatoria dentro de los últimos X días,
    asegurando que caiga de lunes a viernes.

    Genera una data casuale negli ultimi X giorni,
    assicurandosi che cada tra lunedì e venerdì.
    """
    while True:
        dt = random_datetime_last_days(days_back)
        if dt.weekday() < 5:
            return dt


def random_datetime_in_schedule(base_date: datetime, start_t: time, end_t: time):
    """
    Genera una fecha y hora aleatoria dentro del horario principal del usuario.
    No incluye la tolerancia, por lo que representa un caso claramente normal.

    Genera una data e un'ora casuale all'interno dell'orario principale dell'utente.
    Non include la tolleranza, quindi rappresenta un caso chiaramente normale.
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
    Genera una fecha y hora claramente fuera del horario permitido,
    teniendo en cuenta también el margen de tolerancia.

    Genera una data e un'ora chiaramente fuori dall'orario consentito,
    tenendo conto anche del margine di tolleranza.
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
    Genera una fecha y hora aleatoria en fin de semana.

    Genera una data e un'ora casuale nel fine settimana.
    """
    while True:
        dt = random_datetime_last_days(days_back)
        if dt.weekday() >= 5:
            return dt


def build_normal_login_timestamp(profile):
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


def build_anomalous_login_timestamp(profile):
    """
    Genera un timestamp de login anómalo.
    Tipos posibles:
    - fin de semana
    - muy fuera de horario

    Genera un timestamp di login anomalo.
    Tipi possibili:
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


def generate_login_rows(num_logins):
    """
    Genera filas sintéticas para la tabla login_log.
    Devuelve:
    - login_rows: lista de registros listos para insertar en login_log
    - successful_sessions_normal: sesiones correctas y normales
    - successful_sessions_all: todas las sesiones correctas, tanto normales como anómalas

    Genera righe sintetiche per la tabella login_log.
    Restituisce:
    - login_rows: lista di record pronti per essere inseriti in login_log
    - successful_sessions_normal: sessioni corrette e normali
    - successful_sessions_all: tutte le sessioni corrette, sia normali sia anomale
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
            print(f"Login generati: {i + 1}")

    return login_rows, successful_sessions_normal, successful_sessions_all


def build_normal_activity_row(session, profile):
    """
    Construye una actividad claramente normal:
    - dentro de una sesión normal
    - entre semana
    - dentro del horario principal
    - con elemento, entidad y acción permitidos

    Costruisce un'attività chiaramente normale:
    - all'interno di una sessione normale
    - nei giorni lavorativi
    - all'interno dell'orario principale
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


def build_anomalous_activity_row(session, profile):
    """
    Construye una actividad anómala rompiendo una sola regla fuerte.
    Esto ayuda a que el dataset sea más realista y evita que el número total
    de anomalías aumente artificialmente por múltiples incumplimientos simultáneos.

    Costruisce un'attività anomala violando una sola regola forte.
    Questo aiuta a rendere il dataset più realistico ed evita che il numero totale
    di anomalie aumenti artificialmente a causa di più violazioni contemporanee.
    """
    session_seconds = int((session["end"] - session["start"]).total_seconds())
    offset_seconds = random.randint(0, max(1, session_seconds))
    base_logged_at = session["start"] + timedelta(seconds=offset_seconds)

    anomaly_type = random.choice(["element", "entity", "action", "time", "weekend"])

    # Partimos siempre de un caso válido y rompemos una sola regla.
    # Partiamo sempre da un caso valido e violiamo una sola regola.
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
    Genera las filas sintéticas para activity_log.

    Para los casos normales utiliza únicamente sesiones normales.
    Para los casos anómalos puede utilizar cualquier sesión correcta.

    Genera le righe sintetiche per activity_log.

    Per i casi normali utilizza solo sessioni normali.
    Per i casi anomali può utilizzare qualsiasi sessione corretta.
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


def insert_login_rows(cursor, login_rows):
    """
    Inserta por lotes las filas generadas para login_log.

    Inserisce in batch le righe generate per login_log.
    """
    query = """
        INSERT INTO login_log (user_id, result, attempt, logged_at, logout_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, query, login_rows, page_size=1000)


def insert_activity_rows(cursor, activity_rows):
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
    Flujo principal del script:
    1. Genera los registros de login.
    2. Genera los registros de actividad.
    3. Inserta ambos conjuntos de datos en PostgreSQL.

    Flusso principale dello script:
    1. Genera i record di login.
    2. Genera i record di attività.
    3. Inserisce entrambi i dataset in PostgreSQL.
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