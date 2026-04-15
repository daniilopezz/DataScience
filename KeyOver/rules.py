from datetime import datetime, time

"""
Este archivo define las reglas de negocio utilizadas para detectar anomalías
en los eventos de login y en las actividades realizadas por los usuarios.

Se validan los siguientes casos:
- acceso en fin de semana
- acceso fuera del horario laboral
- acceso dentro o fuera del margen de tolerancia horaria
- acceso a elementos no permitidos
- acceso a entidades no permitidas
- ejecución de acciones no permitidas

Questo file definisce le regole di business utilizzate per rilevare anomalie
negli eventi di login e nelle attività eseguite dagli utenti.

Vengono controllati i seguenti casi:
- accesso nel fine settimana
- accesso fuori dall'orario di lavoro
- accesso entro o fuori dal margine di tolleranza oraria
- accesso a elementi non consentiti
- accesso a entità non consentite
- esecuzione di azioni non consentite
"""

USER_NAMES = {
    1: "Matteo Nicolosi",
    2: "Diego Scardino",
    3: "Emilio Sardo"
}

ACTION_NAMES = {
    1000000: "Visualize",
    1000001: "Create",
    1000002: "Edit",
    1000003: "Delete",
    1000004: "Copy",
    1000005: "Share"
}

ENTITY_NAMES = {
    1: "Password",
    2: "Folder",
    3: "Group"
}

# Horarios reales asignados a cada usuario.
# "start" y "end" representan la franja horaria habitual de trabajo.
# "tolerance_minutes" indica el margen permitido antes y después del horario normal.
#
# Orari reali assegnati a ciascun utente.
# "start" e "end" rappresentano la fascia oraria abituale di lavoro.
# "tolerance_minutes" indica il margine consentito prima e dopo l'orario normale.
USER_SCHEDULES = {
    1: {"start": time(9, 0), "end": time(13, 0), "tolerance_minutes": 15},   # Matteo
    2: {"start": time(9, 0), "end": time(17, 0), "tolerance_minutes": 20},   # Diego
    3: {"start": time(10, 0), "end": time(18, 0), "tolerance_minutes": 20}   # Emilio
}
 
# Elementos a los que cada usuario tiene permiso de acceso.
#
# Elementi ai quali ciascun utente ha il permesso di accedere.
ALLOWED_ELEMENTS = {
    1: [1, 2],          # Matteo -> FVG, AMCO
    2: [3],             # Diego -> VETTING
    3: [1, 4, 5, 6]     # Emilio -> FVG, RHODENSE, PAPARDO, PULEJO
}

# Entidades permitidas para cada usuario.
# 
# Entità consentite per ciascun utente.
ALLOWED_ENTITIES = {
    1: [1],        # Matteo -> Password
    2: [1, 2],     # Diego -> Password, Folder
    3: [1, 3]      # Emilio -> Password, Group
}

# Acciones que cada usuario puede ejecutar según las reglas de negocio.
# 
# Azioni che ciascun utente può eseguire secondo le regole di business.
ALLOWED_ACTIONS = {
    1: [1000000, 1000004, 1000005],                  # Matteo -> Visualize, Copy, Share
    2: [1000000, 1000001, 1000002, 1000005],         # Diego -> Visualize, Create, Edit, Share
    3: [1000000, 1000002, 1000004, 1000005]          # Emilio -> Visualize, Edit, Copy, Share
}


def is_weekday(dt: datetime) -> bool:
    """
    Devuelve True si la fecha proporcionada corresponde a un día laborable,
    es decir, de lunes a viernes.

    Restituisce True se la data fornita corrisponde a un giorno lavorativo,
    cioè dal lunedì al venerdì.
    """
    return dt.weekday() in [0, 1, 2, 3, 4]


def is_within_main_schedule(user_id: int, dt: datetime) -> bool:
    """
    Comprueba si un usuario está accediendo dentro de su horario laboral habitual.
    Si el usuario no tiene un horario definido en USER_SCHEDULES, se considera válido.

    Controlla se un utente sta accedendo all'interno del proprio orario di lavoro abituale.
    Se l'utente non ha un orario definito in USER_SCHEDULES, l'accesso viene considerato valido.
    """
    if user_id not in USER_SCHEDULES:
        return True

    schedule = USER_SCHEDULES[user_id]
    current_time = dt.time()
    return schedule["start"] <= current_time <= schedule["end"]


def is_within_tolerance_schedule(user_id: int, dt: datetime) -> bool:
    """
    Comprueba si un acceso o una actividad se realiza dentro del horario permitido
    considerando el margen de tolerancia configurado para el usuario.

    Primero convierte las horas a minutos para facilitar la comparación:
    - hora de inicio
    - hora de fin
    - hora actual

    Después verifica si la hora actual está entre:
    (inicio - tolerancia) y (fin + tolerancia).

    Si el usuario no tiene un horario definido, se considera válido.

    Controlla se un accesso o un'attività viene effettuato entro l'orario consentito
    considerando il margine di tolleranza configurato per l'utente.

    Per farlo converte gli orari in minuti così da semplificare il confronto:
    - orario di inizio
    - orario di fine
    - orario attuale

    Successivamente verifica se l'orario attuale rientra tra:
    (inizio - tolleranza) e (fine + tolleranza).

    Se l'utente non ha un orario definito, viene considerato valido.
    """
    if user_id not in USER_SCHEDULES:
        return True

    schedule = USER_SCHEDULES[user_id]
    tolerance = schedule["tolerance_minutes"]

    start_minutes = schedule["start"].hour * 60 + schedule["start"].minute
    end_minutes = schedule["end"].hour * 60 + schedule["end"].minute
    current_minutes = dt.hour * 60 + dt.minute

    return (start_minutes - tolerance) <= current_minutes <= (end_minutes + tolerance)


def is_allowed_element(user_id: int, element_id: int) -> bool:
    """
    Verifica si el elemento indicado está permitido para el usuario.
    Si el usuario no tiene restricciones definidas, el acceso se considera válido.

    Verifica se l'elemento indicato è consentito per l'utente.
    Se l'utente non ha restrizioni definite, l'accesso viene considerato valido.
    """
    if user_id not in ALLOWED_ELEMENTS:
        return True

    return element_id in ALLOWED_ELEMENTS[user_id]


def is_allowed_entity(user_id: int, entity_id: int) -> bool:
    """
    Verifica si la entidad indicada está permitida para el usuario.
    Si el usuario no tiene restricciones definidas, el acceso se considera válido.

    Verifica se l'entità indicata è consentita per l'utente.
    Se l'utente non ha restrizioni definite, l'accesso viene considerato valido.
    """
    if user_id not in ALLOWED_ENTITIES:
        return True

    return entity_id in ALLOWED_ENTITIES[user_id]


def is_allowed_action(user_id: int, action_id: int) -> bool:
    """
    Verifica si la acción indicada está permitida para el usuario.
    Si el usuario no tiene restricciones definidas, la acción se considera válida.

    Verifica se l'azione indicata è consentita per l'utente.
    Se l'utente non ha restrizioni definite, l'azione viene considerata valida.
    """
    if user_id not in ALLOWED_ACTIONS:
        return True

    return action_id in ALLOWED_ACTIONS[user_id]


def evaluate_login_anomaly(user_id: int, dt: datetime) -> list[str]:
    """
    Evalúa si un evento de login presenta alguna anomalía o aviso.
    Devuelve una lista de mensajes con las incidencias detectadas.

    Valuta se un evento di login presenta qualche anomalia o avviso.
    Restituisce una lista di messaggi con le anomalie o gli avvisi rilevati.
    """
    messages = []

    if not is_weekday(dt):
        messages.append("Anomalia: stai tentando di accedere durante il fine settimana.")

    if not is_within_tolerance_schedule(user_id, dt):
        messages.append("Anomalia: stai tentando di accedere molto fuori dal tuo orario di lavoro.")
    elif not is_within_main_schedule(user_id, dt):
        messages.append("Avviso: stai accedendo fuori dal tuo orario abituale, ma entro la tolleranza.")

    return messages


def evaluate_activity_anomaly(
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    dt: datetime
) -> list[str]:
    """
    Evalúa si una actividad realizada por un usuario presenta anomalías o avisos.
    Devuelve una lista de mensajes con las incidencias detectadas.

    Valuta se un'attività eseguita da un utente presenta anomalie o avvisi.
    Restituisce una lista di messaggi con le anomalie o gli avvisi rilevati.
    """
    messages = []

    if not is_weekday(dt):
        messages.append("Anomalia: attività eseguita durante il fine settimana.")

    if not is_within_tolerance_schedule(user_id, dt):
        messages.append("Anomalia: attività eseguita molto fuori dall'orario di lavoro.")
    elif not is_within_main_schedule(user_id, dt):
        messages.append("Avviso: attività eseguita fuori dall'orario abituale, ma entro la tolleranza.")

    if not is_allowed_element(user_id, element_id):
        messages.append("Anomalia: stai accedendo a un elemento che non ti è consentito.")

    if not is_allowed_entity(user_id, entity_id):
        messages.append("Anomalia: stai accedendo a un'entità non consentita per il tuo utente.")

    if not is_allowed_action(user_id, action_id):
        messages.append("Anomalia: stai tentando di eseguire un'azione non consentita per il tuo utente.")

    return messages