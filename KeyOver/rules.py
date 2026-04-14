from datetime import datetime, time

"""
Este archivo define las reglas de negocio para detectar anomalías
en login y actividad.

Se comprueba:
- acceso en fin de semana
- acceso fuera de horario laboral
- acceso con tolerancia horaria
- acceso a elementos no permitidos
- acceso a entidades no permitidas
- ejecución de acciones no permitidas
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

# Horarios reales por usuario
# start y end son la franja normal
# tolerance_minutes permite cierto margen antes y después
USER_SCHEDULES = {
    1: {"start": time(9, 0), "end": time(13, 0), "tolerance_minutes": 15},   # Matteo
    2: {"start": time(9, 0), "end": time(17, 0), "tolerance_minutes": 20},   # Diego
    3: {"start": time(10, 0), "end": time(18, 0), "tolerance_minutes": 20}   # Emilio
}

# Elementos permitidos por usuario
ALLOWED_ELEMENTS = {
    1: [1, 2],          # Matteo -> FVG, AMCO
    2: [3],             # Diego -> VETTING
    3: [1, 4, 5, 6]     # Emilio -> FVG, RHODENSE, PAPARDO, PULEJO
}

# Entidades permitidas por usuario
ALLOWED_ENTITIES = {
    1: [1],        # Matteo -> Password
    2: [1, 2],     # Diego -> Password, Folder
    3: [1, 3]      # Emilio -> Password, Group
}

# Acciones permitidas por usuario
ALLOWED_ACTIONS = {
    1: [1000000, 1000004, 1000005],                  # Matteo -> Visualize, Copy, Share
    2: [1000000, 1000001, 1000002, 1000005],         # Diego -> Visualize, Create, Edit, Share
    3: [1000000, 1000002, 1000004, 1000005]          # Emilio -> Visualize, Edit, Copy, Share
}


def is_weekday(dt: datetime) -> bool:
    """
    Devuelve True si la fecha cae entre lunes y viernes.
    """
    return dt.weekday() in [0, 1, 2, 3, 4]


def is_within_main_schedule(user_id: int, dt: datetime) -> bool:
    """
    Comprueba si el usuario está dentro de su horario laboral normal.
    """
    if user_id not in USER_SCHEDULES:
        return True

    schedule = USER_SCHEDULES[user_id]
    current_time = dt.time()
    return schedule["start"] <= current_time <= schedule["end"]


def is_within_tolerance_schedule(user_id: int, dt: datetime) -> bool:
    """
    Comprueba si el usuario está dentro del horario permitido contando la tolerancia.
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
    Comprueba si el elemento está permitido para ese usuario.
    """
    if user_id not in ALLOWED_ELEMENTS:
        return True

    return element_id in ALLOWED_ELEMENTS[user_id]


def is_allowed_entity(user_id: int, entity_id: int) -> bool:
    """
    Comprueba si la entidad está permitida para ese usuario.
    """
    if user_id not in ALLOWED_ENTITIES:
        return True

    return entity_id in ALLOWED_ENTITIES[user_id]


def is_allowed_action(user_id: int, action_id: int) -> bool:
    """
    Comprueba si la acción está permitida para ese usuario.
    """
    if user_id not in ALLOWED_ACTIONS:
        return True

    return action_id in ALLOWED_ACTIONS[user_id]


def evaluate_login_anomaly(user_id: int, dt: datetime) -> list[str]:
    """
    Evalúa anomalías de login.
    """
    messages = []

    if not is_weekday(dt):
        messages.append("Anomalía: estás intentando acceder en fin de semana.")

    if not is_within_tolerance_schedule(user_id, dt):
        messages.append("Anomalía: estás intentando acceder muy fuera de tu horario laboral.")
    elif not is_within_main_schedule(user_id, dt):
        messages.append("Aviso: estás accediendo fuera de tu horario habitual, pero dentro de la tolerancia.")

    return messages


def evaluate_activity_anomaly(
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    dt: datetime
) -> list[str]:
    """
    Evalúa anomalías de actividad.
    """
    messages = []

    if not is_weekday(dt):
        messages.append("Anomalía: actividad realizada en fin de semana.")

    if not is_within_tolerance_schedule(user_id, dt):
        messages.append("Anomalía: actividad realizada muy fuera del horario laboral.")
    elif not is_within_main_schedule(user_id, dt):
        messages.append("Aviso: actividad realizada fuera del horario habitual, pero dentro de la tolerancia.")

    if not is_allowed_element(user_id, element_id):
        messages.append("Anomalía: estás entrando en un elemento que no te corresponde.")

    if not is_allowed_entity(user_id, entity_id):
        messages.append("Anomalía: estás accediendo a una entidad no permitida para tu usuario.")

    if not is_allowed_action(user_id, action_id):
        messages.append("Anomalía: estás intentando realizar una acción no permitida para tu usuario.")

    return messages