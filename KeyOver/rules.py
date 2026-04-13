'''
 Este archivo te permitirá comprobar:
	•	si el login se hace fuera de horario
	•	si el login se hace en fin de semana
	•	si un usuario entra en un element_id que no le toca
	•	si intenta usar una acción no permitida
	•	si la entidad no es la correcta

'''

from datetime import datetime

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

LOGIN_WINDOWS = {
    1: (9, 12),   # Matteo
    2: (12, 15),  # Diego
    3: (15, 18)   # Emilio
}

ALLOWED_ELEMENTS = {
    1: [1, 2],          # Matteo -> FVG, AMCO
    2: [3],             # Diego -> VETTING
    3: [4, 1, 5, 6]     # Emilio -> RHODENSE, FVG, PAPARDO, PULEJO
}

ALLOWED_ENTITY_ID = 1

ALLOWED_ACTIONS = [1000000, 1000004, 1000005]  # Visualize, Copy, Share


def is_weekday(dt: datetime) -> bool:
    return dt.weekday() in [0, 1, 2, 3, 4]


def is_within_user_schedule(user_id: int, dt: datetime) -> bool:
    if user_id not in LOGIN_WINDOWS:
        return True

    start_hour, end_hour = LOGIN_WINDOWS[user_id]
    return start_hour <= dt.hour < end_hour


def is_allowed_element(user_id: int, element_id: int) -> bool:
    if user_id not in ALLOWED_ELEMENTS:
        return True

    return element_id in ALLOWED_ELEMENTS[user_id]


def is_allowed_entity(entity_id: int) -> bool:
    return entity_id == ALLOWED_ENTITY_ID


def is_allowed_action(action_id: int) -> bool:
    return action_id in ALLOWED_ACTIONS


def evaluate_login_anomaly(user_id: int, dt: datetime) -> list[str]:
    messages = []

    if not is_weekday(dt):
        messages.append("Anomalía: estás intentando acceder en fin de semana.")

    if not is_within_user_schedule(user_id, dt):
        messages.append("Anomalía: estás intentando acceder fuera de tu horario laboral.")

    return messages


def evaluate_activity_anomaly(
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    dt: datetime
) -> list[str]:
    messages = []

    if not is_weekday(dt):
        messages.append("Anomalía: actividad realizada en fin de semana.")

    if not is_within_user_schedule(user_id, dt):
        messages.append("Anomalía: actividad realizada fuera del horario laboral.")

    if not is_allowed_element(user_id, element_id):
        messages.append("Anomalía: estás entrando en un elemento que no te corresponde.")

    if not is_allowed_entity(entity_id):
        messages.append("Anomalía: estás accediendo a una entidad no permitida.")

    if not is_allowed_action(action_id):
        messages.append("Anomalía: estás intentando realizar una acción no permitida.")

    return messages