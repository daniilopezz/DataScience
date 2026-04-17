from datetime import datetime, time

"""
Este archivo define las reglas de negocio de referencia del proyecto.
Estas reglas describen el comportamiento esperado de cada usuario en términos de:
- horario habitual
- tolerancia horaria
- elementos permitidos
- entidades permitidas
- acciones permitidas
Aunque el modelo de machine learning final no utilice estas reglas para detectar
anomalías en tiempo real, este archivo sigue siendo útil como documentación
funcional y como apoyo para la generación de datos sintéticos.

Questo file definisce le regole di business di riferimento del progetto.
Queste regole descrivono il comportamento atteso di ciascun utente in termini di:
- orario abituale
- tolleranza oraria
- elementi consentiti
- entità consentite
- azioni consentite
Anche se il modello finale di machine learning non utilizza queste regole per
rilevare anomalie in tempo reale, questo file rimane utile come documentazione
funzionale e come supporto per la generazione di dati sintetici.
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

# Horarios habituales definidos para cada usuario.
# "start" y "end" representan la franja normal de trabajo.
# "tolerance_minutes" indica el margen permitido antes y después del horario principal.
#
# Orari abituali definiti per ciascun utente.
# "start" e "end" rappresentano la fascia normale di lavoro.
# "tolerance_minutes" indica il margine consentito prima e dopo l'orario principale.
USER_SCHEDULES = {
    1: {"start": time(9, 0), "end": time(13, 0), "tolerance_minutes": 15},   # Matteo
    2: {"start": time(9, 0), "end": time(17, 0), "tolerance_minutes": 20},   # Diego
    3: {"start": time(10, 0), "end": time(18, 0), "tolerance_minutes": 20}   # Emilio
}

# Elementos permitidos para cada usuario.
#
# Elementi consentiti per ciascun utente.
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

# Acciones permitidas para cada usuario.
#
# Azioni consentite per ciascun utente.
ALLOWED_ACTIONS = {
    1: [1000000, 1000004, 1000005],                  # Matteo -> Visualize, Copy, Share
    2: [1000000, 1000001, 1000002, 1000005],         # Diego -> Visualize, Create, Edit, Share
    3: [1000000, 1000002, 1000004, 1000005]          # Emilio -> Visualize, Edit, Copy, Share
}


def is_weekday(dt: datetime) -> bool:
    """
    Devuelve True si la fecha cae entre lunes y viernes.

    Restituisce True se la data cade tra lunedì e venerdì.
    """
    return dt.weekday() < 5


def is_within_main_schedule(user_id: int, dt: datetime) -> bool:
    """
    Comprueba si el usuario está dentro de su horario habitual principal.
    Si el usuario no tiene un horario definido, se considera válido.

    Controlla se l'utente rientra nel proprio orario abituale principale.
    Se l'utente non ha un orario definito, viene considerato valido.
    """
    if user_id not in USER_SCHEDULES:
        return True

    schedule = USER_SCHEDULES[user_id]
    current_time = dt.time()

    return schedule["start"] <= current_time <= schedule["end"]


def is_within_tolerance_schedule(user_id: int, dt: datetime) -> bool:
    """
    Comprueba si la fecha y hora están dentro de la franja permitida
    incluyendo la tolerancia del usuario.
    Se convierte cada hora a minutos para facilitar la comparación.
    Si el usuario no tiene un horario definido, se considera válido.

    Controlla se la data e l'ora rientrano nella fascia consentita
    includendo la tolleranza dell'utente.
    Ogni orario viene convertito in minuti per facilitare il confronto.
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
    Comprueba si el elemento indicado está permitido para el usuario.
    Si el usuario no tiene restricciones definidas, se considera válido.

    Controlla se l'elemento indicato è consentito per l'utente.
    Se l'utente non ha restrizioni definite, viene considerato valido.
    """
    if user_id not in ALLOWED_ELEMENTS:
        return True

    return element_id in ALLOWED_ELEMENTS[user_id]


def is_allowed_entity(user_id: int, entity_id: int) -> bool:
    """
    Comprueba si la entidad indicada está permitida para el usuario.
    Si el usuario no tiene restricciones definidas, se considera válido.

    Controlla se l'entità indicata è consentita per l'utente. 
    Se l'utente non ha restrizioni definite, viene considerato valido.
    """
    if user_id not in ALLOWED_ENTITIES:
        return True

    return entity_id in ALLOWED_ENTITIES[user_id]


def is_allowed_action(user_id: int, action_id: int) -> bool:
    """
    Comprueba si la acción indicada está permitida para el usuario.
    Si el usuario no tiene restricciones definidas, se considera válida.

    Controlla se l'azione indicata è consentita per l'utente.
    Se l'utente non ha restrizioni definite, viene considerata valida.
    """
    if user_id not in ALLOWED_ACTIONS:
        return True

    return action_id in ALLOWED_ACTIONS[user_id]


def evaluate_login_anomaly(user_id: int, dt: datetime) -> list[str]:
    """
    Evalúa si un evento de login incumple alguna de las reglas de referencia.
    Devuelve una lista de mensajes:
    - "Anomalia: ..." para incumplimientos fuertes
    - "Avviso: ..." para casos fuera del horario principal pero dentro de tolerancia

    Valuta se un evento di login viola una delle regole di riferimento.
    Restituisce una lista di messaggi:
    - "Anomalia: ..." per violazioni forti
    - "Avviso: ..." per casi fuori dall'orario principale ma entro la tolleranza
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
    Evalúa si una actividad incumple alguna de las reglas de referencia.
    Devuelve una lista de mensajes:
    - "Anomalia: ..." para incumplimientos fuertes
    - "Avviso: ..." para casos tolerados pero fuera del patrón principal

    Valuta se un'attività viola una delle regole di riferimento.
    Restituisce una lista di messaggi:
    - "Anomalia: ..." per violazioni forti
    - "Avviso: ..." per casi tollerati ma fuori dal pattern principale
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