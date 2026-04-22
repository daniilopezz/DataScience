from datetime import datetime
import hashlib
import getpass
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import Error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from security.anomaly_guard import (
    build_login_profile,
    evaluate_activity_with_model,
    evaluate_login_with_profile,
    evaluate_session_with_model,
    get_activity_model_bundle,
    get_session_model_bundle,
    get_session_anomaly_threshold,
)

"""
Este archivo gestiona el flujo principal de interacción con la aplicación:
- login
- selección de elementos
- ejecución de acciones
- control de anomalías de actividad y de sesión

La lógica de expulsión depende de machine learning:
- si la actividad individual es anómala
- o si la sesión parcial/completa es anómala

Questo file gestisce il flusso principale di interazione con l'applicazione:
- login
- selezione degli elementi
- esecuzione delle azioni
- controllo delle anomalie di attività e di sessione

La logica di espulsione dipende dal machine learning:
- se la singola attività è anomala
- oppure se la sessione parziale/completa è anomala
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

MAX_ATTEMPTS = 3
DEFAULT_ENTITY_ID = 1

ACTION_DEFINITIONS = {
    "1": {"action_id": 1000000, "label": "Visualize", "text": "visualizzazione"},
    "2": {"action_id": 1000001, "label": "Create", "text": "creazione"},
    "3": {"action_id": 1000002, "label": "Edit", "text": "modifica"},
    "4": {"action_id": 1000003, "label": "Delete", "text": "eliminazione"},
    "5": {"action_id": 1000004, "label": "Copy", "text": "copia"},
    "6": {"action_id": 1000005, "label": "Share", "text": "condivisione"},
}


def hash_password(password: str) -> str:
    """
    Convierte la contraseña en un hash SHA-256.
    Converte la password in un hash SHA-256.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def get_connection():
    """
    Abre una conexión con PostgreSQL.
    Apre una connessione a PostgreSQL.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def get_next_attempt(cursor, user_id: int) -> int:
    """
    Obtiene el siguiente número de intento de login.
    Ottiene il numero del tentativo di login successivo.
    """
    query = """
        SELECT COALESCE(MAX(attempt), 0) + 1
        FROM login_log
        WHERE user_id = %s
    """
    cursor.execute(query, (user_id,))
    return cursor.fetchone()[0]


def save_login_log(cursor, user_id: int, result: bool, attempt: int) -> int:
    """
    Guarda un intento de login y devuelve login_log_id.
    Salva un tentativo di login e restituisce login_log_id.
    """
    query = """
        INSERT INTO login_log (user_id, result, attempt, logged_at, logout_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP, NULL)
        RETURNING login_log_id
    """
    cursor.execute(query, (user_id, result, attempt))
    return cursor.fetchone()[0]


def update_logout(cursor, login_log_id: int):
    """
    Actualiza logout_at al cerrar sesión.
    Aggiorna logout_at alla chiusura della sessione.
    """
    query = """
        UPDATE login_log
        SET logout_at = CURRENT_TIMESTAMP
        WHERE login_log_id = %s
    """
    cursor.execute(query, (login_log_id,))


def save_activity_log(cursor, user_id: int, action_id: int, element_id: int, entity_id: int) -> int:
    """
    Guarda una acción en activity_log y devuelve activity_log_id.
    Salva un'azione in activity_log e restituisce activity_log_id.
    """
    query = """
        INSERT INTO activity_log (user_id, element_id, entity_id, action_id, logged_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING activity_log_id
    """
    cursor.execute(query, (user_id, element_id, entity_id, action_id))
    return cursor.fetchone()[0]


def save_ml_prediction_log(
    cursor,
    activity_log_id: int,
    login_log_id: int,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    prediction: bool,
    anomaly_probability: float,
    session_cumulative_cost: float,
    session_threshold: float,
    threshold_exceeded: bool
):
    """
    Guarda la predicción ML y el contexto de sesión.
    Salva la previsione ML e il contesto di sessione.
    """
    query = """
        INSERT INTO ml_prediction_log (
            activity_log_id,
            login_log_id,
            user_id,
            element_id,
            entity_id,
            action_id,
            prediction,
            anomaly_probability,
            session_cumulative_cost,
            session_threshold,
            threshold_exceeded,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    """
    cursor.execute(
        query,
        (
            activity_log_id,
            login_log_id,
            user_id,
            element_id,
            entity_id,
            action_id,
            bool(prediction),
            float(anomaly_probability),
            float(session_cumulative_cost),
            float(session_threshold),
            bool(threshold_exceeded),
        )
    )


def get_elements(cursor):
    """
    Obtiene los elementos disponibles.
    Recupera gli elementi disponibili.
    """
    query = """
        SELECT element_id, name
        FROM element
        ORDER BY element_id
    """
    cursor.execute(query)
    return cursor.fetchall()


def load_login_history_df(connection) -> pd.DataFrame:
    """
    Carga el historial de login para construir perfiles.
    Carica lo storico dei login per costruire i profili.
    """
    query = """
        SELECT
            login_log_id,
            user_id,
            result,
            attempt,
            logged_at,
            logout_at
        FROM login_log
        ORDER BY login_log_id
    """
    cursor = connection.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    return pd.DataFrame(rows, columns=columns)


def show_main_menu():
    """
    Muestra el menú principal.
    Mostra il menu principale.
    """
    print("\n=== MENU PRINCIPALE ===")
    print("1 - Accedi")
    print("0 - Esci")


def show_element_menu(elements):
    """
    Muestra el menú de elementos.
    Mostra il menu degli elementi.
    """
    print("\n=== MENU ELEMENT ===")
    for element_id, name in elements:
        print(f"{element_id} - {name}")
    print("0 - Torna al menu principale")


def show_action_menu(selected_element_name: str):
    """
    Muestra el menú de acciones del elemento.
    Mostra il menu delle azioni dell'elemento.
    """
    print(f"\n=== MENU AZIONI | ELEMENTO: {selected_element_name} ===")
    print("0 - Torna al menu element")
    for option, data in ACTION_DEFINITIONS.items():
        print(f"{option} - {data['label']}")


def force_logout_due_to_anomaly(cursor, connection, login_log_id: int):
    """
    Fuerza el cierre de sesión por anomalía.
    Forza la chiusura della sessione per anomalia.
    """
    update_logout(cursor, login_log_id)
    connection.commit()


def format_session_cost_expression(session_cost_parts: list[float]) -> str:
    """
    Construye una cadena con la suma de costes de sesión.
    Costruisce una stringa con la somma dei costi di sessione.
    """
    if not session_cost_parts:
        return "0.000000 = 0.000000"

    formatted_parts = [f"{value:.6f}" for value in session_cost_parts]
    total = sum(session_cost_parts)

    if len(formatted_parts) <= 10:
        return " + ".join(formatted_parts) + f" = {total:.6f}"

    return "... + " + " + ".join(formatted_parts[-5:]) + f" = {total:.6f}"


def build_session_features(
    session_started_at: datetime,
    cost_parts: list[float],
    action_timestamps: list[datetime],
    elements_used: set[int],
    action_ids_used: set[int]
) -> dict:
    """
    Construye las variables de la sesión actual.

    Costruisce le variabili della sessione corrente.
    """
    now = datetime.now()
    action_count = len(cost_parts)

    cumulative_cost = float(sum(cost_parts))
    avg_cost = float(cumulative_cost / action_count) if action_count > 0 else 0.0
    max_cost = float(max(cost_parts)) if cost_parts else 0.0

    distinct_elements = len(elements_used)
    distinct_actions = len(action_ids_used)

    session_duration_min = max((now - session_started_at).total_seconds() / 60.0, 0.0001)

    start_hour = (
        session_started_at.hour
        + session_started_at.minute / 60.0
        + session_started_at.second / 3600.0
    )
    day_of_week = int(session_started_at.weekday())

    actions_per_minute = float(action_count / session_duration_min)

    diffs_seconds = []
    if len(action_timestamps) >= 2:
        for i in range(1, len(action_timestamps)):
            diff = (action_timestamps[i] - action_timestamps[i - 1]).total_seconds()
            diffs_seconds.append(max(float(diff), 0.0))

    if diffs_seconds:
        avg_seconds_between_actions = float(sum(diffs_seconds) / len(diffs_seconds))
        min_seconds_between_actions = float(min(diffs_seconds))
        max_seconds_between_actions = float(max(diffs_seconds))
    else:
        fallback_seconds = session_duration_min * 60.0
        avg_seconds_between_actions = float(fallback_seconds)
        min_seconds_between_actions = float(fallback_seconds)
        max_seconds_between_actions = float(fallback_seconds)

    repeated_action_ratio = 1.0
    repeated_element_ratio = 1.0

    if action_count > 0:
        action_counts: dict[int, int] = {}
        for action_id in action_ids_used:
            action_counts[action_id] = 0
        # Reconstruimos recuento real usando el historial temporal paralelo.
        # Ricostruiamo il conteggio reale usando la cronologia temporale parallela.
        # Para esta PoC, al no guardar la lista completa de action_ids repetidos,
        # usamos la variedad como aproximación.
        #
        # Per questa PoC, non salvando la lista completa degli action_ids ripetuti,
        # usiamo la varietà come approssimazione.
        repeated_action_ratio = 1.0 / max(distinct_actions, 1)
        repeated_element_ratio = 1.0 / max(distinct_elements, 1)

    return {
        "action_count": action_count,
        "cumulative_cost": cumulative_cost,
        "avg_cost": avg_cost,
        "max_cost": max_cost,
        "distinct_elements": distinct_elements,
        "distinct_actions": distinct_actions,
        "session_duration_min": session_duration_min,
        "start_hour": start_hour,
        "day_of_week": day_of_week,
        "actions_per_minute": actions_per_minute,
        "avg_seconds_between_actions": avg_seconds_between_actions,
        "min_seconds_between_actions": min_seconds_between_actions,
        "max_seconds_between_actions": max_seconds_between_actions,
        "repeated_action_ratio": repeated_action_ratio,
        "repeated_element_ratio": repeated_element_ratio,
    }


def print_ml_status(
    action_text: str,
    activity_result: dict,
    session_result: dict,
    operation_cost: float,
    session_features: dict,
    session_cost_parts: list[float]
):
    """
    Muestra por pantalla el estado ML de la acción y de la sesión.
    Mostra a schermo lo stato ML dell'azione e della sessione.
    """
    print(f"\nStai eseguendo l'azione: {action_text}.")
    print(
        f"[ML-ACTIVITY] Previsione del modello: "
        f"{'attività anomala' if activity_result['prediction'] == 1 else 'attività normale'}."
    )
    print(f"[ML-ACTIVITY] Costo operazione: {operation_cost:.6f}")
    print(f"[ML-ACTIVITY] Costo cumulato sessione: {session_features['cumulative_cost']:.6f}")
    print(f"[ML-ACTIVITY] Costo sessione = {format_session_cost_expression(session_cost_parts)}")

    print(
        f"[ML-SESSION] Previsione del modello: "
        f"{'sessione anomala' if session_result['prediction'] == 1 else 'sessione normale'}."
    )
    print(f"[ML-SESSION] Score sessione: {session_result['anomaly_score']:.6f}")
    print(f"[ML-SESSION] Actions/min: {session_features['actions_per_minute']:.4f}")
    print(f"[ML-SESSION] Avg sec/action: {session_features['avg_seconds_between_actions']:.4f}")
    print(f"[ML-SESSION] Min sec/action: {session_features['min_seconds_between_actions']:.4f}")
    print(f"[ML-SESSION] Max sec/action: {session_features['max_seconds_between_actions']:.4f}")


def process_action(
    cursor,
    connection,
    activity_model_bundle,
    session_model_bundle,
    login_log_id: int,
    user_id: int,
    element_id: int,
    action_option: str,
    session_started_at: datetime,
    session_cost_parts: list[float],
    session_action_timestamps: list[datetime],
    session_elements_used: set[int],
    session_action_ids_used: set[int]
) -> tuple[bool, list[float], list[datetime], set[int], set[int]]:
    """
    Procesa una acción:
    - evalúa la actividad con ML
    - actualiza la sesión
    - evalúa la sesión con ML
    - guarda logs
    - decide si debe cerrar la sesión

    Elabora un'azione:
    - valuta l'attività con ML
    - aggiorna la sessione
    - valuta la sessione con ML
    - salva i log
    - decide se deve chiudere la sessione
    """
    action_def = ACTION_DEFINITIONS[action_option]
    action_id = action_def["action_id"]
    action_text = action_def["text"]

    if activity_model_bundle is None:
        print(f"\nStai eseguendo l'azione: {action_text}.")
        activity_log_id = save_activity_log(
            cursor=cursor,
            user_id=user_id,
            action_id=action_id,
            element_id=element_id,
            entity_id=DEFAULT_ENTITY_ID
        )
        connection.commit()
        print("[ML] Nessun modello caricato. Nessuna previsione salvata.")
        print(f"Attività registrata correttamente. ID attività: {activity_log_id}")
        return True, session_cost_parts, session_action_timestamps, session_elements_used, session_action_ids_used

    try:
        activity_result = evaluate_activity_with_model(
            model_bundle=activity_model_bundle,
            user_id=user_id,
            element_id=element_id,
            entity_id=DEFAULT_ENTITY_ID,
            action_id=action_id,
            logged_at=datetime.now(),
            session_started_at=session_started_at,
            previous_action_timestamps=session_action_timestamps
        )

        operation_cost = float(activity_result["anomaly_score"])
        current_action_time = datetime.now()

        new_session_cost_parts = session_cost_parts + [operation_cost]
        new_session_action_timestamps = session_action_timestamps + [current_action_time]

        new_session_elements_used = set(session_elements_used)
        new_session_elements_used.add(element_id)

        new_session_action_ids_used = set(session_action_ids_used)
        new_session_action_ids_used.add(action_id)

        session_features = build_session_features(
            session_started_at=session_started_at,
            cost_parts=new_session_cost_parts,
            action_timestamps=new_session_action_timestamps,
            elements_used=new_session_elements_used,
            action_ids_used=new_session_action_ids_used
        )

        if session_model_bundle is not None:
            session_result = evaluate_session_with_model(
                model_bundle=session_model_bundle,
                user_id=user_id,
                action_count=session_features["action_count"],
                cumulative_cost=session_features["cumulative_cost"],
                avg_cost=session_features["avg_cost"],
                max_cost=session_features["max_cost"],
                distinct_elements=session_features["distinct_elements"],
                distinct_actions=session_features["distinct_actions"],
                session_duration_min=session_features["session_duration_min"],
                start_hour=session_features["start_hour"],
                day_of_week=session_features["day_of_week"],
                actions_per_minute=session_features["actions_per_minute"],
                avg_seconds_between_actions=session_features["avg_seconds_between_actions"],
                min_seconds_between_actions=session_features["min_seconds_between_actions"],
                max_seconds_between_actions=session_features["max_seconds_between_actions"],
                repeated_action_ratio=session_features["repeated_action_ratio"],
                repeated_element_ratio=session_features["repeated_element_ratio"]
            )
        else:
            session_result = {"prediction": 0, "anomaly_score": 0.0, "message": ""}

        print_ml_status(
            action_text=action_text,
            activity_result=activity_result,
            session_result=session_result,
            operation_cost=operation_cost,
            session_features=session_features,
            session_cost_parts=new_session_cost_parts
        )

        final_prediction = bool(
            activity_result["prediction"] == 1
            or session_result["prediction"] == 1
        )

        activity_log_id = save_activity_log(
            cursor=cursor,
            user_id=user_id,
            action_id=action_id,
            element_id=element_id,
            entity_id=DEFAULT_ENTITY_ID
        )

        save_ml_prediction_log(
            cursor=cursor,
            activity_log_id=activity_log_id,
            login_log_id=login_log_id,
            user_id=user_id,
            element_id=element_id,
            entity_id=DEFAULT_ENTITY_ID,
            action_id=action_id,
            prediction=final_prediction,
            anomaly_probability=operation_cost,
            session_cumulative_cost=session_features["cumulative_cost"],
            session_threshold=get_session_anomaly_threshold(),
            threshold_exceeded=False
        )

        connection.commit()
        print(f"Attività registrata correttamente. ID attività: {activity_log_id}")

        if activity_result["prediction"] == 1:
            print("\nAttività anomala rilevata dal modello.")
            print("Logout automatico eseguito.")
            force_logout_due_to_anomaly(cursor, connection, login_log_id)
            return False, new_session_cost_parts, new_session_action_timestamps, new_session_elements_used, new_session_action_ids_used

        if session_result["prediction"] == 1:
            print("\nSessione anomala rilevata dal modello di sessione.")
            print("Logout automatico eseguito.")
            force_logout_due_to_anomaly(cursor, connection, login_log_id)
            return False, new_session_cost_parts, new_session_action_timestamps, new_session_elements_used, new_session_action_ids_used

        return True, new_session_cost_parts, new_session_action_timestamps, new_session_elements_used, new_session_action_ids_used

    except Exception as e:
        connection.rollback()
        print(f"[ML] Errore durante la previsione: {e}")
        return False, session_cost_parts, session_action_timestamps, session_elements_used, session_action_ids_used


def action_menu(
    cursor,
    connection,
    activity_model_bundle,
    session_model_bundle,
    login_log_id: int,
    user_id: int,
    element_id: int,
    element_name: str,
    session_started_at: datetime,
    session_cost_parts: list[float],
    session_action_timestamps: list[datetime],
    session_elements_used: set[int],
    session_action_ids_used: set[int]
) -> tuple[bool, list[float], list[datetime], set[int], set[int]]:
    """
    Muestra el menú de acciones y gestiona la opción elegida.
    Mostra il menu delle azioni e gestisce l'opzione scelta.
    """
    while True:
        show_action_menu(element_name)
        option = input("Scegli un'opzione: ").strip()

        if option == "0":
            print("\nRitorno al menu element...")
            return True, session_cost_parts, session_action_timestamps, session_elements_used, session_action_ids_used

        if option not in ACTION_DEFINITIONS:
            print("\nOpzione non valida.")
            continue

        result = process_action(
            cursor=cursor,
            connection=connection,
            activity_model_bundle=activity_model_bundle,
            session_model_bundle=session_model_bundle,
            login_log_id=login_log_id,
            user_id=user_id,
            element_id=element_id,
            action_option=option,
            session_started_at=session_started_at,
            session_cost_parts=session_cost_parts,
            session_action_timestamps=session_action_timestamps,
            session_elements_used=session_elements_used,
            session_action_ids_used=session_action_ids_used
        )

        session_still_active, session_cost_parts, session_action_timestamps, session_elements_used, session_action_ids_used = result

        if not session_still_active:
            return result


def element_menu(
    cursor,
    connection,
    activity_model_bundle,
    session_model_bundle,
    login_log_id: int,
    user_id: int
):
    """
    Muestra el menú de elementos y abre el menú de acciones.
    Mostra il menu degli elementi e apre il menu delle azioni.
    """
    session_started_at = datetime.now()
    session_cost_parts: list[float] = []
    session_action_timestamps: list[datetime] = []
    session_elements_used: set[int] = set()
    session_action_ids_used: set[int] = set()

    while True:
        elements = get_elements(cursor)
        element_map = {str(element_id): (element_id, name) for element_id, name in elements}

        show_element_menu(elements)
        option = input("Seleziona un element: ").strip()

        if option == "0":
            print("\nUscita verso il menu principale...")
            update_logout(cursor, login_log_id)
            connection.commit()
            print("Logout registrato correttamente.")
            break

        if option not in element_map:
            print("\nOpzione non valida.")
            continue

        element_id, element_name = element_map[option]
        print(f"\nHai selezionato l'elemento: {element_name}")

        result = action_menu(
            cursor=cursor,
            connection=connection,
            activity_model_bundle=activity_model_bundle,
            session_model_bundle=session_model_bundle,
            login_log_id=login_log_id,
            user_id=user_id,
            element_id=element_id,
            element_name=element_name,
            session_started_at=session_started_at,
            session_cost_parts=session_cost_parts,
            session_action_timestamps=session_action_timestamps,
            session_elements_used=session_elements_used,
            session_action_ids_used=session_action_ids_used
        )

        session_still_active, session_cost_parts, session_action_timestamps, session_elements_used, session_action_ids_used = result

        if not session_still_active:
            break


def login_user() -> bool:
    """
    Gestiona todo el proceso de inicio de sesión.
    Gestisce l'intero processo di login.
    """
    connection = get_connection()
    if connection is None:
        return False

    cursor = None

    try:
        cursor = connection.cursor()

        try:
            activity_model_bundle = get_activity_model_bundle()
        except Exception:
            activity_model_bundle = None
            print("Impossibile caricare il modello ML di activity.")

        try:
            session_model_bundle = get_session_model_bundle()
        except Exception:
            session_model_bundle = None
            print("Impossibile caricare il modello ML di sessione.")

        try:
            login_history_df = load_login_history_df(connection)
            login_profiles = build_login_profile(login_history_df)
        except Exception as e:
            login_profiles = {}
            print(f"Impossibile costruire i profili di login: {e}")

        for _ in range(MAX_ATTEMPTS):
            print("\n--- LOGIN ---")
            email = input("Email: ").strip()
            password = getpass.getpass("Password: ").strip()

            query = """
                SELECT user_id, name, surname, password_hash, is_active
                FROM users
                WHERE email = %s
            """
            cursor.execute(query, (email,))
            user = cursor.fetchone()

            if user is None:
                print("Utente non trovato.")
                continue

            user_id, name, surname, stored_password_hash, is_active = user
            attempt = get_next_attempt(cursor, user_id)

            if not is_active:
                save_login_log(cursor, user_id, False, attempt)
                connection.commit()
                print("Utente inattivo. Accesso negato.")
                return False

            if stored_password_hash == hash_password(password):
                login_result = evaluate_login_with_profile(
                    login_profiles=login_profiles,
                    user_id=int(user_id),
                    logged_at=datetime.now()
                )

                if login_result["is_anomalous"]:
                    save_login_log(cursor, user_id, False, attempt)
                    connection.commit()
                    print(f"\n{login_result['message']}")
                    return False

                login_log_id = save_login_log(cursor, user_id, True, attempt)
                connection.commit()

                print(f"\nLogin effettuato correttamente. Benvenuto/a, {name} {surname}.")
                element_menu(
                    cursor=cursor,
                    connection=connection,
                    activity_model_bundle=activity_model_bundle,
                    session_model_bundle=session_model_bundle,
                    login_log_id=login_log_id,
                    user_id=user_id
                )
                return True

            save_login_log(cursor, user_id, False, attempt)
            connection.commit()
            print("Password errata.")

        print("\nAccesso bloccato per troppi tentativi falliti.")
        return False

    except Error as e:
        print(f"Errore nel database: {e}")
        return False

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def main():
    """
    Función principal del programa.
    Funzione principale del programma.
    """
    while True:
        show_main_menu()
        option = input("Scegli un'opzione: ").strip()

        if option == "1":
            login_user()
        elif option == "0":
            print("Chiusura del programma.")
            break
        else:
            print("Opzione non valida.")


if __name__ == "__main__":
    main()