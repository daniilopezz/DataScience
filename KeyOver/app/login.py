from datetime import datetime
import hashlib
import getpass
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import Error

# Añadimos la raíz del proyecto al path para poder importar módulos
# también cuando este archivo se ejecuta directamente desde app/login.py.
#
# Aggiungiamo la radice del progetto al path per poter importare moduli
# anche quando questo file viene eseguito direttamente da app/login.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from security.anomaly_guard import (
    build_login_profile,
    evaluate_activity_with_model,
    evaluate_login_with_profile,
    get_activity_model_bundle,
    get_session_anomaly_threshold,
)

"""
Este archivo gestiona el flujo principal de interacción con la aplicación:
inicio de sesión, registro de accesos, selección de elementos, ejecución de acciones
y control de anomalías.

En esta versión:
- el login se evalúa usando un perfil histórico del usuario
- las actividades se evalúan usando el modelo de machine learning
- cada acción genera un coste de sesión basado en anomaly_probability
- el coste acumulado de sesión se compara con un umbral configurable
- si se supera el umbral, la sesión se cierra automáticamente

Questo file gestisce il flusso principale di interazione con l'applicazione:
login, registrazione degli accessi, selezione degli elementi, esecuzione delle azioni
e controllo delle anomalie.

In questa versione:
- il login viene valutato usando un profilo storico dell'utente
- le attività vengono valutate usando il modello di machine learning
- ogni azione genera un costo di sessione basato su anomaly_probability
- il costo cumulato di sessione viene confrontato con una soglia configurabile
- se la soglia viene superata, la sessione viene chiusa automaticamente
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

ACTION_IDS = {
    "1": 1000000,  # Visualize
    "2": 1000001,  # Create
    "3": 1000002,  # Edit
    "4": 1000003,  # Delete
    "5": 1000004,  # Copy
    "6": 1000005   # Share
}


def hash_password(password: str) -> str:
    """
    Convierte la contraseña introducida por el usuario en un hash SHA-256.

    Converte la password inserita dall'utente in un hash SHA-256.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def get_connection():
    """
    Abre una conexión con PostgreSQL utilizando la configuración definida.

    Apre una connessione a PostgreSQL utilizzando la configurazione definita.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def get_next_attempt(cursor, user_id: int) -> int:
    """
    Obtiene el siguiente número de intento de login para un usuario.

    Ottiene il numero del tentativo di login successivo per un utente.
    """
    query = """
        SELECT COALESCE(MAX(attempt), 0) + 1
        FROM login_log
        WHERE user_id = %s
    """
    cursor.execute(query, (user_id,))
    return cursor.fetchone()[0]


def save_login_log(cursor, user_id: int, result: bool, attempt: int):
    """
    Guarda un nuevo intento de login en login_log y devuelve el login_log_id generado.

    Salva un nuovo tentativo di login in login_log e restituisce il login_log_id generato.
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
    Actualiza logout_at cuando el usuario cierra sesión.

    Aggiorna logout_at quando l'utente chiude la sessione.
    """
    query = """
        UPDATE login_log
        SET logout_at = CURRENT_TIMESTAMP
        WHERE login_log_id = %s
    """
    cursor.execute(query, (login_log_id,))


def save_activity_log(cursor, user_id: int, action_id: int, element_id: int, entity_id: int):
    """
    Guarda una acción del usuario en activity_log y devuelve el activity_log_id generado.

    Salva un'azione dell'utente in activity_log e restituisce l'activity_log_id generato.
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
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    prediction,
    anomaly_probability
):
    """
    Guarda en ml_prediction_log la predicción asociada a una actividad concreta.

    Salva in ml_prediction_log la previsione associata a una specifica attività.
    """
    prediction = bool(prediction)

    if anomaly_probability is not None:
        anomaly_probability = float(anomaly_probability)

    query = """
        INSERT INTO ml_prediction_log (
            activity_log_id,
            user_id,
            element_id,
            entity_id,
            action_id,
            prediction,
            anomaly_probability,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    """
    cursor.execute(
        query,
        (
            activity_log_id,
            user_id,
            element_id,
            entity_id,
            action_id,
            prediction,
            anomaly_probability
        )
    )


def get_elements(cursor):
    """
    Obtiene los elementos disponibles desde la tabla element.

    Recupera gli elementi disponibili dalla tabella element.
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
    Carga el historial de login desde la base de datos para construir perfiles.

    Carica lo storico dei login dal database per costruire i profili.
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
    Muestra el menú principal del programa.

    Mostra il menu principale del programma.
    """
    print("\n=== MENU PRINCIPALE ===")
    print("1 - Accedi")
    print("0 - Esci")


def show_element_menu(elements):
    """
    Muestra el menú de elementos disponible.

    Mostra il menu degli elementi disponibile.
    """
    print("\n=== MENU ELEMENT ===")
    for element_id, name in elements:
        print(f"{element_id} - {name}")
    print("0 - Torna al menu principale")


def show_action_menu(selected_element_name: str):
    """
    Muestra el menú de acciones posibles sobre el elemento seleccionado.

    Mostra il menu delle azioni possibili sull'elemento selezionato.
    """
    print(f"\n=== MENU AZIONI | ELEMENTO: {selected_element_name} ===")
    print("0 - Torna al menu element")
    print("1 - Visualize")
    print("2 - Create")
    print("3 - Edit")
    print("4 - Delete")
    print("5 - Copy")
    print("6 - Share")


def force_logout_due_to_anomaly(cursor, connection, login_log_id: int):
    """
    Fuerza el cierre de sesión del usuario por una anomalía detectada.

    Forza la chiusura della sessione dell'utente a causa di un'anomalia rilevata.
    """
    update_logout(cursor, login_log_id)
    connection.commit()


def process_action(
    cursor,
    connection,
    model_bundle,
    login_log_id: int,
    user_id: int,
    element_id: int,
    action_option: str,
    action_text: str,
    session_anomaly_cost: float
) -> tuple[bool, float]:
    """
    Procesa una acción del usuario sobre un elemento.

    Flujo:
    1. Evalúa la actividad con el modelo.
    2. Obtiene el coste de la operación (anomaly_probability).
    3. Suma el coste al total acumulado de la sesión.
    4. Muestra coste individual, coste acumulado y umbral.
    5. Si se supera el umbral de sesión, cierra la sesión.
    6. Si no se supera, guarda la actividad y la predicción en base de datos.

    Devuelve:
    - (True, nuevo_coste) si la sesión puede continuar
    - (False, nuevo_coste) si la sesión ha sido cerrada

    Elabora un'azione dell'utente su un elemento.

    Flusso:
    1. Valuta l'attività con il modello.
    2. Ottiene il costo dell'operazione (anomaly_probability).
    3. Somma il costo al totale cumulato della sessione.
    4. Mostra costo singolo, costo cumulato e soglia.
    5. Se la soglia di sessione viene superata, chiude la sessione.
    6. Se non viene superata, salva l'attività e la previsione nel database.

    Restituisce:
    - (True, nuovo_costo) se la sessione può continuare
    - (False, nuovo_costo) se la sessione è stata chiusa
    """
    action_id = ACTION_IDS[action_option]
    session_threshold = get_session_anomaly_threshold()

    if model_bundle is not None:
        try:
            result = evaluate_activity_with_model(
                model_bundle=model_bundle,
                user_id=user_id,
                element_id=element_id,
                entity_id=DEFAULT_ENTITY_ID,
                action_id=action_id
            )

            operation_cost = float(result["anomaly_score"])
            new_session_cost = session_anomaly_cost + operation_cost

            print(f"\nStai eseguendo l'azione: {action_text}.")
            print(
                f"[ML] Previsione del modello: "
                f"{'attività anomala' if result['prediction'] == 1 else 'attività normale'}."
            )
            print(f"[ML] Costo operazione: {operation_cost:.6f}")
            print(f"[ML] Costo cumulato sessione: {new_session_cost:.6f}")
            print(f"[ML] Soglia di sessione: {session_threshold:.6f}")

            # Si se supera el umbral acumulado, se cierra la sesión y no se guarda la acción.
            # Se viene superata la soglia cumulata, la sessione viene chiusa e l'azione non viene salvata.
            if new_session_cost >= session_threshold:
                print("\nSoglia di sessione superata. Attività marcata come anomala.")
                print("Logout automatico eseguito.")
                force_logout_due_to_anomaly(cursor, connection, login_log_id)
                return False, new_session_cost

            # Si no se supera el umbral, guardamos la actividad.
            # Se la soglia non viene superata, salviamo l'attività.
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
                user_id=user_id,
                element_id=element_id,
                entity_id=DEFAULT_ENTITY_ID,
                action_id=action_id,
                prediction=result["prediction"],
                anomaly_probability=operation_cost
            )

            connection.commit()
            print(f"Attività registrata correttamente. ID attività: {activity_log_id}")
            return True, new_session_cost

        except Exception as e:
            connection.rollback()
            print(f"[ML] Errore durante la previsione: {e}")
            return False, session_anomaly_cost

    # Si no hay modelo, guardamos la actividad sin evaluación ML.
    # Se non c'è modello, salviamo l'attività senza valutazione ML.
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
    return True, session_anomaly_cost


def action_menu(
    cursor,
    connection,
    model_bundle,
    login_log_id: int,
    user_id: int,
    element_id: int,
    element_name: str,
    session_anomaly_cost: float
) -> tuple[bool, float]:
    """
    Muestra el menú de acciones para un elemento y gestiona la opción elegida.

    Devuelve:
    - (True, coste_actualizado) si la sesión sigue activa
    - (False, coste_actualizado) si la sesión se ha cerrado por anomalía

    Mostra il menu delle azioni per un elemento e gestisce l'opzione scelta.

    Restituisce:
    - (True, costo_aggiornato) se la sessione resta attiva
    - (False, costo_aggiornato) se la sessione è stata chiusa per anomalia
    """
    while True:
        show_action_menu(element_name)
        option = input("Scegli un'opzione: ").strip()

        match option:
            case "0":
                print("\nRitorno al menu element...")
                return True, session_anomaly_cost

            case "1":
                session_still_active, session_anomaly_cost = process_action(
                    cursor, connection, model_bundle, login_log_id, user_id,
                    element_id, "1", "visualizzazione", session_anomaly_cost
                )
                if not session_still_active:
                    return False, session_anomaly_cost

            case "2":
                session_still_active, session_anomaly_cost = process_action(
                    cursor, connection, model_bundle, login_log_id, user_id,
                    element_id, "2", "creazione", session_anomaly_cost
                )
                if not session_still_active:
                    return False, session_anomaly_cost

            case "3":
                session_still_active, session_anomaly_cost = process_action(
                    cursor, connection, model_bundle, login_log_id, user_id,
                    element_id, "3", "modifica", session_anomaly_cost
                )
                if not session_still_active:
                    return False, session_anomaly_cost

            case "4":
                session_still_active, session_anomaly_cost = process_action(
                    cursor, connection, model_bundle, login_log_id, user_id,
                    element_id, "4", "eliminazione", session_anomaly_cost
                )
                if not session_still_active:
                    return False, session_anomaly_cost

            case "5":
                session_still_active, session_anomaly_cost = process_action(
                    cursor, connection, model_bundle, login_log_id, user_id,
                    element_id, "5", "copia", session_anomaly_cost
                )
                if not session_still_active:
                    return False, session_anomaly_cost

            case "6":
                session_still_active, session_anomaly_cost = process_action(
                    cursor, connection, model_bundle, login_log_id, user_id,
                    element_id, "6", "condivisione", session_anomaly_cost
                )
                if not session_still_active:
                    return False, session_anomaly_cost

            case _:
                print("\nOpzione non valida.")


def element_menu(cursor, connection, model_bundle, login_log_id: int, user_id: int):
    """
    Muestra el menú de elementos disponibles y abre el menú de acciones.

    Mostra il menu degli elementi disponibili e apre il menu delle azioni.
    """
    session_anomaly_cost = 0.0

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

        if option in element_map:
            element_id, element_name = element_map[option]
            print(f"\nHai selezionato l'elemento: {element_name}")

            session_still_active, session_anomaly_cost = action_menu(
                cursor=cursor,
                connection=connection,
                model_bundle=model_bundle,
                login_log_id=login_log_id,
                user_id=user_id,
                element_id=element_id,
                element_name=element_name,
                session_anomaly_cost=session_anomaly_cost
            )

            if not session_still_active:
                break
        else:
            print("\nOpzione non valida.")


def login_user() -> bool:
    """
    Gestiona todo el proceso de inicio de sesión.

    Flujo:
    1. Abre conexión a PostgreSQL.
    2. Carga el modelo de activity si existe.
    3. Construye perfiles históricos de login.
    4. Permite hasta MAX_ATTEMPTS intentos.
    5. Busca al usuario por email.
    6. Comprueba si está activo.
    7. Valida la contraseña comparando hashes.
    8. Evalúa si el login es extraño.
    9. Si el login es correcto y no es extraño, abre la sesión.

    Gestisce l'intero processo di login.
    """
    connection = get_connection()
    if connection is None:
        return False

    cursor = None

    try:
        cursor = connection.cursor()

        try:
            model_bundle = get_activity_model_bundle()
        except Exception:
            model_bundle = None
            print("Impossibile caricare il modello ML di activity. Il sistema continuerà senza controllo ML sulle azioni.")

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
                element_menu(cursor, connection, model_bundle, login_log_id, user_id)
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