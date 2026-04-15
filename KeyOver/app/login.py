import hashlib
import getpass
import sys
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2 import Error

# Añadimos la raíz del proyecto al path para poder importar ml_model.py
# también cuando este archivo se ejecuta directamente desde app/login.py.
#
# Aggiungiamo la radice del progetto al path per poter importare ml_model.py
# anche quando questo file viene eseguito direttamente da app/login.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml_model import load_model, predict_activity_with_model

"""
Este archivo gestiona el flujo principal de interacción con la aplicación:
inicio de sesión, registro de accesos, selección de elementos, ejecución de acciones
y almacenamiento de predicciones generadas por el modelo de machine learning.

En esta versión no se utilizan reglas fijas para detectar anomalías.
La autenticación sigue funcionando de forma normal, pero la detección de
comportamientos anómalos se realiza a través del modelo entrenado a partir
de los datos históricos.

Questo file gestisce il flusso principale di interazione con l'applicazione:
login, registrazione degli accessi, selezione degli elementi, esecuzione delle azioni
e salvataggio delle previsioni generate dal modello di machine learning.

In questa versione non vengono utilizzate regole fisse per rilevare anomalie.
L'autenticazione continua a funzionare normalmente, ma il rilevamento dei
comportamenti anomali viene eseguito tramite il modello addestrato sui dati storici.
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
ML_MODEL_PATH = PROJECT_ROOT / "models" / "activity_model.pkl"

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
    Así puede compararse con el hash guardado en la base de datos.

    Converte la password inserita dall'utente in un hash SHA-256.
    In questo modo può essere confrontata con l'hash salvato nel database.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def get_connection():
    """
    Abre una conexión con PostgreSQL utilizando la configuración definida.
    Si ocurre un error, muestra el mensaje y devuelve None.

    Apre una connessione a PostgreSQL utilizzando la configurazione definita.
    Se si verifica un errore, mostra il messaggio e restituisce None.
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


def show_ml_prediction(model_bundle, user_id: int, element_id: int, action_id: int):
    """
    Ejecuta una predicción del modelo para la actividad actual.

    Devuelve:
    - prediction: 1 si la actividad es anómala, 0 si es normal
    - probability: score numérico entre 0 y 1

    Esegue una previsione del modello per l'attività attuale.

    Restituisce:
    - prediction: 1 se l'attività è anomala, 0 se è normale
    - probability: score numerico tra 0 e 1
    """
    prediction, probability = predict_activity_with_model(
        model_bundle=model_bundle,
        user_id=user_id,
        element_id=element_id,
        entity_id=DEFAULT_ENTITY_ID,
        action_id=action_id,
        logged_at=datetime.now()
    )

    if prediction == 1:
        print("\n[ML] Previsione del modello: attività anomala.")
    else:
        print("\n[ML] Previsione del modello: attività normale.")

    if probability is not None:
        print(f"[ML] Score stimato di anomalia: {probability:.4f}")

    return int(prediction), probability


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


def process_action(
    cursor,
    connection,
    model_bundle,
    user_id: int,
    element_id: int,
    action_option: str,
    action_text: str
):
    """
    Procesa una acción del usuario sobre un elemento.

    Flujo:
    1. Obtiene el action_id real.
    2. Guarda la actividad en activity_log.
    3. Si existe modelo cargado, ejecuta la predicción ML.
    4. Guarda la predicción en ml_prediction_log.
    5. Confirma la transacción.

    Elabora un'azione dell'utente su un elemento.

    Flusso:
    1. Ottiene l'action_id reale.
    2. Salva l'attività in activity_log.
    3. Se esiste un modello caricato, esegue la previsione ML.
    4. Salva la previsione in ml_prediction_log.
    5. Conferma la transazione.
    """
    action_id = ACTION_IDS[action_option]

    print(f"\nStai eseguendo l'azione: {action_text}.")
    activity_log_id = save_activity_log(
        cursor=cursor,
        user_id=user_id,
        action_id=action_id,
        element_id=element_id,
        entity_id=DEFAULT_ENTITY_ID
    )

    if model_bundle is not None:
        try:
            prediction, probability = show_ml_prediction(
                model_bundle=model_bundle,
                user_id=user_id,
                element_id=element_id,
                action_id=action_id
            )

            save_ml_prediction_log(
                cursor=cursor,
                activity_log_id=activity_log_id,
                user_id=user_id,
                element_id=element_id,
                entity_id=DEFAULT_ENTITY_ID,
                action_id=action_id,
                prediction=prediction,
                anomaly_probability=probability
            )

            if prediction == 1:
                print("[ML] Avviso: questa attività si discosta dal comportamento abituale.")
        except Exception as e:
            print(f"[ML] Errore durante la previsione: {e}")
    else:
        print("[ML] Nessun modello caricato. Nessuna previsione salvata.")

    connection.commit()
    print(f"Attività registrata correttamente. ID attività: {activity_log_id}")


def action_menu(cursor, connection, model_bundle, user_id: int, element_id: int, element_name: str):
    """
    Muestra el menú de acciones para un elemento y gestiona la opción elegida.

    Mostra il menu delle azioni per un elemento e gestisce l'opzione scelta.
    """
    while True:
        show_action_menu(element_name)
        option = input("Scegli un'opzione: ").strip()

        match option:
            case "0":
                print("\nRitorno al menu element...")
                break
            case "1":
                process_action(cursor, connection, model_bundle, user_id, element_id, "1", "visualizzazione")
            case "2":
                process_action(cursor, connection, model_bundle, user_id, element_id, "2", "creazione")
            case "3":
                process_action(cursor, connection, model_bundle, user_id, element_id, "3", "modifica")
            case "4":
                process_action(cursor, connection, model_bundle, user_id, element_id, "4", "eliminazione")
            case "5":
                process_action(cursor, connection, model_bundle, user_id, element_id, "5", "copia")
            case "6":
                process_action(cursor, connection, model_bundle, user_id, element_id, "6", "condivisione")
            case _:
                print("\nOpzione non valida.")


def element_menu(cursor, connection, model_bundle, login_log_id: int, user_id: int):
    """
    Muestra el menú de elementos disponibles y abre el menú de acciones.

    Mostra il menu degli elementi disponibili e apre il menu delle azioni.
    """
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
            action_menu(cursor, connection, model_bundle, user_id, element_id, element_name)
        else:
            print("\nOpzione non valida.")


def login_user() -> bool:
    """
    Gestiona todo el proceso de inicio de sesión.

    Flujo:
    1. Abre conexión a PostgreSQL.
    2. Carga el modelo ML si existe.
    3. Permite hasta MAX_ATTEMPTS intentos.
    4. Busca al usuario por email.
    5. Comprueba si está activo.
    6. Valida la contraseña comparando hashes.
    7. Si el login es correcto, lo guarda en login_log.
    8. Abre el menú de elementos.

    Gestisce l'intero processo di login.

    Flusso:
    1. Apre la connessione a PostgreSQL.
    2. Carica il modello ML se esiste.
    3. Consente fino a MAX_ATTEMPTS tentativi.
    4. Cerca l'utente tramite email.
    5. Verifica se è attivo.
    6. Valida la password confrontando gli hash.
    7. Se il login è corretto, lo salva in login_log.
    8. Apre il menu degli elementi.
    """
    connection = get_connection()
    if connection is None:
        return False

    cursor = None

    try:
        cursor = connection.cursor()

        try:
            model_bundle = load_model(str(ML_MODEL_PATH))
        except Exception:
            model_bundle = None
            print("Impossibile caricare il modello ML. Il sistema continuerà senza previsione di anomalie.")

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