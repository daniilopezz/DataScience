import hashlib
import getpass
from datetime import datetime

import psycopg2
from psycopg2 import Error

from ml_model import load_model, predict_activity_with_model
from rules import evaluate_login_anomaly, evaluate_activity_anomaly

"""
Este archivo gestiona el flujo principal de interacción con la aplicación:
inicio de sesión, registro de accesos, selección de elementos, ejecución de acciones
y almacenamiento de predicciones generadas por el modelo de machine learning.
Combina dos capas de control:
- reglas de negocio, para detectar anomalías según permisos y horarios
- modelo de machine learning, para estimar si una actividad es anómala

Questo file gestisce il flusso principale di interazione con l'applicazione:
login, registrazione degli accessi, selezione degli elementi, esecuzione delle azioni
e salvataggio delle previsioni generate dal modello di machine learning.
Combina due livelli di controllo:
- regole di business, per rilevare anomalie in base a permessi e orari
- modello di machine learning, per stimare se un'attività è anomala
"""

# Configuración de conexión a PostgreSQL.
# Configurazione della connessione a PostgreSQL.
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

# Número máximo de intentos permitidos en el login.
# Numero massimo di tentativi consentiti nel login.
MAX_ATTEMPTS = 3

# Entidad por defecto que se utilizará en las acciones.
# En esta base de datos, 1 corresponde a "Password".
#
# Entità predefinita che verrà utilizzata nelle azioni.
# In questo database, 1 corrisponde a "Password".
DEFAULT_ENTITY_ID = 1

# Ruta del modelo de machine learning ya entrenado.
# Percorso del modello di machine learning già addestrato.
ML_MODEL_PATH = "activity_model.pkl"

# Diccionario que relaciona cada opción del menú con su action_id real de la base de datos.
# Dizionario che collega ogni opzione del menu con il relativo action_id reale del database.
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
    De esta forma se puede comparar con la contraseña almacenada en la base de datos
    sin trabajar con la contraseña en texto plano.

    Converte la password inserita dall'utente in un hash SHA-256.
    In questo modo può essere confrontata con la password memorizzata nel database
    senza lavorare con la password in chiaro.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def get_connection():
    """
    Intenta abrir una conexión con PostgreSQL usando la configuración definida.
    Si falla, muestra el error por pantalla y devuelve None.

    Tenta di aprire una connessione a PostgreSQL usando la configurazione definita.
    Se fallisce, mostra l'errore a schermo e restituisce None.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def get_next_attempt(cursor, user_id: int) -> int:
    """
    Obtiene el siguiente número de intento de login para un usuario concreto.
    Busca el valor máximo de la columna attempt en login_log para ese usuario
    y le suma 1. Si no existen intentos previos, comienza en 1.

    Ottiene il numero di tentativo successivo di login per uno specifico utente.
    Cerca il valore massimo della colonna attempt in login_log per quell'utente
    e aggiunge 1. Se non esistono tentativi precedenti, parte da 1.
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
    Guarda en login_log un nuevo intento de inicio de sesión.
    Registra:
    - user_id
    - resultado del login
    - número de intento
    - fecha y hora del acceso
    - logout_at inicialmente en NULL
    Devuelve el login_log_id generado por la inserción.

    Salva in login_log un nuovo tentativo di accesso.
    Registra:
    - user_id
    - risultato del login
    - numero di tentativo
    - data e ora dell'accesso
    - logout_at inizialmente a NULL
    Restituisce il login_log_id generato dall'inserimento.
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
    Actualiza el campo logout_at del registro de login correspondiente
    cuando el usuario cierra la sesión.

    Aggiorna il campo logout_at del record di login corrispondente
    quando l'utente chiude la sessione.
    """
    query = """
        UPDATE login_log
        SET logout_at = CURRENT_TIMESTAMP
        WHERE login_log_id = %s
    """
    cursor.execute(query, (login_log_id,))


def save_activity_log(cursor, user_id: int, action_id: int, element_id: int, entity_id: int):
    """
    Guarda una acción realizada por el usuario en la tabla activity_log.
    Registra:
    - usuario
    - elemento seleccionado
    - entidad
    - acción ejecutada
    - fecha y hora de la operación

    Devuelve el activity_log_id generado, para poder relacionarlo después
    con la predicción del modelo de machine learning.

    Salva un'azione eseguita dall'utente nella tabella activity_log.
    Registra:
    - utente
    - elemento selezionato
    - entità
    - azione eseguita
    - data e ora dell'operazione

    Restituisce l'activity_log_id generato, così da poterlo collegare in seguito
    alla previsione del modello di machine learning.
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
    Guarda en la tabla ml_prediction_log la predicción generada por el modelo
    de machine learning asociada a una actividad concreta.
    El registro queda enlazado con activity_log mediante activity_log_id,
    lo que permite saber exactamente qué predicción corresponde a cada acción.

    Salva nella tabella ml_prediction_log la previsione generata dal modello
    di machine learning associata a una specifica attività.
    Il record rimane collegato ad activity_log tramite activity_log_id,
    permettendo di sapere esattamente quale previsione corrisponde a ogni azione.
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
    Obtiene desde la tabla element todos los elementos disponibles,
    ordenados por element_id.
    Se utilizan para construir dinámicamente el menú de elementos.

    Recupera dalla tabella element tutti gli elementi disponibili,
    ordinati per element_id.
    Vengono utilizzati per costruire dinamicamente il menu degli elementi.
    """
    query = """
        SELECT element_id, name
        FROM element
        ORDER BY element_id
    """
    cursor.execute(query)
    return cursor.fetchall()


def show_ml_prediction(model, user_id: int, element_id: int, action_id: int):
    """
    Llama al modelo de machine learning para predecir si la actividad actual
    es normal o anómala.
    Muestra por pantalla:
    - la clasificación del modelo
    - la probabilidad estimada de anomalía, si está disponible

    Devuelve:
    - prediction: True/False
    - probability: probabilidad numérica o None

    Richiama il modello di machine learning per prevedere se l'attività attuale
    è normale o anomala.
    Mostra a schermo:
    - la classificazione del modello
    - la probabilità stimata di anomalia, se disponibile

    Restituisce:
    - prediction: True/False
    - probability: probabilità numerica o None
    """
    prediction, probability = predict_activity_with_model(
        model=model,
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
        print(f"[ML] Probabilità stimata di anomalia: {probability:.4f}")

    return bool(prediction), probability


def ask_user_confirmation(messages: list[str]) -> bool:
    """
    Muestra los mensajes de anomalía detectados por las reglas
    y pregunta al usuario si desea continuar igualmente.
    Devuelve:
    - True si el usuario responde 's'
    - False si el usuario responde 'n'

    Mostra i messaggi di anomalia rilevati dalle regole
    e chiede all'utente se desidera continuare comunque.
    Restituisce:
    - True se l'utente risponde 's'
    - False se l'utente risponde 'n'
    """
    print("\n*** ANOMALIA RILEVATA ***")
    for message in messages:
        print(f"- {message}")

    while True:
        answer = input("Vuoi continuare comunque? (s/n): ").strip().lower()
        if answer == "s":
            return True
        if answer == "n":
            return False
        print("Risposta non valida. Scrivi 's' oppure 'n'.")


def show_main_menu():
    """
    Muestra el menú principal del programa.
    Desde aquí el usuario puede iniciar sesión o salir de la aplicación.

    Mostra il menu principale del programma.
    Da qui l'utente può effettuare il login oppure uscire dall'applicazione.
    """
    print("\n=== MENU PRINCIPALE ===")
    print("1 - Accedi")
    print("0 - Esci")


def show_element_menu(elements):
    """
    Muestra el menú de elementos disponibles.
    Los elementos se cargan dinámicamente desde la base de datos.

    Mostra il menu degli elementi disponibili.
    Gli elementi vengono caricati dinamicamente dal database.
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
    model,
    user_id: int,
    element_id: int,
    action_option: str,
    action_text: str
):
    """
    Procesa una acción del usuario sobre un elemento concreto.
    Flujo:
    1. Obtiene el action_id real según la opción elegida.
    2. Evalúa si la acción es anómala según las reglas.
    3. Si hay anomalías, pide confirmación al usuario.
    4. Si el usuario acepta, guarda la actividad en activity_log.
    5. Si existe un modelo cargado, ejecuta la predicción ML.
    6. Guarda la predicción en ml_prediction_log enlazada con la actividad real.

    Elabora un'azione dell'utente su uno specifico elemento.
    Flusso:
    1. Ottiene l'action_id reale in base all'opzione scelta.
    2. Valuta se l'azione è anomala secondo le regole.
    3. Se ci sono anomalie, chiede conferma all'utente.
    4. Se l'utente accetta, salva l'attività in activity_log.
    5. Se esiste un modello caricato, esegue la previsione ML.
    6. Salva la previsione in ml_prediction_log collegata all'attività reale.
    """
    action_id = ACTION_IDS[action_option]
    now = datetime.now()

    anomaly_messages = evaluate_activity_anomaly(
        user_id=user_id,
        element_id=element_id,
        entity_id=DEFAULT_ENTITY_ID,
        action_id=action_id,
        dt=now
    )

    if anomaly_messages:
        confirmed = ask_user_confirmation(anomaly_messages)
        if not confirmed:
            print("Operazione annullata dall'utente.")
            return

    print(f"\nStai eseguendo l'azione: {action_text}.")
    activity_log_id = save_activity_log(cursor, user_id, action_id, element_id, DEFAULT_ENTITY_ID)

    if model is not None:
        prediction, probability = show_ml_prediction(model, user_id, element_id, action_id)

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
    else:
        print("[ML] Nessun modello caricato. Nessuna previsione salvata.")

    connection.commit()
    print(f"Attività registrata correttamente. ID attività: {activity_log_id}")


def action_menu(cursor, connection, model, user_id: int, element_id: int, element_name: str):
    """
    Muestra el menú de acciones para un elemento concreto y gestiona
    la opción elegida por el usuario.

    Mostra il menu delle azioni per uno specifico elemento e gestisce
    l'opzione scelta dall'utente.
    """
    while True:
        show_action_menu(element_name)
        option = input("Scegli un'opzione: ").strip()

        match option:
            case "0":
                print("\nRitorno al menu element...")
                break
            case "1":
                process_action(cursor, connection, model, user_id, element_id, "1", "visualizzazione")
            case "2":
                process_action(cursor, connection, model, user_id, element_id, "2", "creazione")
            case "3":
                process_action(cursor, connection, model, user_id, element_id, "3", "modifica")
            case "4":
                process_action(cursor, connection, model, user_id, element_id, "4", "eliminazione")
            case "5":
                process_action(cursor, connection, model, user_id, element_id, "5", "copia")
            case "6":
                process_action(cursor, connection, model, user_id, element_id, "6", "condivisione")
            case _:
                print("\nOpzione non valida.")


def element_menu(cursor, connection, model, login_log_id: int, user_id: int):
    """
    Muestra el menú de elementos disponibles.
    Cuando el usuario selecciona uno, se abre el menú de acciones.
    Si elige salir, se registra el logout del login actual.

    Mostra il menu degli elementi disponibili.
    Quando l'utente ne seleziona uno, si apre il menu delle azioni.
    Se sceglie di uscire, viene registrato il logout del login attuale.
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
            action_menu(cursor, connection, model, user_id, element_id, element_name)
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
    7. Evalúa anomalías de login según reglas.
    8. Si el login es correcto, lo guarda en login_log.
    9. Abre el menú de elementos.

    Gestisce l'intero processo di login.
    Flusso:
    1. Apre la connessione a PostgreSQL.
    2. Carica il modello ML se esiste.
    3. Consente fino a MAX_ATTEMPTS tentativi.
    4. Cerca l'utente tramite email.
    5. Verifica se è attivo.
    6. Valida la password confrontando gli hash.
    7. Valuta eventuali anomalie di login secondo le regole.
    8. Se il login è corretto, lo salva in login_log.
    9. Apre il menu degli elementi.
    """
    connection = get_connection()
    if connection is None:
        return False

    cursor = None

    try:
        cursor = connection.cursor()

        try:
            model = load_model(ML_MODEL_PATH)
        except Exception:
            model = None
            print("Impossibile caricare il modello ML. Il sistema continuerà solo con le regole.")

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
                now = datetime.now()
                anomaly_messages = evaluate_login_anomaly(user_id, now)

                if anomaly_messages:
                    confirmed = ask_user_confirmation(anomaly_messages)
                    if not confirmed:
                        save_login_log(cursor, user_id, False, attempt)
                        connection.commit()
                        print("Accesso annullato dall'utente.")
                        return False

                login_log_id = save_login_log(cursor, user_id, True, attempt)
                connection.commit()

                print(f"\nLogin effettuato correttamente. Benvenuto/a, {name} {surname}.")
                element_menu(cursor, connection, model, login_log_id, user_id)
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
    Muestra el menú principal y dirige el flujo general de ejecución.

    Funzione principale del programma.
    Mostra il menu principale e dirige il flusso generale di esecuzione.
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