import hashlib
import getpass
from datetime import datetime

import psycopg2
from psycopg2 import Error

from ml_model import load_model, predict_activity_with_model
from rules import evaluate_login_anomaly, evaluate_activity_anomaly

# Configuración de conexión a PostgreSQL.
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

# Número máximo de intentos permitidos en el login.
MAX_ATTEMPTS = 3

# Entidad por defecto que vamos a usar en las acciones.
# En tu base, 1 corresponde a "Password".
DEFAULT_ENTITY_ID = 1

# Ruta del modelo de machine learning ya entrenado.
ML_MODEL_PATH = "activity_model.pkl"

# Diccionario que relaciona cada opción del menú con su action_id real de la base de datos.
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
    Así podemos compararla con la contraseña hasheada guardada en la base de datos.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def get_connection():
    """
    Intenta abrir una conexión con PostgreSQL usando la configuración definida arriba.
    Si falla, muestra el error y devuelve None.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Error al conectar con PostgreSQL: {e}")
        return None


def get_next_attempt(cursor, user_id: int) -> int:
    """
    Obtiene el número de intento siguiente para un usuario concreto en la tabla login_log.
    Busca el máximo valor de 'attempt' de ese usuario y le suma 1.
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
    - si el login fue correcto o no
    - número de intento
    - fecha y hora del login
    - logout_at inicialmente en NULL

    Devuelve el login_log_id generado en la inserción.
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
    Actualiza el campo logout_at del login_log correspondiente cuando el usuario sale.
    De esta forma queda registrado el cierre de sesión.
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
    - acción realizada
    - fecha y hora

    Devuelve el activity_log_id generado en la inserción para poder enlazarlo
    después con la predicción del modelo ML.
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
    Guarda en la tabla ml_prediction_log la predicción del modelo de machine learning
    asociada a una actividad concreta.

    Se enlaza con activity_log mediante activity_log_id, para que cada predicción
    quede unida exactamente a la acción que la ha generado.
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
    Obtiene de la tabla element todos los elementos disponibles.
    Se usan para construir el menú dinámico de elementos.
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
    - la probabilidad estimada de anomalía

    Devuelve:
    - prediction: True/False
    - probability: probabilidad numérica
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
        print("\n[ML] Predicción del modelo: actividad anómala.")
    else:
        print("\n[ML] Predicción del modelo: actividad normal.")

    if probability is not None:
        print(f"[ML] Probabilidad estimada de anomalía: {probability:.4f}")

    return bool(prediction), probability


def ask_user_confirmation(messages: list[str]) -> bool:
    """
    Muestra por pantalla los mensajes de anomalía detectados por las reglas
    y pregunta al usuario si quiere continuar o no.

    Devuelve:
    - True si el usuario responde 's'
    - False si el usuario responde 'n'
    """
    print("\n*** ANOMALÍA DETECTADA ***")
    for message in messages:
        print(f"- {message}")

    while True:
        answer = input("¿Quieres continuar igualmente? (s/n): ").strip().lower()
        if answer == "s":
            return True
        if answer == "n":
            return False
        print("Respuesta no válida. Escribe 's' o 'n'.")


def show_main_menu():
    """
    Muestra el menú principal del programa.
    Desde aquí el usuario puede iniciar sesión o salir.
    """
    print("\n=== MENÚ PRINCIPAL ===")
    print("1 - Iniciar sesión")
    print("0 - Salir")


def show_element_menu(elements):
    """
    Muestra el menú de elementos disponible.
    Los elementos se cargan dinámicamente desde la base de datos.
    """
    print("\n=== MENÚ DE ELEMENT ===")
    for element_id, name in elements:
        print(f"{element_id} - {name}")
    print("0 - Salir al menú principal")


def show_action_menu(selected_element_name: str):
    """
    Muestra el menú de acciones posibles sobre el elemento seleccionado.
    """
    print(f"\n=== MENÚ DE ACCIONES | ELEMENTO: {selected_element_name} ===")
    print("0 - Volver al menú de element")
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
    Procesa una acción del usuario sobre un elemento.

    Flujo:
    1. Obtiene el action_id real según la opción elegida.
    2. Evalúa si la acción es anómala según las reglas.
    3. Si hay anomalías, pide confirmación al usuario.
    4. Si el usuario acepta, guarda la actividad en activity_log.
    5. Si hay modelo cargado, realiza predicción ML.
    6. Guarda la predicción en ml_prediction_log enlazada con la actividad real.
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
            print("Operación cancelada por el usuario.")
            return

    print(f"\nEstas {action_text}.")
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
        print("[ML] No hay modelo cargado. No se guardó predicción.")

    connection.commit()
    print(f"Actividad registrada correctamente. ID actividad: {activity_log_id}")


def action_menu(cursor, connection, model, user_id: int, element_id: int, element_name: str):
    """
    Muestra el menú de acciones para un elemento concreto y gestiona
    la opción elegida por el usuario.
    """
    while True:
        show_action_menu(element_name)
        option = input("Elige una opción: ").strip()

        match option:
            case "0":
                print("\nVolviendo al menú de element...")
                break
            case "1":
                process_action(cursor, connection, model, user_id, element_id, "1", "visualizando")
            case "2":
                process_action(cursor, connection, model, user_id, element_id, "2", "creando")
            case "3":
                process_action(cursor, connection, model, user_id, element_id, "3", "editando")
            case "4":
                process_action(cursor, connection, model, user_id, element_id, "4", "eliminando")
            case "5":
                process_action(cursor, connection, model, user_id, element_id, "5", "copiando")
            case "6":
                process_action(cursor, connection, model, user_id, element_id, "6", "compartiendo")
            case _:
                print("\nOpción no válida.")


def element_menu(cursor, connection, model, login_log_id: int, user_id: int):
    """
    Muestra el menú de elementos disponibles.
    Cuando el usuario selecciona uno, se abre el menú de acciones.
    Si elige salir, se registra el logout del login actual.
    """
    while True:
        elements = get_elements(cursor)
        element_map = {str(element_id): (element_id, name) for element_id, name in elements}

        show_element_menu(elements)
        option = input("Selecciona un element: ").strip()

        if option == "0":
            print("\nSaliendo al menú principal...")
            update_logout(cursor, login_log_id)
            connection.commit()
            print("Logout registrado correctamente.")
            break

        if option in element_map:
            element_id, element_name = element_map[option]
            print(f"\nHas seleccionado el elemento: {element_name}")
            action_menu(cursor, connection, model, user_id, element_id, element_name)
        else:
            print("\nOpción no válida.")


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
            print("No se pudo cargar el modelo ML. El sistema seguirá solo con reglas.")

        for _ in range(MAX_ATTEMPTS):
            print("\n--- LOGIN ---")
            email = input("Email: ").strip()
            password = getpass.getpass("Contraseña: ").strip()

            query = """
                SELECT user_id, name, surname, password_hash, is_active
                FROM users
                WHERE email = %s
            """
            cursor.execute(query, (email,))
            user = cursor.fetchone()

            if user is None:
                print("Usuario no encontrado.")
                continue

            user_id, name, surname, stored_password_hash, is_active = user
            attempt = get_next_attempt(cursor, user_id)

            if not is_active:
                save_login_log(cursor, user_id, False, attempt)
                connection.commit()
                print("Usuario inactivo. Acceso denegado.")
                return False

            if stored_password_hash == hash_password(password):
                now = datetime.now()
                anomaly_messages = evaluate_login_anomaly(user_id, now)

                if anomaly_messages:
                    confirmed = ask_user_confirmation(anomaly_messages)
                    if not confirmed:
                        save_login_log(cursor, user_id, False, attempt)
                        connection.commit()
                        print("Acceso cancelado por el usuario.")
                        return False

                login_log_id = save_login_log(cursor, user_id, True, attempt)
                connection.commit()

                print(f"\nLogin correcto. Bienvenido/a, {name} {surname}.")
                element_menu(cursor, connection, model, login_log_id, user_id)
                return True

            save_login_log(cursor, user_id, False, attempt)
            connection.commit()
            print("Contraseña incorrecta.")

        print("\nAcceso bloqueado por demasiados intentos fallidos.")
        return False

    except Error as e:
        print(f"Error en la base de datos: {e}")
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
    """
    while True:
        show_main_menu()
        option = input("Elige una opción: ").strip()

        if option == "1":
            login_user()
        elif option == "0":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()