# Punto de entrada principal de la aplicación KeyOver1.
# Punto di ingresso principale dell'applicazione KeyOver1.
#
# Flujo:
#   1. Menú principal → login o salida
#   2. Login: email + contraseña, máx. 3 intentos
#   3. Verificación de anomalía de login (perfil estadístico)
#   4. Apertura de sesión interactiva (menú de elementos y acciones con ML)
#
# Flusso:
#   1. Menu principale → login o uscita
#   2. Login: email + password, max 3 tentativi
#   3. Verifica anomalia di login (profilo statistico)
#   4. Apertura sessione interattiva (menu elementi e azioni con ML)

import getpass
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.db import get_connection, get_engine
from utils.hash import hash_password
from security.anomaly_guard import (
    build_login_profile, evaluate_login, get_combined_model,
    get_user_known_elements, get_model_session_threshold,
)
from app.session import run_session

MAX_ATTEMPTS = 3


# ─── Utilidades de BD / Utilità DB ────────────────────────────────────────────

def _get_next_attempt(cursor, user_id: int) -> int:
    cursor.execute(
        "SELECT COALESCE(MAX(attempt), 0) + 1 FROM login_log WHERE user_id = %s",
        (user_id,),
    )
    return cursor.fetchone()[0]


def _save_login_log(cursor, user_id: int, result: bool, attempt: int) -> int:
    # Registra el intento de login y devuelve login_log_id.
    # Registra il tentativo di login e restituisce login_log_id.
    cursor.execute(
        """
        INSERT INTO login_log (user_id, result, attempt, logged_at, logout_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP, NULL)
        RETURNING login_log_id
        """,
        (user_id, result, attempt),
    )
    return cursor.fetchone()[0]


def _load_login_history(engine) -> pd.DataFrame:
    # Carga el historial de logins para construir el perfil estadístico.
    # Carica la cronologia dei login per costruire il profilo statistico.
    return pd.read_sql(
        """
        SELECT login_log_id, user_id, result, logged_at
        FROM login_log
        ORDER BY login_log_id
        """,
        engine,
    )


def _load_session_thresholds(engine) -> dict[int, float]:
    # Calcula el umbral de coste de sesión por usuario como el percentil 95
    # del coste acumulado máximo observado en sesiones históricas completadas.
    # Calcola la soglia di costo sessione per utente come il 95° percentile
    # del costo cumulativo massimo osservato nelle sessioni storiche completate.
    try:
        df = pd.read_sql(
            """
            SELECT
                user_id,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY max_cost) AS threshold
            FROM (
                SELECT
                    mpl.user_id,
                    mpl.login_log_id,
                    MAX(mpl.session_cumulative_cost) AS max_cost
                FROM ml_prediction_log mpl
                JOIN login_log ll ON mpl.login_log_id = ll.login_log_id
                WHERE ll.logout_at IS NOT NULL
                GROUP BY mpl.user_id, mpl.login_log_id
            ) session_costs
            GROUP BY user_id
            """,
            engine,
        )
        if df.empty:
            return {}
        return {int(row.user_id): float(row.threshold) for _, row in df.iterrows()}
    except Exception as e:
        print(f"[THRESHOLD] Impossibile calcolare soglie: {e}")
        return {}


# ─── Flujo de login / Flusso di login ─────────────────────────────────────────

def _do_login() -> bool:
    # Gestiona el ciclo completo de login de un usuario.
    # Gestisce il ciclo completo di login di un utente.
    conn = get_connection()
    if conn is None:
        print("[ERRORE] Impossibile connettersi al database.")
        return False

    engine = get_engine()
    cursor = conn.cursor()

    try:
        # Cargar modelo ML (puede fallar si aún no se ha entrenado)
        # Caricare il modello ML (può fallire se non è ancora stato addestrato)
        try:
            combined_model = get_combined_model()
        except Exception as e:
            combined_model = None
            print(f"[ML] Modello non disponibile: {e}")

        # Construir perfil de login a partir del historial
        # Costruire il profilo di login a partire dalla cronologia
        try:
            history_df    = _load_login_history(engine)
            login_profiles = build_login_profile(history_df)
        except Exception as e:
            login_profiles = {}
            print(f"[PROFILO] Impossibile costruire i profili: {e}")

        # Cargar umbrales de coste de sesión por usuario
        # Caricare le soglie di costo sessione per utente
        try:
            session_thresholds = _load_session_thresholds(engine)
        except Exception as e:
            session_thresholds = {}
            print(f"[THRESHOLD] {e}")

        for attempt_num in range(MAX_ATTEMPTS):
            print("\n--- LOGIN ---")
            email    = input("Email: ").strip()
            password = getpass.getpass("Password: ").strip()

            cursor.execute(
                """
                SELECT user_id, name, surname, password_hash, is_active
                FROM users
                WHERE email = %s
                """,
                (email,),
            )
            user = cursor.fetchone()

            if user is None:
                print("Utente non trovato.")
                continue

            user_id, name, surname, stored_hash, is_active = user
            attempt_n = _get_next_attempt(cursor, user_id)

            if not is_active:
                _save_login_log(cursor, user_id, False, attempt_n)
                conn.commit()
                print("Account inattivo. Accesso negato.")
                return False

            if stored_hash != hash_password(password):
                _save_login_log(cursor, user_id, False, attempt_n)
                conn.commit()
                print("Password errata.")
                continue

            # Contraseña correcta → evaluar anomalía de login
            # Password corretta → valutare anomalia di login
            login_eval = evaluate_login(login_profiles, int(user_id))
            if login_eval["is_anomalous"]:
                _save_login_log(cursor, user_id, False, attempt_n)
                conn.commit()
                print(f"\n{login_eval['message']}")
                return False

            login_log_id = _save_login_log(cursor, user_id, True, attempt_n)
            conn.commit()

            known_elements = get_user_known_elements(combined_model, int(user_id))

            # Umbral: usar el valor del DB si existe, si no el del modelo entrenado.
            # Soglia: usare il valore del DB se esiste, altrimenti quello del modello.
            model_threshold = get_model_session_threshold(combined_model, int(user_id))
            user_threshold = session_thresholds.get(int(user_id), model_threshold)

            threshold_display = f"{user_threshold:.4f}" if user_threshold != float("inf") else "∞"
            elements_display = str(sorted(known_elements)) if known_elements is not None else "tutti"
            print(f"\nBenvenuto/a, {name} {surname}! Accesso effettuato.")
            print(f"  [SOGLIA SESSIONE] costo massimo consentito: {threshold_display}")
            print(f"  [ELEMENTI AUTORIZZATI] {elements_display}")
            run_session(
                conn=conn,
                cursor=cursor,
                user_id=int(user_id),
                login_log_id=login_log_id,
                combined_model=combined_model,
                session_cost_threshold=user_threshold,
                known_elements=known_elements,
            )
            return True

        print("\nAccesso bloccato: troppi tentativi falliti.")
        return False

    except Exception as e:
        print(f"[ERRORE] {e}")
        return False

    finally:
        cursor.close()
        conn.close()


# ─── Main / Principale ────────────────────────────────────────────────────────

def _show_main_menu():
    print("\n=== KEYOVER1 — MENU PRINCIPALE ===")
    print("  1 - Accedi")
    print("  0 - Esci")


def main():
    while True:
        _show_main_menu()
        choice = input("Scegli un'opzione: ").strip()

        if choice == "1":
            _do_login()
        elif choice == "0":
            print("Chiusura del programma. Arrivederci!")
            break
        else:
            print("Opzione non valida.")


if __name__ == "__main__":
    main()
