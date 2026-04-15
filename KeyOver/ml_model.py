import pandas as pd
import psycopg2
from psycopg2 import Error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from rules import evaluate_activity_anomaly

"""
Este archivo se encarga de cargar los datos de actividad desde PostgreSQL,
preparar las variables necesarias para el entrenamiento, etiquetar las anomalías
según las reglas de negocio y entrenar un modelo de Random Forest para detectar
comportamientos anómalos.
También incluye funciones para guardar el modelo entrenado, volver a cargarlo
desde disco y realizar predicciones sobre nuevas actividades.

Questo file si occupa di caricare i dati di attività da PostgreSQL,
preparare le variabili necessarie per l'addestramento, etichettare le anomalie
secondo le regole di business e addestrare un modello Random Forest per rilevare
comportamenti anomali.
Include anche funzioni per salvare il modello addestrato, ricaricarlo dal disco
ed eseguire previsioni su nuove attività.
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}


def get_connection():
    """
    Abre una conexión con PostgreSQL utilizando la configuración definida
    en DB_CONFIG.
    Si ocurre un error durante la conexión, muestra el mensaje por pantalla
    y devuelve None.

    Apre una connessione a PostgreSQL utilizzando la configurazione definita
    in DB_CONFIG.
    Se si verifica un errore durante la connessione, mostra il messaggio a schermo
    e restituisce None.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def load_activity_data() -> pd.DataFrame:
    """
    Carga los datos de la tabla activity_log desde PostgreSQL y los devuelve
    en un DataFrame de pandas.
    Si la conexión falla o se produce un error durante la lectura,
    se devuelve un DataFrame vacío.

    Carica i dati della tabella activity_log da PostgreSQL e li restituisce
    in un DataFrame pandas.
    Se la connessione fallisce o si verifica un errore durante la lettura,
    viene restituito un DataFrame vuoto.
    """
    connection = get_connection()
    if connection is None:
        return pd.DataFrame()

    query = """
        SELECT 
            activity_log_id,
            user_id,
            element_id,
            entity_id,
            action_id,
            logged_at
        FROM activity_log
        ORDER BY activity_log_id
    """

    try:
        df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        print(f"Errore durante il caricamento di activity_log: {e}")
        return pd.DataFrame()
    finally:
        connection.close()


def prepare_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara las variables necesarias para el entrenamiento del modelo.

    Pasos realizados:
    - crea una copia del DataFrame original
    - convierte la columna logged_at a tipo datetime
    - extrae variables temporales:
        - hour
        - minute
        - day_of_week
    - evalúa cada fila con las reglas de negocio definidas en rules.py
    - marca como anomalía únicamente los casos considerados anomalías fuertes

    El resultado es un nuevo DataFrame con la columna adicional "anomaly".

    Prepara le variabili necessarie per l'addestramento del modello.

    Passaggi eseguiti:
    - crea una copia del DataFrame originale
    - converte la colonna logged_at in tipo datetime
    - estrae variabili temporali:
        - hour
        - minute
        - day_of_week
    - valuta ogni riga con le regole di business definite in rules.py
    - marca come anomalia solo i casi considerati anomalie forti

    Il risultato è un nuovo DataFrame con la colonna aggiuntiva "anomaly".
    """
    if df.empty:
        return df

    df = df.copy()

    df["logged_at"] = pd.to_datetime(df["logged_at"])
    df["hour"] = df["logged_at"].dt.hour
    df["minute"] = df["logged_at"].dt.minute
    df["day_of_week"] = df["logged_at"].dt.dayofweek

    anomalies = []

    for _, row in df.iterrows():
        messages = evaluate_activity_anomaly(
            user_id=int(row["user_id"]),
            element_id=int(row["element_id"]),
            entity_id=int(row["entity_id"]),
            action_id=int(row["action_id"]),
            dt=row["logged_at"].to_pydatetime()
        )
        hard_anomaly = any(msg.startswith("Anomalia:") for msg in messages)
        anomalies.append(1 if hard_anomaly else 0)

    df["anomaly"] = anomalies
    return df


def train_activity_model(df: pd.DataFrame):
    """
    Entrena un modelo Random Forest para clasificar actividades normales
    y anómalas.
    Flujo del proceso:
    - comprueba si el DataFrame está vacío
    - define las variables de entrada
    - separa características (X) y variable objetivo (y)
    - divide los datos en entrenamiento y prueba
    - entrena el modelo RandomForestClassifier
    - genera predicciones sobre el conjunto de prueba
    - muestra métricas de evaluación por pantalla

    Devuelve:
    - modelo entrenado
    - X_train
    - X_test
    - y_train
    - y_test

    Addestra un modello Random Forest per classificare attività normali
    e anomale.
    Flusso del processo:
    - verifica se il DataFrame è vuoto
    - definisce le variabili di input
    - separa caratteristiche (X) e variabile target (y)
    - divide i dati in training e test
    - addestra il modello RandomForestClassifier
    - genera previsioni sul set di test
    - mostra a schermo le metriche di valutazione

    Restituisce:
    - modello addestrato
    - X_train
    - X_test
    - y_train
    - y_test
    """
    if df.empty:
        print("Non ci sono dati per l'addestramento.")
        return None, None, None, None, None

    features = [
        "user_id",
        "element_id",
        "entity_id",
        "action_id",
        "hour",
        "minute",
        "day_of_week"
    ]

    X = df[features]
    y = df["anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_features="sqrt"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== RISULTATI DEL MODELLO ACTIVITY ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test


def save_model(model, path: str = "activity_model.pkl"):
    """
    Guarda el modelo entrenado en disco utilizando joblib.
    Parámetros:
    - model: modelo ya entrenado
    - path: ruta y nombre del archivo de salida

    Salva il modello addestrato su disco utilizzando joblib.
    Parametri:
    - model: modello già addestrato
    - path: percorso e nome del file di output
    """
    joblib.dump(model, path)
    print(f"Modello salvato in: {path}")


def load_model(path: str = "activity_model.pkl"):
    """
    Carga desde disco un modelo previamente guardado con joblib.
    Parámetros:
    - path: ruta del archivo del modelo

    Devuelve el modelo cargado.

    Carica da disco un modello precedentemente salvato con joblib.
    Parametri:
    - path: percorso del file del modello

    Restituisce il modello caricato.
    """
    return joblib.load(path)


def predict_activity_with_model(
    model,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at
):
    """
    Realiza una predicción individual utilizando el modelo ya entrenado.
    A partir de los datos de entrada:
    - user_id
    - element_id
    - entity_id
    - action_id
    - logged_at
    construye una fila con las mismas variables utilizadas en el entrenamiento:
    - user_id
    - element_id
    - entity_id
    - action_id
    - hour
    - minute
    - day_of_week
    Devuelve:
    - prediction: clase predicha
    - probability: probabilidad estimada de anomalía si el modelo soporta predict_proba

    Esegue una previsione singola utilizzando il modello già addestrato.
    A partire dai dati di input:
    - user_id
    - element_id
    - entity_id
    - action_id
    - logged_at
    costruisce una riga con le stesse variabili utilizzate durante l'addestramento:
    - user_id
    - element_id
    - entity_id
    - action_id
    - hour
    - minute
    - day_of_week
    Restituisce:
    - prediction: classe predetta
    - probability: probabilità stimata di anomalia se il modello supporta predict_proba
    """
    logged_at = pd.to_datetime(logged_at)

    row = pd.DataFrame([{
        "user_id": user_id,
        "element_id": element_id,
        "entity_id": entity_id,
        "action_id": action_id,
        "hour": logged_at.hour,
        "minute": logged_at.minute,
        "day_of_week": logged_at.dayofweek
    }])

    prediction = model.predict(row)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(row)[0][1]
    else:
        probability = None

    return prediction, probability


if __name__ == "__main__":
    print("Caricamento dei dati di activity_log...")
    df = load_activity_data()
    print("Shape originale:", df.shape)

    print("Preparazione delle feature di activity...")
    df_prepared = prepare_activity_features(df)
    print("Shape preparata:", df_prepared.shape)

    if not df_prepared.empty:
        print("\nDistribuzione delle anomalie:")
        print(df_prepared["anomaly"].value_counts())

    print("Addestramento del modello...")
    model, X_train, X_test, y_train, y_test = train_activity_model(df_prepared)

    if model is not None:
        save_model(model)