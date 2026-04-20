import pandas as pd
import psycopg2
from psycopg2 import Error
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
import sys

"""
Este archivo se encarga de cargar los datos de actividad desde PostgreSQL,
preparar las variables necesarias para el entrenamiento y entrenar modelos
de detección de anomalías basados en el comportamiento habitual de cada usuario.

La idea no es utilizar reglas fijas para marcar anomalías, sino permitir que
el modelo aprenda qué combinaciones son normales para cada usuario y detecte
desviaciones respecto a ese patrón aprendido.

Además, el modelo devuelve una anomaly_probability que:
- nunca puede ser 0
- refleja la rareza de la operación para ese usuario
- puede usarse como "coste" acumulado de sesión

Questo file si occupa di caricare i dati di attività da PostgreSQL,
preparare le variabili necessarie per l'addestramento e addestrare modelli
di rilevamento delle anomalie basati sul comportamento abituale di ciascun utente.

L'idea non è utilizzare regole fisse per marcare le anomalie, ma permettere
al modello di imparare quali combinazioni sono normali per ogni utente e
rilevare deviazioni rispetto a quel comportamento appreso.

Inoltre, il modello restituisce una anomaly_probability che:
- non può mai essere 0
- riflette la rarità dell'operazione per quell'utente
- può essere usata come "costo" cumulato di sessione
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Audit",
    "user": "dani",
    "password": ""
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "activity_model.pkl"

# Probabilidad mínima para evitar valores exactamente iguales a 0.
MIN_ANOMALY_PROBABILITY = 0.000001


def get_connection():
    """
    Abre una conexión con PostgreSQL utilizando la configuración definida
    en DB_CONFIG. Si ocurre un error, devuelve None.

    Apre una connessione a PostgreSQL utilizzando la configurazione definita
    in DB_CONFIG. Se si verifica un errore, restituisce None.
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Errore durante la connessione a PostgreSQL: {e}")
        return None


def load_activity_data() -> pd.DataFrame:
    """
    Carga los datos de activity_log desde PostgreSQL y los devuelve
    en un DataFrame.

    Carica i dati di activity_log da PostgreSQL e li restituisce
    in un DataFrame.
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
    Prepara las variables necesarias para el entrenamiento.
    A partir de logged_at se generan:
    - hour
    - minute
    - day_of_week

    No se crea ninguna etiqueta manual de anomalía, porque el objetivo
    es que el modelo aprenda el comportamiento normal directamente
    desde los datos históricos.

    Prepara le variabili necessarie per l'addestramento.
    A partire da logged_at vengono generate:
    - hour
    - minute
    - day_of_week

    Non viene creata nessuna etichetta manuale di anomalia, perché
    l'obiettivo è che il modello impari il comportamento normale
    direttamente dai dati storici.
    """
    if df.empty:
        return df

    df = df.copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce")
    df = df.dropna(subset=["logged_at"])

    df["hour"] = df["logged_at"].dt.hour
    df["minute"] = df["logged_at"].dt.minute
    df["day_of_week"] = df["logged_at"].dt.dayofweek

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz de variables que se utilizará en el modelo.
    Se aplican variables dummificadas para capturar mejor las categorías
    de element_id, entity_id, action_id y day_of_week.

    Costruisce la matrice delle variabili che verrà utilizzata nel modello.
    Vengono applicate variabili dummy per catturare meglio le categorie
    di element_id, entity_id, action_id e day_of_week.
    """
    feature_df = df[[
        "element_id",
        "entity_id",
        "action_id",
        "hour",
        "minute",
        "day_of_week"
    ]].copy()

    categorical_columns = ["element_id", "entity_id", "action_id", "day_of_week"]

    feature_df = pd.get_dummies(
        feature_df,
        columns=categorical_columns,
        prefix=categorical_columns,
        dtype=int
    )

    return feature_df


def build_user_frequency_stats(user_df: pd.DataFrame) -> dict:
    """
    Construye estadísticas históricas simples por usuario para capturar
    hábitos de comportamiento.

    Se calculan frecuencias relativas de:
    - element_id
    - entity_id
    - action_id
    - combinación (element_id, action_id)

    Esto permite que una operación muy poco frecuente para un usuario
    tenga un coste de anomalía más alto.

    Costruisce statistiche storiche semplici per utente per catturare
    le abitudini di comportamento.

    Si calcolano frequenze relative di:
    - element_id
    - entity_id
    - action_id
    - combinazione (element_id, action_id)

    Questo permette che un'operazione molto poco frequente per un utente
    abbia un costo di anomalia più alto.
    """
    total = len(user_df)

    element_freq = (user_df["element_id"].value_counts(normalize=True)).to_dict()
    entity_freq = (user_df["entity_id"].value_counts(normalize=True)).to_dict()
    action_freq = (user_df["action_id"].value_counts(normalize=True)).to_dict()

    combo_series = (
        user_df.groupby(["element_id", "action_id"]).size() / total
    )
    combo_freq = {tuple(map(int, k)): float(v) for k, v in combo_series.to_dict().items()}

    return {
        "element_freq": {int(k): float(v) for k, v in element_freq.items()},
        "entity_freq": {int(k): float(v) for k, v in entity_freq.items()},
        "action_freq": {int(k): float(v) for k, v in action_freq.items()},
        "combo_freq": combo_freq,
        "total_rows": int(total)
    }


def train_activity_model(df: pd.DataFrame, contamination: float = 0.05):
    """
    Entrena un modelo de detección de anomalías por cada usuario.
    Para cada user_id:
    - filtra sus actividades
    - construye su matriz de variables
    - entrena un IsolationForest
    - guarda también las columnas utilizadas en el entrenamiento
    - guarda estadísticas históricas de frecuencia para ese usuario

    Devuelve un diccionario con esta estructura:
    {
        user_id: {
            "model": modelo_entrenado,
            "feature_columns": columnas_entrenadas,
            "frequency_stats": estadísticas históricas
        }
    }

    Addestra un modello di rilevamento anomalie per ogni utente.
    Per ogni user_id:
    - filtra le sue attività
    - costruisce la sua matrice di variabili
    - addestra un IsolationForest
    - salva anche le colonne utilizzate durante l'addestramento
    - salva statistiche storiche di frequenza per quell'utente
    """
    if df.empty:
        print("Non ci sono dati per l'addestramento.")
        return None

    user_models = {}
    user_ids = sorted(df["user_id"].dropna().unique())

    for user_id in user_ids:
        user_df = df[df["user_id"] == user_id].copy()

        if user_df.empty:
            continue

        X_user = build_feature_matrix(user_df)

        if X_user.shape[0] < 10:
            print(f"Utente {user_id}: dati insufficienti per addestrare il modello.")
            continue

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )

        model.fit(X_user)

        predictions = model.predict(X_user)
        anomaly_count = int((predictions == -1).sum())

        frequency_stats = build_user_frequency_stats(user_df)

        user_models[int(user_id)] = {
            "model": model,
            "feature_columns": list(X_user.columns),
            "frequency_stats": frequency_stats
        }

        print(
            f"Utente {int(user_id)} -> attività: {len(user_df)} | "
            f"anomalia stimate nel training: {anomaly_count}"
        )

    if not user_models:
        print("Nessun modello è stato addestrato.")
        return None

    return user_models


def save_model(model_bundle, path: str = str(MODEL_PATH)):
    """
    Guarda en disco el conjunto de modelos entrenados.

    Salva su disco l'insieme dei modelli addestrati.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path_obj)
    print(f"Modelli salvati in: {path_obj}")


def load_model(path: str = str(MODEL_PATH)):
    """
    Carga desde disco el conjunto de modelos entrenados.

    Carica da disco l'insieme dei modelli addestrati.
    """
    return joblib.load(path)


def build_single_row_features(
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at
) -> pd.DataFrame:
    """
    Construye una fila de variables con el mismo formato lógico usado
    durante el entrenamiento.

    Costruisce una riga di variabili con lo stesso formato logico usato
    durante l'addestramento.
    """
    logged_at = pd.to_datetime(logged_at)

    row = pd.DataFrame([{
        "element_id": element_id,
        "entity_id": entity_id,
        "action_id": action_id,
        "hour": logged_at.hour,
        "minute": logged_at.minute,
        "day_of_week": logged_at.dayofweek
    }])

    return build_feature_matrix(row)


def align_features_to_training(row_features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Alinea una fila de inferencia con las columnas exactas usadas
    durante el entrenamiento del modelo.
    Si falta alguna columna, se rellena con 0.
    Si sobra alguna, se elimina.

    Allinea una riga di inferenza con le colonne esatte usate
    durante l'addestramento del modello.
    Se manca qualche colonna, viene riempita con 0.
    Se ce n'è qualcuna in più, viene eliminata.
    """
    aligned = row_features.copy()

    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0

    aligned = aligned[feature_columns]
    return aligned


def compute_behavior_rarity_score(
    frequency_stats: dict,
    element_id: int,
    entity_id: int,
    action_id: int
) -> float:
    """
    Calcula un score de rareza en función de los hábitos previos del usuario.

    Cuanto menos frecuente sea para ese usuario:
    - el elemento
    - la entidad
    - la acción
    - la combinación elemento + acción

    más alto será el score.

    El valor final se mantiene en el rango [0, 1].

    Calcola uno score di rarità in funzione delle abitudini precedenti dell'utente.
    Più un elemento/entità/azione/combinazione è rara per quell'utente,
    più alto sarà lo score.
    """
    if not frequency_stats:
        return 0.0

    element_freq = float(frequency_stats.get("element_freq", {}).get(int(element_id), 0.0))
    entity_freq = float(frequency_stats.get("entity_freq", {}).get(int(entity_id), 0.0))
    action_freq = float(frequency_stats.get("action_freq", {}).get(int(action_id), 0.0))
    combo_freq = float(
        frequency_stats.get("combo_freq", {}).get((int(element_id), int(action_id)), 0.0)
    )

    # Rareza = 1 - frecuencia.
    # Si algo nunca ha ocurrido para ese usuario, su rareza es máxima (=1).
    element_rarity = 1.0 - element_freq
    entity_rarity = 1.0 - entity_freq
    action_rarity = 1.0 - action_freq
    combo_rarity = 1.0 - combo_freq

    # Damos más peso a la combinación elemento+acción y al elemento,
    # porque suelen describir mejor el hábito real del usuario.
    rarity_score = (
        0.35 * combo_rarity +
        0.30 * element_rarity +
        0.20 * action_rarity +
        0.15 * entity_rarity
    )

    return min(max(float(rarity_score), 0.0), 1.0)


def combine_model_and_behavior_scores(model_probability: float, behavior_rarity: float) -> float:
    """
    Combina:
    - la probabilidad de anomalía del modelo
    - la rareza del comportamiento histórico del usuario

    El objetivo es que una operación poco habitual para ese usuario
    incremente el coste final, aunque la predicción puntual del modelo
    no sea extrema.

    Combina:
    - la probabilità di anomalia del modello
    - la rarità del comportamento storico dell'utente
    """
    combined = (0.65 * float(model_probability)) + (0.35 * float(behavior_rarity))
    combined = min(max(combined, MIN_ANOMALY_PROBABILITY), 1.0)
    return float(combined)


def predict_activity_with_model(
    model_bundle,
    user_id: int,
    element_id: int,
    entity_id: int,
    action_id: int,
    logged_at
):
    """
    Realiza una predicción individual para una actividad concreta.

    Como se trabaja con un modelo por usuario:
    - busca el modelo correspondiente al user_id
    - construye la fila de variables
    - alinea las columnas con las usadas en entrenamiento
    - calcula la predicción del IsolationForest
    - ajusta la anomaly_probability con la rareza histórica del usuario

    Convención de salida:
    - prediction = 1  -> actividad anómala
    - prediction = 0  -> actividad normal
    - probability     -> score final de anomalía entre 0 y 1, nunca igual a 0

    Esegue una previsione singola per una specifica attività.
    Siccome si lavora con un modello per utente:
    - cerca il modello corrispondente al user_id
    - costruisce la riga di variabili
    - allinea le colonne con quelle usate in addestramento
    - calcola la previsione dell'IsolationForest
    - aggiusta la anomaly_probability con la rarità storica dell'utente
    """
    if model_bundle is None:
        raise ValueError("Il bundle dei modelli è vuoto.")

    user_id = int(user_id)

    if user_id not in model_bundle:
        raise ValueError(f"Nessun modello trovato per user_id={user_id}")

    user_model_data = model_bundle[user_id]
    model = user_model_data["model"]
    feature_columns = user_model_data["feature_columns"]
    frequency_stats = user_model_data.get("frequency_stats", {})

    row_features = build_single_row_features(
        element_id=element_id,
        entity_id=entity_id,
        action_id=action_id,
        logged_at=logged_at
    )

    row_aligned = align_features_to_training(row_features, feature_columns)

    raw_prediction = model.predict(row_aligned)[0]
    raw_score = float(model.decision_function(row_aligned)[0])

    # En IsolationForest:
    #  1  -> normal
    # -1  -> anómalo
    prediction = 1 if raw_prediction == -1 else 0

    # Cuanto más negativo sea raw_score, más anómalo es el caso.
    anomaly_score = max(0.0, -raw_score)

    # Normalización estable al rango [0, 1].
    model_probability = anomaly_score / (1.0 + anomaly_score)

    # Ajuste por rareza histórica del comportamiento del usuario.
    behavior_rarity = compute_behavior_rarity_score(
        frequency_stats=frequency_stats,
        element_id=element_id,
        entity_id=entity_id,
        action_id=action_id
    )

    probability = combine_model_and_behavior_scores(
        model_probability=model_probability,
        behavior_rarity=behavior_rarity
    )

    # Nunca permitimos probabilidad igual a 0.
    probability = max(probability, MIN_ANOMALY_PROBABILITY)

    return prediction, float(probability)


if __name__ == "__main__":
    print("Caricamento dei dati di activity_log...")
    df = load_activity_data()
    print("Shape originale:", df.shape)

    print("Preparazione delle feature di activity...")
    df_prepared = prepare_activity_features(df)
    print("Shape preparata:", df_prepared.shape)

    print("Addestramento dei modelli per utente...")
    model_bundle = train_activity_model(df_prepared, contamination=0.05)

    if model_bundle is not None:
        save_model(model_bundle)