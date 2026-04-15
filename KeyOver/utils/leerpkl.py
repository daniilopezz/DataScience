import joblib
from pathlib import Path

"""
Este archivo sirve para cargar y revisar el contenido del fichero .pkl
que almacena los modelos de detección de anomalías entrenados.
En la versión actual del proyecto no se guarda un único modelo,
sino un conjunto de modelos, uno por cada usuario.
El objetivo de este script es verificar:
- que el archivo se carga correctamente
- qué usuarios tienen modelo entrenado
- qué tipo de modelo se ha guardado
- cuántas variables utiliza cada modelo
- qué columnas se usaron en el entrenamiento

Questo file serve per caricare e controllare il contenuto del file .pkl
che memorizza i modelli di rilevamento anomalie addestrati.
Nella versione attuale del progetto non viene salvato un unico modello,
ma un insieme di modelli, uno per ciascun utente.
L'obiettivo di questo script è verificare:
- che il file venga caricato correttamente
- quali utenti hanno un modello addestrato
- che tipo di modello è stato salvato
- quante variabili utilizza ciascun modello
- quali colonne sono state usate durante l'addestramento
"""

MODEL_PATH = Path("models/activity_model.pkl")


def load_model_bundle(path: str | Path = MODEL_PATH):
    """
    Carga desde disco el bundle de modelos entrenados.
    Parámetros:
    - path: ruta del archivo .pkl

    Devuelve:
    - diccionario con un modelo por usuario

    Carica da disco il bundle dei modelli addestrati.
    Parametri:
    - path: percorso del file .pkl

    Restituisce:
    - dizionario con un modello per utente
    """
    return joblib.load(path)


def show_model_bundle_info(model_bundle: dict):
    """
    Muestra información general sobre el bundle de modelos cargado.
    Para cada usuario enseña:
    - identificador del usuario
    - tipo de modelo
    - número de árboles estimadores
    - número de variables de entrada
    - columnas utilizadas en el entrenamiento

    Mostra informazioni generali sul bundle di modelli caricato.
    Per ogni utente mostra:
    - identificativo dell'utente
    - tipo di modello
    - numero di alberi stimatori
    - numero di variabili di input
    - colonne utilizzate durante l'addestramento
    """
    if not isinstance(model_bundle, dict):
        print("Il contenuto caricato non ha il formato atteso.")
        return

    if not model_bundle:
        print("Il bundle dei modelli è vuoto.")
        return

    print("Modelli caricati correttamente.")
    print(f"Numero di utenti con modello: {len(model_bundle)}")

    for user_id, model_data in sorted(model_bundle.items()):
        print("\n" + "=" * 50)
        print(f"Utente: {user_id}")

        model = model_data.get("model")
        feature_columns = model_data.get("feature_columns", [])

        print(f"Tipo modello: {type(model)}")

        if hasattr(model, "n_estimators"):
            print(f"Numero di alberi: {model.n_estimators}")

        if hasattr(model, "n_features_in_"):
            print(f"Numero di variabili: {model.n_features_in_}")
        else:
            print(f"Numero di variabili: {len(feature_columns)}")

        print("Colonne utilizzate nel training:")
        for column in feature_columns:
            print(f"- {column}")


if __name__ == "__main__":
    try:
        model_bundle = load_model_bundle(MODEL_PATH)
        show_model_bundle_info(model_bundle)
    except FileNotFoundError:
        print(f"File non trovato: {MODEL_PATH}")
    except Exception as e:
        print(f"Errore durante il caricamento del file .pkl: {e}")