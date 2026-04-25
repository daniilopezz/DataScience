### Español | Italiano
> Este documento esta disponible en **Español** e **Italiano**. / Questo documento è disponibile in **Spagnolo** e **Italiano**.

---

### Español

# COMO FUNCIONA LA APLICACION — KeyOver1

## 1. ¿Que Hace la Aplicacion?

KeyOver1 es un sistema de deteccion de anomalias de prueba de concepto (PoC) para un gestor de contrasenas o una herramienta de auditoria interna. Supervisa el comportamiento del usuario en tiempo real y utiliza machine learning para decidir si cada accion individual o toda la sesion parece sospechosa, sin depender de reglas fijas `if/else` como mecanismo principal.

El sistema:
- Permite a los usuarios iniciar sesion desde la terminal
- Les permite seleccionar elementos y ejecutar acciones sobre ellos
- Registra cada intento de login y cada accion en PostgreSQL
- Puntua cada accion individualmente (modelo de actividad)
- Puntua la sesion en evolucion despues de cada accion (modelo de sesion)
- Fuerza el cierre automatico de sesion cuando detecta una anomalia

---

## 2. Flujo Completo

```
El usuario inicia main.py
      │
      ▼
Menu principal (1=Login / 0=Salir)
      │
      ▼
Login: email + contrasena (maximo 3 intentos)
      │
      ├── Contrasena incorrecta → guardar log de login fallido, reintentar
      ├── Usuario inactivo      → denegar
      ├── Login anomalo         → denegar (fuera de horario / dia inusual para ese usuario)
      │
      └── Login OK ─────────────────────────────────────────────────────┐
                                                                        │
                                                                        ▼
                                                Menu de elementos (FVG / AMCO / VETTING / ...)
                                                         │
                                                         ▼
                                                Menu de acciones (Visualize / Create / Edit / ...)
                                                         │
                                                         ▼
                                                [ML-ACTIVITY] Puntuar esta accion
                                                         │
                                                         ▼
                                                [ML-SESSION] Puntuar la sesion en curso
                                                         │
                                                         ▼
                                                [COST THRESHOLD] cumulative_cost >= user_threshold?
                                                         │
                                               ┌─────────┴──────────┐
                                               │                    │
                                          Normal            Anomalo / Umbral superado
                                               │                    │
                                        continuar bucle      forzar logout
                                               │                    │
                                  (volver al menu de elementos) (la sesion termina)
```

---

## 3. Conexion con la Base de Datos

**Motor:** PostgreSQL  
**Archivo de configuracion:** `config/db.py`

```py
DB_CONFIG = {
    "host":   "localhost",
    "port":   5432,
    "dbname": "Audit",
    "user":   "dani",
    "password": ""
}
```

Todos los modulos importan `get_connection()` desde `config/db.py`. No hay credenciales duplicadas en ninguna otra parte.

---

## 4. Descripcion Archivo por Archivo

| Archivo | Responsabilidad |
|------|---------------|
| `config/db.py` | Constantes de configuracion de BD + factoría `get_connection()` |
| `data_generation/generate_data.py` | Genera e inserta logins y actividades sinteticas |
| `MachineLearning/train_models.py` | Pipeline completo de entrenamiento; exporta `models/combined_model.pkl` |
| `models/combined_model.pkl` | *(generado)* Archivo unico que contiene ambos modelos ML |
| `security/anomaly_guard.py` | Carga el modelo combinado; `evaluate_login`, `evaluate_activity`, `evaluate_session` |
| `app/session.py` | Bucle interactivo de sesion, procesamiento de acciones, llamadas de evaluacion ML |
| `main.py` | Punto de entrada; flujo de login, carga de perfil, lanzamiento de sesion |
| `HOW_THE_APPLICATION_WORKS.md` | Este archivo |
| `requirements.txt` | Dependencias de Python |

---

## 5. Como Se Entrena el Modelo de Actividad

**Script:** `MachineLearning/train_models.py`  
**Algoritmo:** `sklearn.ensemble.IsolationForest` — un modelo por usuario.

### Ingenieria de caracteristicas
Cada fila representa una accion dentro de una sesion:

| Caracteristica | Descripcion |
|---------|-------------|
| `element_id` (one-hot) | Que elemento fue accedido |
| `entity_id` (one-hot) | Que entidad estuvo implicada |
| `action_id` (one-hot) | Que accion se realizo |
| `hour`, `minute` | Hora del dia |
| `day_of_week` (one-hot) | Dia de la semana |
| `log_sec_prev` | log(1 + segundos desde la accion anterior) |
| `log_sec_start` | log(1 + segundos desde el inicio de la sesion) |

### Formula de probabilidad de anomalia
```
probability = 0.75 × model_score + 0.25 × combo_rarity
```
- `model_score` = puntuacion bruta de IsolationForest normalizada por el percentil 95
- `combo_rarity` = 1 − frecuencia historica de la combinacion (`element`, `entity`, `action`)

Esto combina la confianza del modelo con una senal de rareza basada en datos.

### Contamination
`contamination=0.05` — el modelo espera aproximadamente un 5% de anomalias en los datos de entrenamiento.

---

## 6. Como Se Entrena el Modelo de Sesion

**Script:** `MachineLearning/train_models.py`  
**Algoritmo:** `sklearn.ensemble.IsolationForest` — un modelo por usuario.

### Idea clave: prefijos de sesion
En lugar de entrenar solo con sesiones completas, el modelo se entrena con **todos los estados intermedios** de cada sesion (prefijo 1, prefijo 1-2, prefijo 1-2-3, ...). Esto permite que el modelo evalúe una sesion parcial durante la ejecucion usando la misma distribucion de datos con la que fue entrenado.

### Caracteristicas de sesion (15 en total)

| Caracteristica | Descripcion |
|---------|-------------|
| `action_count` | Total de acciones hasta el momento |
| `distinct_elements` | Elementos unicos accedidos |
| `distinct_actions` | Tipos de accion unicos utilizados |
| `session_duration_min` | Tiempo transcurrido desde el inicio de la sesion (minutos) |
| `start_hour` | Hora en la que comenzo la sesion (float) |
| `day_of_week` | Dia de la semana del inicio de sesion |
| `actions_per_minute` | Ritmo de la sesion |
| `avg_seconds_between_actions` | Tiempo medio entre acciones consecutivas |
| `min_seconds_between_actions` | Menor intervalo (indicador de bot) |
| `max_seconds_between_actions` | Mayor intervalo |
| `cumulative_cost` | Suma de las puntuaciones de anomalia de actividad |
| `avg_cost` | Promedio de la puntuacion de anomalia de actividad |
| `max_cost` | Pico de la puntuacion de anomalia de actividad |
| `repeated_action_ratio` | Frecuencia con la que aparece la accion mas repetida |
| `repeated_element_ratio` | Frecuencia con la que aparece el elemento mas repetido |

### ¿Por que modelos por usuario?
Por ejemplo, Matteo normalmente hace unas 3 acciones por sesion; Diego hace unas 10. Un unico umbral global generaria falsos positivos para Diego y pasaria por alto anomalias de Matteo. Los modelos por usuario aprenden de forma independiente el patron normal de cada persona.

### Contamination
`contamination=0.03` — mas conservador, ya que las sesiones ya estan filtradas por el modelo de actividad.

---

## 7. Como Ejecutar la Aplicacion

### Paso 0 — Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 1 — Generar datos sinteticos
```bash
cd KeyOver1
python data_generation/generate_data.py
```
Inserta aproximadamente 100.000 registros de login y 100.000 registros de actividad en la base de datos `Audit`.

### Paso 2 — Entrenar los modelos
```bash
python MachineLearning/train_models.py
```
Salida: `models/combined_model.pkl` (aproximadamente 10–20 MB segun el tamano de los datos de entrenamiento).

### Paso 3 — Ejecutar la aplicacion
```bash
python main.py
```

---

## 8. Como Se Registra la Informacion en la Base de Datos

### `login_log`
Una fila por intento de login (correcto o fallido).  
Columnas: `user_id`, `result` (bool), `attempt` (int), `logged_at`, `logout_at` (actualizado al cerrar sesion).

### `activity_log`
Una fila por accion realizada dentro de una sesion.  
Columnas: `user_id`, `element_id`, `entity_id`, `action_id`, `logged_at`.

### `ml_prediction_log`
Una fila por accion, despues de la evaluacion ML.  
Columnas clave:
- `activity_log_id`, `login_log_id` — enlaces a la accion y a la sesion
- `prediction` — `true` si es anomala (el modelo de actividad o el de sesion la marcaron)
- `anomaly_probability` — puntuacion de anomalia de actividad (0–1)
- `session_cumulative_cost` — suma de las puntuaciones de anomalia acumuladas en la sesion
- `threshold_exceeded` — reservado para futuras reglas basadas en umbral

---

## 9. Como Se Toman las Decisiones de Deteccion de Anomalias

### Anomalia en el login (basada en perfil)
Antes de abrir una sesion, el sistema compara la marca temporal del login con el patron historico del usuario:
- Si la hora de login cae fuera del rango `[p10 − 0.5h, p90 + 0.5h]` → se marca
- Si el dia de la semana no esta entre los dias habituales del usuario → se marca

Si se cumple cualquiera de las dos condiciones, el login se **deniega**.

### Anomalia en la actividad (basada en ML)
Despues de cada accion:
1. Se construye el vector de caracteristicas para esa accion
2. Se ejecuta `IsolationForest.predict()` — si la salida es `-1`, entonces `prediction=1` (anomalia)
3. Se calcula `anomaly_probability` = puntuacion del modelo combinada con la rareza de la combinacion
4. Si `prediction == 1` → **logout automatico**

### Anomalia de sesion (basada en ML)
Al mismo tiempo que la comprobacion de actividad:
1. Se recalculan las 15 caracteristicas de sesion tras anadir la nueva accion
2. Se ejecuta `IsolationForest.predict()` de sesion sobre el estado actual de la sesion
3. Si `prediction == 1` → **logout automatico**

### Umbral de coste de sesion (estadistico, por usuario)
Despues de ambas comprobaciones ML:
1. Se compara `cumulative_cost` (suma de todas las puntuaciones de accion hasta el momento) con el umbral personal del usuario
2. El umbral = percentil 95 del maximo coste acumulado historico por sesion para ese usuario
3. Si `cumulative_cost >= threshold` → **logout automatico**

La terminal muestra la suma acumulada despues de cada accion:
```
costo sessione: 0.2000 + 0.4000 + 0.2000 = 0.8000  [soglia: 1.2340]
```

La `prediction` final guardada en `ml_prediction_log` es `True` si **cualquiera** de las tres comprobaciones (modelo de actividad, modelo de sesion, umbral de coste) marca la accion.

---

## 10. Umbral de Coste de Sesion (Maximo por Usuario)

### ¿Que Es el Coste de Sesion?

Cada vez que un usuario realiza una accion, el **IsolationForest de actividad** (modelo ML) asigna una puntuacion de anomalia a esa accion: un valor flotante entre aproximadamente 0 y 1, donde un valor mas alto significa mayor sospecha. Esta puntuacion se guarda en `ml_prediction_log.anomaly_probability` y se denomina internamente `op_cost`.

El **coste de sesion** es la suma acumulada de todos los costes de accion dentro de una sesion:

```
costo sessione = 0.2000 + 0.4000 + 0.2000 = 0.8000  [soglia: 1.2340]
```

Esta linea se imprime en la terminal despues de cada accion y va creciendo con cada nuevo paso. Le da al operador una vista intuitiva e interpretable de cuanta "sospecha" se ha acumulado durante la sesion.

---

### ¿Por que un Umbral por Usuario?

Un unico umbral global seria injusto e inexacto:

- **Diego** suele realizar unas 3 acciones por sesion → su coste acumulado normal es bajo
- **Matteo** suele realizar unas 8 acciones por sesion → su coste acumulado normal es naturalmente mas alto

Un umbral calibrado para Diego nunca se activaria en las sesiones anomalas de Matteo, y viceversa. Cada usuario necesita su propio limite basado en su patron personal de uso.

---

### Como Se Calcula el Umbral (Estadistico, No ML)

Los costes de las acciones individuales provienen del modelo ML (IsolationForest). Sin embargo, **el propio umbral** se deriva estadisticamente, usando el mismo enfoque que el perfil de login (basado en percentiles).

**Formula:**
```
user_threshold = PERCENTILE_CONT(0.95) of MAX(session_cumulative_cost) per completed session
```

**Paso a paso:**
1. Consultar `ml_prediction_log` unido con `login_log` (donde `logout_at IS NOT NULL` = sesiones completadas)
2. Para cada sesion, tomar el **coste acumulado maximo** alcanzado (= coste en la ultima accion)
3. Agrupar por `user_id` y calcular el **percentil 95** de esos costes maximos
4. El resultado es el umbral para ese usuario: el 95% de sus sesiones historicas quedaron por debajo de ese valor

**¿Por que el percentil 95?**  
Es la convencion estadistica estandar para el "limite superior normal". Absorbe la variacion natural del dia a dia sin ser tan estricto como para que salten sesiones rutinarias, ni tan laxo como para dejar pasar anomalias reales.

**Fallback:**  
Si un usuario no tiene historial de sesiones completadas, el umbral se fija en `∞` (sin limite). En ese caso, el modelo ML de sesion sigue aportando proteccion.

---

### ¿Cuando Fuerza un Logout?

Despues de cada accion, el sistema comprueba:

```
if cumulative_cost >= user_threshold → force logout
```

Esta comprobacion se ejecuta **despues** de las comprobaciones ML de actividad y sesion. El mensaje de logout es:

```
⚠  Soglia costo sessione superata (1.3500 >= 1.2340) → logout automatico.
```

El evento se registra en `ml_prediction_log` con:
- `session_threshold` = el umbral calculado para el usuario
- `threshold_exceeded` = `True`
- `prediction` = `True` (marcado como anomalo)

---

### ¿Por que el Umbral es Estadistico y No Otro Modelo ML?

| Enfoque | Ventajas | Inconvenientes |
|---|---|---|
| **Percentil estadistico** | Transparente, interpretable ("el limite es 1.23"), facil de auditar | Asume que los datos historicos son representativos |
| **Nuevo modelo ML para el umbral** | Puede aprender patrones complejos no lineales | Caja negra: dificil explicar "por que este limite?", redundante con el IsolationForest de sesion |
| **IsolationForest de sesion (existente)** | Ya aprende patrones normales por usuario relacionados con el coste | Implicito: no puede mostrar un limite numerico al operador |

El IsolationForest de sesion **ya impone de forma implicita un limite de coste** porque `cumulative_cost` es una de sus 15 caracteristicas. El umbral estadistico anade una capa **explicita, visible y auditable** que satisface el requisito operativo de mostrar un limite, sin duplicar la complejidad del ML.

---

## 11. Usuarios

| user_id | Nombre | Horario habitual | Volumen de sesion | Elementos preferidos |
|---------|------|--------------|----------------|-------------------|
| 1 | Matteo Nicolosi | 09:00–13:00 | ~8 acciones | FVG, AMCO |
| 2 | Diego Scardino | 09:00–17:00 | ~3 acciones | VETTING |
| 3 | Emilio Sardo | 10:00–18:00 | ~5 acciones | FVG, RHODENSE, PAPARDO, PULEJO |

Los modelos por usuario son esenciales porque cada usuario tiene un patron de uso realmente diferente.

---
### Italiano

# COME FUNZIONA L'APPLICAZIONE — KeyOver1

## 1. Cosa Fa l'Applicazione?

KeyOver1 e un sistema di rilevamento delle anomalie proof-of-concept (PoC) per un gestore di password o uno strumento di audit interno. Monitora il comportamento dell'utente in tempo reale e utilizza il machine learning per decidere se ogni singola azione o l'intera sessione appare sospetta, senza basarsi principalmente su regole fisse `if/else`.

Il sistema:
- Consente agli utenti di effettuare il login dal terminale
- Permette di selezionare elementi ed eseguire azioni su di essi
- Registra ogni tentativo di login e ogni azione in PostgreSQL
- Assegna un punteggio a ogni azione individualmente (modello di attivita)
- Assegna un punteggio alla sessione in evoluzione dopo ogni azione (modello di sessione)
- Forza il logout automatico quando viene rilevata un'anomalia

---

## 2. Flusso Completo

```
L'utente avvia main.py
      │
      ▼
Menu principale (1=Login / 0=Esci)
      │
      ▼
Login: email + password (massimo 3 tentativi)
      │
      ├── Password errata    → salva il log di login fallito, riprova
      ├── Utente inattivo    → nega l'accesso
      ├── Login anomalo      → nega l'accesso (fuori orario / giorno insolito per quell'utente)
      │
      └── Login OK ────────────────────────────────────────────────────┐
                                                                       │
                                                                       ▼
                                               Menu elementi (FVG / AMCO / VETTING / ...)
                                                         │
                                                         ▼
                                              Menu azioni (Visualize / Create / Edit / ...)
                                                         │
                                                         ▼
                                               [ML-ACTIVITY] Valuta questa azione
                                                         │
                                                         ▼
                                                [ML-SESSION] Valuta la sessione in corso
                                                         │
                                                         ▼
                                              [COST THRESHOLD] cumulative_cost >= user_threshold?
                                                         │
                                               ┌─────────┴──────────┐
                                               │                    │
                                            Normale         Anomalo / Soglia superata
                                               │                    │
                                        continua il ciclo     forza il logout
                                               │                    │
                                  (torna al menu elementi)   (la sessione termina)
```

---

## 3. Connessione al Database

**Motore:** PostgreSQL  
**File di configurazione:** `config/db.py`

```python
DB_CONFIG = {
    "host":   "localhost",
    "port":   5432,
    "dbname": "Audit",
    "user":   "dani",
    "password": ""
}
```

Tutti i moduli importano `get_connection()` da `config/db.py`. Le credenziali non sono duplicate altrove.

---

## 4. Descrizione File per File

| File | Responsabilita |
|------|---------------|
| `config/db.py` | Costanti di configurazione DB + factory `get_connection()` |
| `utils/hash.py` | Hash delle password con SHA-256 |
| `data_generation/generate_data.py` | Genera e inserisce login e attivita sintetiche |
| `MachineLearning/train_models.py` | Pipeline completa di training; esporta `models/combined_model.pkl` |
| `models/combined_model.pkl` | *(generato)* File unico che contiene entrambi i modelli ML |
| `security/anomaly_guard.py` | Carica il modello combinato; `evaluate_login`, `evaluate_activity`, `evaluate_session` |
| `app/session.py` | Ciclo interattivo della sessione, elaborazione delle azioni, chiamate di valutazione ML |
| `main.py` | Punto di ingresso; flusso di login, caricamento del profilo, avvio della sessione |
| `HOW_THE_APPLICATION_WORKS.md` | Questo file |
| `requirements.txt` | Dipendenze Python |

---

## 5. Come Viene Addestrato il Modello di Attivita

**Script:** `MachineLearning/train_models.py`  
**Algoritmo:** `sklearn.ensemble.IsolationForest` — un modello per utente.

### Feature engineering
Ogni riga rappresenta un'azione all'interno di una sessione:

| Caratteristica | Descrizione |
|---------|-------------|
| `element_id` (one-hot) | Quale elemento e stato aperto |
| `entity_id` (one-hot) | Quale entita e stata coinvolta |
| `action_id` (one-hot) | Quale azione e stata eseguita |
| `hour`, `minute` | Ora del giorno |
| `day_of_week` (one-hot) | Giorno della settimana |
| `log_sec_prev` | log(1 + secondi dall'azione precedente) |
| `log_sec_start` | log(1 + secondi dall'inizio della sessione) |

### Formula della probabilita di anomalia
```
probability = 0.75 × model_score + 0.25 × combo_rarity
```
- `model_score` = punteggio grezzo di IsolationForest normalizzato con il 95o percentile
- `combo_rarity` = 1 − frequenza storica della combinazione (`element`, `entity`, `action`)

Questo unisce la confidenza del modello con un segnale di rarita guidato dai dati.

### Contamination
`contamination=0.05` — il modello si aspetta circa il 5% di anomalie nei dati di training.

---

## 6. Come Viene Addestrato il Modello di Sessione

**Script:** `MachineLearning/train_models.py`  
**Algoritmo:** `sklearn.ensemble.IsolationForest` — un modello per utente.

### Idea chiave: prefissi di sessione
Invece di addestrarsi solo su sessioni complete, il modello viene addestrato su **tutti gli stati intermedi** di ogni sessione (prefisso 1, prefisso 1-2, prefisso 1-2-3, ...). Questo permette al modello di valutare una sessione parziale durante l'esecuzione usando la stessa distribuzione di dati su cui e stato addestrato.

### Caratteristiche della sessione (15 totali)

| Caratteristica | Descrizione |
|---------|-------------|
| `action_count` | Totale delle azioni finora |
| `distinct_elements` | Elementi unici consultati |
| `distinct_actions` | Tipi di azione unici utilizzati |
| `session_duration_min` | Tempo trascorso dall'inizio della sessione (minuti) |
| `start_hour` | Ora di inizio della sessione (float) |
| `day_of_week` | Giorno della settimana di inizio sessione |
| `actions_per_minute` | Ritmo della sessione |
| `avg_seconds_between_actions` | Intervallo medio tra azioni consecutive |
| `min_seconds_between_actions` | Intervallo minimo (indicatore di bot) |
| `max_seconds_between_actions` | Intervallo massimo |
| `cumulative_cost` | Somma dei punteggi di anomalia delle attivita |
| `avg_cost` | Media del punteggio di anomalia delle attivita |
| `max_cost` | Picco del punteggio di anomalia delle attivita |
| `repeated_action_ratio` | Frequenza con cui compare l'azione piu ripetuta |
| `repeated_element_ratio` | Frequenza con cui compare l'elemento piu ripetuto |

### Perche modelli per utente?
Per esempio, Matteo normalmente esegue circa 3 azioni per sessione; Diego circa 10. Un'unica soglia globale produrrebbe falsi positivi per Diego e non rileverebbe anomalie per Matteo. I modelli per utente imparano separatamente il comportamento normale di ogni persona.

### Contamination
`contamination=0.03` — piu conservativo, dato che le sessioni sono gia filtrate dal modello di attivita.

---

## 7. Come Eseguire l'Applicazione

### Passo 0 — Installare le dipendenze
```bash
pip install -r requirements.txt
```

### Passo 1 — Generare dati sintetici
```bash
cd KeyOver1
python data_generation/generate_data.py
```
Inserisce circa 100.000 record di login e 100.000 record di attivita nel database `Audit`.

### Passo 2 — Addestrare i modelli
```bash
python MachineLearning/train_models.py
```
Output: `models/combined_model.pkl` (circa 10–20 MB a seconda della dimensione dei dati di training).

### Passo 3 — Avviare l'applicazione
```bash
python main.py
```

---

## 8. Come Vengono Registrate le Informazioni nel Database

### `login_log`
Una riga per ogni tentativo di login (riuscito o fallito).  
Colonne: `user_id`, `result` (bool), `attempt` (int), `logged_at`, `logout_at` (aggiornato al logout).

### `activity_log`
Una riga per ogni azione eseguita all'interno di una sessione.  
Colonne: `user_id`, `element_id`, `entity_id`, `action_id`, `logged_at`.

### `ml_prediction_log`
Una riga per ogni azione, dopo la valutazione ML.  
Colonne chiave:
- `activity_log_id`, `login_log_id` — collegamenti all'azione e alla sessione
- `prediction` — `true` se anomala (il modello di attivita oppure quello di sessione l'ha segnalata)
- `anomaly_probability` — punteggio di anomalia dell'attivita (0–1)
- `session_cumulative_cost` — somma dei punteggi di anomalia accumulati nella sessione
- `threshold_exceeded` — riservato a future regole basate su soglia

---

## 9. Come Vengono Prese le Decisioni di Rilevamento delle Anomalie

### Anomalia di login (basata sul profilo)
Prima di aprire una sessione, il sistema confronta il timestamp del login con il modello storico dell'utente:
- Se l'ora di login cade fuori dall'intervallo `[p10 − 0.5h, p90 + 0.5h]` → viene segnalata
- Se il giorno della settimana non rientra tra i giorni abituali dell'utente → viene segnalato

Se una delle due condizioni e vera, il login viene **negato**.

### Anomalia dell'attivita (basata su ML)
Dopo ogni azione:
1. Viene costruito il vettore di caratteristiche per quell'azione
2. Viene eseguito `IsolationForest.predict()` — se l'output e `-1`, allora `prediction=1` (anomalia)
3. Viene calcolata `anomaly_probability` = punteggio del modello combinato con la rarita della combinazione
4. Se `prediction == 1` → **logout automatico**

### Anomalia di sessione (basata su ML)
Contemporaneamente al controllo dell'attivita:
1. Vengono ricalcolate tutte e 15 le caratteristiche della sessione dopo l'aggiunta della nuova azione
2. Viene eseguito `IsolationForest.predict()` di sessione sullo stato corrente della sessione
3. Se `prediction == 1` → **logout automatico**

### Soglia del costo di sessione (statistica, per utente)
Dopo entrambi i controlli ML:
1. Si confronta `cumulative_cost` (somma di tutti i punteggi delle azioni fino a quel momento) con la soglia personale dell'utente
2. La soglia = 95o percentile del massimo costo cumulativo storico per sessione di quell'utente
3. Se `cumulative_cost >= threshold` → **logout automatico**

Il terminale mostra la somma crescente dopo ogni azione:
```
costo sessione: 0.2000 + 0.4000 + 0.2000 = 0.8000  [soglia: 1.2340]
```

La `prediction` finale memorizzata in `ml_prediction_log` e `True` se **uno qualsiasi** dei tre controlli (modello di attivita, modello di sessione, soglia di costo) segnala l'azione.

---

## 10. Soglia del Costo di Sessione (Massimo per Utente)

### Che Cos'e il Costo di Sessione?

Ogni volta che un utente esegue un'azione, l'**IsolationForest di attivita** (modello ML) assegna un punteggio di anomalia a quell'azione: un valore float compreso circa tra 0 e 1, dove un valore piu alto significa maggiore sospetto. Questo punteggio viene memorizzato in `ml_prediction_log.anomaly_probability` ed e indicato internamente come `op_cost`.

Il **costo di sessione** e la somma cumulativa di tutti i costi delle azioni all'interno di una sessione:

```
costo sessione = 0.2000 + 0.4000 + 0.2000 = 0.8000  [soglia: 1.2340]
```

Questa riga viene stampata nel terminale dopo ogni azione e cresce a ogni nuovo passaggio. Offre all'operatore una vista intuitiva e interpretabile di quanta "sospettosita" si sia accumulata durante la sessione.

---

### Perche una Soglia per Utente?

Un'unica soglia globale sarebbe ingiusta e imprecisa:

- **Diego** esegue tipicamente circa 3 azioni per sessione → il suo costo cumulativo normale e basso
- **Matteo** esegue tipicamente circa 8 azioni per sessione → il suo costo cumulativo normale e naturalmente piu alto

Una soglia calibrata per Diego non scatterebbe mai sulle sessioni anomale di Matteo, e viceversa. Ogni utente ha bisogno del proprio limite in base al proprio schema di utilizzo personale.

---

### Come Viene Calcolata la Soglia (Statistica, Non ML)

I costi delle singole azioni provengono dal modello ML (IsolationForest). Tuttavia, **la soglia stessa** viene derivata statisticamente, usando lo stesso approccio del profilo di login (basato sui percentili).

**Formula:**
```
user_threshold = PERCENTILE_CONT(0.95) of MAX(session_cumulative_cost) per completed session
```

**Passo dopo passo:**
1. Interrogare `ml_prediction_log` unito a `login_log` (dove `logout_at IS NOT NULL` = sessioni completate)
2. Per ogni sessione, prendere il **massimo costo cumulativo** raggiunto (= costo dell'ultima azione)
3. Raggruppare per `user_id` e calcolare il **95o percentile** di quei costi massimi
4. Il risultato e la soglia per quell'utente: il 95% delle sue sessioni storiche e rimasto sotto questo valore

**Perche il 95o percentile?**  
E la convenzione statistica standard per il "limite superiore normale". Assorbe la variazione naturale del giorno per giorno senza essere cosi stretta da far scattare le sessioni ordinarie, ne cosi larga da lasciar passare anomalie reali.

**Fallback:**  
Se un utente non ha alcuna cronologia di sessioni completate, la soglia viene impostata a `∞` (nessun limite). In quel caso, il modello ML di sessione continua comunque a fornire protezione.

---

### Quando Fa Scattare il Logout?

Dopo ogni azione, il sistema controlla:

```
if cumulative_cost >= user_threshold → force logout
```

Questo controllo viene eseguito **dopo** i controlli ML di attivita e sessione. Il messaggio di logout e:

```
⚠  Soglia costo sessione superata (1.3500 >= 1.2340) → logout automatico.
```

L'evento viene registrato in `ml_prediction_log` con:
- `session_threshold` = la soglia calcolata per l'utente
- `threshold_exceeded` = `True`
- `prediction` = `True` (segnalato come anomalo)

---

### Perche la Soglia e Statistica e Non un Altro Modello ML?

| Approccio | Vantaggi | Svantaggi |
|---|---|---|
| **Percentile statistico** | Trasparente, interpretabile ("il limite e 1.23"), facile da auditare | Presuppone che i dati storici siano rappresentativi |
| **Nuovo modello ML per la soglia** | Puo apprendere pattern complessi non lineari | Scatola nera: difficile spiegare "perche questo limite?", ridondante con l'IsolationForest di sessione |
| **IsolationForest di sessione (esistente)** | Impara gia pattern normali per utente legati al costo | Implicito: non puo mostrare un limite numerico all'operatore |

L'IsolationForest di sessione **impone gia implicitamente un limite di costo** perche `cumulative_cost` e una delle sue 15 caratteristiche. La soglia statistica aggiunge un livello **esplicito, visibile e auditabile** che soddisfa il requisito operativo di mostrare un limite, senza duplicare la complessita del ML.

---

## 11. Utenti

| user_id | Nome | Orario tipico | Volume di sessione | Elementi preferiti |
|---------|------|--------------|----------------|-------------------|
| 1 | Matteo Nicolosi | 09:00–13:00 | ~8 azioni | FVG, AMCO |
| 2 | Diego Scardino | 09:00–17:00 | ~3 azioni | VETTING | 
| 3 | Emilio Sardo | 10:00–18:00 | ~5 azioni | FVG, RHODENSE, PAPARDO, PULEJO |

I modelli per utente sono essenziali perche ogni utente ha un modello di utilizzo realmente diverso.


prueba automatizada?