# Cálculo de Actividades y Costo de Sesión

Este documento explica únicamente **cómo se calculan los números** detrás de cada tipo de actividad y del costo de sesión por usuario. No describe el flujo general de la aplicación.

---

## 1. Tipos de actividad y sus IDs

Los tipos de actividad son etiquetas que se asignan a un `action_id` numérico. El mapeo está en **`app/session.py` — líneas 20–27**:

```
"1" → (1000000, "Visualize")
"2" → (1000001, "Create")
"3" → (1000002, "Edit")
"4" → (1000003, "Delete")
"5" → (1000004, "Copy")
"6" → (1000005, "Share")
```

El tipo en sí no cambia directamente el cálculo del coste. Lo que importa es qué tan frecuente es ese tipo de acción dentro de la sesión actual y si el comportamiento encaja con el historial del usuario.

---

## 2. Tres modelos de ML (resumen)

El sistema entrena tres modelos independientes por usuario:

| Modelo | Archivo | Qué aprende |
|---|---|---|
| **activity** | `train_models.py` — `train_activity_models()` | Si una acción individual es anómala (timing, elemento, hora) |
| **action_frequency** | `train_models.py` — `train_action_frequency_models()` | Si la distribución de tipos de acción en la sesión es inusual |
| **session** | `train_models.py` — `train_session_models()` | Si el patrón global de la sesión (ritmo, coste acumulado, diversidad) es anómalo |

Todos usan `IsolationForest` de scikit-learn. Los tres se guardan juntos en `models/combined_model.pkl`.

---

## 3. Costo de una acción individual (`op_cost`)

Cada vez que el usuario ejecuta una acción, el sistema calcula un **`op_cost`** que representa qué tan costosa es esa acción para el presupuesto de la sesión. El proceso tiene tres pasos.

### Paso 1 — Coste base: modelo de frecuencia de acciones

**`app/session.py` — línea 230**
```python
op_cost = evaluate_action_cost(combined_model, user_id, new_counts)
```

Esto llama a `predict_action_frequency_cost()` en **`train_models.py` — líneas 374–394**.

#### ¿Qué mide este modelo?

Durante el entrenamiento (**`train_models.py` — `train_action_frequency_models()` — líneas 343–371**), por cada usuario se construye un dataset donde cada fila es una sesión histórica y las features son los conteos de cada tipo de acción en esa sesión:

| Feature | Descripción |
|---|---|
| `action_1000000` | Cuántas veces se hizo "Visualize" en la sesión |
| `action_1000001` | Cuántas veces se hizo "Create" |
| `action_1000002` | Cuántas veces se hizo "Edit" |
| `action_1000003` | Cuántas veces se hizo "Delete" |
| `action_1000004` | Cuántas veces se hizo "Copy" |
| `action_1000005` | Cuántas veces se hizo "Share" |

Se entrena un `IsolationForest(n_estimators=200, contamination=0.05)` sobre estos vectores. El modelo aprende cuáles distribuciones de tipos de acción son normales para ese usuario (por ejemplo, este usuario hace muchas lecturas y pocas eliminaciones).

#### Fórmula del coste de frecuencia

**`train_models.py` — líneas 390–396** (también `add_frequency_costs()` para batch):

```
raw_score  = decision_function(X_session_counts_so_far)
a_score    = max(0.0, -raw_score)          # negativo = más anómalo en IsolationForest
freq_cost  = clip(a_score / score_scale, 0.0, 1.0)
freq_cost  = max(freq_cost, min_action_cost)   # piso ML por usuario (NUNCA 0)
```

- **`X_session_counts_so_far`**: vector de conteos de la sesión actual hasta este momento (no histórico, sino en tiempo real).
- **`score_scale`**: percentil 95 de los `a_score` del entrenamiento — normaliza el score a [0, 1].
- **`min_action_cost`**: coste mínimo derivado del historial del usuario: `1 / (longitud_media_sesión × 4)`, con piso global de `0.01`. Se aprende durante el entrenamiento y se almacena en el modelo.
- **`freq_cost`**: resultado en `[min_action_cost, 1.0]`. **Nunca puede ser 0.** Cerca del mínimo si la distribución de acciones es normal; cerca de 1 si es inusual.

> Ejemplo: un usuario que históricamente usa muchas "Visualize" y pocas "Delete", pero en esta sesión ya realizó 10 "Delete", recibirá un `freq_cost` alto porque ese patrón de conteo es raro en su historial.

### Paso 2 — Detección de anomalía de actividad individual

En paralelo al coste de frecuencia, se evalúa la acción con el modelo de actividad:

**`app/session.py` — líneas 217–238**
```python
act_result = evaluate_activity(combined_model, user_id, element_id, ...)
is_activity_anomaly = bool(act_result["prediction"] == 1 or element_is_unknown)
```

`act_result["prediction"]` viene de `predict_activity()` en **`train_models.py` — líneas 587–650**. El modelo de actividad considera:

| Feature | Descripción |
|---|---|
| `element_id` (one-hot) | Elemento sobre el que se actuó |
| `entity_id` (one-hot) | Entidad implicada |
| `action_id` (one-hot) | Tipo de acción |
| `hour`, `minute` | Momento del día |
| `day_of_week` (one-hot) | Día de la semana |
| `log_sec_prev` | `log(1 + segundos desde la acción anterior)` |
| `log_sec_start` | `log(1 + segundos desde el inicio de sesión)` |

Se entrena con `contamination=0.05`. La predicción es binaria: **0 = normal**, **1 = anómalo**.

`element_is_unknown` es `True` si el elemento no está en el perfil ML del usuario (ver sección 7).

### Paso 3 — Override del coste final

**`app/session.py` — líneas 239–243**
```python
if is_activity_anomaly:
    op_cost = 1.0
else:
    op_cost = min(op_cost, 0.5)
op_cost = max(op_cost, MIN_ACTION_COST)   # garantía absoluta: nunca 0
```

La lógica es:

| Situación | `op_cost` resultante |
|---|---|
| Acción marcada como anómala por el modelo de actividad | **1.0** (coste máximo) |
| Elemento fuera del perfil ML del usuario | **1.0** (coste máximo) |
| Acción normal en elemento conocido | **max(min(freq_cost, 0.5), MIN_ACTION_COST)** |

`MIN_ACTION_COST = 0.01` es el piso global absoluto. El piso real por usuario (almacenado en el modelo) es mayor y se aplica antes de llegar aquí, por lo que en la práctica nunca se llega al 0.01 en producción.

> La razón del tope de 0.5 para acciones normales es evitar que el uso diversificado pero legítimo consuma el presupuesto de sesión demasiado rápido. El coste máximo de 1.0 para anomalías acelera la acumulación hacia el umbral.

### Resumen visual del cálculo de `op_cost`

```
                        ┌────────────────────────────────┐
                        │  Modelo action_frequency (IF)  │
  session_counts ──────▶│  decision_function(X) → score  │──▶ freq_cost ∈ [0, 1]
                        └────────────────────────────────┘
                                                                      │
                        ┌────────────────────────────────┐            │
                        │  Modelo activity (IF)          │            ▼
  acción actual  ──────▶│  predict(X) → prediction       │──▶ is_activity_anomaly
                        └────────────────────────────────┘            │
                                                                      ▼
                                             ┌─────────────────────────────────┐
                                             │  if is_activity_anomaly:        │
                                             │      op_cost = 1.0              │
                                             │  else:                          │
                                             │      op_cost = min(freq_cost,   │
                                             │                    0.5)         │
                                             └─────────────────────────────────┘
```

---

## 4. Coste acumulado de sesión

El **costo de sesión** es la suma de todos los `op_cost` de las acciones ejecutadas en la sesión activa.

**`app/session.py` — línea 119 (`_build_session_features`)**
```python
cum_cost = float(sum(cost_parts))
```

Donde `cost_parts` crece con cada acción (**`app/session.py` — línea 244**):
```python
new_cp = cost_parts + [op_cost]
```

### Ejemplo

Si un usuario ejecuta 4 acciones con costes `0.05`, `0.10`, `1.0` (elemento desconocido), `0.08`:
```
cost_parts       = [0.05, 0.10, 1.0, 0.08]
cumulative_cost  = 0.05 + 0.10 + 1.0 + 0.08 = 1.23
```

### Métricas derivadas (features del modelo de sesión)

Calculadas en **`app/session.py` — líneas 119–121**:

| Variable | Fórmula |
|---|---|
| `cumulative_cost` | `sum(cost_parts)` |
| `avg_cost` | `cumulative_cost / n` |
| `max_cost` | `max(cost_parts)` |

---

## 5. Umbral de coste de sesión (`session_cost_threshold`)

Durante el entrenamiento se calcula un **umbral por usuario**. Si el coste acumulado supera ese umbral, se fuerza el logout automático.

### Cómo se calcula el umbral

**`train_models.py` — `train_session_models()` — líneas 553–557**:

```python
session_max_costs = udf.groupby("login_log_id")["cumulative_cost"].max()
if len(session_max_costs) >= 5:
    cost_threshold = float(np.percentile(session_max_costs.values, 95))
else:
    cost_threshold = float("inf")
```

El proceso paso a paso:

1. Para cada sesión histórica del usuario, se toma el **coste máximo acumulado** que llegó a alcanzar esa sesión.
2. Se recogen los costes máximos de **todas las sesiones** del usuario en un array.
3. Se calcula el **percentil 95** de ese array.
4. **Condición mínima**: si el usuario tiene menos de 5 sesiones históricas, el umbral es `∞` (no se aplica límite). Esto evita umbrales basados en datos insuficientes.

### Por qué el percentil 95

El percentil 95 significa que el 95% de las sesiones históricas del usuario quedaron por debajo de ese coste. Cualquier sesión nueva que supere ese valor es estadísticamente inusual en comparación con el comportamiento habitual.

### Ejemplo de cálculo del umbral

Supongamos que un usuario tiene 10 sesiones históricas con costes máximos:
```
[0.30, 0.45, 0.22, 0.55, 0.40, 0.35, 0.28, 0.50, 0.42, 0.38]
percentil 95 → ~0.53
```
El umbral de ese usuario sería `0.53`. Si en una sesión nueva el coste acumulado llega a `0.54`, se activa el logout automático.

### Dónde se usa

- Guardado en `combined_model.pkl` → `session[uid]["session_cost_threshold"]`
- Consultado en **`security/anomaly_guard.py` — línea 73** (`get_model_session_threshold`)
- Comprobado en **`app/session.py` — línea 264**:
  ```python
  threshold_exceeded = sf["cumulative_cost"] >= session_cost_threshold
  ```

---

## 6. Evaluación completa por acción (resumen del flujo)

Cada vez que el usuario ejecuta una acción, `_process_action()` en **`app/session.py` — líneas 173–327** ejecuta estos pasos:

1. **Calcula el coste de frecuencia** → `evaluate_action_cost()` → `predict_action_frequency_cost()` (modelo `action_frequency`).
2. **Evalúa la acción individualmente** → `evaluate_activity()` → `predict_activity()` (modelo `activity`) → devuelve `(prediction, anomaly_score)`.
3. **Determina `is_activity_anomaly`** → `prediction == 1` OR `element_is_unknown`.
4. **Fija `op_cost`** → `1.0` si es anómala, `min(freq_cost, 0.5)` si es normal.
5. **Acumula el coste** → añade `op_cost` a `cost_parts`.
6. **Construye features de sesión** → `_build_session_features()` en `app/session.py:107`.
7. **Evalúa la sesión** → `evaluate_session()` → `predict_session()` (modelo `session`) → devuelve `(prediction, score)`.
8. **Decide el resultado final** (**`app/session.py` — línea 265**):
   ```python
   final_pred = bool(is_activity_anomaly or is_session_anomaly or threshold_exceeded)
   ```
9. **Guarda log** → siempre guarda en `ml_prediction_log` con el coste y el threshold.
10. **Logout automático** → **únicamente** si `threshold_exceeded`. Las anomalías individuales o de sesión muestran advertencias en pantalla pero la sesión continúa.

> El único trigger de logout automático es superar el umbral de coste acumulado. Detectar una acción anómala o un patrón de sesión anómalo solo incrementa más rápido el coste (via `op_cost = 1.0`), acercando al usuario al umbral.

---

## 7. ¿Cómo se determina qué elementos conoce cada usuario?

Durante el entrenamiento (**`train_models.py` — líneas 246–247**):
```python
element_freq = udf["element_id"].value_counts(normalize=True)
known_elements = sorted([int(eid) for eid, freq in element_freq.items() if freq >= 0.01])
```

Un elemento es "conocido" para el usuario si aparece en al menos el **1% de sus actividades históricas**. Si el usuario accede a un elemento que no cumple ese umbral, `element_is_unknown = True`.

Cuando `element_is_unknown = True`:
- La acción queda marcada como `is_activity_anomaly = True`.
- `op_cost` se fija a `1.0` (coste máximo).
- Se muestra una advertencia en pantalla.
- **La sesión no termina** — solo el coste acumulado sube más rápido.

Los elementos conocidos se consultan en **`security/anomaly_guard.py` — línea 60** (`get_user_known_elements`).

---

## 8. Umbral de coste vs. las otras señales de anomalía

| Señal | ¿Provoca logout inmediato? | Efecto real |
|---|---|---|
| `is_activity_anomaly = True` | No | `op_cost = 1.0` → coste acumulado sube más rápido |
| `is_session_anomaly = True` | No | Se registra en log; coste no se modifica aquí |
| `threshold_exceeded = True` | **Sí** | Logout automático inmediato |
| `element_is_unknown = True` | No (ya incluido en `is_activity_anomaly`) | `op_cost = 1.0` → mismo efecto |

El diseño es **de presión acumulada**: una acción anómala aislada no termina la sesión, pero si el usuario acumula suficientes acciones anómalas (cada una con coste 1.0), el `cumulative_cost` llegará al umbral y el logout ocurrirá automáticamente.

---

---

# Calcolo delle Attività e del Costo di Sessione

Questo documento spiega unicamente **come vengono calcolati i numeri** dietro ogni tipo di attività e il costo di sessione per utente.

---

## 1. Tipi di attività e relativi ID

I tipi di attività sono etichette assegnate a un `action_id` numerico. La mappatura si trova in **`app/session.py` — righe 20–27**:

```
"1" → (1000000, "Visualize")
"2" → (1000001, "Create")
"3" → (1000002, "Edit")
"4" → (1000003, "Delete")
"5" → (1000004, "Copy")
"6" → (1000005, "Share")
```

Il tipo in sé non modifica direttamente il calcolo del costo. Ciò che conta è quanto quel tipo di azione sia frequente nella sessione corrente e se il comportamento corrisponde allo storico dell'utente.

---

## 2. Tre modelli ML (riepilogo)

Il sistema addestra tre modelli indipendenti per utente:

| Modello | File | Cosa impara |
|---|---|---|
| **activity** | `train_models.py` — `train_activity_models()` | Se una singola azione è anomala (timing, elemento, ora) |
| **action_frequency** | `train_models.py` — `train_action_frequency_models()` | Se la distribuzione dei tipi di azione nella sessione è insolita |
| **session** | `train_models.py` — `train_session_models()` | Se il pattern globale della sessione (ritmo, costo accumulato, diversità) è anomalo |

Tutti usano `IsolationForest` di scikit-learn. I tre vengono salvati insieme in `models/combined_model.pkl`.

---

## 3. Costo di una singola azione (`op_cost`)

Ogni volta che l'utente esegue un'azione, il sistema calcola un **`op_cost`** che rappresenta quanto quell'azione costi al budget della sessione. Il processo si articola in tre passi.

### Passo 1 — Costo base: modello di frequenza delle azioni

**`app/session.py` — riga 230**
```python
op_cost = evaluate_action_cost(combined_model, user_id, new_counts)
```

Questo chiama `predict_action_frequency_cost()` in **`train_models.py` — righe 374–394**.

#### Cosa misura questo modello

Durante l'addestramento (**`train_models.py` — `train_action_frequency_models()` — righe 343–371**), per ogni utente si costruisce un dataset dove ogni riga è una sessione storica e le feature sono i conteggi di ogni tipo di azione in quella sessione:

| Feature | Descrizione |
|---|---|
| `action_1000000` | Quante volte è stata eseguita "Visualize" nella sessione |
| `action_1000001` | Quante volte è stata eseguita "Create" |
| `action_1000002` | Quante volte è stata eseguita "Edit" |
| `action_1000003` | Quante volte è stata eseguita "Delete" |
| `action_1000004` | Quante volte è stata eseguita "Copy" |
| `action_1000005` | Quante volte è stata eseguita "Share" |

Viene addestrato un `IsolationForest(n_estimators=200, contamination=0.05)` su questi vettori. Il modello impara quali distribuzioni di tipi di azione sono normali per quell'utente.

#### Formula del costo di frequenza

**`train_models.py` — righe 390–396** (anche `add_frequency_costs()` per batch):

```
raw_score  = decision_function(X_conteggi_sessione_finora)
a_score    = max(0.0, -raw_score)          # negativo = più anomalo in IsolationForest
freq_cost  = clip(a_score / score_scale, 0.0, 1.0)
freq_cost  = max(freq_cost, min_action_cost)   # floor ML per utente (MAI 0)
```

- **`X_conteggi_sessione_finora`**: vettore dei conteggi della sessione corrente fino a questo momento (non storico, ma in tempo reale).
- **`score_scale`**: percentile 95 degli `a_score` del training — normalizza lo score a [0, 1].
- **`min_action_cost`**: costo minimo derivato dallo storico dell'utente: `1 / (lunghezza_media_sessione × 4)`, con floor globale di `0.01`. Viene calcolato durante il training e salvato nel modello.
- **`freq_cost`**: risultato in `[min_action_cost, 1.0]`. **Non può mai essere 0.** Vicino al minimo se la distribuzione è normale; vicino a 1 se è insolita.

> Esempio: un utente che storicamente esegue molte "Visualize" e poche "Delete", ma in questa sessione ha già eseguito 10 "Delete", riceverà un `freq_cost` alto perché quel pattern di conteggio è raro nel suo storico.

### Passo 2 — Rilevamento anomalia di attività individuale

In parallelo al costo di frequenza, l'azione viene valutata con il modello di attività:

**`app/session.py` — righe 217–238**
```python
act_result = evaluate_activity(combined_model, user_id, element_id, ...)
is_activity_anomaly = bool(act_result["prediction"] == 1 or element_is_unknown)
```

Le feature del modello di attività (**`train_models.py` — `predict_activity()` — righe 587–650**):

| Feature | Descrizione |
|---|---|
| `element_id` (one-hot) | Elemento su cui è stata eseguita l'azione |
| `entity_id` (one-hot) | Entità coinvolta |
| `action_id` (one-hot) | Tipo di azione |
| `hour`, `minute` | Momento della giornata |
| `day_of_week` (one-hot) | Giorno della settimana |
| `log_sec_prev` | `log(1 + secondi dall'azione precedente)` |
| `log_sec_start` | `log(1 + secondi dall'inizio della sessione)` |

Addestrato con `contamination=0.05`. La previsione è binaria: **0 = normale**, **1 = anomalo**.

`element_is_unknown` è `True` se l'elemento non è nel profilo ML dell'utente (vedi sezione 7).

### Passo 3 — Override del costo finale

**`app/session.py` — righe 239–243**
```python
if is_activity_anomaly:
    op_cost = 1.0
else:
    op_cost = min(op_cost, 0.5)
op_cost = max(op_cost, MIN_ACTION_COST)   # garanzia assoluta: mai 0
```

La logica è:

| Situazione | `op_cost` risultante |
|---|---|
| Azione segnata come anomala dal modello di attività | **1.0** (costo massimo) |
| Elemento fuori dal profilo ML dell'utente | **1.0** (costo massimo) |
| Azione normale su elemento noto | **max(min(freq_cost, 0.5), MIN_ACTION_COST)** |

`MIN_ACTION_COST = 0.01` è il floor globale assoluto. Il floor reale per utente (salvato nel modello) è maggiore e viene applicato prima di arrivare qui, quindi in produzione non si arriva mai a 0.01.

> Il tetto di 0.5 per le azioni normali evita che l'uso diversificato ma legittimo consumi il budget di sessione troppo velocemente. Il costo massimo di 1.0 per le anomalie accelera l'accumulo verso la soglia.

### Schema visivo del calcolo di `op_cost`

```
                        ┌────────────────────────────────┐
                        │  Modello action_frequency (IF) │
  conteggi_sessione ───▶│  decision_function(X) → score  │──▶ freq_cost ∈ [0, 1]
                        └────────────────────────────────┘
                                                                      │
                        ┌────────────────────────────────┐            │
                        │  Modello activity (IF)         │            ▼
  azione corrente ─────▶│  predict(X) → prediction       │──▶ is_activity_anomaly
                        └────────────────────────────────┘            │
                                                                      ▼
                                             ┌─────────────────────────────────┐
                                             │  if is_activity_anomaly:        │
                                             │      op_cost = 1.0              │
                                             │  else:                          │
                                             │      op_cost = min(freq_cost,   │
                                             │                    0.5)         │
                                             └─────────────────────────────────┘
```

---

## 4. Costo cumulativo della sessione

Il **costo di sessione** è la somma di tutti gli `op_cost` delle azioni eseguite nella sessione attiva.

**`app/session.py` — riga 119 (`_build_session_features`)**
```python
cum_cost = float(sum(cost_parts))
```

Dove `cost_parts` cresce con ogni azione (**`app/session.py` — riga 244**):
```python
new_cp = cost_parts + [op_cost]
```

### Esempio

Se un utente esegue 4 azioni con costi `0.05`, `0.10`, `1.0` (elemento sconosciuto), `0.08`:
```
cost_parts       = [0.05, 0.10, 1.0, 0.08]
cumulative_cost  = 0.05 + 0.10 + 1.0 + 0.08 = 1.23
```

### Metriche derivate (feature del modello di sessione)

Calcolate in **`app/session.py` — righe 119–121**:

| Variabile | Formula |
|---|---|
| `cumulative_cost` | `sum(cost_parts)` |
| `avg_cost` | `cumulative_cost / n` |
| `max_cost` | `max(cost_parts)` |

---

## 5. Soglia del costo di sessione (`session_cost_threshold`)

Durante l'addestramento viene calcolata una **soglia per utente**. Se il costo cumulativo supera quella soglia, viene forzato il logout automatico.

### Come viene calcolata la soglia

**`train_models.py` — `train_session_models()` — righe 553–557**:

```python
session_max_costs = udf.groupby("login_log_id")["cumulative_cost"].max()
if len(session_max_costs) >= 5:
    cost_threshold = float(np.percentile(session_max_costs.values, 95))
else:
    cost_threshold = float("inf")
```

Il processo passo per passo:

1. Per ogni sessione storica dell'utente, si prende il **costo massimo cumulativo** raggiunto da quella sessione.
2. Si raccolgono i costi massimi di **tutte le sessioni** dell'utente in un array.
3. Si calcola il **percentile 95** di quell'array.
4. **Condizione minima**: se l'utente ha meno di 5 sessioni storiche, la soglia è `∞` (nessun limite). Questo evita soglie basate su dati insufficienti.

### Perché il percentile 95

Il percentile 95 significa che il 95% delle sessioni storiche dell'utente è rimasto al di sotto di quel costo. Qualsiasi nuova sessione che supera quel valore è statisticamente insolita rispetto al comportamento abituale.

### Esempio di calcolo della soglia

Supponiamo che un utente abbia 10 sessioni storiche con costi massimi:
```
[0.30, 0.45, 0.22, 0.55, 0.40, 0.35, 0.28, 0.50, 0.42, 0.38]
percentile 95 → ~0.53
```
La soglia di quell'utente sarebbe `0.53`. Se in una nuova sessione il costo cumulativo raggiunge `0.54`, scatta il logout automatico.

### Dove viene utilizzata

- Salvata in `combined_model.pkl` → `session[uid]["session_cost_threshold"]`
- Consultata in **`security/anomaly_guard.py` — riga 73** (`get_model_session_threshold`)
- Verificata in **`app/session.py` — riga 264**:
  ```python
  threshold_exceeded = sf["cumulative_cost"] >= session_cost_threshold
  ```

---

## 6. Valutazione completa per azione (riassunto del flusso)

Ogni volta che l'utente esegue un'azione, `_process_action()` in **`app/session.py` — righe 173–327** esegue questi passi:

1. **Calcola il costo di frequenza** → `evaluate_action_cost()` → `predict_action_frequency_cost()` (modello `action_frequency`).
2. **Valuta la singola azione** → `evaluate_activity()` → `predict_activity()` (modello `activity`) → restituisce `(prediction, anomaly_score)`.
3. **Determina `is_activity_anomaly`** → `prediction == 1` OR `element_is_unknown`.
4. **Fissa `op_cost`** → `1.0` se è anomala, `min(freq_cost, 0.5)` se è normale.
5. **Accumula il costo** → aggiunge `op_cost` a `cost_parts`.
6. **Costruisce le feature di sessione** → `_build_session_features()` in `app/session.py:107`.
7. **Valuta la sessione** → `evaluate_session()` → `predict_session()` (modello `session`) → restituisce `(prediction, score)`.
8. **Decide il risultato finale** (**`app/session.py` — riga 265**):
   ```python
   final_pred = bool(is_activity_anomaly or is_session_anomaly or threshold_exceeded)
   ```
9. **Salva il log** → sempre salva in `ml_prediction_log` con il costo e la soglia.
10. **Logout automatico** → **solo** se `threshold_exceeded`. Le anomalie singole o di sessione mostrano avvisi a schermo ma la sessione continua.

> L'unico trigger di logout automatico è il superamento della soglia di costo cumulativo. Rilevare un'azione anomala o un pattern di sessione anomalo incrementa più velocemente il costo (via `op_cost = 1.0`), avvicinando l'utente alla soglia.

---

## 7. Come si determina quali elementi conosce ogni utente?

Durante l'addestramento (**`train_models.py` — righe 246–247**):
```python
element_freq = udf["element_id"].value_counts(normalize=True)
known_elements = sorted([int(eid) for eid, freq in element_freq.items() if freq >= 0.01])
```

Un elemento è "conosciuto" per l'utente se compare in almeno l'**1% delle sue attività storiche**. Se l'utente accede a un elemento che non soddisfa questa condizione, `element_is_unknown = True`.

Quando `element_is_unknown = True`:
- L'azione viene segnata come `is_activity_anomaly = True`.
- `op_cost` viene fissato a `1.0` (costo massimo).
- Viene mostrato un avviso a schermo.
- **La sessione non termina** — solo il costo cumulativo aumenta più velocemente.

Gli elementi conosciuti vengono consultati in **`security/anomaly_guard.py` — riga 60** (`get_user_known_elements`).

---

## 8. Soglia di costo vs. altri segnali di anomalia

| Segnale | Provoca logout immediato? | Effetto reale |
|---|---|---|
| `is_activity_anomaly = True` | No | `op_cost = 1.0` → costo cumulativo sale più velocemente |
| `is_session_anomaly = True` | No | Viene registrato nel log; il costo non si modifica qui |
| `threshold_exceeded = True` | **Sì** | Logout automatico immediato |
| `element_is_unknown = True` | No (già incluso in `is_activity_anomaly`) | `op_cost = 1.0` → stesso effetto |

Il progetto è basato su **pressione cumulativa**: una singola azione anomala non termina la sessione, ma se l'utente accumula abbastanza azioni anomale (ognuna con costo 1.0), il `cumulative_cost` raggiungerà la soglia e il logout avverrà automaticamente.
