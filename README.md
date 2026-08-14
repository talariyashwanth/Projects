# Cognisense — Behavioral Cognitive Load Detector

> Estimate cognitive overload from interaction behavior — before performance breaks down.

Cognisense is an end-to-end behavioral ML system that estimates a user's cognitive-load
state (**Low / Medium / High**) from *how* they interact with a computer — typing dynamics,
pauses, mouse movement, error patterns, response latency, and context switching. It watches
interaction *patterns*, never content.

> ### ⚠️ Read this first
> Cognisense is an **experimental ML prototype trained on synthetic data**. It is **not** a
> clinical, medical, or psychological diagnostic tool. Reported scores demonstrate that the
> pipeline is correct and honestly evaluated; they carry **no real-world validity**.
> See [Scientific & Ethical Limitations](#-scientific--ethical-limitations).

---

## Why this project is interesting

Most ML portfolio projects stop at `input → model → prediction`. Cognisense is a **system**:

```
Behavior → Feature Engineering → Classification → Probability
   → Cognitive Load Score → Temporal Analysis → Explainability
   → Personal Baseline → Actionable Insight
```

The engineering decisions that matter most here are the ones that make the results
*believable*, and they are documented in [Honest ML Engineering](#-honest-ml-engineering).

---

## Results

Best model: **Logistic Regression**, selected by cross-validated macro-F1.

| Model | Test Accuracy | Macro F1 | Precision | Recall | CV Macro-F1 (5-fold) |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** ✅ | **0.839** | **0.839** | 0.845 | 0.834 | **0.773** ± 0.025 |
| Random Forest | 0.816 | 0.817 | 0.824 | 0.811 | 0.766 ± 0.031 |
| XGBoost | 0.811 | 0.812 | 0.819 | 0.807 | 0.751 ± 0.025 |

Per-class performance (best model):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Low | 0.863 | 0.894 | 0.878 | 226 |
| Medium | 0.775 | 0.798 | 0.786 | 198 |
| High | 0.898 | 0.810 | 0.852 | 142 |

Confusion matrix (rows = actual, cols = predicted):

```
              Low   Med  High
Actual Low    202    24     0
       Med     27   158    13
       High     5    22   115
```

**How to read these numbers.** Three findings are worth more than the headline accuracy:

1. **The linear baseline wins.** Random Forest and XGBoost were both beaten by Logistic
   Regression on cross-validated F1. The behavioral signals are largely monotonic with load,
   so added model capacity bought overfitting rather than accuracy. The simplest adequate
   model was selected, not the most impressive-sounding one.
2. **Medium is the hardest class** (F1 0.786 vs 0.878 / 0.852). This is the expected and
   correct result: "Medium" is the band between two cut points on a continuum, so it absorbs
   the boundary ambiguity from both neighbours.
3. **Low↔High confusion is almost nil** (0 and 5 samples). The model rarely makes the
   *serious* error, and its mistakes are concentrated on adjacent states.

---

## 🔬 Honest ML Engineering

Four deliberate choices keep the reported numbers meaningful. Each one *lowers* the headline
score, which is the point.

### 1. The synthetic data is built to be hard
A naive generator draws each feature independently per class. With 12 such features, classes
separate trivially and every model scores ~99.7% — a number that measures the generator, not
the model. Instead ([`src/dataset_generator.py`](src/dataset_generator.py)):

- **A shared latent load factor** drives all 12 channels, so features *co-vary* as they do in
  real interaction data instead of being conditionally independent.
- **Per-subject baselines and sensitivities** (60 simulated individuals) mean absolute values
  aren't diagnostic on their own — a fast typist under load may out-type a slow typist at rest.
- **Boundary-band label sampling** assigns stochastic labels near class cut points, because a
  sample sitting on a boundary is genuinely ambiguous.
- **5% label noise** represents subjective/mismeasured self-report ground truth.

A regression test (`test_classes_genuinely_overlap`) fails if the classes ever drift back
toward being trivially separable.

### 2. Subject-aware splitting (no leakage)
Behavioral windows from one person are highly correlated. A random split puts the same
subject in train *and* test, leaking their personal baseline and inflating scores. Cognisense
uses `GroupShuffleSplit` for the split and `StratifiedGroupKFold` for CV, so **no individual
ever spans the boundary**. Every score reported here is generalisation **to a new person** —
the metric that actually matters for deployment.

### 3. Model selection never peeks at the test set
The winner is chosen by cross-validated macro-F1 on training folds. The test set is scored
exactly once, at the end, so it remains an unbiased estimate.

### 4. Permutation importance over impurity importance
Tree impurity importance is biased toward high-cardinality features. Cognisense reports
**permutation importance measured on held-out subjects** — the actual macro-F1 drop when a
feature is shuffled.

| Feature | Relative importance |
|---|---:|
| Pause frequency | 17.8% |
| Response time | 15.7% |
| Idle time | 11.0% |
| Context switches | 10.4% |
| Click frequency | 8.9% |
| Retry count | 7.8% |

> Feature importance indicates which features **the trained model relies on** — not proof
> that those behaviors *cause* cognitive load.

---

## Features

- **12 engineered behavioral features** across keyboard, mouse, and task-performance channels
- **Three-model comparison** — Logistic Regression / Random Forest / XGBoost
- **Cognitive Load Score (0–100)** — probability-weighted, so the gauge moves continuously
  instead of jumping when the argmax flips
- **Explainability engine** — per-feature z-score attribution against a reference baseline
- **Personalized baselines** — calibrate to an individual's own norm so a naturally slow
  typist isn't flagged as overloaded for typing slowly
- **Session timeline** — load progression across a session, with warm-up → focus →
  difficulty → break → recovery phases
- **Recovery detection** — quantifies behavioral change across a rest break
- **Live browser playground** — real keystroke/mouse capture with rolling prediction
- **React dashboard** — 4 pages, incl. a Model Insights page exposing the full ML evaluation

---

## Quickstart

**Requirements:** Python 3.11+, Node 18+

```bash
# 1. Backend setup
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Train (regenerates dataset, trains 3 models, writes artifacts + plots)
python src/train.py

# 3. Run tests
pytest tests/ -v

# 4. Start the API  → http://127.0.0.1:8000  (docs at /docs)
uvicorn backend.main:app --reload

# 5. Start the frontend (separate terminal) → http://localhost:5173
cd frontend && npm install && npm run dev
```

Model artifacts and the synthetic dataset are committed, so steps 4–5 work without training.

---

## API

Interactive OpenAPI docs at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service status + active model |
| `POST` | `/api/predict` | Predict from a 12-feature vector |
| `POST` | `/api/predict-personalized` | Predict against a personal baseline |
| `POST` | `/api/predict-raw-events` | Extract features from raw events, then predict |
| `POST` | `/api/calibrate` | Build a personal baseline from calibration windows |
| `GET` | `/api/model-insights` | Metrics, confusion matrices, feature importance |
| `GET` | `/api/feature-baselines` | Population reference statistics |
| `POST` | `/api/simulate-session` | Scripted session timeline for demos |

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"typing_speed_wpm":24,"error_rate":0.26,"response_time_seconds":16,
       "pause_frequency_per_min":13,"backspace_ratio":0.24}'
```

```json
{
  "predicted_state": "High",
  "cognitive_load_score": 88,
  "score_band": "High",
  "probabilities": {"Low": 0.0002, "Medium": 0.1161, "High": 0.8837},
  "top_contributing_signals": [
    {"display_name": "Error Rate", "value": 0.26, "z_score": 2.26, "is_elevating_load": true}
  ]
}
```

---

## Project structure

```
cognisense/
├── backend/main.py              # FastAPI service
├── data/raw/                    # Synthetic dataset (regenerable)
├── models/                      # Trained model, scaler, metrics, baselines
├── reports/                     # Confusion matrices + importance plots
├── src/
│   ├── dataset_generator.py     # Synthetic data w/ realistic overlap
│   ├── preprocessing.py         # Cleaning + subject-aware splitting
│   ├── feature_engineering.py   # Raw events → 12-feature vector
│   ├── train.py                 # 3-model comparison + evaluation
│   └── predict.py               # Inference, scoring, explainability
├── frontend/src/components/     # Dashboard, LiveSession, Analytics, Insights
├── tests/test_pipeline.py       # 31 tests
├── requirements.txt
└── LICENSE
```

## Feature vector

| Feature | Description |
|---|---|
| `typing_speed_wpm` | Words per minute |
| `avg_keystroke_interval_ms` | Mean time between keypresses |
| `backspace_ratio` | Backspaces / total keystrokes |
| `pause_frequency_per_min` | Pauses > 2s per minute (threshold is a tunable parameter) |
| `mouse_velocity_px_s` | Mean cursor velocity |
| `mouse_distance_px` | Total cursor travel |
| `click_frequency_per_min` | Clicks per minute |
| `idle_time_seconds` | Total no-interaction time |
| `error_rate` | Incorrect attempts / total attempts |
| `response_time_seconds` | Latency to act |
| `retry_count` | Repeated attempts at one action |
| `context_switches_per_min` | Window/task switches per minute |

Score bands (a **product visualization convention**, not a validated scale):
`0–35 Low` · `36–70 Medium` · `71–100 High`

---

## Tech stack

**ML:** scikit-learn, XGBoost, pandas, NumPy, joblib
**Backend:** FastAPI, Uvicorn, Pydantic
**Frontend:** React 19, Vite, Recharts, Lucide
**Viz & testing:** Matplotlib, pytest

---

## ⚠️ Scientific & Ethical Limitations

Cognisense estimates cognitive-load states from observable behavioral interaction patterns.
It is an experimental ML prototype and **not** a clinical or psychological diagnostic tool.

**Data limitations**
- Training data is **synthetic**. Synthetic data cannot establish real-world validity, and no
  claim about accuracy on real humans is supported by these results.
- The 3-class discretisation is a modelling convenience; cognitive load is continuous.

**Inference limitations**
- Typing and mouse behavior differ substantially between individuals.
- Context (task type, environment, fatigue, input device) heavily influences interaction.
- Cognitive load **cannot** be perfectly inferred from keyboard/mouse behavior.
- A model prediction is **not** evidence of a person's underlying psychological state.
- Behavioral labels involve subjective measurement.
- Feature importance shows model reliance, not causation.
- A lower score after a break is an *observed behavioral change*, not proof of psychological
  recovery.

**Ethical use**
- Behavioral monitoring must be **transparent and consent-based**. Covert monitoring is an
  inappropriate use of this work.
- Cognisense records interaction *dynamics*, never keystroke content.
- Behavioral data is sensitive: minimize collection, protect it, and delete it when no longer
  needed.
- Do not use for employee surveillance, performance ranking, or any consequential decision
  about a person.

---

## Roadmap

- **v2:** OS-level event capture, adaptive personal baselines, SHAP explanations, persistent
  session history
- **v3:** Temporal models (LSTM/Transformer) over feature sequences, multimodal signals with
  explicit consent, online learning
- **Validation:** the essential next step is a small IRB-reviewed study with an established
  instrument (e.g. NASA-TLX) to gather real labels. Until then, all metrics here describe
  behavior on synthetic data only.

## License

MIT — see [LICENSE](LICENSE).
