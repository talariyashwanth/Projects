# Cognisense — Cognitive Load Detector
## Product Requirements Document (PRD)

> **Tagline:** Detect cognitive overload before performance breaks down.

---

# 1. Product Overview

### Product Name

**Cognisense**

### One-Line Description

Cognisense is an ML-powered behavioral analytics system that estimates a user's cognitive load from interaction patterns such as typing behavior, response time, mouse movement, errors, pauses, and task performance.

The system watches *how* someone interacts with a computer—not what they type—and predicts whether their current workload is **Low, Medium, or High**.

> **Important:** Cognisense is an experimental behavioral ML prototype, not a medical or psychological diagnostic system.

---

# 2. The Problem

When someone becomes cognitively overloaded, their observable interaction behavior may change.

For example, a developer debugging a difficult problem may:

- Type more slowly
- Make more corrections
- Pause frequently
- Switch between windows repeatedly
- Make more errors
- Take longer to respond
- Abandon tasks
- Exhibit increasingly irregular interaction patterns

The problem is:

> **These signals exist, but they are usually invisible to the user.**

Cognisense attempts to convert these behavioral signals into a measurable ML estimate.

---

# 3. Target Users

## Primary Users

### Students and Developers

Example:

> A student is solving a programming problem.

Cognisense observes their interaction behavior and estimates:

**Current cognitive load: HIGH**

## Future Users

Potential future applications include:

- Students
- Programmers
- Researchers
- Online learning platforms
- Workplaces
- Aviation/training systems
- Gaming
- Human-computer interaction research

---

# 4. Core Product Experience

## Step 1 — Start Session

The user opens the application and starts a monitored task session.

```text
COGNISENSE

Cognitive Load Monitoring

Task:
[ Solve a programming problem ]

Session duration:
[ 15 minutes ]

        START SESSION
```

## Step 2 — Perform Task

The application records behavioral signals.

For the MVP, the system does not need invasive monitoring.

Potential signals:

- Typing speed
- Keystroke intervals
- Backspaces
- Typing errors
- Pauses
- Mouse movement
- Clicks
- Task completion time
- Response time
- Window/task switching
- Interaction frequency

## Step 3 — Feature Extraction

Raw behavioral signals:

```text
Keystrokes
Mouse events
Clicks
Timestamps
Errors
Task events
```

↓

Feature engineering:

```text
WPM
Average keystroke interval
Backspace ratio
Pause frequency
Mouse velocity
Click frequency
Error rate
Task completion time
Window-switch frequency
Interaction variance
```

↓

ML model.

---

# 5. The Three Cognitive States

The initial MVP should use three classes rather than immediately attempting complicated continuous prediction.

## 🟢 LOW

The user is operating comfortably.

Typical characteristics:

- Fast response
- Low error rate
- Consistent typing
- Low hesitation
- Stable interaction

## 🟡 MEDIUM

The user is experiencing moderate mental effort.

Typical characteristics:

- Increased pauses
- Slightly slower interaction
- Moderate corrections
- Occasional task switching

## 🔴 HIGH

The user may be experiencing cognitive overload.

Typical characteristics:

- Long pauses
- High error rate
- Frequent corrections
- Inconsistent interaction
- Slower responses
- Increased task switching

> **Important:** The system should describe this as an *estimated cognitive-load state*, not a medical diagnosis.

---

# 6. The ML Problem

The core problem is:

> **Supervised classification**

### Input

Behavioral features:

```text
typing speed
error rate
pause frequency
mouse behavior
response time
task performance
context switching
...
```

### Output

```text
LOW
MEDIUM
HIGH
```

A later version can treat cognitive load as a regression problem:

```text
Cognitive Load Score = 0–100
```

However, three-class classification is easier to implement, evaluate, and explain for the MVP.

---

# 7. Feature Engineering

Feature engineering is one of the most important parts of Cognisense.

---

## 7.1 Keyboard Features

### Typing Speed

Measure words per minute (WPM).

Conceptually:

```text
WPM = words typed / time
```

### Keystroke Interval

Average time between consecutive key presses.

### Backspace Ratio

```text
backspaces / total keystrokes
```

A higher value can indicate frequent correction.

### Pause Frequency

Count pauses longer than a predefined threshold.

For example:

```text
> 2 seconds = pause
```

The threshold should be treated as a configurable engineering parameter, not as a universal psychological threshold.

---

# 8. Mouse Features

Potential mouse-derived features include:

## Mouse Velocity

How quickly the cursor moves.

## Movement Distance

Total distance traveled by the mouse.

## Click Frequency

Clicks per minute.

## Idle Time

Time during which no interaction occurs.

## Movement Irregularity

Variance or other measures of mouse movement behavior.

---

# 9. Task Performance Features

The system should not rely entirely on keyboard/mouse behavior.

## Response Time

How long the user takes to answer or perform an action.

## Error Rate

```text
incorrect attempts / total attempts
```

## Task Completion

Whether the user completes the assigned task.

## Retry Count

How many times the user retries the same action.

These features provide context about actual task performance.

---

# 10. Context Switching

A particularly useful behavioral signal is:

> **How frequently does the user switch between tasks or windows?**

Example:

```text
IDE → Browser → IDE → Browser → IDE → Browser
```

A sudden increase may indicate:

- Searching for information
- Uncertainty
- Debugging
- Distraction

Do **not** claim that context switching proves cognitive overload.

Instead, the application should say:

> "Frequent context switching contributed to the model's high-load prediction."

This is more scientifically defensible.

---

# 11. Feature Vector

At a given time window, the model can receive a feature vector such as:

```text
[
    typing_speed,
    avg_keystroke_interval,
    backspace_ratio,
    pause_frequency,
    mouse_velocity,
    mouse_distance,
    click_frequency,
    idle_time,
    error_rate,
    response_time,
    retry_count,
    context_switches
]
```

Example:

```text
[
  31.2,
  0.42,
  0.14,
  8,
  240.5,
  1520,
  12,
  8.2,
  0.19,
  14.7,
  4,
  9
]
```

---

# 12. Dataset Strategy

The biggest practical challenge is obtaining reliable cognitive-load labels.

For the MVP, use either:

1. A suitable public behavioral/cognitive-load dataset, or
2. A controlled/simulated dataset for demonstrating the complete ML pipeline.

The eventual dataset should contain observations similar to:

| WPM | Error Rate | Pause Frequency | Context Switches | Response Time | Load |
|---:|---:|---:|---:|---:|---|
| 62 | 0.03 | 2 | 1 | 4.2 | Low |
| 54 | 0.07 | 4 | 2 | 6.1 | Low |
| 43 | 0.13 | 7 | 4 | 9.8 | Medium |
| 35 | 0.21 | 11 | 7 | 14.2 | High |
| 29 | 0.27 | 15 | 10 | 18.6 | High |

### Data Integrity Requirement

Do not randomly assign labels based on arbitrary rules and present them as genuine cognitive-load measurements.

If synthetic data is used:

> Explicitly label it as **synthetic/simulated training data**.

If a public dataset is used:

> Cite the original dataset and document its labeling methodology.

---

# 13. Model Pipeline

Use multiple models so the project includes an actual ML experiment.

## Baseline

**Logistic Regression**

Why:

- Easy to explain
- Strong baseline
- Interpretable coefficients
- Useful for comparison

## Main Model

**Random Forest**

Why:

- Captures nonlinear relationships
- Works well with mixed behavioral features
- Doesn't require huge datasets
- Feature importance is easy to demonstrate

## Optional Experiment

**XGBoost**

Compare:

```text
Logistic Regression
        vs
Random Forest
        vs
XGBoost
```

Select the best model based on validation performance rather than automatically assuming the most complex model is best.

---

# 14. Training Pipeline

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Standardization (where appropriate)
     ↓
Model Training
     ↓
Cross Validation
     ↓
Evaluation
     ↓
Model Selection
     ↓
Save Model
```

Recommended libraries:

```text
scikit-learn
pandas
numpy
matplotlib
joblib
```

---

# 15. Model Evaluation

Do not report only:

> Accuracy = 92%

Use multiple evaluation metrics.

## Accuracy

Overall correctness.

## Precision

How many predicted high-load observations were actually high-load.

## Recall

How many actual high-load observations were detected.

## F1-Score

Balances precision and recall.

## Confusion Matrix

Example:

```text
                 Predicted

              Low  Med  High

Actual Low     82    7    1
       Med      8   71    6
       High     1    7   84
```

For a serious implementation, also consider:

- Stratified train/test split
- Cross-validation
- Class imbalance
- Per-class metrics
- Calibration if probabilities are shown to users

---

# 16. Explainability

This should be one of the strongest product features.

Do not simply say:

> HIGH LOAD — 89%

Instead show:

```text
WHY?

↑ Error rate                 +24%
↑ Pause frequency            +19%
↑ Context switching          +16%
↓ Typing speed               +14%
↑ Response time              +11%
```

Potential explainability methods:

- Random Forest feature importance for the MVP
- SHAP in a later version

For the weekend MVP, feature importance is sufficient if presented carefully.

> Feature importance indicates which features influenced the trained model, not proof that those features directly caused cognitive load.

---

# 17. Cognitive Load Score

The classifier can output probabilities.

Example:

```text
Low probability     = 0.05
Medium probability  = 0.21
High probability    = 0.74
```

The product can convert these probabilities into a user-facing score.

Example:

```text
Cognitive Load Score = 84/100
```

Suggested product convention:

```text
0–35     LOW
36–70    MEDIUM
71–100   HIGH
```

This score is a product visualization convention and should not be presented as a clinically validated cognitive-load scale.

---

# 18. Dashboard

The dashboard should make the project visually impressive while still exposing the ML logic.

Example:

```text
┌──────────────────────────────────────────────┐
│ COGNISENSE                                   │
│ Behavioral Cognitive Load Monitor            │
├──────────────────────────────────────────────┤
│                                              │
│ CURRENT COGNITIVE LOAD                       │
│                                              │
│                 78 / 100                     │
│                                              │
│                 HIGH 🔴                      │
│                                              │
├──────────────────────┬───────────────────────┤
│ Typing Speed         │ Error Rate            │
│ 34 WPM ↓             │ 21% ↑                 │
├──────────────────────┼───────────────────────┤
│ Pause Frequency      │ Context Switching     │
│ 11/min ↑             │ 8/min ↑               │
└──────────────────────┴───────────────────────┘
```

---

# 19. Load Timeline

Show how estimated cognitive load changes throughout a session.

Example:

```text
100 ┤                         ╭──────╮
 80 ┤                    ╭────╯      ╰───
 60 ┤            ╭───────╯
 40 ┤      ╭─────╯
 20 ┤──────╯
  0 └────────────────────────────────────
       0    3    6    9    12    15 min
```

This lets the user see:

> "My estimated load increased sharply around minute 9."

---

# 20. Real-Time Mode

For the MVP, real-time behavior can be simulated or implemented with a controlled session.

Every 10–30 seconds:

```text
Collect window
      ↓
Extract features
      ↓
Model prediction
      ↓
Update dashboard
```

Example:

```text
10:01 → LOW
10:02 → LOW
10:03 → MEDIUM
10:04 → MEDIUM
10:05 → HIGH
10:06 → HIGH
10:07 → HIGH
```

Then display:

> ⚠ Sustained high-load pattern detected.

The alert should be framed as a behavioral model signal, not a medical warning.

---

# 21. Session Summary

When the user finishes:

```text
SESSION COMPLETE

Duration: 18m 42s

Average Load
━━━━━━━━━━━━━━━━
67 / 100

Peak Load
━━━━━━━━━━━━━━━━
91 / 100

Time in states

LOW       32%
MEDIUM    41%
HIGH      27%

────────────────────────

Top contributing signals

1. Error rate
2. Pause frequency
3. Context switching
4. Response time
```

---

# 22. Recovery Feature

A unique feature is to detect whether estimated load decreases after a break.

Example:

```text
Before break:     86
After break:      52
Recovery:         34 points
```

Display:

> **Recovery detected**

This turns Cognisense from a simple classifier into a behavioral analytics system.

Important caveat:

> A decrease in the model score after a break is an observed behavioral change, not proof that the person's psychological cognitive load has objectively recovered.

---

# 23. Personalized Baseline

A more advanced version should account for individual differences.

Different people naturally behave differently.

Example:

```text
Person A
Normal typing = 45 WPM

Person B
Normal typing = 80 WPM
```

Therefore, the system should not assume:

> 45 WPM = high cognitive load.

Instead, the first session can establish a personal baseline.

Example:

```text
YOUR BASELINE

Typing speed: 62 WPM
Error rate: 4%
Pause frequency: 3/min
```

Later:

```text
Current typing: 41 WPM

Deviation: -34%
```

---

# 24. Personalized Detection Architecture

```text
                USER
                 ↓
         Baseline Calibration
                 ↓
        Personal Behavioral Profile
                 ↓
        Current Session Features
                 ↓
          Deviation Calculation
                 ↓
             ML MODEL
                 ↓
       Personalized Load Estimate
```

This is a strong Phase 2 feature if there is not enough time to implement it for the MVP.

---

# 25. Recommended Tech Stack

## Frontend

**Streamlit**

## Backend

**Python**

## ML

```text
scikit-learn
```

## Data Processing

```text
pandas
numpy
```

## Visualization

```text
matplotlib
plotly
```

## Model Persistence

```text
joblib
```

## Optional Explainability

```text
SHAP
```

## Optional Report Generation

```text
reportlab
```

---

# 26. Project Folder Structure

Use a proper engineering structure:

```text
cognisense/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── cognitive_load_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app/
│   └── app.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 27. Application Pages

Build four primary pages.

## 1. Dashboard

Displays the current estimated state.

## 2. Live Session

Displays current behavioral signals and predictions.

## 3. Session Analytics

Displays historical graphs and session statistics.

## 4. Model Insights

Displays:

- Model accuracy
- Confusion matrix
- Feature importance
- Classification report
- Model comparison

The Model Insights page is important because it demonstrates that the application is backed by an actual ML pipeline rather than being only a frontend.

---

# 28. MVP vs Advanced Version

## 🟢 MVP — Sunday

Implement:

- Public or clearly labeled synthetic dataset
- Data preprocessing
- Feature engineering
- Random Forest
- Three-class prediction
- Streamlit dashboard
- Load score
- Feature importance
- Session timeline
- GitHub README

This is enough for a strong portfolio MVP.

---

## 🟡 Version 2

Add:

- Real keyboard event capture
- Mouse tracking
- Personalized baseline
- Real-time prediction
- SHAP explanations
- Session history
- User profiles

---

## 🔴 Version 3

Potential advanced features:

- Multimodal behavioral signals
- Webcam-based facial/eye signals with explicit consent
- Physiological signals from wearable devices
- Adaptive baseline
- Temporal ML
- LSTM/Transformer models
- Online learning
- Personalized models

Do not attempt these features for the weekend MVP.

---

# 29. What Makes Cognisense Stand Out

Most student ML projects follow:

```text
Input → ML Model → Prediction
```

Cognisense should follow:

```text
Behavior
   ↓
Feature Engineering
   ↓
ML Classification
   ↓
Probability
   ↓
Cognitive Load Score
   ↓
Temporal Analysis
   ↓
Explainability
   ↓
Personal Baseline
   ↓
Actionable Insight
```

This makes Cognisense a **system**, rather than merely a trained model.

---

# 30. Scientific and Ethical Limitations

This section should be included in the README.

Do **not** claim:

> "Our system accurately detects a person's mental state."

Use:

> "Cognisense estimates cognitive-load states from observable behavioral interaction patterns. The system is an experimental ML prototype and is not a clinical or psychological diagnostic tool."

Important limitations:

- Typing behavior differs between individuals.
- Mouse behavior differs between individuals.
- Context influences interaction patterns.
- Cognitive load cannot be perfectly inferred from keyboard/mouse behavior.
- Behavioral labels may contain subjective measurement.
- Synthetic data cannot establish real-world validity.
- A model prediction is not evidence of a person's underlying psychological state.
- Monitoring should be transparent and consent-based.
- Sensitive behavioral data should be minimized, protected, and deleted when no longer required.

---

# 31. Final Product Flow

```text
                  COGNISENSE
                      │
                      ▼
                Start Session
                      │
                      ▼
              Perform a Task
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Keyboard      Mouse       Task
          │           │           │
          └───────────┼───────────┘
                      ▼
               Feature Engine
                      │
                      ▼
               ML Classifier
                      │
                      ▼
          ┌───────────┴───────────┐
          ▼                       ▼
    Load Prediction          Explanation
          │                       │
          └───────────┬───────────┘
                      ▼
                 Dashboard
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Current      Timeline    Insights
        Load
                      │
                      ▼
                Session Report
```

---

# 32. Project Pitch

When asked:

### "What did you build?"

Do not say:

> "I made a machine learning model to predict cognitive load."

Instead:

> **"I built Cognisense, a behavioral ML system that estimates cognitive-load states from keyboard, mouse, and task-performance signals. I engineered behavioral features, compared classification models, generated explainable predictions, and built a real-time dashboard showing how estimated cognitive load changes throughout a task session."**

This communicates:

- ML
- Feature engineering
- Model comparison
- Explainability
- Product thinking
- Data visualization
- Software engineering
- Responsible AI

---

# 33. Weekend Implementation Principle

The objective is **not** to build every feature described in this PRD.

The objective is to produce a polished, technically honest MVP.

Prioritize:

```text
1. Reliable data pipeline
2. Meaningful features
3. Proper ML training/evaluation
4. Explainable predictions
5. Strong dashboard
6. Clean GitHub repository
7. Clear README
```

Deprioritize:

```text
- Deep learning
- Webcam monitoring
- Physiological sensors
- Complex personalization
- LSTM/Transformers
- Production-scale infrastructure
```

A clean, defensible ML system with a polished demo is substantially better than a huge "AI" project with questionable data and an unstable implementation.
