# MITRE ATT&CK Command Classification using Machine Learning and Deep Learning

> From raw command-line telemetry to actionable ATT&CK technique classification

This project investigates how **command-line process titles (`proctitle`)** can be automatically classified into **MITRE ATT&CK techniques** using both **Machine Learning (ML)** and **Deep Learning (DL)** approaches.

The focus is not only on model performance, but on **realistic threat modeling**, **data imbalance handling**, and **domain-aware feature engineering**, which are critical for operational cybersecurity systems.

---

## 🎯 Motivation

Command-line executions are one of the strongest indicators of attacker activity on compromised hosts.  
They directly expose reconnaissance, lateral movement, persistence, and execution behaviors.

However, command data is challenging:
- commands are short and noisy,
- semantics depend on arguments, flags, paths, and IPs,
- the same command may correspond to different ATT&CK techniques,
- datasets are naturally imbalanced and frequency-driven.

This project addresses these challenges by combining:
- careful exploratory analysis,
- frequency-aware learning,
- custom tokenization for cyber artifacts,
- and a comparative study between ML and DL models.

---

## 🧠 Dataset Description

### Dataset origin

The dataset originates from the **Casino Limit Capture-The-Flag (CTF)** challenge, a realistic offensive security scenario where **multiple attacking teams** operate against a defended infrastructure.

During the exercise:
- attacker activity was fully monitored,
- all executed commands were collected from system logs,
- each command was later annotated with **MITRE ATT&CK techniques**.

References:
- Casino Limit challenge: https://casinolimit.inria.fr/challenge.html  
- Dataset description paper: https://inria.hal.science/hal-05224264  

---

### Raw data characteristics

Each row in the dataset corresponds to an **observed command execution pattern**, described by:

- `proctitle`  
  The full command-line string as executed by the attacker.

- `technique`  
  The original MITRE ATT&CK technique label.

- `technique_grouped`  
  A consolidated ATT&CK technique label used for modeling.

- `count`  
  The number of times this exact `(proctitle, technique)` pair appeared in the logs.

Important clarification:
> The dataset does **not** represent unique commands only.  
> Execution frequency (`count`) is a first-class signal and is explicitly used during training.

---

### Dataset size and label distribution

- Initial number of ATT&CK techniques: **67**
- Final number of techniques after consolidation: **37**
- Number of unique command strings: **~100**
- Strong class imbalance:
  - some techniques appear hundreds of times,
  - others only a few times.

This reflects **real attacker behavior**, where certain actions (discovery, network scanning) are far more frequent than others.

---

### Technique consolidation strategy

To ensure meaningful training and stratified splits, rare techniques are grouped into a dedicated class:

""Z999 – Other Low-Frequency Techniques""


This grouping:
- prevents extreme overfitting,
- avoids empty classes in validation/test splits,
- preserves semantic interpretability.

Crucially:
> Apparent duplicates after grouping are expected and intentional.  
> They represent **different techniques mapping to the same command**, or **frequency aggregation**, not data leakage.

---

### Dataset availability in this repository

- `data/raw/`  
  Contains the full processed dataset used in the experiments.

- `data/sample/`  
  Contains a lightweight CSV illustrating the dataset format, suitable for quick inspection.

- `data/README.md`  
  Provides additional details on dataset usage and structure.

---

## 🧪 Methodology Overview

The workflow follows three main stages.

---

### 1️⃣ Exploratory Data Analysis (EDA)

**Notebook:** `notebooks/01_eda.ipynb`

Key objectives:
- inspect dataset structure and types,
- analyze technique vs grouped-technique distributions,
- understand the semantic meaning of the `count` column,
- validate the need for frequency-weighted learning,
- consolidate `(proctitle, technique_grouped)` pairs by summing counts.

Key insight:
> In cybersecurity telemetry, frequency is part of the signal, not noise.

---

### 2️⃣ Machine Learning Baseline  
**TF-IDF + Logistic Regression**

**Notebook:** `notebooks/02_ml_lr.ipynb`

#### Feature engineering

Commands are vectorized using **TF-IDF (unigrams + bigrams)**.

A **custom token pattern** is introduced to preserve cybersecurity-specific artifacts:
- IP addresses and ports,
- URLs,
- filesystem paths,
- command flags,
- environment variables.

This step significantly improves semantic consistency compared to default NLP tokenization.

#### Training strategy

- Stratified split: **60% train / 20% validation / 20% test**
- **Sample weights = execution count**
- Weighted evaluation metrics (accuracy, F1-score)
- Hyperparameter tuning via grid search

#### Observed results (ML)

- Strong and stable weighted performance.
- High accuracy on frequent techniques.
- Confusions mainly between semantically close ATT&CK techniques.
- High interpretability and low computational cost.

This makes the ML model a **robust baseline** for operational environments.
Results (ML)

Default LR model (custom TF-IDF, count-weighted)

Accuracy (weighted): 0.7084

F1-score (weighted): 0.7908
GridSearchCV tuning
Parameter grid includes:

C ∈ {0.1, 1, 10, 100}

solver ∈ {lbfgs, saga}

class_weight ∈ {None, balanced}

scoring = f1_weighted
Best CV score:

0.8414 (weighted F1 in CV)
Best params:

C=10, penalty='l2', solver='lbfgs', class_weight=None, max_iter=1000

Optimized LR model (final)

Accuracy: 0.8502

F1-score (weighted): 0.8358

Classification report (custom pattern):

report_custom = classification_report(
    y_test_encoded,
    y_test_pred_final,
    target_names=label_encoder.classes_,
    zero_division=0
)
print(report_custom)


Confusion matrix was generated as well (37×37). For reference:

total predictions = 14,197

correct on diagonal = 12,070

accuracy = 0.8502

---

### 3️⃣ Deep Learning Baseline  
**Tokenizer + LSTM**

**Notebook:** `notebooks/03_dl_lstm.ipynb`

Preprocessing

Tokenizer configuration:

num_words=5000

oov_token="<unk>"

lower=True

Sequence preprocessing:

MAX_SEQUENCE_LENGTH = 20

pad_sequences(..., padding='post', truncating='post')

Label handling:

LabelEncoder + one-hot targets (to_categorical)
Why: prevents false ordinal relationships between class indices and matches softmax training.

Model

Embedding dim: 100

LSTM units: 64

Dropout: 0.5

Output: Dense(NUM_CLASSES, activation='softmax')

Optimizer: adam

Loss: categorical_crossentropy

Early stopping on val_loss (patience=3, restore_best_weights=True)

Training: epochs=10, batch_size=32, with sample weights

Results (DL)

On the test set:

Accuracy: 0.8345

F1-score (weighted): 0.7854

(Optional) classification report:

report_lstm = classification_report(
    y_test_encoded,
    y_test_pred_lstm,
    target_names=label_encoder.classes_,
    zero_division=0
)
print(report_lstm)

ML vs DL — What we learned
Score comparison (test set)
Approach                           	           Accuracy                             	Weighted F1
Optimized LR (custom token pattern)	           0.8502                                  	0.8358
LSTM baseline                                 	0.8345	                                  0.7854

Interpretation

On this dataset, domain-aware ML is extremely competitive and provides the best overall weighted F1.

LSTM reaches a similar accuracy but a lower weighted F1, likely due to:

heavy imbalance and frequency effects,

limited dataset size for deep semantic generalization,

sensitivity to hyperparameters.

---

## 📁 Project Structure

mitre-attack-command-classifier/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_ml_lr.ipynb
│ └── 03_dl_lstm.ipynb
│
├── command_classifier/
│ ├── train_lr.py
│ ├── train_lstm.py
│ ├── predict.py
│ ├── common.py
│ └── init.py
│
├── data/
│ ├── raw/
│ ├── sample/
│ └── README.md
│
├── requirements.txt
├── requirements-dl.txt
├── LICENSE
└── README.md


---

## ⚠️ Limitations and Future Work

- Commands analyzed independently (no temporal correlation).
- No host-level or user-level context.
- Deep learning constrained by dataset size.

Future work:
- command sequence modeling,
- attack chain reconstruction,
- integration with SIEM / EDR pipelines,
- human-in-the-loop validation.

---

## 🎓 Academic Context

This project was originally developed as part of an advanced course on **AI-based threat detection**, and was later **refactored and extended** into a standalone portfolio project suitable for professional and research evaluation.

---

## 📜 License

MIT License

---

## 👤 Author

**Thierry Armel Tchomo Kombou**  
Cybersecurity & AI Engineering  
Télécom Paris  
GitHub: https://github.com/0xTchomo

