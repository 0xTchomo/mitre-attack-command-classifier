# MITRE ATT&CK Command Classification using ML & DL

> **From raw command-line telemetry to actionable ATT&CK technique classification**

This project explores how **machine learning (ML)** and **deep learning (DL)** models can be used to automatically classify **command-line process titles (`proctitle`)** into **MITRE ATT&CK techniques**, a key challenge in modern **host-based threat detection**.

The work combines **careful data analysis**, **feature engineering tailored to cybersecurity artifacts**, and **model comparison**, bridging the gap between academic experimentation and realistic defensive use cases.

---

## 🎯 Project Motivation

Command-line executions are one of the most informative traces of adversarial behavior on a system.  
However, raw commands are:

- highly variable,
- context-dependent,
- noisy (paths, flags, IPs, URLs, credentials),
- and often reused across multiple attack techniques.

This project investigates:

- **Can we automatically map commands to MITRE ATT&CK techniques?**
- **How far can classical ML go with the right feature engineering?**
- **What does deep learning add when modeling command semantics?**

---

## 🧠 Dataset & Threat Model

### Dataset origin
The dataset comes from the **Casino Limit CTF scenario**, where multiple attacking teams were monitored during a realistic attack simulation.

Each sample represents:
- a **command-line process title (`proctitle`)**
- a **MITRE ATT&CK technique label**
- an associated **execution count** (frequency in the original logs)

> 📌 The raw dataset is not generated synthetically: it reflects real attacker behavior observed during a competitive offensive scenario.

### Label consolidation
- Initial labels: **67 ATT&CK techniques**
- Final labels: **37 techniques**
- Rare techniques are grouped into a dedicated class:  
  **`Z999: Other Low-Frequency Techniques`**

This avoids extreme class imbalance while preserving semantic meaning.

---

## 🧪 Methodology Overview

The project follows a **progressive, reproducible pipeline**:

### 1️⃣ Exploratory Data Analysis (EDA)
Notebook: `notebooks/01_eda.ipynb`

Key analyses:
- distribution of techniques vs grouped techniques,
- difference between **unique commands** and **real execution frequency** (`count`),
- explanation of artificial duplicates created by technique grouping,
- consolidation of `(proctitle, technique_grouped)` pairs by summing counts.

**Key insight**  
> A command can legitimately map to multiple attack techniques depending on context — grouping and weighting are not data bugs, but threat modeling decisions.

---

### 2️⃣ Machine Learning Baseline (TF-IDF + Logistic Regression)
Notebook: `notebooks/02_ml_lr.ipynb`

#### Feature representation
- **TF-IDF** with:
  - unigrams + bigrams,
  - maximum vocabulary size = 5,000.

#### ⚠️ Cybersecurity-aware tokenization (key contribution)
A **custom token pattern** is introduced to preserve:
- IP addresses (`10.35.108.10:22`)
- URLs (`http://malicious.com/payload.sh`)
- file paths (`/etc/passwd`)
- command flags (`--color=auto`)
- environment variables (`$HOME`, `${PATH}`)

This significantly improves semantic signal compared to default NLP tokenization.

#### Training strategy
- Stratified **60 / 20 / 20** split (train / validation / test)
- **Sample weights = execution count**
- Weighted metrics (`accuracy_weighted`, `f1_weighted`)
- Hyperparameter tuning with `GridSearchCV`

**Result**  
Classical ML, when combined with domain-specific feature engineering, provides a **strong and interpretable baseline**.

---

### 3️⃣ Deep Learning Baseline (LSTM)
Notebook: `notebooks/03_dl_lstm.ipynb`

The DL approach models commands as **sequences**, rather than bags of tokens.

Pipeline:
- Tokenization with OOV handling
- Fixed-length padding
- Embedding layer
- LSTM for sequential modeling
- Softmax output over 37 ATT&CK classes

Why LSTM?
- captures ordering and structure (`command → flags → arguments → paths`)
- learns implicit patterns missed by linear models

This model serves as a **semantic baseline**, not a replacement for ML.

---

## 📊 Key Findings

- **Execution frequency matters**: ignoring the `count` column severely misrepresents attacker behavior.
- **Grouping rare techniques is necessary**, but must be explained and handled carefully.
- **Custom tokenization is critical** in cybersecurity NLP.
- **ML models remain highly competitive** when domain knowledge is injected.
- **DL models add flexibility**, but at higher computational and interpretability cost.

---

## 📁 Project Structure

mitre-attack-command-classifier/
│
├── notebooks/
│ ├── 01_eda.ipynb # Dataset exploration and consolidation
│ ├── 02_ml_lr.ipynb # TF-IDF + Logistic Regression baseline
│ └── 03_dl_lstm.ipynb # LSTM deep learning model
│
├── command_classifier/
│ ├── train_lr.py # ML training pipeline
│ ├── train_lstm.py # DL training pipeline
│ ├── predict.py # Inference on new commands
│ ├── common.py # Shared utilities
│ └── init.py
│
├── data/
│ ├── raw/ # Full dataset (see data/README.md)
│ ├── sample/ # Lightweight sample CSV
│ └── README.md # Dataset access instructions
│
├── requirements.txt # ML dependencies
├── requirements-dl.txt # DL dependencies
├── LICENSE
└── README.md


---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/0xTchomo/mitre-attack-command-classifier.git
cd mitre-attack-command-classifier
pip install -r requirements.txt

