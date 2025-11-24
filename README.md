# LoL Esports Prediction – MassiveNumber Model 

This project builds a full game-level prediction system for professional League of Legends (LoL) matches using historical player, team, champion, and league statistics.  
It was developed for the STAT 440 “Learning From Big Data” competition and achieved a leaderboard score of **24.177%**.

The model generates:
- **Winner prediction** (probability Red wins)
- **Game length prediction** (seconds)
- **Player kill predictions** (10 players per match)

All modeling code is in:  
`src/UpdatedMassiveNumber.R`

---

## 1. Project Overview

The competition provided multi-year match data (2018–2022) and required predicting outcomes for 2023 matches.  
Because the test set contained only match structure (teams, players, champions, league), the core challenge was **building accurate historical statistics** and **engineering game-level features** using only data before each test match date.

### Tasks
1. Predict the probability that **Red side wins**
2. Predict **10 kill counts** (5 Red, 5 Blue)
3. Predict **game length** (seconds)

---

## 2. Data Used

The competition dataset included:

### **Train (player-level)**
- `pid`, `tid`, `lid`
- `position`, `champion`, `side`
- `k`, `d`, `a`, `dpm`, `gpm`
- `gamelength`
- `date` (2018–2022)
- `result` (1 = win, 0 = loss)

### **Game-level design matrices**
- `Xtr.csv` – game structure (teams, 10 players, 10 champions, league id, date)
- `Ytr.csv` – game-level labels (winner, total kills, game length)
- `Xte.csv` – test set structure

---

## 3. Feature Engineering

All features were computed using **only matches prior to each test match** to avoid leakage.

### **3.1 Temporal Features**
- Days since dataset start
- Year, month, day of year
- Meta-based expected game length  
- Meta-based expected kill rate  
  (captures changes in game pace over seasons)

### **3.2 League-Level Stats**
- average game length  
- variance in game length  
- average kills  
- league competitiveness score (scaled)
- league tier (5-level discretized)

### **3.3 Team Stats**
- team win rate  
- team games played  
(missing teams imputed with global averages)

### **3.4 Player Stats**
Per-player historical averages:
- kills, deaths, assists  
- damage per minute, gold per minute  
- win rate  
- games played

### **3.5 Champion Stats**
- pick rate  
- win rate  
- kills / deaths / assists  
- DPM / GPM averages

### **3.6 Game-Level Aggregates**
Computed for both teams:
- average player win rate  
- average player kill stats  
- average champion win rate  
- expected kills (player-based and champion-based)

### **3.7 Differential Features**
Key predictors:
- player winrate diff (Red − Blue)
- team winrate diff
- champion winrate diff
- expected kills diff

---

## 4. Modeling

The project uses **independent models for winner, length, and kills**, then ensembles them.

### **4.1 Winner Prediction (Binary Classification)**

**Models**
- Logistic Regression (glm) using numeric features  
- CatBoost using categorical features (teams, players, champs, positions)

**Ensemble**
- 40% Logistic (4 random seeds)
- 60% CatBoost (2 random seeds)

---

### **4.2 Game Length Prediction (Regression)**

**Models**
- Ridge Regression (glmnet)
- XGBoost Regression (reg:squarederror)

**Ensemble**
- Ridge + XGBoost (multiple seeds)

---

### **4.3 Player Kill Prediction (10 regressions)**

For each player (1–5 Red, 1–5 Blue):

**Models**
- XGBoost (MSE)
- XGBoost (Poisson)
- Ridge

Custom weight sets were tuned per position.

---

## 5. Final Submission

Two full pipelines (v2 and v8) were generated using different ensemble weights.  
The final competition file was:
v10 = (v2 + v8) / 2

This achieved the **24.177%** competition score.

---

## 6. File Structure
lol-esports-prediction/
└── src/
    └── UpdatedMassiveNumber.R # Full modeling pipeline

---

## 7. Run Instructions

Install required R packages:

```r
install.packages(c(
  "tidyverse", "pROC", "xgboost",
  "glmnet", "catboost", "lubridate"
))
Place data files in the working directory:
train.csv
Xtr.csv
Ytr.csv
Xte.csv

Run:
source("src/UpdatedMassiveNumber.R")

Outputs:
BIGWIN_v2.csv
BIGWIN_v8.csv
BIGWIN_v10t.csv (final submission)

## Why This Project Matters

League of Legends match prediction is much more than a game-related task.  
This project demonstrates practical, real-world machine learning skills used in modern data science roles.

### 1. Realistic large-scale data complexity
Professional LoL esports produces thousands of matches and tens of thousands of player–team–champion combinations.  
This dataset behaves like real sports analytics or financial prediction data, with high dimensionality, temporal variation, and non-linear interactions.

### 2. Full end-to-end ML pipeline
This project covers every major ML component in one system:
- Binary classification (winner)
- Regression (game length)
- Multi-output regression (10 player kills)
- Feature engineering
- Categorical modeling
- Time-based modeling and leakage prevention
- Ensemble blending (logistic + CatBoost + XGBoost + Ridge)

This mirrors actual industry-level machine learning workflows.

### 3. Modeling meta changes over time (meta drift)
League patches change champion balance, game tempo, kill frequency, and strategy every year.  
To handle this, the model integrates:
- Year fraction  
- Expected kills by season  
- Expected game length by season  

This shows the ability to model **non-stationary data**, a core challenge in real-world ML systems such as finance, forecasting, and recommendations.

### 4. High-cardinality categorical modeling
Players, teams, leagues, champions — hundreds of unique categorical values.  
CatBoost and engineered aggregations allow the model to capture:
- Player skill history  
- Team performance consistency  
- Champion pick/win trends  
- League differences  

High-cardinality categorical modeling is essential in ecommerce, sports analytics, and personalization.

### 5. Industrial relevance
This modeling structure is the same as what is used in:
- Esports analytics companies  
- Sports prediction platforms  
- Betting probability engines  
- Game analytics teams (including Riot Games)

The project demonstrates the ability to build **production-grade predictive systems** on real, evolving data.

### 6. Correct time-aware feature construction (no leakage)
Only past matches before each test date were used for historical statistics.  
This ensures:
- Proper chronological separation  
- Realistic generalization  
- No future information leakage  

Leakage prevention is one of the most important skills in applied machine learning.

---

### **One-Sentence Summary**
**This project is a fully engineered, production-style machine learning system that handles temporal drift, high-cardinality data, structured feature engineering, and multi-task prediction—far beyond a simple game model.**


9. Author

Dahyeon Choi
Data Science, Simon Fraser University
Oct 2025
