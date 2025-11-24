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

### **Train (player-level)**
- `pid`, `tid`, `lid`
- `position`, `champion`, `side`
- `k`, `d`, `a`, `dpm`, `gpm`
- `gamelength`
- `date` (2018–2022)
- `result` (1 = win, 0 = loss)

### **Game-level design matrices**
- `Xtr.csv` – game structure (teams, 10 players, champions, league id, date)
- `Ytr.csv` – game-level labels (winner, kills, game length)
- `Xte.csv` – test set structure

---

## 3. Feature Engineering  
All features were computed using **only matches prior to each test match** to avoid leakage.

### **3.1 Temporal Features**
- Days since dataset start  
- Year, month, day of year  
- Expected game length by season  
- Expected kill rate by season  

### **3.2 League-Level Stats**
- Average game length  
- Game length variance  
- Average kills  
- Competitiveness score  
- Tier category (5 bins)

### **3.3 Team Stats**
- Team win rate  
- Team games played  
- Missing teams → imputed with global averages  

### **3.4 Player Stats**
- K/D/A  
- DPM, GPM  
- Win rate  
- Games played  

### **3.5 Champion Stats**
- Pick rate  
- Win rate  
- K/D/A averages  
- DPM, GPM averages  

### **3.6 Game-Level Aggregates**
- Average player win rate per team  
- Average champion win rate per team  
- Expected kills (player-based + champion-based)  

### **3.7 Differential Features**
- winrate diff (Red − Blue)  
- team winrate diff  
- champion winrate diff  
- expected kills diff  

---

## 4. Modeling

### **4.1 Winner Prediction (Binary Classification)**  
**Models**
- Logistic Regression (4 seeds)  
- CatBoost (2 seeds)  

**Ensemble**
- 40% Logistic  
- 60% CatBoost  

---

### **4.2 Game Length Prediction (Regression)**
**Models**
- Ridge Regression  
- XGBoost  

**Ensemble**
- Ridge + XGBoost  

---

### **4.3 Player Kill Prediction (10 regressions)**
For each of 10 players:

**Models**
- XGBoost (MSE)  
- XGBoost (Poisson)  
- Ridge Regression  

**Ensemble**
- Custom weights per position  

---

## 5. Final Submission

Two versions (v2, v8) were generated using different ensemble weights.  
The final file was:
v10 = (v2 + v8) / 2

This achieved **24.177%** on the competition leaderboard.

---

## 6. File Structure
```txt
lol-esports-prediction/
└── src/
    └── UpdatedMassiveNumber.R   # Full modeling pipeline

7. Run Instructions
Install required R packages:
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
BIGWIN_v10t.csv   (final submission)


Why This Project Matters

League of Legends match prediction is much more than a game-related task.
This project demonstrates practical, real-world machine learning skills used in modern data science roles.

1. Realistic large-scale data complexity

Professional LoL esports produces thousands of matches and tens of thousands of player–team–champion combinations.
This dataset behaves like real sports analytics or financial prediction data, with high dimensionality, temporal variation, and non-linear interactions.

2. Full end-to-end ML pipeline

This project covers every major ML component in one system:

Binary classification (winner)

Regression (game length)

Multi-output regression (10 player kills)

Feature engineering

Categorical modeling

Time-based modeling and leakage prevention

Ensemble blending (Logistic + CatBoost + XGBoost + Ridge)

3. Modeling meta changes over time (meta drift)

League patches change champion balance, game tempo, kill frequency, and strategy every year.
The model accounts for this using:

Year fraction

Expected kills by season

Expected match duration by season

4. High-cardinality categorical modeling

Players, teams, leagues, champions — hundreds of unique categorical values.
CatBoost and engineered aggregations capture:

Player skill history

Team performance consistency

Champion pick/win trends

League differences

5. Industrial relevance

This modeling structure is identical to what is used in:

Esports analytics companies

Sports prediction platforms

Betting probability engines

Game analytics teams (including Riot Games)

6. Time-aware feature construction (no leakage)

Only past matches before each test date were used.
This ensures correct chronological separation and prevents future information from leaking into training.

One-Sentence Summary

A production-style machine learning system that handles temporal drift, high-cardinality data, structured feature engineering, and multi-task prediction—far beyond a simple game model.

8. Author

Dahyeon Choi
Data Science, Simon Fraser University
Oct 2025
---
