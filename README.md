# LoL Esports Prediction – MassiveNumber Model 

<p align="center">
  <img src="47aca76b-f395-45ce-8305-87261b5b2fb0.png" width="75%" />
</p>

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

8. Author

Dahyeon Choi
Data Science, Simon Fraser University
Oct 2025
