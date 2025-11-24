# ============================================================================
# MODEL: 24.177% - MassiveNumber
# ============================================================================

# MassiveNumber


# ============================================================================
# SETUP AND DATA PREP
# ============================================================================
rm(list = ls())
gc()

library(tidyverse)
library(pROC)
library(xgboost)
library(glmnet)
library(catboost)
library(lubridate)

data <- read.csv('train.csv', stringsAsFactors = FALSE)
Xtr <- read.csv('Xtr.csv', stringsAsFactors = FALSE)
Ytr <- read.csv('Ytr.csv', stringsAsFactors = FALSE)
Xte <- read.csv('Xte.csv', stringsAsFactors = FALSE)
kill_cols <- c(paste0("k", 1:5, "_Red"), paste0("k", 1:5, "_Blue"))

data$date <- as.Date(data$date)
Xtr$date <- as.Date(Xtr$date)
Xte$date <- as.Date(Xte$date)

Xtr <- Xtr[order(Xtr$date), ]
Ytr <- Ytr[match(Xtr$gid, Ytr$gid), ]

categorical_features <- c(
  'tid_Red', 'tid_Blue', 'lid',
  paste0('pid', 1:5, '_Red'), paste0('pid', 1:5, '_Blue'),
  paste0('position', 1:5, '_Red'), paste0('position', 1:5, '_Blue'),
  paste0('champion', 1:5, '_Red'), paste0('champion', 1:5, '_Blue')
)


# ============================================================================
# STATS
# ============================================================================
calculate_historical_stats_full <- function(data, cutoff_date) {
  past_data <- data[data$date < cutoff_date, ]
  
  player_stats <- past_data %>%
    group_by(pid) %>%
    summarise(
      avg_k = mean(k, na.rm = TRUE),
      avg_d = mean(d, na.rm = TRUE),
      avg_a = mean(a, na.rm = TRUE),
      avg_dpm = mean(dpm, na.rm = TRUE),
      avg_gpm = mean(gpm, na.rm = TRUE),
      win_rate = mean(result, na.rm = TRUE),
      games_played = n(),
      .groups = 'drop'
    )
  
  team_stats <- past_data %>%
    group_by(tid) %>%
    summarise(
      team_win_rate = mean(result, na.rm = TRUE),
      team_games_played = n(),
      .groups = 'drop'
    )
  
  champion_stats <- past_data %>%
    group_by(champion) %>%
    summarise(
      champ_win_rate = mean(result, na.rm = TRUE),
      champ_pick_rate = n() / length(unique(past_data$gid)),
      champ_avg_k = mean(k, na.rm = TRUE),
      champ_avg_d = mean(d, na.rm = TRUE),
      champ_avg_a = mean(a, na.rm = TRUE),
      champ_avg_dpm = mean(dpm, na.rm = TRUE),
      champ_avg_gpm = mean(gpm, na.rm = TRUE),
      .groups = 'drop'
    )
  
  league_stats <- past_data %>%
    group_by(lid) %>%
    summarise(
      league_avg_length = mean(gamelength, na.rm = TRUE),
      league_std_length = sd(gamelength, na.rm = TRUE),
      league_avg_kills = mean(k, na.rm = TRUE),
      league_n_games = n(),
      .groups = 'drop'
    ) %>%
    mutate(
      league_competitiveness = scale(league_std_length)[,1] + scale(league_avg_kills)[,1],
      league_tier = cut(league_avg_length, breaks = 5, labels = FALSE)
    )
  
  global_stats <- list(
    global_wr = mean(past_data$result, na.rm = TRUE),
    global_avg_k = mean(past_data$k, na.rm = TRUE),
    global_avg_d = mean(past_data$d, na.rm = TRUE),
    global_avg_a = mean(past_data$a, na.rm = TRUE),
    global_avg_dpm = mean(past_data$dpm, na.rm = TRUE),
    global_avg_gpm = mean(past_data$gpm, na.rm = TRUE)
  )
  
  return(list(
    player = player_stats,
    team = team_stats,
    champion = champion_stats,
    league = league_stats,
    global = global_stats
  ))
}

stats_test <- calculate_historical_stats_full(data, min(Xte$date))


# ============================================================================
# FEATURES
# ============================================================================
create_feature_matrix_complete <- function(X, stats) {
  features <- data.frame(matrix(nrow = nrow(X), ncol = 0))
  
  first_date <- min(c(Xtr$date, Xte$date))
  features$days_since_start <- as.numeric(X$date - first_date)
  features$year <- as.numeric(format(X$date, "%Y"))
  features$month <- as.numeric(format(X$date, "%m"))
  features$year_fraction <- features$year + (as.numeric(format(X$date, "%j")) - 1) / 365
  features$expected_length_by_year <- 2100 - (features$year_fraction - 2018) * 46
  features$expected_kills_by_year <- 2.1 + (features$year_fraction - 2018) * 0.18
  
  league_data <- stats$league[match(X$lid, stats$league$lid), ]
  features$league_avg_length <- league_data$league_avg_length
  features$league_std_length <- league_data$league_std_length
  features$league_competitiveness <- league_data$league_competitiveness
  features$league_tier <- league_data$league_tier
  
  features$league_avg_length[is.na(features$league_avg_length)] <- 1900
  features$league_std_length[is.na(features$league_std_length)] <- 200
  features$league_competitiveness[is.na(features$league_competitiveness)] <- 0
  features$league_tier[is.na(features$league_tier)] <- 3
  
  features$team_Red_winrate <- stats$team$team_win_rate[match(X$tid_Red, stats$team$tid)]
  features$team_Blue_winrate <- stats$team$team_win_rate[match(X$tid_Blue, stats$team$tid)]
  features$team_Red_winrate[is.na(features$team_Red_winrate)] <- stats$global$global_wr
  features$team_Blue_winrate[is.na(features$team_Blue_winrate)] <- stats$global$global_wr
  
  for (side in c("Red", "Blue")) {
    for (i in 1:5) {
      pid_col <- paste0("pid", i, "_", side)
      pid <- X[[pid_col]]
      
      player_data <- stats$player[match(pid, stats$player$pid), ]
      features[[paste0("p", i, "_", side, "_winrate")]] <- player_data$win_rate
      features[[paste0("p", i, "_", side, "_avgk")]] <- player_data$avg_k
      features[[paste0("p", i, "_", side, "_avgd")]] <- player_data$avg_d
      features[[paste0("p", i, "_", side, "_avga")]] <- player_data$avg_a
      features[[paste0("p", i, "_", side, "_avgdpm")]] <- player_data$avg_dpm
      features[[paste0("p", i, "_", side, "_avggpm")]] <- player_data$avg_gpm
      features[[paste0("p", i, "_", side, "_games")]] <- player_data$games_played
      
      champ_col <- paste0("champion", i, "_", side)
      champion <- X[[champ_col]]
      champ_data <- stats$champion[match(champion, stats$champion$champion), ]
      
      features[[paste0("c", i, "_", side, "_winrate")]] <- champ_data$champ_win_rate
      features[[paste0("c", i, "_", side, "_pickrate")]] <- champ_data$champ_pick_rate
      features[[paste0("c", i, "_", side, "_avgk")]] <- champ_data$champ_avg_k
      features[[paste0("c", i, "_", side, "_avgd")]] <- champ_data$champ_avg_d
      features[[paste0("c", i, "_", side, "_avga")]] <- champ_data$champ_avg_a
      features[[paste0("c", i, "_", side, "_avgdpm")]] <- champ_data$champ_avg_dpm
      features[[paste0("c", i, "_", side, "_avggpm")]] <- champ_data$champ_avg_gpm
    }
  }
  
  red_player_wr_cols <- paste0("p", 1:5, "_Red_winrate")
  blue_player_wr_cols <- paste0("p", 1:5, "_Blue_winrate")
  features$Red_avg_player_winrate <- rowMeans(features[, red_player_wr_cols], na.rm = TRUE)
  features$Blue_avg_player_winrate <- rowMeans(features[, blue_player_wr_cols], na.rm = TRUE)
  
  red_player_k_cols <- paste0("p", 1:5, "_Red_avgk")
  blue_player_k_cols <- paste0("p", 1:5, "_Blue_avgk")
  features$Red_avg_player_kills <- rowMeans(features[, red_player_k_cols], na.rm = TRUE)
  features$Blue_avg_player_kills <- rowMeans(features[, blue_player_k_cols], na.rm = TRUE)
  
  red_champ_wr_cols <- paste0("c", 1:5, "_Red_winrate")
  blue_champ_wr_cols <- paste0("c", 1:5, "_Blue_winrate")
  features$Red_avg_champ_winrate <- rowMeans(features[, red_champ_wr_cols], na.rm = TRUE)
  features$Blue_avg_champ_winrate <- rowMeans(features[, blue_champ_wr_cols], na.rm = TRUE)
  
  red_champ_k_cols <- paste0("c", 1:5, "_Red_avgk")
  blue_champ_k_cols <- paste0("c", 1:5, "_Blue_avgk")
  features$Red_avg_champ_kills <- rowMeans(features[, red_champ_k_cols], na.rm = TRUE)
  features$Blue_avg_champ_kills <- rowMeans(features[, blue_champ_k_cols], na.rm = TRUE)
  
  features$total_expected_kills_players <- (
    rowSums(features[, paste0("p", 1:5, "_Red_avgk")], na.rm = TRUE) +
      rowSums(features[, paste0("p", 1:5, "_Blue_avgk")], na.rm = TRUE)
  )
  
  features$total_expected_kills_champs <- (
    rowSums(features[, paste0("c", 1:5, "_Red_avgk")], na.rm = TRUE) +
      rowSums(features[, paste0("c", 1:5, "_Blue_avgk")], na.rm = TRUE)
  )
  
  features$total_expected_kills_avg <- (features$total_expected_kills_players + features$total_expected_kills_champs) / 2
  
  features$wr_diff <- features$Red_avg_player_winrate - features$Blue_avg_player_winrate
  features$team_wr_diff <- features$team_Red_winrate - features$team_Blue_winrate
  features$champ_wr_diff <- features$Red_avg_champ_winrate - features$Blue_avg_champ_winrate
  features$player_kills_diff <- features$Red_avg_player_kills - features$Blue_avg_player_kills
  
  for (col in names(features)) {
    if (is.numeric(features[[col]]) && any(is.na(features[[col]]))) {
      if (grepl("winrate|_wr", col)) {
        features[[col]][is.na(features[[col]])] <- stats$global$global_wr
      } else if (grepl("avgk|kills", col)) {
        features[[col]][is.na(features[[col]])] <- stats$global$global_avg_k
      } else if (grepl("avgd|deaths", col)) {
        features[[col]][is.na(features[[col]])] <- stats$global$global_avg_d
      } else if (grepl("avga|assists", col)) {
        features[[col]][is.na(features[[col]])] <- stats$global$global_avg_a
      } else if (grepl("avgdpm|dpm", col)) {
        features[[col]][is.na(features[[col]])] <- stats$global$global_avg_dpm
      } else if (grepl("avggpm|gpm", col)) {
        features[[col]][is.na(features[[col]])] <- stats$global$global_avg_gpm
      } else {
        features[[col]][is.na(features[[col]])] <- 0
      }
    }
  }
  
  return(features)
}

X_full_features <- create_feature_matrix_complete(Xtr, stats_test)
X_test_features <- create_feature_matrix_complete(Xte, stats_test)


# ============================================================================
# MODELS
# ============================================================================
# Winner:
#--------
winner_finals <- list()
for(seed in c(42, 41, 40, 39)) {
  set.seed(seed)
  logistic_final <- glm(
    winner_Red ~ .,
    data = cbind(winner_Red = Ytr$winner_Red, X_full_features),
    family = binomial(link = "logit")
  )
  winner_finals[[paste0("logistic_", seed)]] <- predict(logistic_final, X_test_features, type = "response")
}

X_full_cat <- Xtr[, categorical_features]
X_full_cat$day_of_year <- yday(Xtr$date)
X_full_cat$month <- month(Xtr$date)

X_test_cat <- Xte[, categorical_features]
X_test_cat$day_of_year <- yday(Xte$date)
X_test_cat$month <- month(Xte$date)

for (col in categorical_features) {
  if (col %in% names(X_full_cat)) {
    all_levels <- unique(c(X_full_cat[[col]], X_test_cat[[col]]))
    X_full_cat[[col]] <- factor(X_full_cat[[col]], levels = all_levels)
    X_test_cat[[col]] <- factor(X_test_cat[[col]], levels = all_levels)
  }
}

cat_indices <- which(names(X_full_cat) %in% categorical_features) - 1
full_pool <- catboost.load_pool(data = X_full_cat, label = Ytr$winner_Red, cat_features = cat_indices)
test_pool <- catboost.load_pool(data = X_test_cat, cat_features = cat_indices)

catboost_final_1 <- catboost.train(
  learn_pool = full_pool,
  params = list(iterations = 1100, learning_rate = 0.01, max_ctr_complexity = 14,
                loss_function = 'Logloss', verbose = 200, random_seed = 42,
                task_type = "CPU", thread_count = 22)
)
winner_finals$catboost_1 <- catboost.predict(catboost_final_1, test_pool, prediction_type = 'Probability')

catboost_final_2 <- catboost.train(
  learn_pool = full_pool,
  params = list(iterations = 1100, learning_rate = 0.01, max_ctr_complexity = 14,
                loss_function = 'Logloss', verbose = 200, random_seed = 123,
                task_type = "CPU", thread_count = 22)
)
winner_finals$catboost_2 <- catboost.predict(catboost_final_2, test_pool, prediction_type = 'Probability')

pred_winner_test_enhanced2 <- (
  0.1 * winner_finals$logistic_42 + 0.1 * winner_finals$logistic_41 +
    0.1 * winner_finals$logistic_40 + 0.1 * winner_finals$logistic_39 +
    0.30 * winner_finals$catboost_1 + 0.30 * winner_finals$catboost_2
)


# Length:
#--------
set.seed(42)
cv_ridge_length <- cv.glmnet(x = as.matrix(X_full_features), y = Ytr$gamelength,
                             type.measure = "mse", alpha = 0, nfolds = 6)
pred_length_test <- predict(cv_ridge_length, as.matrix(X_test_features), s = "lambda.1se")[,1]

set.seed(41)
cv_ridge_length2 <- cv.glmnet(x = as.matrix(X_full_features), y = Ytr$gamelength,
                              type.measure = "mse", alpha = 0, nfolds = 6)
pred_length_test2 <- predict(cv_ridge_length2, as.matrix(X_test_features), s = "lambda.1se")[,1]

set.seed(42)
dtrain_lengtht <- xgb.DMatrix(data = as.matrix(X_full_features), label = Ytr$gamelength)
xgb_length_t <- xgb.train(
  params = list(objective = "reg:squarederror", max_depth = 10, eta = 0.03,
                colsample_bytree = 0.8, subsample = 0.8, nthread=22),
  data = dtrain_lengtht, nrounds = 151, verbose = 0
)
pred_xgb_length_t <- predict(xgb_length_t, as.matrix(X_test_features))

dtrain_lengtht2 <- xgb.DMatrix(data = as.matrix(X_full_features), label = Ytr$gamelength)
xgb_length_t2 <- xgb.train(
  params = list(objective = "reg:squarederror", max_depth = 10, eta = 0.03,
                colsample_bytree = 0.8, subsample = 0.8, nthread=22),
  data = dtrain_lengtht2, nrounds = 151, verbose = 0
)
pred_xgb_length_t2 <- predict(xgb_length_t2, as.matrix(X_test_features))

pred_ensemble_length <- 0.3 * pred_length_test + 0.3 * pred_xgb_length_t +
  0.1 * pred_length_test2 + 0.3 * pred_xgb_length_t2


# Kill:
#--------
submission_2 <- data.frame(gid = Xte$gid)
submission_2$winner_Red <- pred_winner_test_enhanced2
submission_2$gamelength <- pred_ensemble_length

weight_sets <- list(
  w2 = list(c(0,0.6,0.2,0.2), c(0,0.4,0.4,0.2), c(0.1,0.4,0.3,0.2), c(0.1,0.2,0.5,0.2), c(0,0.4,0.4,0.2),
            c(0.1,0.3,0.4,0.2), c(0,0.5,0.3,0.2), c(0.1,0.3,0.4,0.2), c(0.2,0.2,0.4,0.2), c(0,0.7,0.1,0.2)),
  
  w8 = list(c(0,0.6,0.25,0.15), c(0,0.45,0.35,0.2), c(0.05,0.5,0.25,0.2), c(0.05,0.25,0.45,0.25), c(0,0.65,0.25,0.1),
            c(0.05,0.4,0.35,0.2), c(0,0.55,0.3,0.15), c(0.05,0.4,0.35,0.2), c(0.2,0.3,0.35,0.15), c(0,0.75,0.15,0.1))
)

set.seed(42)
for(version in c(2, 8)) {
  pred_kills_test_enhanced <- matrix(0, nrow = nrow(X_test_features), ncol = 10)
  
  for(i in 1:10) {
    dtrain_k_final <- xgb.DMatrix(as.matrix(X_full_features), label = Ytr[[kill_cols[i]]])
    
    xgb_mse <- xgb.train(
      params = list(objective = "reg:squarederror", max_depth = 4, eta = 0.05,
                    subsample = 0.8, colsample_bytree = 0.8, nthread=22),
      data = dtrain_k_final, nrounds = 151, verbose = 0
    )
    
    xgb_poisson <- xgb.train(
      params = list(objective = "count:poisson", max_depth = 4, eta = 0.05,
                    subsample = 0.8, colsample_bytree = 0.8, nthread=22),
      data = dtrain_k_final, nrounds = 151, verbose = 0
    )
    
    ridge <- cv.glmnet(as.matrix(X_full_features), Ytr[[kill_cols[i]]], alpha = 0, nfolds = 5)
    
    xgb_poisson2 <- xgb.train(
      params = list(objective = "count:poisson", max_depth = 4, eta = 0.05,
                    subsample = 0.7, colsample_bytree = 0.7, nthread=22),
      data = dtrain_k_final, nrounds = 151, verbose = 0
    )
    
    pred1 <- predict(xgb_mse, as.matrix(X_test_features))
    pred2 <- predict(xgb_poisson, as.matrix(X_test_features))
    pred3 <- predict(ridge, as.matrix(X_test_features), s = "lambda.min")[,1]
    pred4 <- predict(xgb_poisson2, as.matrix(X_test_features))
    
    w <- weight_sets[[paste0("w", version)]][[i]]
    pred_kills_test_enhanced[, i] <- w[1]*pred1 + w[2]*pred2 + w[3]*pred3 + w[4]*pred4
  }
  
  for (i in 1:10) {
    submission_2[[kill_cols[i]]] <- pred_kills_test_enhanced[, i]
  }
  
  write.csv(submission_2, paste0("BIGWIN_v", version, ".csv"), row.names = FALSE)
}


# ============================================================================
# Submitted file: v10 = (v2 + v8) / 2
# ============================================================================
v2 <- read.csv("BIGWIN_v2.csv")
v8 <- read.csv("BIGWIN_v8.csv")

v10 <- data.frame(gid = Xte$gid)
v10$winner_Red <- (v2$winner_Red + v8$winner_Red) / 2
for(i in 1:10) {
  v10[[kill_cols[i]]] <- (v2[[kill_cols[i]]] + v8[[kill_cols[i]]]) / 2
}
v10$gamelength <- (v2$gamelength + v8$gamelength) / 2

write.csv(v10, "BIGWIN_v10t.csv", row.names = FALSE)
