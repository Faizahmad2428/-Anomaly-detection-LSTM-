

# ----------------------------
# 📦 Load Libraries
# ----------------------------
library(tidyverse)
library(lubridate)
library(dplyr)
library(keras)
library(tensorflow)

# ----------------------------
# 🧼 Load and Preprocess Data
# ----------------------------
data <- read.delim("E:/Umer/University/sem4/AS/pbl/household_power_2.txt", sep=";", header = TRUE)

# Convert Date + Time to datetime
data$datetime <- dmy_hms(paste(data$Date, data$Time))

# Convert columns to numeric
cols_to_numeric <- c("Global_active_power", "Global_reactive_power", "Voltage", 
                     "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3")
data[cols_to_numeric] <- lapply(data[cols_to_numeric], function(x) as.numeric(as.character(x)))

# Remove NA
data <- na.omit(data)

# Normalize selected columns
ts_data <- data %>% select(all_of(cols_to_numeric))
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
ts_data_norm <- as.data.frame(lapply(ts_data, normalize))

# ----------------------------
# ✂️ Split Data
# ----------------------------
set.seed(42)
total_rows <- nrow(ts_data_norm)
train_index <- sample(1:total_rows, size = 0.8 * total_rows)

train_data <- as.matrix(ts_data_norm[train_index, ])
test_data  <- as.matrix(ts_data_norm[-train_index, ])

# ----------------------------
# 🧠 Build Autoencoder Model
# ----------------------------
input_dim <- ncol(train_data)

model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', input_shape = input_dim) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = input_dim, activation = 'sigmoid')

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam'
)

# ----------------------------
# 🏋️ Train the Model
# ----------------------------
history <- model %>% fit(
  train_data,
  train_data,
  epochs = 50,
  batch_size = 1024,         # large batch size for speed
  validation_split = 0.1,
  verbose = 1
)

# -----------------------------
# Save model
# -----------------------------
save_model_hdf5(model, "household_power_lstm_autoencoder.h5")

# Save normalization params (min/max) for inverse transform & UI use
norm_params <- data.frame(
  feature = cols_to_numeric,
  min = sapply(ts_data, min),
  max = sapply(ts_data, max)
)
saveRDS(history, "train_history.rds")

sampled <- ts_data_norm[-train_index, ][1:100, ]
write.csv(sampled, "sampled_test_data.csv", row.names = FALSE)
write.csv(norm_params, "norm_params.csv", row.names = FALSE)