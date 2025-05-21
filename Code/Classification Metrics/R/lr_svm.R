# Start timing the process
start_time <- Sys.time()

# Load necessary libraries
library(readr)
library(ggplot2)
library(glmnet)
library(e1071)
library(caret)
library(pROC)
library(dplyr) 
library(ROSE)
library(randomForest)

# Read the CSV file
df <- read_csv("diabetes.csv")

# Convert the target variable 'Outcome' to a factor
df$Outcome <- as.factor(df$Outcome)

# Define features (X) and target variable (y)
X <- df[, -ncol(df)]  # All columns except the last one (features)
y <- df$Outcome       # The last column (target/label) as a factor

# Split the data into training and testing sets
set.seed(42)
sample_indices <- sample(seq_len(nrow(df)), size = 0.2 * nrow(df))

X_train <- X[-sample_indices, ]  # 80% of data for training
X_test <- X[sample_indices, ]    # 20% of data for testing
y_train <- y[-sample_indices]    # Corresponding y values for training
y_test <- y[sample_indices]      # Corresponding y values for testing

# Function to calculate entropy for a probability vector
calculate_entropy <- function(p) {
  return(-p * log2(p) - (1 - p) * log2(1 - p))
}

# Function to calculate purity (proportion of correct classifications)
calculate_purity <- function(actual, predicted) {
  return(sum(actual == predicted) / length(actual))
}

# ---------------------- LOGISTIC REGRESSION ----------------------

# Train Logistic Regression model using glmnet
log_clf <- glmnet(as.matrix(X_train), y_train, family = "binomial", maxit = 1000)

# Select a specific lambda value (e.g., the first one or any other by index)
selected_lambda <- log_clf$lambda[1]

# Predict using the selected lambda value on training data
log_pred_train <- predict(log_clf, as.matrix(X_train), s = selected_lambda, type = "response")
log_pred_train <- as.vector(log_pred_train)

# Predict using the selected lambda value on test data
log_pred_test <- predict(log_clf, as.matrix(X_test), s = selected_lambda, type = "response")
log_pred_test <- as.vector(log_pred_test)

# Binarize predictions using the manual threshold
manual_threshold <- 0.5
log_pred_train_bin <- ifelse(log_pred_train > manual_threshold, 1, 0)
log_pred_test_bin <- ifelse(log_pred_test > manual_threshold, 1, 0)

# Force the levels of predictions and actual labels to align
log_pred_train_bin <- factor(log_pred_train_bin, levels = c(0, 1))
log_pred_test_bin <- factor(log_pred_test_bin, levels = c(0, 1))

# Confusion matrix and metrics for Logistic Regression (Test)
conf_matrix_log_test <- confusionMatrix(log_pred_test_bin, y_test)
accuracy_log_test <- conf_matrix_log_test$overall["Accuracy"]
error_rate_log_test <- 1 - accuracy_log_test
precision_log <- conf_matrix_log_test$byClass["Precision"]
recall_log <- conf_matrix_log_test$byClass["Recall"]
f_measure_log <- conf_matrix_log_test$byClass["F1"]
specificity_log <- conf_matrix_log_test$byClass["Specificity"]
purity_log_test <- calculate_purity(y_test, log_pred_test_bin)

# Confusion matrix and metrics for Logistic Regression (Training)
conf_matrix_log_train <- confusionMatrix(log_pred_train_bin, y_train)
accuracy_log_train <- conf_matrix_log_train$overall["Accuracy"]
error_rate_log_train <- 1 - accuracy_log_train
purity_log_train <- calculate_purity(y_train, log_pred_train_bin)

# Calculate average entropy for Logistic Regression (Test)
entropy_log_test <- mean(calculate_entropy(log_pred_test))

# Calculate Generalization Error for Logistic Regression
generalization_error_log <- abs(error_rate_log_train - error_rate_log_test)

# Print Logistic Regression metrics in requested format
cat("--- Logistic Regression Metrics ---\n")
cat("Logistic Regression Accuracy (Test): ", accuracy_log_test, "\n")
cat("Logistic Regression Error Rate (Test): ", error_rate_log_test, "\n")
cat("Logistic Regression Precision: ", precision_log, "\n")
cat("Logistic Regression Recall: ", recall_log, "\n")
cat("Logistic Regression F1 Score (F-Measure): ", f_measure_log, "\n")
cat("Logistic Regression Specificity: ", specificity_log, "\n")
cat("Logistic Regression Sensitivity: ", recall_log, "\n")
cat("Logistic Regression Purity (Test): ", purity_log_test, "\n")
cat("Logistic Regression Average Entropy (Test): ", entropy_log_test, "\n")

# Print training error for Logistic Regression
cat("--- Logistic Regression Training Error ---\n")
cat("Logistic Regression Accuracy (Training): ", accuracy_log_train, "\n")
cat("Logistic Regression Error Rate (Training): ", error_rate_log_train, "\n")
cat("Logistic Regression Purity (Training): ", purity_log_train, "\n")

# Print Generalization Error for Logistic Regression
cat("--- Logistic Regression Generalization Error ---\n")
cat("Logistic Regression Generalization Error: ", generalization_error_log, "\n")

# ---------------------- LIFT CHART for Logistic Regression ----------------------

# Function to calculate lift
calculate_lift <- function(actuals, predicted_probs, groups = 10) {
  df <- data.frame(actuals = actuals, predicted_probs = predicted_probs)
  df <- df[order(-df$predicted_probs), ]  # Sort by predicted probabilities (descending)
  
  # Correct decile grouping using dplyr's ntile() function
  df$group <- ntile(df$predicted_probs, groups)
  
  lift_data <- df %>%
    group_by(group) %>%
    summarise(actual_positives = sum(actuals == 1), 
              total = n(), 
              lift = sum(actuals == 1) / (sum(actuals == 1) / length(actuals)))
  
  lift_data$cumulative_lift <- cumsum(lift_data$actual_positives) / cumsum(lift_data$total)
  return(lift_data)
}

# Calculate lift for Logistic Regression
lift_log <- calculate_lift(y_test, log_pred_test)

# Plot Lift Chart for Logistic Regression
ggplot(lift_log, aes(x = group, y = cumulative_lift)) +
  geom_line(color = "blue", size = 1) +
  ggtitle("Lift Chart - Logistic Regression") +
  xlab("Decile Group") + ylab("Cumulative Lift") +
  theme_minimal()

# ---------------------- ROC Curve for Logistic Regression ----------------------

roc_curve_log <- roc(y_test, as.numeric(log_pred_test))

# Plot the ROC curve for Logistic Regression
plot(roc_curve_log, col = "blue", lwd = 2, main = "ROC Curve - Logistic Regression", 
     xlab = "1 - Specificity (False Positive Rate)", ylab = "Sensitivity (Recall)")
grid()


# ---------------------- SUPPORT VECTOR MACHINE (SVM) ----------------------

# Apply ROSE to balance the classes in the training set
train_data <- cbind(X_train, Outcome = y_train)
train_data_rose <- ROSE(Outcome ~ ., data = train_data, seed = 1)$data

# Separate features and target after ROSE
X_train_rose <- train_data_rose[, -ncol(train_data_rose)]
y_train_rose <- train_data_rose$Outcome

# Ensure the target variable is treated as a factor after ROSE
y_train_rose <- as.factor(y_train_rose)

# Fix factor levels for SVM compatibility
levels(y_train_rose) <- c("Class0", "Class1")
levels(y_test) <- c("Class0", "Class1")

# Standardize the features (scaling is important for SVM performance)
scaler <- preProcess(X_train_rose, method = c("center", "scale"))
X_train_scaled <- predict(scaler, X_train_rose)
X_test_scaled <- predict(scaler, X_test)

# Perform Recursive Feature Elimination (RFE) for feature selection
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
results <- rfe(X_train_scaled, y_train_rose, sizes = c(1:5), rfeControl = control)

# Use the best features selected by RFE
best_features <- results$optVariables
X_train_scaled <- X_train_scaled[, best_features]
X_test_scaled <- X_test_scaled[, best_features]

# Ensure the feature columns are aligned
X_train_scaled <- as.data.frame(X_train_scaled)
X_test_scaled <- as.data.frame(X_test_scaled)
colnames(X_train_scaled) <- best_features
colnames(X_test_scaled) <- best_features

# Set up cross-validation and expanded tuning grid for SVM
tune_grid <- expand.grid(C = 10^(-3:3), sigma = c(0.0001, 0.001, 0.01, 0.1, 1, 10))

control <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Train the SVM model using caret with expanded tuning
svm_tuned <- train(x = X_train_scaled, y = y_train_rose, 
                   method = "svmRadial", 
                   tuneGrid = tune_grid, 
                   trControl = control)

# Predict on the training set to calculate training error
svm_train_pred_prob <- predict(svm_tuned, X_train_scaled, type = "prob")[, 2]

# Binarize training predictions using the manual threshold
svm_train_pred_bin <- ifelse(svm_train_pred_prob > manual_threshold, "Class1", "Class0")

# Confusion matrix and metrics for training data
conf_matrix_svm_train <- confusionMatrix(factor(svm_train_pred_bin), y_train_rose)

accuracy_svm_train <- conf_matrix_svm_train$overall["Accuracy"]
error_rate_svm_train <- 1 - accuracy_svm_train  # Training Error Rate

# Predict using the best tuned SVM model on test data
svm_pred_prob <- predict(svm_tuned, X_test_scaled, type = "prob")[, 2]

# Adjust the classification threshold
manual_threshold <- 0.4  # You can adjust this and test different values
svm_pred_test_bin <- ifelse(svm_pred_prob > manual_threshold, "Class1", "Class0")

# Confusion matrix and metrics for test data
conf_matrix_svm <- confusionMatrix(factor(svm_pred_test_bin), y_test)

accuracy_svm <- conf_matrix_svm$overall["Accuracy"]
precision_svm <- conf_matrix_svm$byClass["Precision"]
recall_svm <- conf_matrix_svm$byClass["Recall"]
f_measure_svm <- conf_matrix_svm$byClass["F1"]
specificity_svm <- conf_matrix_svm$byClass["Specificity"]
sensitivity_svm <- recall_svm
error_rate_svm <- 1 - accuracy_svm  # Test Error Rate
purity_svm <- calculate_purity(y_test, svm_pred_test_bin)  # Calculate Purity

# Calculate average entropy for SVM on test data
entropy_svm <- mean(calculate_entropy(svm_pred_prob))

# Calculate Generalization Error
generalization_error_svm <- abs(error_rate_svm_train - error_rate_svm)

# Print SVM metrics in requested format
cat("--- SVM Metrics ---\n")
cat("SVM Accuracy (Test): ", accuracy_svm, "\n")
cat("SVM Error Rate (Test): ", error_rate_svm, "\n")
cat("SVM Precision: ", precision_svm, "\n")
cat("SVM Recall: ", recall_svm, "\n")
cat("SVM F1 Score (F-Measure): ", f_measure_svm, "\n")
cat("SVM Specificity: ", specificity_svm, "\n")
cat("SVM Sensitivity: ", recall_svm, "\n")
cat("SVM Purity: ", purity_svm, "\n")
cat("SVM Average Entropy (Test): ", entropy_svm, "\n")

# Print training error
cat("--- SVM Training Error ---\n")
cat("SVM Accuracy (Training): ", accuracy_svm_train, "\n")
cat("SVM Error Rate (Training): ", error_rate_svm_train, "\n")

# Print Generalization Error
cat("--- SVM Generalization Error ---\n")
cat("SVM Generalization Error: ", generalization_error_svm, "\n")
# ---------------------- LIFT CHART for SVM ----------------------

# Function to calculate lift
calculate_lift <- function(actuals, predicted_probs, groups = 10) {
  df <- data.frame(actuals = actuals, predicted_probs = predicted_probs)
  df <- df[order(-df$predicted_probs), ]  # Sort by predicted probabilities (descending)
  
  df$group <- ntile(df$predicted_probs, groups)
  
  lift_data <- df %>%
    group_by(group) %>%
    summarise(actual_positives = sum(actuals == "Class1"), 
              total = n(), 
              lift = sum(actuals == "Class1") / (sum(actuals == "Class1") / length(actuals)))
  
  lift_data$cumulative_lift <- cumsum(lift_data$actual_positives) / cumsum(lift_data$total)
  return(lift_data)
}

# Calculate lift for SVM
lift_svm <- calculate_lift(y_test, svm_pred_prob)

# Plot Lift Chart for SVM
ggplot(lift_svm, aes(x = group, y = cumulative_lift)) +
  geom_line(color = "green", size = 1) +
  ggtitle("Lift Chart - SVM") +
  xlab("Decile Group") + ylab("Cumulative Lift") +
  theme_minimal()

# ---------------------- ROC Curve for SVM ----------------------

roc_curve_svm <- roc(y_test, svm_pred_prob)

# Plot the ROC curve for SVM
plot(roc_curve_svm, col = "green", lwd = 2, main = "ROC Curve - SVM", 
     xlab = "1 - Specificity (False Positive Rate)", ylab = "Sensitivity (Recall)")
grid()

# End timing the process
end_time <- Sys.time()
cat("--- Platform Comparison Metrics ---\n")
cat("Total Build/Execution Time: ", end_time - start_time, "\n")

