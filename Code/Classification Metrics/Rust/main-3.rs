use csv::ReaderBuilder;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::{precision, recall};
use smartcore::svm::svc::{SVC, SVCParameters};
use std::error::Error;
use std::time::{Instant, SystemTime};
use std::fs::File;
use std::io::{self, BufRead};
use rand::{seq::SliceRandom, SeedableRng, rngs::StdRng};

fn main() -> Result<(), Box<dyn Error>> {
    // Record the overall build start time (for platform comparison)
    let build_start = SystemTime::now();

    // Load dataset from a CSV file
    let file_path = "src/diabetes.csv";
    let mut reader = ReaderBuilder::new().from_path(file_path)?;

    let mut records = vec![];
    // Parse each row into floating point numbers and collect all rows into `records`
    for result in reader.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().map(|x| x.parse::<f64>().unwrap()).collect();
        records.push(row);
    }

    // Split data into features and labels
    let n_features = records[0].len() - 1;
    let n_samples = records.len();

    let mut features = vec![0.0; n_samples * n_features];
    let mut targets = vec![];

    // Separate features and targets from the dataset
    for (i, row) in records.iter().enumerate() {
        for j in 0..n_features {
            features[i * n_features + j] = row[j];
        }
        targets.push(row[n_features] as f64);
    }

    // Balance the dataset by undersampling the majority class
    standardize_features(&mut features, n_features);
    undersample_majority_class(&mut features, &mut targets, 0.0);

    // Recalculate number of samples after balancing, if necessary
    let n_samples_balanced = features.len() / n_features;
    let train_size = (0.8 * n_samples_balanced as f64) as usize; // Use 80% of the data for training

    let train_x = DenseMatrix::from_array(train_size, n_features, &features[..train_size * n_features]);
    let test_x = DenseMatrix::from_array(n_samples_balanced - train_size, n_features, &features[train_size * n_features..]);
    let train_y = targets[..train_size].to_vec();
    let test_y = targets[train_size..n_samples_balanced].to_vec();

    // --------------- LOGISTIC REGRESSION ---------------
    let start_lr = Instant::now(); // Start timer for logistic regression model training

    // Train logistic regression model on the training data
    let lr_model = LogisticRegression::fit(&train_x, &train_y, Default::default())?;
    let lr_duration = start_lr.elapsed(); // End timer

    // Measure prediction time for logistic regression
    let start_lr_predict = Instant::now();
    let lr_predictions = lr_model.predict(&test_x)?;
    let lr_predict_duration = start_lr_predict.elapsed();

    // Calculate accuracy, precision, recall, and F1 score for logistic regression
    let lr_accuracy = lr_predictions.iter().zip(&test_y)
        .filter(|(pred, target)| pred == target)
        .count() as f64 / test_y.len() as f64;
    let lr_precision = precision(&test_y, &lr_predictions);
    let lr_recall = recall(&test_y, &lr_predictions);
    let lr_f1 = 2.0 * (lr_precision * lr_recall) / (lr_precision + lr_recall);

    // Add Error Rate
    let lr_error_rate = 1.0 - lr_accuracy;

    // Add Specificity (TN / (TN + FP))
    let lr_specificity = specificity(&test_y, &lr_predictions);

    // Calculate Training Error manually
    let lr_train_predictions = lr_model.predict(&train_x)?;
    let lr_training_error = 1.0 - (lr_train_predictions.iter().zip(&train_y)
        .filter(|(pred, target)| pred == target)
        .count() as f64 / train_y.len() as f64);

    // Calculate Rand Statistic
    let lr_rand_statistic = calculate_rand_statistic(&lr_predictions, &test_y);

    // Calculate Jaccard Coefficient
    let lr_jaccard = calculate_jaccard_coefficient(&lr_predictions, &test_y);

    // Calculate Confidence Interval for Accuracy
    let lr_ci = calculate_confidence_interval(lr_accuracy, test_y.len());

    // Print logistic regression metrics
    println!("\n--- Logistic Regression Metrics ---");
    println!("Logistic Regression Accuracy: {:.2}%", lr_accuracy * 100.0);
    println!("Logistic Regression Error Rate: {:.2}%", lr_error_rate * 100.0);
    println!("Logistic Regression Precision: {:.2}", lr_precision);
    println!("Logistic Regression Recall (Sensitivity): {:.2}", lr_recall);
    println!("Logistic Regression F1 Score (F-Measure): {:.2}", lr_f1);
    println!("Logistic Regression Specificity: {:.2}", lr_specificity);
    println!("Logistic Regression Training Error: {:.2}%", lr_training_error * 100.0);
    println!("Logistic Regression Rand Statistic: {:.2}", lr_rand_statistic);
    println!("Logistic Regression Jaccard Coefficient: {:.2}", lr_jaccard);
    println!("Logistic Regression 95% Confidence Interval for Accuracy: ({:.2}%, {:.2}%)", lr_ci.0 * 100.0, lr_ci.1 * 100.0);
    println!("Logistic Regression Training Time: {:.2?}", lr_duration);
    println!("Logistic Regression Prediction Time: {:.2?}", lr_predict_duration);

    // --------------- SVM ---------------
    let start_svm = Instant::now(); // Start timer for SVM model training

    // Set SVM parameters and train the SVM model on the training data
    let svm_params = SVCParameters::default().with_c(1.0);  // Adjust the regularization parameter (C)
    let svm_model = SVC::fit(&train_x, &train_y, svm_params)?;
    let svm_duration = start_svm.elapsed(); // End timer

    // Measure prediction time for SVM
    let start_svm_predict = Instant::now();
    let svm_predictions = svm_model.predict(&test_x)?;
    let svm_predict_duration = start_svm_predict.elapsed();

    // Calculate accuracy, precision, recall, and F1 score for SVM
    let svm_accuracy = svm_predictions.iter().zip(&test_y)
        .filter(|(pred, target)| pred == target)
        .count() as f64 / test_y.len() as f64;
    let svm_precision = precision(&test_y, &svm_predictions);
    let svm_recall = recall(&test_y, &svm_predictions);

    // Handle case where precision or recall is zero to avoid NaN
    let svm_f1 = if svm_precision + svm_recall == 0.0 {
        0.0
    } else {
        2.0 * (svm_precision * svm_recall) / (svm_precision + svm_recall)
    };

    // Add Error Rate for SVM
    let svm_error_rate = 1.0 - svm_accuracy;

    // Add Specificity for SVM
    let svm_specificity = specificity(&test_y, &svm_predictions);

    // Calculate Training Error manually for SVM
    let svm_train_predictions = svm_model.predict(&train_x)?;
    let svm_training_error = 1.0 - (svm_train_predictions.iter().zip(&train_y)
        .filter(|(pred, target)| pred == target)
        .count() as f64 / train_y.len() as f64);

    // Calculate Rand Statistic
    let svm_rand_statistic = calculate_rand_statistic(&svm_predictions, &test_y);

    // Calculate Jaccard Coefficient
    let svm_jaccard = calculate_jaccard_coefficient(&svm_predictions, &test_y);

    // Calculate Confidence Interval for Accuracy
    let svm_ci = calculate_confidence_interval(svm_accuracy, test_y.len());

    // Print SVM metrics
    println!("\n--- SVM Metrics ---");
    println!("SVM Accuracy: {:.2}%", svm_accuracy * 100.0);
    println!("SVM Error Rate: {:.2}%", svm_error_rate * 100.0);
    println!("SVM Precision: {:.2}", svm_precision);
    println!("SVM Recall: {:.2}", svm_recall);
    println!("SVM F1 Score (F-Measure): {:.2}", svm_f1);
    println!("SVM Specificity: {:.2}", svm_specificity);
    println!("SVM Training Error: {:.2}%", svm_training_error * 100.0);
    println!("SVM Rand Statistic: {:.2}", svm_rand_statistic);
    println!("SVM Jaccard Coefficient: {:.2}", svm_jaccard);
    println!("SVM 95% Confidence Interval for Accuracy: ({:.2}%, {:.2}%)", svm_ci.0 * 100.0, svm_ci.1 * 100.0);
    println!("SVM Training Time: {:.2?}", svm_duration);
    println!("SVM Prediction Time: {:.2?}", svm_predict_duration);

    // --------------- PLATFORM COMPARISON METRICS ---------------
    let build_time = build_start.elapsed().unwrap();
    println!("\n--- Platform Comparison Metrics ---");
    println!("Total Build/Execution Time: {:.2?}", build_time);

    Ok(())
}

// Function to calculate Rand Statistic
fn calculate_rand_statistic(predictions: &[f64], actual: &[f64]) -> f64 {
    let mut tp = 0.0;
    let mut tn = 0.0;
    let mut fp = 0.0;
    let mut false_negatives = 0.0;

    for (p1, a1) in predictions.iter().zip(actual.iter()) {
        for (p2, a2) in predictions.iter().zip(actual.iter()) {
            if (p1 == p2 && a1 == a2) {
                if *p1 == *a1 {
                    tp += 1.0;
                } else {
                    tn += 1.0;
                }
            } else if (p1 != p2 && a1 != a2) {
                fp += 1.0;
            } else {
                false_negatives += 1.0;

            }
        }
    }

    (tp + tn) / (tp + tn + fp + false_negatives)
}

// Function to calculate Jaccard Coefficient
fn calculate_jaccard_coefficient(predictions: &[f64], actual: &[f64]) -> f64 {
    let mut intersection = 0.0;
    let mut union = 0.0;

    for (p, a) in predictions.iter().zip(actual.iter()) {
        if p == a {
            intersection += 1.0;
        }
        union += 1.0;
    }

    intersection / union
}

// Function to calculate Confidence Interval for Accuracy
fn calculate_confidence_interval(accuracy: f64, n: usize) -> (f64, f64) {
    let z = 1.96; // Z-score for 95% confidence level
    let p = accuracy;
    let se = (p * (1.0 - p) / n as f64).sqrt();
    (p - z * se, p + z * se)
}

// Function to calculate Specificity
fn specificity(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mut tn = 0;
    let mut fp = 0;

    for (&actual, &predicted) in y_true.iter().zip(y_pred.iter()) {
        if actual == 0.0 && predicted == 0.0 {
            tn += 1;
        } else if actual == 0.0 && predicted == 1.0 {
            fp += 1;
        }
    }

    if tn + fp == 0 {
        return 0.0;
    }

    tn as f64 / (tn + fp) as f64
}

// Function to standardize features by setting the mean to 0 and the standard deviation to 1
fn standardize_features(features: &mut Vec<f64>, n_features: usize) {
    let n_samples = features.len() / n_features;
    for j in 0..n_features {
        // Extract a column of features
        let col: Vec<f64> = (0..n_samples)
            .map(|i| features[i * n_features + j])
            .collect();
        
        // Calculate mean and standard deviation for the column
        let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
        let stddev: f64 = (col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64).sqrt();
        
        // Standardize each element in the column
        for i in 0..n_samples {
            features[i * n_features + j] = (features[i * n_features + j] - mean) / stddev;
        }
    }
}

// Function to undersample the majority class with a fixed random seed
fn undersample_majority_class(features: &mut Vec<f64>, targets: &mut Vec<f64>, majority_class: f64) {
    let n_features = features.len() / targets.len();
    let majority_indices: Vec<usize> = targets.iter()
        .enumerate()
        .filter(|(_, &y)| y == majority_class)
        .map(|(idx, _)| idx)
        .collect();

    let minority_indices: Vec<usize> = targets.iter()
        .enumerate()
        .filter(|(_, &y)| y != majority_class)
        .map(|(idx, _)| idx)
        .collect();

    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for consistency

    // Randomly select a subset of majority class samples equal in size to the minority class
    let undersampled_majority_indices: Vec<usize> = majority_indices
        .choose_multiple(&mut rng, minority_indices.len())
        .cloned()
        .collect();

    let mut all_indices = minority_indices;
    all_indices.extend(undersampled_majority_indices);
    all_indices.sort();

    // Rebuild the feature and target sets using the selected indices
    *features = all_indices.iter()
        .flat_map(|&idx| features[idx * n_features..(idx + 1) * n_features].to_vec())
        .collect();

    *targets = all_indices.iter()
        .map(|&idx| targets[idx])
        .collect();
}
