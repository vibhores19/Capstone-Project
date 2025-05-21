#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Helper structure to store metrics
struct Metrics {
    double accuracy = 0.0;
    double error_rate = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double specificity = 0.0;
    double auc = 0.0;
    double entropy = 0.0;
    double purity = 0.0;
    double rand_statistic = 0.0;
    double jaccard_coefficient = 0.0;
    double ci_accuracy = 0.0;
    double variance = 0.0;
};

// Function to load the dataset from a CSV file
vector<vector<double>> loadCSV(string diabetes) {
    vector<vector<double>> data;
    ifstream file(diabetes);
    string line, word;

    if (!file.is_open()) {
        cout << "Error: Could not open file." << endl;
        return data;
    }

    bool firstLine = true;  // To skip the header if it exists
    while (getline(file, line)) {
        if (firstLine) {
            firstLine = false;  // Skip the header row
            continue;
        }

        vector<double> row;
        stringstream ss(line);

        // Process each word (i.e., value) in the line
        while (getline(ss, word, ',')) {
            try {
                row.push_back(stod(word));  // Convert to double
            } catch (const invalid_argument& e) {
                cout << "Invalid data encountered: " << word << endl;
                return data;
            }
        }

        // Add the row to the data vector if it's not empty
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}

// Function to standardize the data (Z-score normalization)
void standardize(vector<vector<double>>& data) {
    size_t num_features = data[0].size() - 1;
    for (size_t i = 0; i < num_features; ++i) {
        double mean = 0.0, std_dev = 0.0;

        // Compute mean
        for (size_t j = 0; j < data.size(); ++j) {
            mean += data[j][i];
        }
        mean /= data.size();

        // Compute standard deviation
        for (size_t j = 0; j < data.size(); ++j) {
            std_dev += pow(data[j][i] - mean, 2);
        }
        std_dev = sqrt(std_dev / data.size());

        // Standardize features
        for (size_t j = 0; j < data.size(); ++j) {
            if (std_dev != 0) {
                data[j][i] = (data[j][i] - mean) / std_dev;
            }
        }
    }
}

// Logistic Regression class using Gradient Descent
class LogisticRegression {
public:
    vector<double> weights;
    double bias;
    double learning_rate;

    LogisticRegression(size_t num_features, double lr = 0.01) {
        weights = vector<double>(num_features, 0.0);
        bias = 0.0;
        learning_rate = lr;
    }

    // Sigmoid function to convert output into probability
    double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

    // Predict function (returns 1 or 0 based on probability)
    int predict(const vector<double>& x) {
        double sum = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            sum += weights[i] * x[i];
        }
        return (sigmoid(sum) >= 0.5) ? 1 : 0;
    }

    // Predict the probability instead of class
    double predict_proba(const vector<double>& x) {
        double sum = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            sum += weights[i] * x[i];
        }
        return sigmoid(sum);
    }

    // Training function using Gradient Descent
    void train(const vector<vector<double>>& data, int epochs = 1000) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& row : data) {
                vector<double> x(row.begin(), row.end() - 1);
                int y = (row.back() == 1) ? 1 : 0;

                double prediction = sigmoid(bias + inner_product(weights.begin(), weights.end(), x.begin(), 0.0));
                double error = prediction - y;

                // Update weights and bias
                for (size_t i = 0; i < weights.size(); ++i) {
                    weights[i] -= learning_rate * error * x[i];
                }
                bias -= learning_rate * error;
            }
        }
    }
};

// Simple linear SVM class using Stochastic Gradient Descent (SGD)
class SVM {
public:
    vector<double> weights;
    double bias;
    double learning_rate;
    double lambda;  // Regularization strength

    SVM(size_t num_features, double lr = 0.01, double reg = 0.01) {
        weights = vector<double>(num_features, 0.0);
        bias = 0.0;
        learning_rate = lr;
        lambda = reg;
    }

    // Prediction function (returns -1 or 1)
    int predict(const vector<double>& x) {
        double sum = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            sum += weights[i] * x[i];
        }
        return (sum >= 0) ? 1 : -1;
    }

    // Training function using SGD
    void train(const vector<vector<double>>& data, int epochs = 1000) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& row : data) {
                vector<double> x(row.begin(), row.end() - 1);
                int y = (row.back() == 1) ? 1 : -1;

                if (y * predict(x) < 1) {
                    for (size_t i = 0; i < weights.size(); ++i) {
                        weights[i] += learning_rate * (y * x[i] - 2 * lambda * weights[i]);
                    }
                    bias += learning_rate * y;
                } else {
                    for (size_t i = 0; i < weights.size(); ++i) {
                        weights[i] += learning_rate * (-2 * lambda * weights[i]);
                    }
                }
            }
        }
    }
};

// Helper function to calculate ROC AUC
double calculate_auc(const vector<pair<double, int>>& predictions) {
    vector<pair<double, int>> sorted_predictions = predictions;  // Avoid modifying the original const vector
    sort(sorted_predictions.begin(), sorted_predictions.end(), [](const pair<double, int>& a, const pair<double, int>& b) {
        return a.first > b.first;
    });
    double auc = 0.0;
    double tp = 0.0, fp = 0.0;
    double prev_tp = 0.0, prev_fp = 0.0;

    for (const auto& pred : sorted_predictions) {
        if (pred.second == 1) {
            tp++;
        } else {
            fp++;
            auc += (tp - prev_tp) * ((fp + prev_fp) / 2.0);
            prev_tp = tp;
            prev_fp = fp;
        }
    }
    return auc / (tp * fp);
}

// Helper function to calculate additional metrics
Metrics calculate_additional_metrics(int tp, int tn, int fp, int fn, const vector<pair<double, int>>& lr_predictions_proba, size_t data_size) {
    Metrics metrics;
    // Sensitivity (Recall) and Specificity
    metrics.recall = (double)tp / (tp + fn);
    metrics.specificity = (double)tn / (tn + fp);
    metrics.error_rate = (double)(fp + fn) / data_size;
    metrics.precision = (double)tp / (tp + fp);
    metrics.f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);

    // ROC AUC for Logistic Regression
    metrics.auc = calculate_auc(lr_predictions_proba);

    // Entropy Calculation for Logistic Regression
    for (const auto& pred : lr_predictions_proba) {
        double p = pred.first;
        metrics.entropy += -((p * log(p)) + ((1 - p) * log(1 - p)));
    }
    metrics.entropy /= lr_predictions_proba.size();

    // Purity Calculation (based on accuracy)
    metrics.purity = (double)(tp + tn) / data_size;

    // Rand Statistic and Jaccard Coefficient for similarity
    int agreements = tp + tn;
    int disagreements = fp + fn;

    metrics.rand_statistic = (double)agreements / (agreements + disagreements);
    metrics.jaccard_coefficient = (double)tp / (tp + fp + fn);

    // Confidence Interval for Accuracy
    double accuracy = (double)(tp + tn) / data_size;
    double z = 1.96;  // For 95% confidence interval
    metrics.ci_accuracy = z * sqrt((accuracy * (1 - accuracy)) / data_size);

    // Variance of predicted probabilities for Logistic Regression
    double mean_proba = 0.0;
    for (const auto& pred : lr_predictions_proba) {
        mean_proba += pred.first;
    }
    mean_proba /= lr_predictions_proba.size();

    metrics.variance = 0.0;
    for (const auto& pred : lr_predictions_proba) {
        metrics.variance += pow(pred.first - mean_proba, 2);
    }
    metrics.variance /= lr_predictions_proba.size();

    return metrics;
}

// Function to calculate classification metrics for both models
Metrics evaluate(const vector<vector<double>>& data, SVM& svm_model, LogisticRegression& lr_model) {
    int tp_svm = 0, tn_svm = 0, fp_svm = 0, fn_svm = 0;
    int tp_lr = 0, tn_lr = 0, fp_lr = 0, fn_lr = 0;
    vector<pair<double, int>> lr_predictions_proba;  // For ROC AUC

    for (const auto& row : data) {
        vector<double> x(row.begin(), row.end() - 1);
        int actual = (row.back() == 1) ? 1 : 0;

        // Predict SVM and Logistic Regression
        int predicted_svm = svm_model.predict(x);
        int predicted_lr = lr_model.predict(x);
        double predicted_lr_proba = lr_model.predict_proba(x);

        // SVM confusion matrix
        if (predicted_svm == 1 && actual == 1) tp_svm++;
        else if (predicted_svm == -1 && actual == 0) tn_svm++;
        else if (predicted_svm == 1 && actual == 0) fp_svm++;
        else if (predicted_svm == -1 && actual == 1) fn_svm++;

        // Logistic Regression confusion matrix
        if (predicted_lr == 1 && actual == 1) tp_lr++;
        else if (predicted_lr == 0 && actual == 0) tn_lr++;
        else if (predicted_lr == 1 && actual == 0) fp_lr++;
        else if (predicted_lr == 0 && actual == 1) fn_lr++;

        // Store predictions and actual values for ROC curve and AUC
        lr_predictions_proba.push_back({predicted_lr_proba, actual});
    }

    // Metrics for SVM
    double accuracy_svm = (double)(tp_svm + tn_svm) / data.size();
    double precision_svm = (double)tp_svm / (tp_svm + fp_svm);
    double recall_svm = (double)tp_svm / (tp_svm + fn_svm);
    double f1_svm = 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm);
    double error_rate_svm = (double)(fp_svm + fn_svm) / data.size();

    // Calculate additional metrics for SVM (same as for LR)
    Metrics svm_metrics = calculate_additional_metrics(tp_svm, tn_svm, fp_svm, fn_svm, {}, data.size());

    // Metrics for Logistic Regression
    double accuracy_lr = (double)(tp_lr + tn_lr) / data.size();
    double precision_lr = (double)tp_lr / (tp_lr + fp_lr);
    double recall_lr = (double)tp_lr / (tp_lr + fn_lr);
    double f1_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr);
    double error_rate_lr = (double)(fp_lr + fn_lr) / data.size();

    // Calculate additional metrics for Logistic Regression
    Metrics lr_metrics = calculate_additional_metrics(tp_lr, tn_lr, fp_lr, fn_lr, lr_predictions_proba, data.size());

    // Print SVM metrics
    cout << "\nSVM Metrics:" << endl;
    cout << "Accuracy: " << accuracy_svm << endl;
    cout << "Error Rate: " << error_rate_svm << endl;
    cout << "Precision: " << precision_svm << endl;
    cout << "Recall (Sensitivity): " << recall_svm << endl;
    cout << "F1 Score: " << f1_svm << endl;
    cout << "Specificity: " << svm_metrics.specificity << endl;
    cout << "Rand Statistic: " << svm_metrics.rand_statistic << endl;
    cout << "Jaccard Coefficient: " << svm_metrics.jaccard_coefficient << endl;
    cout << "Purity: " << svm_metrics.purity << endl;
    cout << "Confidence Interval for Accuracy: ±" << svm_metrics.ci_accuracy << endl;

    // Print Logistic Regression metrics
    cout << "\nLogistic Regression Metrics:" << endl;
    cout << "Accuracy: " << accuracy_lr << endl;
    cout << "Error Rate: " << error_rate_lr << endl;
    cout << "Precision: " << precision_lr << endl;
    cout << "Recall (Sensitivity): " << recall_lr << endl;
    cout << "F1 Score: " << f1_lr << endl;
    cout << "Specificity: " << lr_metrics.specificity << endl;
    cout << "ROC AUC: " << lr_metrics.auc << endl;
    cout << "Entropy: " << lr_metrics.entropy << endl;
    cout << "Purity: " << lr_metrics.purity << endl;
    cout << "Rand Statistic: " << lr_metrics.rand_statistic << endl;
    cout << "Jaccard Coefficient: " << lr_metrics.jaccard_coefficient << endl;
    cout << "Confidence Interval for Accuracy: ±" << lr_metrics.ci_accuracy << endl;
    cout << "Variance of Predictions: " << lr_metrics.variance << endl;

    return lr_metrics;
}

int main() {
    // Start time for the entire execution
    auto start_time = high_resolution_clock::now();

    // Load the dataset
    vector<vector<double>> data = loadCSV("diabetes.csv");

    // Check if the data is loaded successfully
    if (data.empty()) {
        cout << "Error: Failed to load dataset." << endl;
        return 1;
    }

    cout << "Data loaded successfully!" << endl;

    // Standardize the dataset
    standardize(data);

    // Create SVM and Logistic Regression models
    SVM svm_model(data[0].size() - 1);
    LogisticRegression lr_model(data[0].size() - 1);

    // Train the models
    svm_model.train(data, 1000); // Train SVM
    lr_model.train(data, 1000);  // Train Logistic Regression

    // Evaluate the models
    evaluate(data, svm_model, lr_model);

    // End time for the entire execution
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);

    // Print execution time
    cout << "Execution Time: " << duration.count() << " milliseconds" << endl;

    return 0;
}
