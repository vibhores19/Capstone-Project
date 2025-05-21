#include "functions.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <map>
#include <tuple>
#include <Eigen/Dense>


//std::vector<std::tuple<std::string, std::string, double>> loadGeneralCSV(const std::string& filename) {
//    std::vector<std::tuple<std::string, std::string, double>> data;
//    std::ifstream file(filename);
//
//    if (!file.is_open()) {
//        std::cerr << "Error: Could not open the file " << filename << std::endl;
//        return data;
//    }
//
//    std::string column1, column2;
//    double column3;
//
//    while (file >> column1 >> column2 >> column3) {
//        data.emplace_back(column1, column2, column3);
//    }
//
//    file.close();
//    return data;
//}
//

// Improved loadGeneralCSV function with correct CSV parsing
std::vector<std::tuple<std::string, int, double, double, double, std::string, double>> loadGeneralCSV(const std::string& filename) {
    std::vector<std::tuple<std::string, int, double, double, double, std::string, double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return data;
    }

    std::string line;
    std::string seeding, echomotion;
    int time;
    double sne, cloudcover, prewetness, rainfall;

    // Ignore the header
    std::getline(file, line);

    // Read the file line by line
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string temp;

        // Handle the quoted values (seeding, echomotion)
        std::getline(ss, temp, ',');
        seeding = temp.substr(1, temp.size() - 2);  // Remove the quotes

        std::getline(ss, temp, ',');
        time = std::stoi(temp);

        std::getline(ss, temp, ',');
        sne = std::stod(temp);

        std::getline(ss, temp, ',');
        cloudcover = std::stod(temp);

        std::getline(ss, temp, ',');
        prewetness = std::stod(temp);

        std::getline(ss, temp, ',');
        echomotion = temp.substr(1, temp.size() - 2);  // Remove the quotes

        std::getline(ss, temp, ',');
        rainfall = std::stod(temp);

        // Add the parsed row to the data vector
        data.emplace_back(seeding, time, sne, cloudcover, prewetness, echomotion, rainfall);
    }

    file.close();
    return data;
}


// Function to calculate the mean
double calculateMean(const std::vector<double>& values) {
    double sum = 0;
    for (double val : values) {
        sum += val;
    }
    return sum / values.size();
}

// Function to calculate sum of squares (ANOVA components)
double calculateSumOfSquares(const std::vector<double>& data, double mean) {
    double sum = 0.0;
    for (double value : data) {
        sum += (value - mean) * (value - mean);
    }
    return sum;
}


// Function to calculate the standard deviation
double calculateStdDev(const std::vector<double>& values, double mean) {
    double sum = 0;
    for (double val : values) {
        sum += (val - mean) * (val - mean);
    }
    return std::sqrt(sum / values.size());
}

// Function to calculate variance
double calculateVariance(const std::vector<double>& values, double mean) {
    double sum = 0.0;
    for (double val : values) {
        sum += (val - mean) * (val - mean);
    }
    return sum / (values.size() - 1);  // Using (n-1) for sample variance
}

// Function to load CSV into a vector of tuples
std::vector<std::tuple<std::string, std::string, double>> loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::tuple<std::string, std::string, double>> data;
    std::string line, var1, var2;
    double value;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::getline(ss, var1, ',');
        std::getline(ss, var2, ',');
        ss >> value;

        data.push_back(std::make_tuple(var1, var2, value));
    }

    return data;
}

// Function to load simple CSV (e.g., "roomwidth.csv", "waves.csv")
std::vector<std::tuple<std::string, double>> loadSimpleCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::tuple<std::string, double>> data;
    std::string line, unit;
    double width;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::getline(ss, unit, ',');
        ss >> width;

        data.push_back(std::make_tuple(unit, width));
    }

    return data;
}
//
// Function to load CSV into a vector of tuples (e.g., for "water.csv")
std::vector<std::tuple<std::string, std::string, double, double>> loadWaterInferenceCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::tuple<std::string, std::string, double, double>> data;
    std::string line, location, town;
    double mortality, hardness;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::getline(ss, location, ',');
        std::getline(ss, town, ',');
        ss >> mortality;
        ss.ignore(1);  // Ignore comma
        ss >> hardness;

        data.push_back(std::make_tuple(location, town, mortality, hardness));
    }

    return data;
}

// Function for independent samples t-test
double t_test(const std::vector<double>& group1, const std::vector<double>& group2) {
    double mean1 = calculateMean(group1);
    double mean2 = calculateMean(group2);

    double var1 = calculateVariance(group1, mean1);
    double var2 = calculateVariance(group2, mean2);

    // Debug output to check values
    std::cout << "Mean of group 1: " << mean1 << std::endl;
    std::cout << "Mean of group 2: " << mean2 << std::endl;
    std::cout << "Variance of group 1: " << var1 << std::endl;
    std::cout << "Variance of group 2: " << var2 << std::endl;

    if (var1 == 0 || var2 == 0) {
        std::cout << "One of the groups has zero variance." << std::endl;
        return 0;  // Return 0 since the test statistic is not valid in this case
    }

    double pooled_se = std::sqrt(var1 / group1.size() + var2 / group2.size());
    return (mean1 - mean2) / pooled_se;
}

// Function to perform linear regression using Eigen library
void linearRegression(const Eigen::VectorXd& X, const Eigen::VectorXd& y) {
    Eigen::VectorXd beta = (X.transpose() * X).inverse() * X.transpose() * y;
    std::cout << "Linear Regression Coefficients: \n" << beta << std::endl;
}

// Function to compute summary statistics
void computeStatistics(const std::vector<std::tuple<std::string, std::string, double>>& data) {
    std::map<std::string, std::vector<double>> firstVarMap;
    std::map<std::string, std::vector<double>> secondVarMap;

    // Organize data by first and second variables
    for (const auto& entry : data) {
        firstVarMap[std::get<0>(entry)].push_back(std::get<2>(entry));
        secondVarMap[std::get<1>(entry)].push_back(std::get<2>(entry));
    }

    // Calculate and print mean and std deviation for first variable (e.g., "source")
    for (const auto& group : firstVarMap) {
        double mean = calculateMean(group.second);
        double stddev = calculateStdDev(group.second, mean);
        std::cout << "First Var: " << group.first << " | Mean: " << mean << " | StdDev: " << stddev << std::endl;
    }

    // Calculate and print mean and std deviation for second variable (e.g., "type")
    for (const auto& group : secondVarMap) {
        double mean = calculateMean(group.second);
        double stddev = calculateStdDev(group.second, mean);
        std::cout << "Second Var: " << group.first << " | Mean: " << mean << " | StdDev: " << stddev << std::endl;
    }
}

// New function implementations for regression analysis

// Function to perform multiple linear regression
Eigen::VectorXd performLinearRegression(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    // Beta = (X'X)^(-1) X'y
    return (X.transpose() * X).inverse() * X.transpose() * y;
}


// Function to calculate standard errors of coefficients
Eigen::VectorXd calculateStandardErrors(const Eigen::MatrixXd& X, const Eigen::VectorXd& residuals, int n, int p) {
    double residual_sum_of_squares = residuals.squaredNorm();
    double sigma_squared = residual_sum_of_squares / (n - p);
    Eigen::MatrixXd cov_matrix = sigma_squared * (X.transpose() * X).inverse();
    return cov_matrix.diagonal().array().sqrt();
}

// Function to compute t-values for the coefficients
Eigen::VectorXd computeTValues(const Eigen::VectorXd& coefficients, const Eigen::VectorXd& standard_errors) {
    return coefficients.array() / standard_errors.array();
}
