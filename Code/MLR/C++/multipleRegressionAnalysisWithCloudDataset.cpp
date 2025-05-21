//#include "functions.h"
//#include <iostream>
//#include <fstream>
//#include <string>
//#include <vector>
//#include <tuple>
//#include <chrono>  // Include chrono for timing
//
//// Function to create a design matrix for the regression
//Eigen::MatrixXd createDesignMatrix(const std::vector<std::tuple<std::string, int, double, double, double, std::string, double>>& data) {
//    int n = data.size();
//    Eigen::MatrixXd X(n, 10);  // 10 columns for intercept + 9 predictors including interactions
//
//    for (int i = 0; i < n; ++i) {
//        std::string seeding, echomotion;
//        int time;
//        double sne, cloudcover, prewetness, rainfall;
//
//        std::tie(seeding, time, sne, cloudcover, prewetness, echomotion, rainfall) = data[i];
//
//        double seeding_val = (seeding == "yes") ? 1.0 : 0.0;
//        double echomotion_val = (echomotion == "stationary") ? 1.0 : 0.0;
//
//        // Intercept
//        X(i, 0) = 1.0;
//        // Main effects
//        X(i, 1) = seeding_val;
//        X(i, 2) = sne;
//        X(i, 3) = cloudcover;
//        X(i, 4) = prewetness;
//        X(i, 5) = echomotion_val;
//        X(i, 6) = time;
//        // Interaction terms
//        X(i, 7) = seeding_val * sne;
//        X(i, 8) = seeding_val * cloudcover;
//        X(i, 9) = seeding_val * prewetness;
//    }
//
//    return X;
//}
//
//// Function to perform OLS regression
//Eigen::VectorXd performOLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
//    return (X.transpose() * X).ldlt().solve(X.transpose() * y);
//}
//
//// Function to calculate residuals
//Eigen::VectorXd calculateResiduals(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& beta) {
//    return y - (X * beta);  // residuals = y - X * beta
//}
//
//int main() {
//    // Start timing
//    auto start = std::chrono::high_resolution_clock::now();
//
//    // Load the dataset
//    std::string filename = "clouds.csv";  // Replace with actual dataset path
//    auto data = loadGeneralCSV(filename);
//
//    // Create the design matrix (X) and the rainfall vector (y)
//    Eigen::MatrixXd X = createDesignMatrix(data);
//    Eigen::VectorXd y(data.size());
//    for (int i = 0; i < data.size(); ++i) {
//        y(i) = std::get<6>(data[i]);  // rainfall is the 7th element in the tuple
//    }
//
//    // Perform OLS regression
//    Eigen::VectorXd beta = performOLS(X, y);
//
//    // Compute fitted values
//    Eigen::VectorXd fitted_values = X * beta;
//
//    // Compute residuals
//    Eigen::VectorXd residuals = calculateResiduals(X, y, beta);
//
//    // Stop timing
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> duration = end - start;
//
//    // Output the coefficients, fitted values, residuals, and runtime
//    std::cout << "Coefficients:\n" << beta << std::endl;
//    std::cout << "\nFitted Values:\n" << fitted_values << std::endl;
//    std::cout << "\nResiduals:\n" << residuals << std::endl;
//    std::cout << "\nRuntime: " << duration.count() << " seconds" << std::endl;
//
//    return 0;
//}
