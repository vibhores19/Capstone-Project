#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <string>
#include <tuple>
#include <Eigen/Dense>

// Declare the functions here
double calculateMean(const std::vector<double>& values);
double calculateSumOfSquares(const std::vector<double>& data, double mean);
double calculateStdDev(const std::vector<double>& values, double mean);
double calculateVariance(const std::vector<double>& values, double mean);


//std::vector<std::tuple<std::string, std::string, double>> loadGeneralCSV(const std::string& filename);

std::vector<std::tuple<std::string, int, double, double, double, std::string, double>> loadGeneralCSV(const std::string& filename);

std::vector<std::tuple<std::string, std::string, double>> loadCSV(const std::string& filename);
std::vector<std::tuple<std::string, double>> loadSimpleCSV(const std::string& filename);
std::vector<std::tuple<std::string, std::string, double, double>> loadWaterInferenceCSV(const std::string& filename);

double t_test(const std::vector<double>& group1, const std::vector<double>& group2);
void linearRegression(const Eigen::VectorXd& X, const Eigen::VectorXd& y);

void computeStatistics(const std::vector<std::tuple<std::string, std::string, double>>& data);

Eigen::VectorXd performLinearRegression(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
Eigen::VectorXd calculateStandardErrors(const Eigen::MatrixXd& X, const Eigen::VectorXd& residuals, int n, int p);
Eigen::VectorXd computeTValues(const Eigen::VectorXd& coefficients, const Eigen::VectorXd& standard_errors);

#endif  // FUNCTIONS_H



