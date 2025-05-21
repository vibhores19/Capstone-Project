//#include "functions.h"
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <cmath>
//#include <vector>
//#include <map>
//#include <tuple>
//#include <Eigen/Dense>
//#include <chrono>  // For measuring runtime
//
//// Function to load water.csv into a vector of tuples (four elements: location, town, mortality, hardness)
//std::vector<std::tuple<std::string, std::string, double, double>> loadWaterCSV(const std::string& filename) {
//    std::ifstream file(filename);
//    std::vector<std::tuple<std::string, std::string, double, double>> data;
//    std::string line, location, town;
//    double mortality, hardness;
//
//    // Skip header
//    std::getline(file, line);
//
//    while (std::getline(file, line)) {
//        std::stringstream ss(line);
//        std::getline(ss, location, ',');  // Location (North or South)
//        std::getline(ss, town, ',');      // Town name
//        ss >> mortality;
//        ss.ignore(1);  // Ignore comma
//        ss >> hardness;
//
//        data.push_back(std::make_tuple(location, town, mortality, hardness));
//    }
//
//    return data;
//}
//
//// Function to compute summary statistics by location for water hardness and mortality
//void computeWaterStatistics(const std::vector<std::tuple<std::string, std::string, double, double>>& data) {
//    std::map<std::string, std::vector<double>> mortalityMap, hardnessMap;
//
//    // Organize data by location (North, South)
//    for (const auto& entry : data) {
//        mortalityMap[std::get<0>(entry)].push_back(std::get<2>(entry)); // Get mortality
//        hardnessMap[std::get<0>(entry)].push_back(std::get<3>(entry));  // Get hardness
//    }
//
//    // Calculate and print mean and standard deviation for mortality and hardness
//    for (const auto& group : mortalityMap) {
//        double mortalityMean = calculateMean(group.second);
//        double mortalityStdDev = calculateStdDev(group.second, mortalityMean);
//        std::cout << "Location: " << group.first << " | Mortality Mean: " << mortalityMean
//                  << " | Mortality StdDev: " << mortalityStdDev << std::endl;
//    }
//    for (const auto& group : hardnessMap) {
//        double hardnessMean = calculateMean(group.second);
//        double hardnessStdDev = calculateStdDev(group.second, hardnessMean);
//        std::cout << "Location: " << group.first << " | Hardness Mean: " << hardnessMean
//                  << " | Hardness StdDev: " << hardnessStdDev << std::endl;
//    }
//}
//
//// Function to perform MANOVA on water hardness and mortality data
//void performWaterMANOVA(const std::vector<std::tuple<std::string, std::string, double, double>>& data) {
//    std::map<std::string, std::vector<Eigen::Vector2d>> groups; // For North and South
//
//    // Organize data into groups (North, South) for MANOVA
//    for (const auto& entry : data) {
//        Eigen::Vector2d vec(std::get<2>(entry), std::get<3>(entry)); // Mortality, Hardness
//        groups[std::get<0>(entry)].push_back(vec);  // Group by location (North/South)
//    }
//
//    // Compute group means and overall mean
//    Eigen::Vector2d grandMean = Eigen::Vector2d::Zero();
//    int totalN = 0;
//
//    for (const auto& group : groups) {
//        for (const auto& vec : group.second) {
//            grandMean += vec;
//            totalN++;
//        }
//    }
//    grandMean /= totalN;
//
//    // Between-group and within-group covariance matrices
//    Eigen::Matrix2d SSB = Eigen::Matrix2d::Zero(); // Between-group sum of squares and products
//    Eigen::Matrix2d SSW = Eigen::Matrix2d::Zero(); // Within-group sum of squares and products
//
//    // Calculate SSB and SSW
//    for (const auto& group : groups) {
//        Eigen::Vector2d groupMean = Eigen::Vector2d::Zero();
//        for (const auto& vec : group.second) {
//            groupMean += vec;
//        }
//        groupMean /= group.second.size();
//
//        // Between-group scatter
//        Eigen::Vector2d meanDiff = groupMean - grandMean;
//        SSB += group.second.size() * (meanDiff * meanDiff.transpose());
//
//        // Within-group scatter
//        for (const auto& vec : group.second) {
//            Eigen::Vector2d withinDiff = vec - groupMean;
//            SSW += (withinDiff * withinDiff.transpose());
//        }
//    }
//
//    // Calculate Wilks' Lambda correctly
//    Eigen::Matrix2d combinedMatrix = SSW + SSB;
//    double wilksLambda = SSW.determinant() / combinedMatrix.determinant();
//
//    // Output results
//    std::cout << "\nMANOVA Results for Water Hardness and Mortality Dataset:" << std::endl;
//    std::cout << "SSB (Between-Group Scatter): \n" << SSB << std::endl;
//    std::cout << "SSW (Within-Group Scatter): \n" << SSW << std::endl;
//    std::cout << "Wilks' Lambda: " << wilksLambda << std::endl;
//}
//
//
//int main() {
//    // Start timing
//    auto start = std::chrono::high_resolution_clock::now();
//
//    // Load and analyze water dataset
//    std::vector<std::tuple<std::string, std::string, double, double>> waterData = loadWaterCSV("water.csv");
//    std::cout << "\nAnalyzing Water Hardness Dataset..." << std::endl;
//    computeWaterStatistics(waterData);
//
//    // Perform MANOVA analysis
//    performWaterMANOVA(waterData);
//
//    // Stop timing
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> duration = end - start;
//
//    // Output the runtime
//    std::cout << "\nRuntime: " << duration.count() << " seconds" << std::endl;
//
//    return 0;
//}
