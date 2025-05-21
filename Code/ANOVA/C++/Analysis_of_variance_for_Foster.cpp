//// variance for Foster Feeding dataset
//#include "functions.h"
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <vector>
//#include <map>
//#include <tuple>
//#include <Eigen/Dense>
//#include <numeric>
//#include <chrono>  // Include for runtime measurement
//
//// Function to load foster feeding data into a map (grouped by litgen and motgen)
//std::map<std::pair<std::string, std::string>, std::vector<double>> loadFosterData(const std::string& filename) {
//    std::map<std::pair<std::string, std::string>, std::vector<double>> data;
//    std::ifstream file(filename);
//    std::string line, litgen, motgen;
//    double weight;
//
//    // Skip header
//    std::getline(file, line);
//
//    while (std::getline(file, line)) {
//        std::stringstream ss(line);
//        std::getline(ss, litgen, ',');
//        std::getline(ss, motgen, ',');
//        ss >> weight;
//
//        data[{litgen, motgen}].push_back(weight);
//    }
//
//    return data;
//}
//
//// Function to perform ANOVA for foster feeding data
//void performFosterANOVA(const std::map<std::pair<std::string, std::string>, std::vector<double>>& data) {
//    double grandMean = 0.0;
//    int totalN = 0;
//
//    // Calculate grand mean
//    for (const auto& group : data) {
//        grandMean += std::accumulate(group.second.begin(), group.second.end(), 0.0);
//        totalN += group.second.size();
//    }
//    grandMean /= totalN;
//
//    // Initialize SSB (between groups) and SSW (within groups)
//    double SSB = 0.0;
//    double SSW = 0.0;
//
//    // Calculate SSB and SSW
//    for (const auto& group : data) {
//        double groupMean = calculateMean(group.second);
//        double sumWithinGroup = calculateSumOfSquares(group.second, groupMean);
//        SSW += sumWithinGroup;
//
//        SSB += group.second.size() * std::pow(groupMean - grandMean, 2);
//    }
//
//    // Degrees of freedom
//    int df_between = data.size() - 1;
//    int df_within = totalN - data.size();
//
//    // Mean squares
//    double MSB = SSB / df_between;
//    double MSW = SSW / df_within;
//
//    // F-statistic
//    double F = MSB / MSW;
//
//    // Output results
//    std::cout << "SSB (Between-Groups Sum of Squares): " << SSB << std::endl;
//    std::cout << "SSW (Within-Groups Sum of Squares): " << SSW << std::endl;
//    std::cout << "MSB (Mean Square Between): " << MSB << std::endl;
//    std::cout << "MSW (Mean Square Within): " << MSW << std::endl;
//    std::cout << "F-statistic: " << F << std::endl;
//}
//
//int main() {
//    // Start timing
//    auto start = std::chrono::high_resolution_clock::now();
//
//    // Load and analyze foster dataset
//    std::map<std::pair<std::string, std::string>, std::vector<double>> fosterData = loadFosterData("foster.csv");
//    std::cout << "\nAnalyzing Foster Feeding Dataset..." << std::endl;
//    performFosterANOVA(fosterData);
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
