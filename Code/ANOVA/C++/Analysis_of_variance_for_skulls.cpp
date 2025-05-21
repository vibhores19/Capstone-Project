//// variance for Skulls dataset with MANOVA
//#include "functions.h"
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <vector>
//#include <cmath>
//#include <map>
//#include <tuple>
//#include <Eigen/Dense>
//#include <numeric>
//#include <chrono>  // Include for runtime measurement
//
//// Function to load CSV into a vector of tuples (for skulls.csv)
//std::vector<std::tuple<std::string, double, double, double, double>> loadSkullsCSV(const std::string& filename) {
//    std::ifstream file(filename);
//    std::vector<std::tuple<std::string, double, double, double, double>> data;
//    std::string line, epoch;
//    double mb, bh, bl, nh;
//
//    // Skip header
//    std::getline(file, line);
//
//    while (std::getline(file, line)) {
//        std::stringstream ss(line);
//        std::getline(ss, epoch, ',');
//        ss >> mb;
//        ss.ignore(1);  // Ignore comma
//        ss >> bh;
//        ss.ignore(1);  // Ignore comma
//        ss >> bl;
//        ss.ignore(1);  // Ignore comma
//        ss >> nh;
//
//        data.push_back(std::make_tuple(epoch, mb, bh, bl, nh));
//    }
//    return data;
//}
//
//// Function to compute summary statistics by epoch for skull measurements
//void computeSkullStatistics(const std::vector<std::tuple<std::string, double, double, double, double>>& data) {
//    std::map<std::string, std::vector<double>> mbMap, bhMap, blMap, nhMap;
//
//    // Organize data by epoch
//    for (const auto& entry : data) {
//        mbMap[std::get<0>(entry)].push_back(std::get<1>(entry));
//        bhMap[std::get<0>(entry)].push_back(std::get<2>(entry));
//        blMap[std::get<0>(entry)].push_back(std::get<3>(entry));
//        nhMap[std::get<0>(entry)].push_back(std::get<4>(entry));
//    }
//
//    // Calculate and print mean and standard deviation for each measurement across epochs
//    for (const auto& group : mbMap) {
//        double mbMean = calculateMean(group.second);
//        double mbStdDev = calculateStdDev(group.second, mbMean);
//        std::cout << "Epoch: " << group.first << " | MB Mean: " << mbMean << " | MB StdDev: " << mbStdDev << std::endl;
//    }
//    // Repeat for other measurements (bh, bl, nh) as in mb
//}
//
//// Function to perform MANOVA on skull data
//void performSkullsMANOVA(const std::vector<std::tuple<std::string, double, double, double, double>>& data) {
//    // Step 1: Create a matrix for the dependent variables (MB, BH, BL, NH) and categorize by epochs
//    std::map<std::string, std::vector<Eigen::Vector4d>> groups; // Grouped by epochs
//    Eigen::MatrixXd totalMatrix(data.size(), 4); // Matrix for the entire dataset
//
//    int i = 0;
//    for (const auto& entry : data) {
//        Eigen::Vector4d skullVec(std::get<1>(entry), std::get<2>(entry), std::get<3>(entry), std::get<4>(entry));
//        totalMatrix.row(i) = skullVec;
//        groups[std::get<0>(entry)].push_back(skullVec); // Group by epoch
//        i++;
//    }
//
//    // Step 2: Compute total mean vector (grand mean)
//    Eigen::Vector4d grandMean = totalMatrix.colwise().mean();
//
//    // Step 3: Between-group and within-group matrices
//    Eigen::Matrix4d SSB = Eigen::Matrix4d::Zero(); // Between-group sum of squares and products
//    Eigen::Matrix4d SSW = Eigen::Matrix4d::Zero(); // Within-group sum of squares and products
//
//    // Calculate SSB and SSW
//    for (const auto& group : groups) {
//        Eigen::Vector4d groupMean = Eigen::Vector4d::Zero();
//        for (const auto& vec : group.second) {
//            groupMean += vec;
//        }
//        groupMean /= group.second.size();
//
//        // Between-group scatter
//        Eigen::Vector4d meanDiff = groupMean - grandMean;
//        SSB += group.second.size() * (meanDiff * meanDiff.transpose());
//
//        // Within-group scatter
//        for (const auto& vec : group.second) {
//            Eigen::Vector4d withinDiff = vec - groupMean;
//            SSW += (withinDiff * withinDiff.transpose());
//        }
//    }
//
//    // Step 4: Calculate the Wilks' Lambda test statistic
//    Eigen::Matrix4d invSSW = SSW.inverse();
//    Eigen::Matrix4d productMatrix = invSSW * SSB;
//    double wilksLambda = (productMatrix.determinant() > 0) ? 1 / productMatrix.determinant() : 0;
//
//    // Step 5: Output results
//    std::cout << "\nMANOVA Results for Egyptian Skulls Dataset:" << std::endl;
//    std::cout << "SSB (Between-Group Scatter): \n" << SSB << std::endl;
//    std::cout << "SSW (Within-Group Scatter): \n" << SSW << std::endl;
//    std::cout << "Wilks' Lambda: " << wilksLambda << std::endl;
//}
//
//int main() {
//    // Start timing
//    auto start = std::chrono::high_resolution_clock::now();
//
//    // Load and analyze skulls dataset
//    std::vector<std::tuple<std::string, double, double, double, double>> skullsData = loadSkullsCSV("skulls.csv");
//    std::cout << "\nAnalyzing Egyptian Skulls Dataset..." << std::endl;
//    computeSkullStatistics(skullsData);
//
//    // Perform MANOVA analysis
//    performSkullsMANOVA(skullsData);
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
