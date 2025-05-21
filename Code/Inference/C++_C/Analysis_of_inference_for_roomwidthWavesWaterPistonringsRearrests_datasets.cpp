//#include "functions.h"  // Include your header file where functions are declared
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <vector>
//#include <tuple>
//#include <string>
//#include <Eigen/Dense>
//#include <chrono>  // For measuring runtime
//
//int main() {
//    // Start timing
//    auto start = std::chrono::high_resolution_clock::now();
//
//    // ---- Roomwidth Data ----
//    std::vector<std::tuple<std::string, double>> roomwidthData = loadSimpleCSV("roomwidth.csv");
//    std::vector<double> roomwidth_feet;
//    std::vector<double> roomwidth_metres;
//
//    for (const auto& entry : roomwidthData) {
//        std::string unit = std::get<0>(entry);
//        double width = std::get<1>(entry);
//        if (unit == "feet") {
//            roomwidth_feet.push_back(width);
//        } else if (unit == "metres") {
//            roomwidth_metres.push_back(width * 3.28);  // Convert metres to feet
//        }
//    }
//
//    if (!roomwidth_feet.empty() && !roomwidth_metres.empty()) {
//        double t_stat = t_test(roomwidth_feet, roomwidth_metres);
//        std::cout << "T-test statistic (roomwidth): " << t_stat << std::endl;
//    }
//
//    // ---- Water Data ----
//    std::vector<std::tuple<std::string, std::string, double, double>> waterData = loadWaterInferenceCSV("water.csv");
//    std::vector<double> mortality_data;
//    std::vector<double> hardness_data;
//
//    for (const auto& entry : waterData) {
//        mortality_data.push_back(std::get<2>(entry));
//        hardness_data.push_back(std::get<3>(entry));
//    }
//
//    if (!mortality_data.empty() && !hardness_data.empty()) {
//        Eigen::VectorXd mortality_vec = Eigen::Map<Eigen::VectorXd>(mortality_data.data(), mortality_data.size());
//        Eigen::VectorXd hardness_vec = Eigen::Map<Eigen::VectorXd>(hardness_data.data(), hardness_data.size());
//
//        linearRegression(hardness_vec, mortality_vec);
//    }
//
//    // ---- Waves Data ----
//    std::vector<std::tuple<std::string, double>> wavesData = loadSimpleCSV("waves.csv");
//    std::vector<double> method1_data;
//    std::vector<double> method2_data;
//
//    for (const auto& entry : wavesData) {
//        std::string method = std::get<0>(entry);
//        double measurement = std::get<1>(entry);
//        if (method == "method1") {
//            method1_data.push_back(measurement);
//        } else if (method == "method2") {
//            method2_data.push_back(measurement);
//        }
//    }
//
//    if (!method1_data.empty() && !method2_data.empty()) {
//        double t_stat = t_test(method1_data, method2_data);
//        std::cout << "T-test statistic (waves): " << t_stat << std::endl;
//    }
//
//    // ---- Pistonrings Data ----
//    std::vector<std::tuple<std::string, double>> pistonringsData = loadSimpleCSV("pistonrings.csv");
//    std::vector<double> diameter_before;
//    std::vector<double> diameter_after;
//
//    for (const auto& entry : pistonringsData) {
//        std::string state = std::get<0>(entry);
//        double diameter = std::get<1>(entry);
//        if (state == "before") {
//            diameter_before.push_back(diameter);
//        } else if (state == "after") {
//            diameter_after.push_back(diameter);
//        }
//    }
//
//    if (!diameter_before.empty() && !diameter_after.empty()) {
//        double t_stat = t_test(diameter_before, diameter_after);
//        std::cout << "T-test statistic (pistonrings): " << t_stat << std::endl;
//    }
//
//    // ---- Rearrests Data ----
//    std::vector<std::tuple<std::string, double>> rearrestsData = loadSimpleCSV("rearrests.csv");
//    std::vector<double> rearrests_treatment;
//    std::vector<double> rearrests_control;
//
//    for (const auto& entry : rearrestsData) {
//        std::string group = std::get<0>(entry);
//        double rearrest_rate = std::get<1>(entry);
//        if (group == "treatment") {
//            rearrests_treatment.push_back(rearrest_rate);
//        } else if (group == "control") {
//            rearrests_control.push_back(rearrest_rate);
//        }
//    }
//
//    if (!rearrests_treatment.empty() && !rearrests_control.empty()) {
//        double t_stat = t_test(rearrests_treatment, rearrests_control);
//        std::cout << "T-test statistic (rearrests): " << t_stat << std::endl;
//    }
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
