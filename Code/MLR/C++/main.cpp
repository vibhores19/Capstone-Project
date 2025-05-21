#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>

using namespace std;

// Structure to hold the dataset row
struct DataRow {
    string source;
    string type;
    double weightgain;
};

// Function to calculate the mean
double mean(const vector<double>& data) {
    double sum = 0;
    for (double val : data) {
        sum += val;
    }
    return sum / static_cast<double>(data.size());  // Cast size_t to double
}

// Function to calculate total sum of squares (SS_Total)
double total_sum_of_squares(const vector<double>& data, double grand_mean) {
    double ss_total = 0;
    for (double val : data) {
        ss_total += pow(val - grand_mean, 2);
    }
    return ss_total;
}

// Function to calculate sum of squares for a factor (SS_Factor)
double sum_of_squares_factor(const map<string, vector<double>>& factor_groups, double grand_mean) {
    double ss_factor = 0;
    for (const auto& group : factor_groups) {
        double group_mean = mean(group.second);
        ss_factor += group.second.size() * pow(group_mean - grand_mean, 2);
    }
    return ss_factor;
}

int main() {
    // Hard-coded dataset based on the provided image
    vector<DataRow> data = {
        {"Beef", "Low", 90}, {"Beef", "Low", 76}, {"Beef", "Low", 90},
        {"Beef", "Low", 64}, {"Beef", "Low", 86}, {"Beef", "Low", 51},
        {"Beef", "Low", 72}, {"Beef", "Low", 90}, {"Beef", "Low", 95},
        {"Beef", "Low", 78}, {"Beef", "High", 73}, {"Beef", "High", 102},
        {"Beef", "High", 118}, {"Beef", "High", 104}, {"Beef", "High", 81},
        {"Beef", "High", 107}, {"Beef", "High", 100}, {"Beef", "High", 87},
        {"Beef", "High", 117}, {"Beef", "High", 111}, {"Cereal", "Low", 107},
        {"Cereal", "Low", 95}, {"Cereal", "Low", 97}, {"Cereal", "Low", 80},
        {"Cereal", "Low", 98}, {"Cereal", "Low", 74}, {"Cereal", "Low", 74},
        {"Cereal", "Low", 67}, {"Cereal", "Low", 89}, {"Cereal", "Low", 58},
        {"Cereal", "High", 98}, {"Cereal", "High", 74}, {"Cereal", "High", 56},
        {"Cereal", "High", 111}, {"Cereal", "High", 95}, {"Cereal", "High", 88},
        {"Cereal", "High", 82}, {"Cereal", "High", 77}, {"Cereal", "High", 86},
        {"Cereal", "High", 92}
    };

    // Organize the data into map structures
    map<string, vector<double>> data_by_source;
    map<string, vector<double>> data_by_type;
    vector<double> all_values;

    for (const auto& row : data) {
        data_by_source[row.source].push_back(row.weightgain);
        data_by_type[row.type].push_back(row.weightgain);
        all_values.push_back(row.weightgain);
    }

    // Compute the grand mean
    double grand_mean = mean(all_values);

    // Compute the total sum of squares (SS_Total)
    double ss_total = total_sum_of_squares(all_values, grand_mean);

    // Compute the sum of squares for each factor (source and type)
    double ss_source = sum_of_squares_factor(data_by_source, grand_mean);
    double ss_type = sum_of_squares_factor(data_by_type, grand_mean);

    // Residual sum of squares (error term) SS_Residual
    double ss_error = ss_total - (ss_source + ss_type);

    // Degrees of freedom
    size_t df_source = data_by_source.size() - 1;
    size_t df_type = data_by_type.size() - 1;
    size_t df_total = all_values.size() - 1;
    size_t df_error = df_total - df_source - df_type;

    // Mean squares
    double ms_source = ss_source / df_source;
    double ms_type = ss_type / df_type;
    double ms_error = ss_error / df_error;

    // F-values
    double f_source = ms_source / ms_error;
    double f_type = ms_type / ms_error;

    // Output the results
    cout << "ANOVA Results:" << endl;
    cout << "Source SS: " << ss_source << ", DF: " << df_source << ", MS: " << ms_source << ", F: " << f_source << endl;
    cout << "Type SS: " << ss_type << ", DF: " << df_type << ", MS: " << ms_type << ", F: " << f_type << endl;
    cout << "Error SS: " << ss_error << ", DF: " << df_error << ", MS: " << ms_error << endl;

    return 0;
}
