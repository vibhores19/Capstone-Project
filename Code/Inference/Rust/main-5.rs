use csv::Reader; // CSV reader for loading datasets
use ndarray::{Array1, Array2, Axis}; // Importing ndarray types and Axis for data operations
use statrs::statistics::Statistics; // For statistical calculations
use statrs::distribution::ContinuousCDF; // Importing ContinuousCDF for cdf method
use std::error::Error;
use std::time::Instant; // Importing for measuring execution time

// Function to calculate the median
fn calculate_median(data: &Array1<f64>) -> f64 {
    let mut sorted_data: Vec<f64> = data.to_vec(); // Convert Array1 to Vec
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); // Sort the Vec
    let len = sorted_data.len();

    if len == 0 {
        return 0.0;
    }

    if len % 2 == 0 {
        (sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0
    } else {
        sorted_data[len / 2]
    }
}

// Function to calculate summary statistics and standard deviation
fn summary_statistics(data: &Array1<f64>, label: &str) {
    let mean = data.mean().unwrap_or(0.0);
    let std_dev = data.std_dev(); // Use statrs crate to calculate the standard deviation
    let min = data.min();
    let max = data.max();
    let median = calculate_median(data);

    println!("Summary Statistics for {}: ", label);
    println!("Mean: {:.2}, Standard Deviation: {:.2}", mean, std_dev);
    println!("Min: {:.2}, Max: {:.2}, Median: {:.2}", min, max, median);
}

// Function to load and convert Room Width data from `roomwidth.csv`
fn load_and_convert_data(file_path: &str) -> Result<(Array1<f64>, Array1<String>, Vec<f64>), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    let mut widths: Vec<f64> = Vec::new();
    let mut units: Vec<String> = Vec::new();
    let mut original_widths: Vec<f64> = Vec::new(); // To store original widths

    for result in reader.records() {
        let record = result?;
        let unit = &record[0]; // Column index 0 for unit
        let width: f64 = record[1].parse()?; // Column index 1 for width

        // Convert units: 1 for feet, 3.28 for meters
        let converted_width = if unit == "feet" { width } else { width * 3.28 };
        widths.push(converted_width);
        original_widths.push(width); // Store the original width value
        units.push(unit.to_string());
    }

    Ok((Array1::from(widths), Array1::from(units), original_widths))
}

// Function to perform two-sample t-test (assuming equal variances)
fn perform_t_test(group1: &Array1<f64>, group2: &Array1<f64>) {
    let mean1 = group1.mean().unwrap_or(0.0);
    let mean2 = group2.mean().unwrap_or(0.0);
    let std_dev1 = group1.std_dev();
    let std_dev2 = group2.std_dev();

    println!(
        "Group 1 - Mean: {:.2}, Standard Deviation: {:.2}",
        mean1, std_dev1
    );
    println!(
        "Group 2 - Mean: {:.2}, Standard Deviation: {:.2}",
        mean2, std_dev2
    );

    // Calculate the pooled standard deviation
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;
    let pooled_variance = (((n1 - 1.0) * std_dev1.powi(2)) + ((n2 - 1.0) * std_dev2.powi(2))) / (n1 + n2 - 2.0);
    let pooled_std_dev = pooled_variance.sqrt();

    // Calculate the t-statistic
    let t_statistic = (mean1 - mean2) / (pooled_std_dev * (1.0 / n1 + 1.0 / n2).sqrt());

    // Calculate degrees of freedom
    let df = n1 + n2 - 2.0;

    // Calculate the p-value
    let p_value = statrs::distribution::StudentsT::new(0.0, 1.0, df).unwrap().cdf(t_statistic);

    // Print the results
    println!("Two-sample t-test assuming equal variances:");
    println!("t = {:.3}, df = {:.3}, p-value = {:.5}", t_statistic, df, p_value);
}

// Function to perform Welch's t-test
fn welch_t_test(group1: &Array1<f64>, group2: &Array1<f64>) {
    let mean1 = group1.mean().unwrap_or(0.0);
    let mean2 = group2.mean().unwrap_or(0.0);
    let var1 = group1.var(0.0);
    let var2 = group2.var(0.0);
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;

    let t_statistic = (mean1 - mean2) / ((var1 / n1) + (var2 / n2)).sqrt();
    let numerator = (var1 / n1 + var2 / n2).powi(2);
    let denominator = ((var1 / n1).powi(2) / (n1 - 1.0)) + ((var2 / n2).powi(2) / (n2 - 1.0));
    let df = numerator / denominator;

    println!("Welch's t-test: t = {:.3}, df = {:.3}", t_statistic, df);
}

// Function to perform Wilcoxon Rank Sum Test
fn wilcoxon_rank_sum_test(group1: &Array1<f64>, group2: &Array1<f64>) {
    let mut combined: Vec<(f64, String)> = group1.iter().map(|&x| (x, String::from("group1"))).collect();
    combined.extend(group2.iter().map(|&x| (x, String::from("group2"))));

    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let ranks: Vec<(f64, String, f64)> = combined.iter().enumerate().map(|(i, &(val, ref group))| (val, group.clone(), i as f64 + 1.0)).collect();

    let rank_sum1: f64 = ranks.iter().filter(|(_, group, _)| group == "group1").map(|(_, _, rank)| rank).sum();
    let rank_sum2: f64 = ranks.iter().filter(|(_, group, _)| group == "group2").map(|(_, _, rank)| rank).sum();

    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;
    let u1 = rank_sum1 - n1 * (n1 + 1.0) / 2.0;
    let u2 = rank_sum2 - n2 * (n2 + 1.0) / 2.0;

    let u = u1.min(u2);
    let mean_u = n1 * n2 / 2.0;
    let std_dev_u = ((n1 * n2 * (n1 + n2 + 1.0)) / 12.0).sqrt();

    let z = (u - mean_u) / std_dev_u;

    println!("Wilcoxon Rank Sum Test:");
    println!("U1: {:.3}, U2: {:.3}, U: {:.3}", u1, u2, u);
    println!("Mean of U: {:.3}, Standard Deviation of U: {:.3}, Z: {:.3}", mean_u, std_dev_u, z);
}

// Function to load `waves.csv` dataset
fn load_waves_data(file_path: &str) -> Result<(Array1<f64>, Array1<f64>), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    let mut method1: Vec<f64> = Vec::new();
    let mut method2: Vec<f64> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let value1: f64 = record[0].parse()?; // Column index 0 for method1
        let value2: f64 = record[1].parse()?; // Column index 1 for method2

        method1.push(value1);
        method2.push(value2);
    }

    Ok((Array1::from(method1), Array1::from(method2)))
}

/// Function to perform Wave Energy Device Mooring Analysis
fn wave_energy_mooring_analysis(file_path: &str) -> Result<(), Box<dyn Error>> {
    let (method1, method2) = load_waves_data(file_path)?;
    let mooringdiff = &method1 - &method2;

    println!("Wave Energy Device Mooring Analysis:");

    // Summary statistics for differences between methods
    summary_statistics(&mooringdiff, "Differences between Mooring Methods");

    // Create a boxplot of the differences
    println!("\nBoxplot of the differences (not implemented in this terminal-based environment).");

    // Perform t-test and Wilcoxon rank sum test on the differences
    println!("\nTwo-sample t-test between Method 1 and Method 2:");
    perform_t_test(&method1, &method2);

    println!("\nWilcoxon Rank Sum Test between Method 1 and Method 2:");
    wilcoxon_rank_sum_test(&method1, &method2);

    println!("Wave Energy Device Mooring Analysis Complete.\n");
    Ok(())
}

// Function to load `rearrests.csv` dataset
fn load_rearrests_data(file_path: &str) -> Result<(i32, i32, i32), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    let mut before: i32 = 0;
    let mut after: i32 = 0;
    let mut no_change: i32 = 0;

    for (i, result) in reader.records().enumerate() {
        let record = result?;
        let rearrest: i32 = record[0].parse()?; // Column index 0 for "Rearrest"
        let no_rearrest: i32 = record[1].parse()?; // Column index 1 for "No rearrest"

        if i == 0 {
            before = rearrest;
            no_change = no_rearrest;
        } else if i == 1 {
            after = rearrest;
        }
    }

    Ok((before, after, no_change))
}

// Function to perform Rearrests of Juveniles Analysis
fn rearrests_of_juveniles_analysis(file_path: &str) -> Result<(), Box<dyn Error>> {
    let (before, after, _no_change) = load_rearrests_data(file_path)?;

    println!("Rearrests of Juveniles Data Analysis:");
    println!("Before Treatment - Rearrest: {}, No Rearrest: {}", before, _no_change);
    println!("After Treatment - Rearrest: {}", after);

    // Perform McNemar's test for matched pairs
    mcnemar_test(before, after, _no_change);

    // Perform Binomial test
    let successes = after;
    let trials = before + after;
    binomial_test(successes, trials);

    println!("Rearrests of Juveniles Analysis Complete.\n");
    Ok(())
}

// Function to perform McNemar's Test
fn mcnemar_test(before: i32, after: i32, _no_change: i32) {
    let chi_squared = (f64::from(before - after).abs() as f64 - 1.0).powi(2) / f64::from(before + after);
    println!("McNemar's test statistic: {:.3}", chi_squared);
}

// Function to perform Binomial Test
fn binomial_test(successes: i32, trials: i32) {
    let p = successes as f64 / trials as f64;
    println!("Binomial Test: Successes = {}, Trials = {}, Proportion = {:.3}", successes, trials, p);
}

// Function to load `pistonrings.csv` dataset
fn load_pistonrings_data(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    let mut data: Vec<Vec<f64>> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().map(|x| x.parse().unwrap_or(0.0)).collect();
        data.push(row);
    }

    let rows = data.len();
    let cols = data[0].len();
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((rows, cols), flat_data)?;

    Ok(array_data)
}

// Function to perform Chi-squared test for independence
fn chisq_test(data: &Array2<f64>) -> f64 {
    let (rows, cols) = data.dim();
    let row_totals: Array1<f64> = data.sum_axis(Axis(1));
    let col_totals: Array1<f64> = data.sum_axis(Axis(0));
    let total = row_totals.sum();
    let mut chi_square = 0.0;

    for i in 0..rows {
        for j in 0..cols {
            let expected = (row_totals[i] * col_totals[j]) / total;
            let observed = data[[i, j]];
            chi_square += (observed - expected).powi(2) / expected;
        }
    }

    chi_square
}

// Function to calculate residuals for Chi-squared test
fn calculate_residuals(data: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = data.dim();
    let row_totals: Array1<f64> = data.sum_axis(Axis(1));
    let col_totals: Array1<f64> = data.sum_axis(Axis(0));
    let total = row_totals.sum();
    let mut residuals = Array2::<f64>::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let expected = (row_totals[i] * col_totals[j]) / total;
            residuals[[i, j]] = (data[[i, j]] - expected) / expected.sqrt();
        }
    }

    residuals
}

// Function to perform Piston-ring Failures Analysis
fn piston_ring_failures_analysis(file_path: &str) -> Result<(), Box<dyn Error>> {
    let piston_data = load_pistonrings_data(file_path)?;

    println!("Piston-ring Failures Data Analysis:");

    // Perform Chi-squared test for independence
    let chi_square = chisq_test(&piston_data);
    println!("Chi-squared test statistic: {:.3}", chi_square);

    // Calculate residuals
    let residuals = calculate_residuals(&piston_data);
    println!("Residuals from Chi-squared test:\n{:?}", residuals);

    println!("Piston-ring Failures Analysis Complete.\n");
    Ok(())
}

// Main function to call all the analysis functions with runtime measurements
fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now(); // Start timer for total runtime

    // File paths for datasets
    let roomwidth_file = "/Users/mozumdertushar/Desktop/Rust_simple_inference/rust_simple_inference/roomwidth.csv";
    let waves_file = "/Users/mozumdertushar/Desktop/Rust_simple_inference/rust_simple_inference/waves.csv";
    let rearrests_file = "/Users/mozumdertushar/Desktop/Rust_simple_inference/rust_simple_inference/rearrests.csv";
    let pistonrings_file = "/Users/mozumdertushar/Desktop/Rust_simple_inference/rust_simple_inference/pistonrings.csv";

    // Measure runtime for Room Width Analysis
    let start = Instant::now();
    let (converted_widths, units, _original_widths) = load_and_convert_data(roomwidth_file)?;
    println!("Room Width Data Loaded in: {:?}", start.elapsed());

    let feet_indices: Vec<usize> = units.iter().enumerate().filter(|(_, u)| *u == "feet").map(|(i, _)| i).collect();
    let meters_indices: Vec<usize> = units.iter().enumerate().filter(|(_, u)| *u == "metres").map(|(i, _)| i).collect();

    let feet_data = Array1::from_iter(feet_indices.iter().map(|&i| converted_widths[i]));
    let meters_data = Array1::from_iter(meters_indices.iter().map(|&i| converted_widths[i]));

    let start = Instant::now();
    summary_statistics(&feet_data, "Widths in Feet");
    summary_statistics(&meters_data, "Widths in Meters (Converted to Feet)");
    println!("Room Width Summary Statistics Calculated in: {:?}", start.elapsed());

    let start = Instant::now();
    perform_t_test(&feet_data, &meters_data);
    println!("Two-sample t-test Completed in: {:?}", start.elapsed());

    let start = Instant::now();
    welch_t_test(&feet_data, &meters_data);
    println!("Welch's t-test Completed in: {:?}", start.elapsed());

    let start = Instant::now();
    wilcoxon_rank_sum_test(&feet_data, &meters_data);
    println!("Wilcoxon Rank Sum Test Completed in: {:?}", start.elapsed());

    // Measure runtime for Wave Energy Device Mooring Analysis
    let start = Instant::now();
    wave_energy_mooring_analysis(waves_file)?;
    println!("Wave Energy Device Mooring Analysis Completed in: {:?}", start.elapsed());

    // Measure runtime for Rearrests of Juveniles Analysis
    let start = Instant::now();
    rearrests_of_juveniles_analysis(rearrests_file)?;
    println!("Rearrests of Juveniles Analysis Completed in: {:?}", start.elapsed());

    // Measure runtime for Piston-ring Failures Analysis
    let start = Instant::now();
    piston_ring_failures_analysis(pistonrings_file)?;
    println!("Piston-ring Failures Analysis Completed in: {:?}", start.elapsed());

    // Print total runtime
    println!("Total Program Runtime: {:?}", total_start.elapsed());

    Ok(())
}
