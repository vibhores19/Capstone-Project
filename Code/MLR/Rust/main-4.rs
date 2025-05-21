use csv::Reader;
use nalgebra::{DMatrix, DVector};
use std::error::Error;
use std::fs;
use std::time::Instant; // Added for tracking execution time

// Function to read the dataset and extract features and target variable
fn load_dataset() -> Result<(DMatrix<f64>, DVector<f64>), Box<dyn Error>> {
    let mut rdr = Reader::from_path("src/data/clouds.csv")?;
    let mut features: Vec<f64> = Vec::new();
    let mut target: Vec<f64> = Vec::new();

    // Iterate over each record (row) in the CSV file and process the data.
    for result in rdr.records() {
        let record = result?;

        // Convert categorical variables into numeric values.
        let seeding = if &record[0] == "yes" { 1.0 } else { 0.0 }; 
        let time: f64 = record[1].parse()?; // Parsing time
        let sne: f64 = record[2].parse()?; // SNe criterion
        let cloudcover: f64 = record[3].parse()?; // Cloud cover
        let prewetness: f64 = record[4].parse()?; // Prewetness
        let echomotion = if &record[5] == "moving" { 1.0 } else { 0.0 }; // Echo motion
        let rainfall: f64 = record[6].parse()?; // Target variable: Rainfall

        // Append the intercept (1.0) and feature values to the features vector.
        features.push(1.0); // Intercept term
        features.push(seeding);
        features.push(time);
        features.push(sne);
        features.push(cloudcover);
        features.push(prewetness);
        features.push(echomotion);
        
        // Append the target value (rainfall) to the target vector.
        target.push(rainfall);
    }

    // Convert the features and target vectors into nalgebra matrices.
    let n_rows = target.len();
    let n_cols = 7; // 6 features + intercept
    let feature_matrix = DMatrix::from_vec(n_rows, n_cols, features);
    let target_vector = DVector::from_vec(target);

    Ok((feature_matrix, target_vector))
}

// Linear regression using least squares: (X^T X)^{-1} X^T y
fn linear_regression(X: &DMatrix<f64>, y: &DVector<f64>) -> Result<DVector<f64>, Box<dyn Error>> {
    let Xt = X.transpose(); // Transpose of X (X^T)
    let XtX = Xt.clone() * X; // X^T * X (matrix multiplication)
    
    // Try to invert X^T * X. If it's not invertible, return an error.
    let XtX_inv = XtX.try_inverse().ok_or("Matrix inversion failed")?; // (X^T X)^-1
    
    // Calculate X^T * y
    let XtY = Xt * y; // X^T * y (matrix multiplication)
    
    // Calculate the coefficients (beta)
    let beta = XtX_inv * XtY; // (X^T X)^-1 * X^T * y
    
    Ok(beta)
}

// Function to count lines of code in the main.rs file
fn count_lines_of_code(file_path: &str) -> usize {
    let content = fs::read_to_string(file_path).expect("Unable to read file");
    let lines: Vec<&str> = content.lines().filter(|line| !line.trim().is_empty() && !line.trim().starts_with("//")).collect();
    lines.len()
}

fn main() -> Result<(), Box<dyn Error>> {
    // Step 1: Start timing for loading dataset
    let start_time = Instant::now();
    let (features, target) = load_dataset()?;
    let load_time = start_time.elapsed();
    
    // Step 2: Perform linear regression
    let regression_start = Instant::now();
    let beta = linear_regression(&features, &target)?;
    let regression_time = regression_start.elapsed();
    
    // Output: Coefficients (similar to what you'd get in R)
    println!("--- Coefficients (Intercept + Variables) ---");
    println!("Intercept: {:.4}", beta[0]);
    println!("Seeding: {:.4}", beta[1]);
    println!("Time: {:.4}", beta[2]);
    println!("SNe: {:.4}", beta[3]);
    println!("Cloudcover: {:.4}", beta[4]);
    println!("Prewetness: {:.4}", beta[5]);
    println!("Echomotion: {:.4}", beta[6]);

    // Step 3: Calculate the fitted values (predicted rainfall)
    let fitted_values = features * &beta;
    
    // Step 4: Calculate residuals (difference between actual and predicted values)
    let residuals = &target - &fitted_values;

    // Output: Fitted values and residuals
    println!("\n--- Fitted Values (Predicted Rainfall) ---");
    for (i, value) in fitted_values.iter().enumerate() {
        println!("Prediction for data point {}: {:.4}", i + 1, value);
    }

    println!("\n--- Residuals (Errors: Actual - Predicted) ---");
    for (i, value) in residuals.iter().enumerate() {
        println!("Residual for data point {}: {:.4}", i + 1, value);
    }

    // Step 5: Count lines of code in the main.rs file and print the result
    let loc = count_lines_of_code("src/main.rs");
    println!("\n--- Lines of Code (LOC) ---");
    println!("Total lines of code: {}", loc);

    // Output: Time taken for dataset loading and regression
    println!("\n--- Performance Metrics ---");
    println!("Time to load dataset: {:?}", load_time);
    println!("Time to perform regression: {:?}", regression_time);

    Ok(())
}
