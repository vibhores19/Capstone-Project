use ndarray::Array1;
use csv::ReaderBuilder;
use std::error::Error;
use std::collections::HashMap;
use ndarray_stats::SummaryStatisticsExt;
use statrs::distribution::{FisherSnedecor, ContinuousCDF};
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    // Start timing the whole program
    let total_start = Instant::now();
    
    // Foster Feeding Analysis
    println!("Foster Feeding Analysis:");
    let foster_start = Instant::now();
    foster_feeding_anova()?;
    let foster_duration = foster_start.elapsed();
    println!("Foster Feeding Run Time: {:.2?}\n", foster_duration);

    // Weight Gain Analysis
    println!("\nWeight Gain Analysis:");
    let weight_start = Instant::now();
    weight_gain_anova()?;
    let weight_duration = weight_start.elapsed();
    println!("Weight Gain Run Time: {:.2?}\n", weight_duration);

    // Male Egyptian Skulls Analysis
    println!("\nMale Egyptian Skulls Analysis:");
    let skulls_start = Instant::now();
    skulls_manova()?;
    let skulls_duration = skulls_start.elapsed();
    println!("Male Egyptian Skulls Run Time: {:.2?}\n", skulls_duration);

    // End timing the whole program
    let total_duration = total_start.elapsed();
    println!("Total Run Time for All Analyses: {:.2?}\n", total_duration);

    // Comparison Metrics
    // Lines of code: Manually calculated (to be done for all implementations)
    let lines_of_code = 169;  // Example value; update this with actual LOC
    println!("Lines of Code: {}", lines_of_code);

    // Add other metrics such as memory usage if required
    Ok(())
}

// Function for Foster Feeding ANOVA
fn foster_feeding_anova() -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path("src/data/foster.csv")?;
    let mut data: Vec<(String, String, f64)> = Vec::new();
    
    for result in rdr.records() {
        let record = result?;
        let litgen = record[0].to_string();
        let motgen = record[1].to_string();
        let weight: f64 = record[2].parse()?;
        data.push((litgen, motgen, weight));
    }

    let mut grouped_data: HashMap<(String, String), Vec<f64>> = HashMap::new();
    for (litgen, motgen, weight) in &data {
        grouped_data.entry((litgen.clone(), motgen.clone()))
            .or_insert_with(Vec::new)
            .push(*weight);
    }

    let total_data_points = data.len();
    let total_groups = grouped_data.len();
    let overall_mean = Array1::from(data.iter().map(|(_, _, weight)| *weight).collect::<Vec<f64>>()).mean().unwrap();

    let mut ss_between = 0.0; 
    let mut ss_within = 0.0;

    for (group, weights) in &grouped_data {
        let array = Array1::from(weights.clone());
        let group_mean = array.mean().unwrap();
        ss_between += (group_mean - overall_mean).powi(2) * array.len() as f64;
        for &val in weights {
            ss_within += (val - group_mean).powi(2);
        }
    }

    let df_between = total_groups - 1;
    let df_within = total_data_points - total_groups;

    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;

    let f_value = ms_between / ms_within;

    let f_dist = FisherSnedecor::new(df_between as f64, df_within as f64).unwrap();
    let p_value = 1.0 - f_dist.cdf(f_value);

    println!("ANOVA Results:");
    println!("SS Between: {:.2}", ss_between);
    println!("SS Within: {:.2}", ss_within);
    println!("DF Between: {}", df_between);
    println!("DF Within: {}", df_within);
    println!("MS Between: {:.2}", ms_between);
    println!("MS Within: {:.2}", ms_within);
    println!("F-Value: {:.2}", f_value);
    println!("P-Value: {:.4}", p_value);

    Ok(())
}

// Function for Weight Gain ANOVA
fn weight_gain_anova() -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path("src/data/weightgain.csv")?;
    let mut data: Vec<(String, String, f64)> = Vec::new();
    
    for result in rdr.records() {
        let record = result?;
        let source = record[0].to_string();
        let r#type = record[1].to_string();
        let weightgain: f64 = record[2].parse()?;
        data.push((source, r#type, weightgain));
    }

    let mut grouped_data: HashMap<(String, String), Vec<f64>> = HashMap::new();
    for (source, r#type, weightgain) in &data {
        grouped_data.entry((source.clone(), r#type.clone()))
            .or_insert_with(Vec::new)
            .push(*weightgain);
    }

    println!("Means and Standard Deviations for each group:");
    for ((source, r#type), weights) in &grouped_data {
        let array = Array1::from(weights.clone());
        let mean = array.mean().unwrap();
        let std_dev = array.std(0.0);
        println!("Source: {}, Type: {}, Mean: {:.2}, Std Dev: {:.2}", source, r#type, mean, std_dev);
    }

    let overall_mean = Array1::from(data.iter().map(|(_, _, weight)| *weight).collect::<Vec<f64>>()).mean().unwrap();

    let mut ss_between_source = 0.0;
    let mut ss_within = 0.0;

    for (source, group) in grouped_data.iter() {
        let array = Array1::from(group.clone());
        let group_mean = array.mean().unwrap();
        ss_between_source += (group_mean - overall_mean).powi(2) * group.len() as f64;
        for &val in group {
            ss_within += (val - group_mean).powi(2);
        }
    }

    println!("SS Between (Source): {:.2}", ss_between_source);
    println!("SS Within: {:.2}", ss_within);

    Ok(())
}

// Function for Male Egyptian Skulls MANOVA
fn skulls_manova() -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path("src/data/skulls.csv")?;
    let mut data: Vec<(String, f64, f64, f64, f64)> = Vec::new();
    
    for result in rdr.records() {
        let record = result?;
        let epoch = record[0].to_string();
        let mb: f64 = record[1].parse()?;
        let bh: f64 = record[2].parse()?;
        let bl: f64 = record[3].parse()?;
        let nh: f64 = record[4].parse()?;
        data.push((epoch, mb, bh, bl, nh));
    }

    let mut grouped_data: HashMap<String, Vec<(f64, f64, f64, f64)>> = HashMap::new();
    for (epoch, mb, bh, bl, nh) in &data {
        grouped_data.entry(epoch.clone())
            .or_insert_with(Vec::new)
            .push((*mb, *bh, *bl, *nh));
    }

    println!("Mean Measurements for each Epoch:");
    for (epoch, measurements) in &grouped_data {
        let mb_values: Vec<f64> = measurements.iter().map(|(mb, _, _, _)| *mb).collect();
        let bh_values: Vec<f64> = measurements.iter().map(|(_, bh, _, _)| *bh).collect();
        let bl_values: Vec<f64> = measurements.iter().map(|(_, _, bl, _)| *bl).collect();
        let nh_values: Vec<f64> = measurements.iter().map(|(_, _, _, nh)| *nh).collect();

        let mean_mb = Array1::from(mb_values).mean().unwrap();
        let mean_bh = Array1::from(bh_values).mean().unwrap();
        let mean_bl = Array1::from(bl_values).mean().unwrap();
        let mean_nh = Array1::from(nh_values).mean().unwrap();

        println!("Epoch: {}", epoch);
        println!("Mean MB: {:.2}, Mean BH: {:.2}, Mean BL: {:.2}, Mean NH: {:.2}", mean_mb, mean_bh, mean_bl, mean_nh);
    }

    Ok(())
}
