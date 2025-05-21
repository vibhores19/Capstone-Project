using CSV, DataFrames, Statistics, HypothesisTests, Distributions, Plots, StatsPlots, PrettyTables, GLM, CategoricalArrays

roomwidth = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/roomwidth.csv", DataFrame)

# Convert estimates of room width from metres to feet
roomwidth.convert = ifelse.(roomwidth.unit .== "feet", 1, 3.28)
roomwidth.converted_width = roomwidth.width .* roomwidth.convert

# Summary statistics for feet and metres
describe(roomwidth[roomwidth.unit .== "feet", :converted_width])
describe(roomwidth[roomwidth.unit .== "metres", :converted_width])


# Standard deviation for feet and metres
feet_sd = std(roomwidth[roomwidth.unit .== "feet", :converted_width])
metres_sd = std(roomwidth[roomwidth.unit .== "metres", :converted_width])

println("Feet SD: ", feet_sd)
println("Metres SD: ", metres_sd)

@df roomwidth boxplot(:unit, :converted_width, ylabel="Estimated width (feet)", legend=false)

# Custom function to add a Q-Q line
function qqline(data::Vector, dist=Normal())
    # Quantiles of the data
    sorted_data = sort(data)
    
    # Theoretical quantiles (for normal distribution)
    n = length(data)
    theoretical_quantiles = quantile(dist, [(i - 0.5) / n for i in 1:n])
    
    # Fit a line through the first and third quantiles
    q1_data, q3_data = quantile(sorted_data, [0.25, 0.75])
    q1_theory, q3_theory = quantile(dist, [0.25, 0.75])
    
    # Calculate slope and intercept for the line
    slope = (q3_data - q1_data) / (q3_theory - q1_theory)
    intercept = q1_data - slope * q1_theory
    
    # Return the line equation
    return slope, intercept
end

using Distributions, Plots

# Example data (feet measurements)
feet_data = roomwidth[roomwidth.unit .== "feet", :converted_width]

# Create the Q-Q plot
qqplot(Normal(), feet_data, title="Q-Q Plot for Feet", xlabel="Theoretical Quantiles", ylabel="Feet Width")

# Add the Q-Q line
slope, intercept = qqline(feet_data)
plot!(x -> slope * x + intercept, label="Q-Q Line", color=:red)

using Distributions, Plots

# Extract original meters data (no conversion)
metres_data_original = roomwidth[roomwidth.unit .== "metres", :width]

# Q-Q plot for original meters data (not converted to feet)
p2 = qqplot(Normal(), metres_data_original, title="Q-Q Plot for Metres (Original)", xlabel="Theoretical Quantiles", ylabel="Estimated Width (Metres)", yticks=0:5:40, ylim=(0, 40))

# Add the Q-Q line
slope, intercept = qqline(metres_data_original)
plot!(p2, x -> slope * x + intercept, label="Q-Q Line for Metres", color=:red)

# Display the plot
display(p2)

using HypothesisTests

# Extracting the numeric vectors from the DataFrame
feet_data = roomwidth[roomwidth.unit .== "feet", :converted_width]  # Extract feet data as a vector
metres_data = roomwidth[roomwidth.unit .== "metres", :converted_width]  # Extract metres data as a vector

# Performing the Equal Variance T-Test
equal_variance_ttest = HypothesisTests.EqualVarianceTTest(feet_data, metres_data)

# Extract the p-value from the t-test result
p_value = pvalue(equal_variance_ttest)

# Calculate the means for each group
mean_feet = mean(feet_data)
mean_metres = mean(metres_data)

# Display the results
println("Equal Variance T-Test Result:")
println(equal_variance_ttest)

println("\nP-Value: ", p_value)
println("Mean in group feet: ", mean_feet)
println("Mean in group metres: ", mean_metres)

using HypothesisTests, Statistics

# Extracting the numeric vectors from the DataFrame
feet_data = roomwidth[roomwidth.unit .== "feet", :converted_width]  # Extract feet data as a vector
metres_data = roomwidth[roomwidth.unit .== "metres", :converted_width]  # Extract metres data as a vector

# Performing the Welch's (Unequal Variance) T-Test
unequal_variance_ttest = HypothesisTests.UnequalVarianceTTest(feet_data, metres_data)

# Extract the p-value from the t-test result
p_value_unequal = pvalue(unequal_variance_ttest)

# Calculate the means for each group
mean_feet = mean(feet_data)
mean_metres = mean(metres_data)

# Display the results
println("Welch's (Unequal Variance) T-Test Result:")
println(unequal_variance_ttest)

println("\nP-Value: ", p_value_unequal)
println("Mean in group feet: ", mean_feet)
println("Mean in group metres: ", mean_metres)

# Wilcoxon rank sum test
wilcox_test = HypothesisTests.MannWhitneyUTest(feet_data, metres_data)
println(wilcox_test)

# Extracting the numeric vectors from the DataFrame
feet_data = roomwidth[roomwidth.unit .== "feet", :converted_width]  # Extract feet data as a vector
metres_data = roomwidth[roomwidth.unit .== "metres", :converted_width]  # Extract metres data as a vector

# Performing the Wilcoxon Rank-Sum (Mann-Whitney U) Test
wilcox_test = MannWhitneyUTest(feet_data, metres_data)

# Extract p-value
p_value_wilcox = pvalue(wilcox_test)

# Extract U statistic (also referred to as W in the output)
u_statistic = wilcox_test.U

# Sample estimate: difference in location
location_diff = median(feet_data) - median(metres_data)

# Display the results
println("Wilcoxon Rank-Sum (Mann-Whitney U) Test Result:")
println(wilcox_test)

println("\nW (U statistic): ", u_statistic)
println("P-Value: ", p_value_wilcox)
println("Sample Estimate (Difference in Location): ", location_diff)

using HypothesisTests, Statistics

# Extracting the numeric vectors from the DataFrame
feet_data = roomwidth[roomwidth.unit .== "feet", :converted_width]  # Extract feet data as a vector
metres_data = roomwidth[roomwidth.unit .== "metres", :converted_width]  # Extract metres data as a vector

# Performing the Wilcoxon Rank-Sum (Mann-Whitney U) Test
wilcox_test = MannWhitneyUTest(feet_data, metres_data)

# Extract p-value
p_value_wilcox = pvalue(wilcox_test)

# Extract U statistic (also referred to as W in the output)
u_statistic = wilcox_test.U

# Manually calculate Hodges-Lehmann estimator (difference in location)
pairwise_diffs = [x - y for x in feet_data for y in metres_data]
location_diff = median(pairwise_diffs)

# Display the results
println("Wilcoxon Rank-Sum (Mann-Whitney U) Test Result:")
println(wilcox_test)

println("\nW (U statistic): ", u_statistic)
println("P-Value: ", p_value_wilcox)
println("Sample Estimate (Difference in Location): ", location_diff)

pistonrings_df = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/pistonrings.csv", DataFrame)

pistonrings = Matrix(pistonrings_df)

#Define the compressor and leg labels
compressor_labels = ["C1", "C2", "C3", "C4"]
leg_labels = ["North", "Centre", "South"] 

chisq_test = ChisqTest(pistonrings)

#Calculate row and column totals and expected frequencies
row_totals = sum(pistonrings, dims=2)
col_totals = sum(pistonrings, dims=1)
total = sum(pistonrings)
expected_frequencies = row_totals * col_totals / total


# Calculate the residuals
residuals = (pistonrings .- expected_frequencies) ./ sqrt.(expected_frequencies)

# Create a DataFrame with compressor labels and residuals
residuals_df = DataFrame(Compressor = compressor_labels)
for i in 1:length(leg_labels)
    residuals_df[!, Symbol(leg_labels[i])] = residuals[:, i]  # Add each residual column
end

# Display the residuals in a nice table format
pretty_table(residuals_df)



#Generate the association plot using rectangles

# Increase the height of the graph by changing the `size` parameter
plot(legend = false, size=(600, 600), xlim=(0.5, 3.5), ylim=(0.5, 4.5))

# Loop over each compressor (y-axis) and leg (x-axis) to place rectangles
for i in 1:4  # Compressors C1 to C4 (now reversed, C1 is at i=4, C4 at i=1)
    for j in 1:3  # Legs North, Centre, South
        
        rect_height = abs(residuals[i, j])  # Height represents the magnitude of the residual
        rect_color = residuals[i, j] > 0 ? :gray : :lightgray  # Color for positive and negative residuals
        
        # Draw a rectangle for each (compressor, leg) pair
        # x-coordinate is j (leg), y-coordinate is 5 - i (compressor, reversed)
        plot!([j-0.4, j-0.4, j+0.4, j+0.4], [5-i-0.4, 5-i+0.4 * rect_height, 5-i+0.4 * rect_height, 5-i-0.4],
              fillcolor=rect_color, linecolor=:black, lw=0.5, seriestype=:shape)
    end
end

# Customize the x-axis and reverse the y-axis labels to show C1 at the top
xticks!(1:3, leg_labels)
yticks!(1:4, compressor_labels)
xlabel!("Leg")
ylabel!("Compressor")
title!("Association Plot of Residuals (Reversed)")



# Load the CSV file 
rearrests = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/rearrests.csv", DataFrame)


using CSV, DataFrames, Distributions, PrettyTables

# Step 1: Load the CSV file (replace with your actual file path)
rearrests_df = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/rearrests.csv", DataFrame)

# Step 2: Convert the data to a matrix (assuming it's a 2x2 contingency table)
rearrests = Matrix(rearrests_df)

# Step 3: Extract values for McNemar's test
b = rearrests[1, 2]  # cell (1,2)
c = rearrests[2, 1]  # cell (2,1)

# Step 4: Calculate McNemar's test statistic (without continuity correction)
chi_square = ((b - c)^2) / (b + c)

# Step 5: Calculate the p-value (using Chi-Square distribution with 1 degree of freedom)
p_value = 1 - cdf(Chisq(1), chi_square)

# Step 6: Display the result
println("McNemar's Test Statistic: ", chi_square)
println("P-value: ", p_value)

using CSV, DataFrames, Distributions, PrettyTables



# Step 2: Convert the data to a matrix (assuming it's a 2x2 contingency table)
rearrests = Matrix(rearrests_df)

# Step 3: Extract values for McNemar's test
b = rearrests[1, 2]  # cell (1,2)
c = rearrests[2, 1]  # cell (2,1)

# Step 4: Calculate McNemar's test statistic (without continuity correction)
chi_square = ((b - c)^2) / (b + c)

# Step 5: Calculate the p-value (using Chi-Square distribution with 1 degree of freedom)
p_value = 1 - cdf(Chisq(1), chi_square)

# Step 6: Display the McNemar's test result
println("McNemar's Test Statistic: ", chi_square)
println("Degrees of Freedom: 1")
println("P-value: ", p_value)

using HypothesisTests

# Step 1: Define number of successes and total number of trials
successes = rearrests[2, 1]  # The value from the second row, first column (290)
n_trials = rearrests[2, 1] + rearrests[1, 2]  # Sum of the second and first off-diagonal values (805)

# Step 2: Perform the binomial test
binom_test = BinomialTest(successes, n_trials, 0.5)

# Step 3: Extract confidence intervals
conf_interval = confint(binom_test)

# Step 4: Display the binomial test result
println("Number of successes: ", successes)
println("Number of trials: ", n_trials)
println("P-value: ", pvalue(binom_test))
println("95% Confidence Interval: ", conf_interval)
println("Estimated Probability of Success: ", successes / n_trials)

waves_df = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/waves.csv", DataFrame)


#Calculate differences between method1 and method2
mooringdiff = waves_df.method1 .- waves_df.method2

boxplot!(mooringdiff, ylabel="Differences (Newton metres)", title="Boxplot")

qqnorm(mooringdiff,xlabel =  ylabel="Differences (Newton metres)", title="QQ Plot")


using HypothesisTests

# Perform a t-test to check if the mean of mooringdiff is significantly different from 0
t_test_result = OneSampleTTest(mooringdiff, 0)

# Print the results of the t-test
println("Two-sample t-test result:")
println("T-statistic: ", t_test_result.t)
println("Degrees of Freedom: ", t_test_result.df)
println("P-value: ", pvalue(t_test_result))
println("95% Confidence Interval: ", confint(t_test_result))
println("Mean: ", mean(mooringdiff))

function wilcoxon_signed_rank_test(x)
    # Differences from the hypothesized median
    diffs = x .- 0
    
    # Remove zeros (ties)
    non_zero_diffs = filter(d -> d != 0, diffs)
    
    # Compute the ranks of the absolute differences
    abs_diffs = abs.(non_zero_diffs)
    ranks = sortperm(abs_diffs)
    
    # Assign signs to the ranks based on the original data
    signed_ranks = sign.(non_zero_diffs)[ranks] .* (1:length(non_zero_diffs))
    
    # Sum of positive and negative ranks
    W_plus = sum(r -> r > 0 ? r : 0, signed_ranks)
    W_minus = sum(r -> r < 0 ? abs(r) : 0, signed_ranks)
    
    # Use the smaller of the two
    W = min(W_plus, W_minus)
    
    # Get the p-value for the test
    # This is an approximation for large samples
    n = length(non_zero_diffs)
    mean_W = n * (n + 1) / 4
    std_W = sqrt(n * (n + 1) * (2n + 1) / 24)
    
    z = (W - mean_W) / std_W
    p_value = 2 * (1 - cdf(Normal(), abs(z)))
    
    return W, z, p_value
end

# Example of using the custom Wilcoxon test
W, z, p_value = wilcoxon_signed_rank_test(mooringdiff)

# Output the results
println("Wilcoxon signed-rank test result:")
println("Test statistic (W): ", W)
println("Z-value: ", z)
println("P-value: ", p_value)

# Custom Wilcoxon signed-rank test with continuity correction
function wilcoxon_signed_rank_test(data::Vector; correction=true)
    # Differences from 0
    diffs = data .- 0

    # Remove zeros (ties)
    non_zero_diffs = filter(d -> d != 0, diffs)

    # Get ranks of the absolute differences
    abs_diffs = abs.(non_zero_diffs)
    sorted_abs_diffs = sort(abs_diffs)
    ranks = [findfirst(x -> x == d, sorted_abs_diffs) for d in abs_diffs]

    # Signed ranks
    signed_ranks = sign.(non_zero_diffs) .* ranks

    # Sum of positive and negative ranks
    W_plus = sum(r -> r > 0 ? r : 0, signed_ranks)
    W_minus = sum(r -> r < 0 ? abs(r) : 0, signed_ranks)

    # Use the smaller of the two
    V = min(W_plus, W_minus)

    # Continuity correction
    correction_value = if correction 0.5 else 0 end
    n = length(non_zero_diffs)
    mean_V = n * (n + 1) / 4
    std_V = sqrt(n * (n + 1) * (2n + 1) / 24)

    # Calculate Z-value with continuity correction
    Z = (V - mean_V - correction_value) / std_V

    # P-value (two-sided test)
    p_value = 2 * (1 - cdf(Normal(), abs(Z)))

    return V, Z, p_value
end

# Perform the test on mooringdiff
V, Z, p_value = wilcoxon_signed_rank_test(mooringdiff)

# Output the results
println("Wilcoxon signed rank test with continuity correction")
println("Test statistic (V): ", V)
println("P-value: ", p_value)

function rank_data(data::Vector{Float64})
    sorted_data = sortperm(data)
    ranks = Vector{Float64}(undef, length(data))
    
    i = 1
    while i <= length(data)
        val = data[sorted_data[i]]
        tie_indices = findall(x -> data[x] == val, sorted_data)
        avg_rank = mean(tie_indices)
        
        for idx in tie_indices
            ranks[sorted_data[idx]] = avg_rank
        end
        i += length(tie_indices)
    end
    return ranks
end

# Custom Wilcoxon signed-rank test with continuity correction
function wilcoxon_signed_rank_test(data::Vector{Float64}; correction=true)
    # Remove zeros (differences of zero are ignored)
    non_zero_diffs = filter(d -> d != 0, data)

    # Get absolute differences and rank them
    abs_diffs = abs.(non_zero_diffs)
    ranks = rank_data(abs_diffs)

    # Apply signs to ranks based on original differences
    signed_ranks = sign.(non_zero_diffs) .* ranks

    # Calculate test statistic V
    W_plus = sum(r -> r > 0 ? r : 0, signed_ranks)
    W_minus = sum(r -> r < 0 ? abs(r) : 0, signed_ranks)

    # Use the smaller of W+ and W-
    V = min(W_plus, W_minus)

    # Apply continuity correction if needed
    correction_value = correction ? 0.5 : 0.0
    n = length(non_zero_diffs)
    mean_V = n * (n + 1) / 4
    std_V = sqrt(n * (n + 1) * (2n + 1) / 24)

    # Z-value with continuity correction
    Z = (V - mean_V - correction_value) / std_V

    # Calculate two-sided p-value
    p_value = 2 * (1 - cdf(Normal(), abs(Z)))

    return V, Z, p_value
end

# Test the function with your mooringdiff data
mooringdiff = [ # Example data, replace with your own data
    -0.1, 0.2, 0.3, -0.4, 0.0, -0.6, 0.7, 0.8, -0.9, 1.0
]

# Perform the Wilcoxon signed-rank test
V, Z, p_value = wilcoxon_signed_rank_test(mooringdiff)

# Output the results
println("Wilcoxon signed-rank test with continuity correction")
println("Test statistic (V): ", V)
println("Z-value: ", Z)
println("P-value: ", p_value)

water_df = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/water.csv", DataFrame)


unique_locations = unique(water.location)
psymb = [findfirst(==(loc), unique_locations) for loc in water.location]

# Convert location column to a categorical variable for better handling
water.location = CategoricalArray(water.location)

# Scatter plot of mortality vs hardness, with different markers and colors based on location
p1 = @df water scatter(:hardness, :mortality, group=:location,
    markershape=[:circle :square], label=["North" "South"], 
    color=[:blue :green], # Distinct colors for groups
    title="Mortality vs Hardness", legend=:topright)

# Fit a linear model (lm)
lm_model = lm(@formula(mortality ~ hardness), water)

# Generate predictions from the linear model
predicted_values = predict(lm_model)

# Plot the regression line based on the fitted model
@df water plot!(water.hardness, predicted_values, label="Regression Line", lw=2)

histogram(water.hardness, 
    title="Histogram of Hardness", 
    xlabel="Hardness", 
    ylabel="Frequency", 
    bins=10, # You can adjust the number of bins
    color=:blue, 
    legend=false)

boxplot(water.mortality, 
    title="Boxplot of Mortality", 
    ylabel="Mortality", 
    color=:lightblue, 
    legend=false)

correlation_coefficient = cor(water.mortality, water.hardness)


correlation_test = CorrelationTest(water.mortality, water.hardness)



