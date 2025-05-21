### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ e67bbc24-85ad-11ef-3b1a-3196ad739f27
using DataFrames, CSV, GLM,Plots,StatsPlots, LinearAlgebra, Distributions,  HypothesisTests, Statistics, PrettyTables, SimpleANOVA, MultivariateStats, LinearAlgebra, CategoricalArrays

# ╔═╡ ac1a58b5-ef21-4cad-bc48-225a83a5eb3f


# ╔═╡ 5d804c63-60ae-47e1-92d5-98fd8815dcc8
md"## Multiple Linear Regression"

# ╔═╡ 5477ed79-0911-41b2-8b6f-93dde49a22c3


# ╔═╡ 0a7c07c1-c0f4-4b1d-865b-ec9b9c506006
# Load the dataset
clouds = CSV.read("/Users/VSR/Desktop/Capstone/ANOVA/clouds.csv", DataFrame)

# ╔═╡ 870c766b-c0fc-459e-bd83-d7f1e438a4b6
# Boxplot for rainfall based on seeding
@df clouds boxplot(:seeding, :rainfall, xlabel="Seeding", ylabel="Rainfall", legend=false)


# ╔═╡ 702adce5-bebe-4662-8b73-f91da402522f
# Boxplot for rainfall based on echomotion
@df clouds boxplot(:echomotion, :rainfall, xlabel="Echo Motion", ylabel="Rainfall", legend=false)

# ╔═╡ c6d2ba6f-0b82-41e7-8431-c62774ccaae7
formula = @formula(rainfall ~ seeding * (sne + cloudcover + prewetness + echomotion) + time)

# ╔═╡ 1e49fc14-de7e-40c0-9145-9f175c063bae
clouds_lm = lm(formula, clouds)

# ╔═╡ 9c49a264-ce78-4cf0-b21b-cbf8b5b04ed9
typeof(clouds_lm)

# ╔═╡ f4924357-e978-4f31-8474-652c3ad73019


# ╔═╡ 8c44db6c-9556-46c0-b60a-b270700025b5


# ╔═╡ cd8d00c2-354b-44c3-b1de-721edf41a41d
begin
	betastar = coef(clouds_lm)
	println("Coefficients: ", betastar)
end

# ╔═╡ 64143f68-9eb9-4e14-855b-123c2813dfb3
begin
	# Covariance matrix
	Vbetastar = vcov(clouds_lm)
	println("Covariance Matrix: ", Vbetastar)
end

# ╔═╡ 6f1b8760-c928-4c40-bb90-cc93daad5577
std_errors = sqrt.(diag(Vbetastar))

# ╔═╡ 8195f5ae-4625-4547-91e4-e3537dfb0338
# Scatterplots for rainfall against each covariate

scatter(clouds.time, clouds.rainfall, xlabel="Time", ylabel="Rainfall")


# ╔═╡ b39c4573-77e6-42b8-85fa-b7664231d67d
scatter(clouds.cloudcover, clouds.rainfall, xlabel="Cloud Cover", ylabel="Rainfall")


# ╔═╡ 21a7b431-b82d-4bfa-b7c2-13646265281a
scatter(clouds.sne, clouds.rainfall, xlabel="S-Ne Criterion", ylabel="Rainfall")

# ╔═╡ cb8aefb1-bb13-4f4f-ac59-fe3c41e7db62
scatter(clouds.prewetness, clouds.rainfall, xlabel="Prewetness", ylabel="Rainfall")

# ╔═╡ c3164907-8584-4d65-b98a-3855b9fd299e
clouds_fitted = fitted(clouds_lm)

# ╔═╡ 3e8df5f8-bb44-40e1-a111-8a6455b36817
clouds_resid = clouds.rainfall - clouds_fitted

# ╔═╡ c16999a7-a93b-4df6-99cf-e7367c06d4b6
begin
    # Set the backend to GR (which supports annotations)
    gr()
    
    # Create a new scatter plot
    scatter(clouds_fitted, clouds_resid, xlabel="Fitted Values", ylabel="Residuals", label=false)
    
    
    
 
end

# ╔═╡ 5c1e3029-1a31-4b15-8357-d2f800cd6535
begin
	# Number of observations and parameters
	m = nrow(clouds)
	p = length(coef(clouds_lm))
	
	# Mean squared error (MSE)
	mse = sum(clouds_resid.^2) / (m - p)
	
	# Hat matrix diagonal (leverage values)
	X = modelmatrix(clouds_lm)  # Extract the model matrix
	H = X * inv(X' * X) * X'    # Compute the hat matrix
	h_ii = diag(H)              # Extract the diagonal elements (leverage)
	
	# Compute Cook's distance
	cooks_distances = (clouds_resid.^2 / (p * mse)) .* (h_ii ./ (1 .- h_ii).^2)
	
	# Plot Cook's distance
	bar(1:length(cooks_distances), cooks_distances, xlabel="Observation", ylabel="Cook's Distance", legend=false)
end

# ╔═╡ c99e9560-8a22-4026-998f-454f6f18af2c


# ╔═╡ 9228ffb0-e188-4dd0-a88b-dafaf7274a9d


# ╔═╡ a0772d4c-2426-4a1c-b62d-569a5b6eee69


# ╔═╡ 20bbedda-2d40-4e7f-8b54-f0fa15282a2b
begin
	# Q-Q Plot for residuals against a normal distribution
	qqplot(clouds_resid, Normal(), xlabel="Theoretical Quantiles", ylabel="Residuals", title="Q-Q Plot")
	
	# Calculate the theoretical quantiles
	sorted_residuals = sort(clouds_resid)
	n = length(clouds_resid)
	theoretical_quantiles = quantile(Normal(), (1:n) ./ (n+1))
	
	# Add Q-Q line (1-to-1 line between theoretical and sample quantiles)
	plot!(theoretical_quantiles, sorted_residuals, line=:dash, label="Q-Q Line", color=:red)
end

# ╔═╡ 1cefa728-c820-4303-820a-ee15ef55e4b1


# Scatter plot of Rainfall vs. S-Ne criterion with regression lines
scatter(clouds.sne, clouds.rainfall, group=clouds.seeding, xlabel="S-Ne Criterion", ylabel="Rainfall", label=["No Seeding" "Seeding"])


# ╔═╡ 320cffaf-12ab-4ef9-8449-014231ae0c8e
md"The linear regression results in Julia match those in R, with nearly identical coefficient estimates, standard errors, t-values, and p-values. Significant predictors in both outputs, like seeding: yes and interaction terms, confirm consistency across platforms."

# ╔═╡ 615ac17b-a752-4dac-85ae-19bb61bd2818
md"# Simple Inference"

# ╔═╡ 82d0e06b-1338-4dc5-be13-eac736de6fd4
md"###### RoomWidth Dataset (Two Sample t-test, Welch Two Sample t-test, Wilcoxon rank sum test)"

# ╔═╡ 3719a7f5-9d24-48c0-becf-6ebde74647b0


# ╔═╡ a4036c49-d98f-408c-81e7-5b3d1579b9a2
roomwidth = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/roomwidth.csv", DataFrame)

# ╔═╡ 5f2b372f-c800-4507-b754-8136b16901ed
begin
	# Convert estimates of room width from metres to feet
	roomwidth.convert = ifelse.(roomwidth.unit .== "feet", 1, 3.28)
	roomwidth.converted_width = roomwidth.width .* roomwidth.convert
end

# ╔═╡ b3f1927e-06aa-413f-bfe1-eed3e9ed08f8
begin
	
	
	# Summary statistics for feet
	feet_summary = describe(roomwidth[roomwidth.unit .== "feet", :converted_width])
	
	# Summary statistics for metres
	metres_summary = describe(roomwidth[roomwidth.unit .== "metres", :converted_width])
	
	
end

# ╔═╡ a590b1f0-33dd-46ad-869a-1ad4e971c8ab
begin
	# Standard deviations for both feet and metre estimates (converted)
	std_devs = combine(groupby(roomwidth, :unit), :converted_width => std => :std_dev)
	println(std_devs)
end

# ╔═╡ df963c9a-6866-4826-ac9a-460cfdbe3b2b
# Boxplot comparing estimates in feet and metres (converted)
@df roomwidth boxplot(:unit, :converted_width, ylabel="Estimated width (feet)", legend=false)

# ╔═╡ effab8c8-b628-42de-adcb-6d53a52d8161
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



# ╔═╡ f626b8f7-a493-49c1-8fae-6a795bf7b84f
begin
	feet_data = roomwidth[roomwidth.unit .== "feet", :converted_width]
	
	# Create the Q-Q plot
	qqplot(Normal(), feet_data, title="Q-Q Plot for Feet", xlabel="Theoretical Quantiles", ylabel="Feet Width")
	# Add the Q-Q line
	slope, intercept = qqline(feet_data)
	plot!(x -> slope * x + intercept, label="Q-Q Line", color=:red)
	
	
	
end

# ╔═╡ 1851053b-b5c4-4639-b448-b3c95cc03353
begin
	metres_data_original = roomwidth[roomwidth.unit .== "metres", :width]
	
	# Q-Q plot for original meters data (not converted to feet)
	p2 = qqplot(Normal(), metres_data_original, title="Q-Q Plot for Metres (Original)", xlabel="Theoretical Quantiles", ylabel="Estimated Width (Metres)", yticks=0:5:40, ylim=(0, 40))
	
	# Add the Q-Q line
	slope1, intercept1 = qqline(metres_data_original)
	plot!(p2, x -> slope1 * x + intercept1, label="Q-Q Line for Metres", color=:red)
	
end

# ╔═╡ f091fbe0-7ff9-4855-90da-43c483581c78
metres_data = roomwidth[roomwidth.unit .== "metres", :converted_width]  # Extract metres data as a vector


# ╔═╡ 6e9b327e-81f4-4fac-a6d3-523d1e97f7e1


# ╔═╡ 8797aed7-1262-4af9-815f-433916f4acc1
equal_variance_ttest = HypothesisTests.EqualVarianceTTest(feet_data, metres_data)


# ╔═╡ 24046ba5-4261-4d97-b86f-e61e062853eb
# ╠═╡ disabled = true
#=╠═╡
p_value = pvalue(equal_variance_ttest)
  ╠═╡ =#

# ╔═╡ 187afb8c-27d8-460e-9ca0-7269a73eeb1d
begin
	mean_feet = mean(feet_data)
end

# ╔═╡ e7dafb0e-5f00-4078-81fc-3210ae21295c
	mean_metres = mean(metres_data)


# ╔═╡ 70e3909f-7f7b-44f8-a413-ff243576e7bd
begin
	println("Equal Variance T-Test Result:")
	println(equal_variance_ttest)
end

# ╔═╡ 1d836aa4-4231-4e80-825a-6271340e8d4f
unequal_variance_ttest = HypothesisTests.UnequalVarianceTTest(feet_data, metres_data)


# ╔═╡ 67ae4170-513f-40db-91db-b6707337c230
p_value_unequal = pvalue(unequal_variance_ttest)


# ╔═╡ 20f97841-2e6d-47fc-a922-847d4e3f1241
begin
	println("Welch's (Unequal Variance) T-Test Result:")
	println(unequal_variance_ttest)
	
end

# ╔═╡ 12623e3a-5b2c-42bd-a409-9fb5b9693283
begin
	println("\nP-Value: ", p_value_unequal)
	println("Mean in group feet: ", mean_feet)
	println("Mean in group metres: ", mean_metres)
	
end

# ╔═╡ 047cc7e7-a107-4c3a-b6d8-5199e80db41d
begin
	wilcox_test = HypothesisTests.MannWhitneyUTest(feet_data, metres_data)
	println(wilcox_test)
	
end

# ╔═╡ 45dbdd86-a05e-48ee-b706-cf4a25a925f0
p_value_wilcox = pvalue(wilcox_test)

# ╔═╡ cf5383b7-233f-4c6f-95d9-783b4997d3e8
u_statistic = wilcox_test.U


# ╔═╡ b99d984d-2213-4f4e-a6e6-96784b61a3e9
# Sample estimate: difference in location
location_diff = median(feet_data) - median(metres_data)


# ╔═╡ ae59c06b-4c65-42a2-8b89-a095456f3726
begin
	# Display the results
	println("Wilcoxon Rank-Sum (Mann-Whitney U) Test Result:")
	println(wilcox_test)
	
end

# ╔═╡ 1c398664-5909-4764-9624-f2782171a624
begin
	println("\nW (U statistic): ", u_statistic)
	println("P-Value: ", p_value_wilcox)
	println("Sample Estimate (Difference in Location): ", location_diff)
	
end

# ╔═╡ 2b415bb9-5836-4fee-9f4b-7b96a8151482
md"### Piston Rings Dataset (Chi Square Test) "

# ╔═╡ f2e09ce7-b4d8-4e17-9649-d7531a274014
begin
	pistonrings_df = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/pistonrings.csv", DataFrame)
	pistonrings = Matrix(pistonrings_df)
	#Define the compressor and leg labels
	compressor_labels = ["C1", "C2", "C3", "C4"]
	leg_labels = ["North", "Centre", "South"] 
	
end

# ╔═╡ 6c50940a-0b8e-4468-8df1-52ee63f1dded
chisq_test = ChisqTest(pistonrings)


# ╔═╡ 9209ef19-8381-4fe4-8410-3362d80fa7cf
begin
	total = sum(pistonrings)
	row_sums = sum(pistonrings, dims=2)
	col_sums = sum(pistonrings, dims=1)
end

# ╔═╡ 1f1e2362-ce8f-498a-8929-5441ad0d3433
# Extract residuals
residuals = chisq_test.residuals

# ╔═╡ 2a13f0e2-d877-4fbc-a90e-998cf98a014f


# ╔═╡ 89e27878-4d50-4c35-a9be-9ba9e5dda66c
begin
	residuals_df = DataFrame(residuals, :auto)
	rename!(residuals_df, Symbol.(leg_labels))  # Rename the columns with leg labels
	residuals_df[!,:Compressor] = compressor_labels  # Add compressor labels as a column
	println(residuals_df)
end

# ╔═╡ 7e32ff8c-0eef-45af-8df9-4258e64ebac6
begin
	# Increase the height of the graph by changing the `size` parameter
	plot(legend = false, size=(600, 600), xlim=(0.5, 3.5), ylim=(0.5, 4.5), title="Association Plot of Residuals")
	
	# Loop over each compressor (y-axis) and leg (x-axis) to place rectangles
	for i in 1:4  # Compressors C1 to C4
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
	yticks!(1:4, reverse(compressor_labels))  # Reversed to match plot order
	xlabel!("Leg")
	ylabel!("Compressor")
end

# ╔═╡ d5a96796-3ce4-419d-a7d1-0d99037706d9
md"Both R and Julia yield the same Chi-squared test statistic, p-value, and residuals, leading to the same conclusion: there is insufficient evidence to reject the null hypothesis of independence. Julia’s output, however, includes additional details such as multinomial probabilities, confidence intervals, and standardized residuals, which may be useful for deeper analysis."

# ╔═╡ e85178af-0684-40ed-9a7f-c39571c2168b


# ╔═╡ 2de6f0f2-2b92-43eb-a73d-9fd45c387981
md"### Rearrests of Juveniles(McNemar and Exact binomial test)"

# ╔═╡ ec67fb3f-a749-43eb-bed3-4f37e8ca5509
rearrests_df = CSV.read("/Users/VSR/Desktop/Capstone/SimpleInference/rearrests.csv", DataFrame)
rearrests = Matrix(rearrests_df)


# ╔═╡ 899cdb7a-4acd-4a24-aeb7-bde129e72574
rearrests = Matrix(rearrests_df)


# ╔═╡ ea3006b1-fb0e-43ed-bcb4-859e3b755be2
begin
	b = rearrests[1, 2]  
	c = rearrests[2, 1] 
	
end

# ╔═╡ e1a67a30-ac33-4dec-bd7c-ffa400d2a311
chi_square = ((b - c)^2) / (b + c)


# ╔═╡ 3e1d39eb-581b-477e-94ba-3e596799296e
p_value = 1 - cdf(Chisq(1), chi_square)


# ╔═╡ 19102cc7-8b46-432e-956e-996aa107c718
begin
	println("\nP-Value: ", p_value)
	println("Mean in group feet: ", mean_feet)
	println("Mean in group metres: ", mean_metres)
	
end

# ╔═╡ 3120cf59-0987-406c-b1ce-aa766e148264
begin
	println("McNemar's Test Statistic: ", chi_square)
	println("P-value: ", p_value)
end

# ╔═╡ fe104779-9dd3-4119-afaa-20f1fd421425
begin
	successes = rearrests[2, 1]  # The value from the second row, first column (290)
	n_trials = rearrests[2, 1] + rearrests[1, 2]  # Sum of the second and first off-diagonal values (805)
	
end

# ╔═╡ 391ebd54-6a27-4b0f-9ce0-85dc9351d056
binom_test = BinomialTest(successes, n_trials, 0.5)


# ╔═╡ e6e9df9f-b50f-4506-8918-dc8a3d5734d9
conf_interval = confint(binom_test)


# ╔═╡ e35237cf-a8a8-4910-ac7f-fa6b6b17e551
begin
	println("Number of successes: ", successes)
	println("Number of trials: ", n_trials)
	println("P-value: ", pvalue(binom_test))
	println("95% Confidence Interval: ", conf_interval)
	println("Estimated Probability of Success: ", successes / n_trials)
	
end

# ╔═╡ bc277181-4705-4eb6-b05e-a8c00afc977c
md"The results from both R and Julia are consistent, indicating that your implementation in Julia is working perfectly."

# ╔═╡ 816d3763-b06e-4f99-84d2-63228f7a0f51


# ╔═╡ 3d86fef6-3cbe-4add-8cb8-c25f110f71a9
md"### Waves Dataset( One Sample t-test, mooringdiff, One Sample t-test)"

# ╔═╡ 5adde739-5dcb-4ac7-98b7-580c00ccbaea
waves_df = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/waves.csv", DataFrame)


# ╔═╡ a1c33c88-02c0-4619-830d-aad0ae3d1615
mooringdiff = waves_df.method1 .- waves_df.method2


# ╔═╡ 76ecb0e6-28f8-4e2c-9e58-4817769506b5
boxplot!(mooringdiff, ylabel="Differences (Newton metres)", title="Boxplot")

# ╔═╡ a9b3e916-0a48-4ca7-b9b5-f09c245b6e62
qqnorm(mooringdiff,xlabel =  ylabel="Differences (Newton metres)", title="QQ Plot")


# ╔═╡ 054aaba5-a36c-43e4-8666-fe411dba62c2
t_test_result = OneSampleTTest(mooringdiff, 0)


# ╔═╡ 440954b0-f92f-4493-a965-6160e2a10249
begin
	# Print the results of the t-test
	println("Two-sample t-test result:")
	println("T-statistic: ", t_test_result.t)
	println("Degrees of Freedom: ", t_test_result.df)
	println("P-value: ", pvalue(t_test_result))
	println("95% Confidence Interval: ", confint(t_test_result))
	println("Mean: ", mean(mooringdiff))
end

# ╔═╡ 2ad9c510-053d-436c-a7ac-f092b4b4db9a
md"The p-value of 0.3797 indicates that there is no statistically significant difference between the sample mean and 0 at the 5% significance level.
	•	The 95% confidence interval for the sample mean is  (-0.08258, 0.2059) , which includes 0, further supporting the conclusion that there is no significant difference.
	•	Thus, we fail to reject the null hypothesis, and there is no evidence to suggest that the true mean is different from 0.

Both R and Julia provided the same results, and the conclusion is consistent across both platforms"

# ╔═╡ 8e6b1681-7702-41dd-9beb-63d909b708b2
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

# ╔═╡ 6d8421e9-2208-441b-bbe7-731215dca23c
W, z, p_value1 = wilcoxon_signed_rank_test(mooringdiff)

# ╔═╡ 32ae55a9-41c5-4f77-8b25-051b65eeb894
begin
	# Output the results
	println("Wilcoxon signed-rank test result:")
	println("Test statistic (W): ", W)
	println("Z-value: ", z)
	println("P-value: ", p_value1)
end

# ╔═╡ 0e8cf9c1-184d-439d-84dd-821f6e7e994e
md"The p-values from both tests are similar, with Julia’s result being 0.3061 and R’s being 0.3165. Both are greater than the typical significance level of 0.05, leading to the same conclusion to Fail to reject the null hypothesis" 

# ╔═╡ d9f1c109-c4a3-4060-b488-ab8c68e9d09d
md"### Water Dataset( correlation)"

# ╔═╡ d5ce1c12-0755-47f9-8e0a-e70d40650bca


# ╔═╡ 6577d7cc-2570-451a-9756-5e439b1d7f0f


# ╔═╡ 25aca87e-ec1a-4980-9bd8-f6eb65fce683
# ╠═╡ disabled = true
#=╠═╡
water = CSV.read("/Users/VSR/Desktop/Capstone/Simple Inference/water.csv", DataFrame)

  ╠═╡ =#

# ╔═╡ 9116ea87-ff8d-4cf0-b603-6105a55178e1


# ╔═╡ 88a1c2bf-5d1c-40e2-97fd-c3cefcc048e5


# ╔═╡ 56277a4c-f567-4398-b633-1d8088916e82
md"The results you obtained in Julia match closely with those in R:

	1.	Correlation Coefficient (ρ): -0.654849 in both R and Julia.
	2.	95% Confidence Interval: (-0.7783, -0.4826), identical in both.
	3.	P-value: Both results show a very small p-value (<1e-07), indicating strong evidence against the null hypothesis of zero correlation.
	4.	T-statistic: -6.65554, matching exactly.
	5.	Degrees of Freedom: 59, also consistent across both platforms.

The Julia and R results align perfectly, showing that there is a significant negative correlation between mortality and hardness in the data."

# ╔═╡ 023f80bd-c317-4379-82c1-e7267edea220
md"# ANOVA"

# ╔═╡ 9134a25a-82de-49d5-8386-3e4231dad1f6
md"### Weightgain dataset"

# ╔═╡ 3ae400f6-607e-47cb-995a-5ccacc2c03c9


# ╔═╡ 152799e2-59a4-4ad0-9916-94b9f996bd92
weightgain = CSV.read("/Users/VSR/Desktop/Capstone/ANOVA/weightgain.csv", DataFrame)

# ╔═╡ ac9ce404-fb41-47c0-b3db-4ff97dba729d
begin
	# Calculate mean
	mean_stats = combine(groupby(weightgain, [:source, :type]), :weightgain => mean => :Mean)
	
	# Calculate standard deviation
	std_stats = combine(groupby(weightgain, [:source, :type]), :weightgain => std => :SD)
	
	# Display results
	println(mean_stats)
	println(std_stats)
end

# ╔═╡ 3344c894-773e-4c01-af0c-1f1ed8353e3d
anova_results = anova(weightgain, :weightgain, [:source, :type])

# ╔═╡ e1ad8092-0e6d-490a-bca2-24de12a5c524
println(anova_results)

# ╔═╡ 26161c4d-ff5e-4013-ad28-2e3cb25c1a84
md"The ANOVA results in Julia align closely with R, showing similar sums of squares, mean squares, F-values, and p-values for each factor and their interaction. Both outputs highlight the significance of the type factor at p = 0.0211, while source and type × source show no significant effect."

# ╔═╡ e81e5d26-38d7-4339-af39-d39b74f9bb29
md"### Foster Feeding of Rats (Two-Way ANOVA and Tukey HSD)"

# ╔═╡ bf4b07ef-a021-4ce5-9191-dbd98466ca3b
foster = CSV.read("/Users/VSR/Desktop/Capstone/ANOVA/foster.csv", DataFrame)

# ╔═╡ 99ec059d-b109-44ce-9c84-21432576c79a
begin
	# Perform ANOVA
	anova_result = anova(foster, :weight, [:litgen, :motgen])
	
	# Display the ANOVA results
	println(anova_result)
end

# ╔═╡ b70e519d-50ba-4d2a-9495-35ff950bdbfd
begin
	# Define the formula for the model
	formula1 = GLM.@formula(weight ~ litgen * motgen)
	
	# Fit the model using GLM, which can handle unbalanced designs
	foster_lm = lm(formula1, foster)
end

# ╔═╡ 507c1ed9-4827-4638-8a11-2c2e264d5701
# Display coefficient table with statistical summary
println(coeftable(foster_lm))

# ╔═╡ 7cd9b362-ebf3-4f00-8f8a-7249eaab7781
begin
	
	# Predicted values and residuals
	y_pred = fitted(foster_lm)                # Fitted (predicted) values
	y_actual = foster.weight                  # Actual response variable
	residuals_foster = y_actual - y_pred             # Residuals
	
	# Total Sum of Squares (SST)
	mean_y = mean(y_actual)
	sst = sum((y_actual .- mean_y).^2)
	
	# Residual Sum of Squares (SSE)
	sse = sum(residuals_foster.^2)
	
	# Model Sum of Squares (SSM) - Explained by the model
	ssm = sst - sse
end

# ╔═╡ f14cc1be-b3fb-4af3-86a9-4589e84b2da6
begin
	# Degrees of Freedom
	no = length(y_actual)              # Total observations
	k = length(coef(foster_lm)) - 1   # Number of predictors (excluding intercept)
	df_total = no - 1
	df_model = k
	df_residual = df_total - df_model
	
	# Mean Squares
	ms_model = ssm / df_model
	ms_residual = sse / df_residual
	
	# F-statistic for the model
	f_statistic = ms_model / ms_residual
	
	# p-value for the F-statistic
	p_value2 = 1 - cdf(FDist(df_model, df_residual), f_statistic)
	
	
end

# ╔═╡ 1355f643-926d-44fa-8334-7f79f247d0d1
begin
	# Display results
		println("ANOVA Summary:")
		println("Total Sum of Squares (SST): ", sst)
		println("Model Sum of Squares (SSM): ", ssm)
		println("Residual Sum of Squares (SSE): ", sse)
		println("Degrees of Freedom (Model): ", df_model)
		println("Degrees of Freedom (Residual): ", df_residual)
		println("Mean Square (Model): ", ms_model)
		println("Mean Square (Residual): ", ms_residual)
		println("F-statistic: ", f_statistic)
		println("P-value: ", p_value2)
end

# ╔═╡ b7c88aec-89e9-413b-9cac-82ffb92f38ad
begin
	motgen_levels = unique(foster.motgen)
	
	# Calculate means for each level of `motgen`
	motgen_means = combine(groupby(foster, :motgen), :weight => mean => :mean_weight)
	
	# Perform pairwise comparisons with Tukey's adjustment
	alpha = 0.05
	n_groups = length(motgen_levels)
	n_total = nrow(foster)
	df_residual1 = n_total - n_groups
	ms_residual1 = deviance(foster_lm) / df_residual1  # Mean square error
	
	# Store pairwise results
	results = DataFrame(Level1 = String[], Level2 = String[], MeanDiff = Float64[], LowerCI = Float64[], UpperCI = Float64[], p_value = Float64[])
	
	for i in 1:(n_groups-1)
	    for j in (i+1):n_groups
	        level1, level2 = motgen_levels[i], motgen_levels[j]
	        mean1 = motgen_means[findfirst(==(level1), motgen_means.motgen), :mean_weight]
	        mean2 = motgen_means[findfirst(==(level2), motgen_means.motgen), :mean_weight]
	        
	        # Calculate mean difference and standard error
	        mean_diff = abs(mean1 - mean2)
	        std_error = sqrt(ms_residual1 * (2 / n_total))
	        
	        # Calculate Tukey's HSD critical value
	        q_critical = quantile(StudentizedRange(n_groups, df_residual1), 1 - alpha / 2)
	        margin = q_critical * std_error
	        
	        # Confidence interval
	        lower_ci = mean_diff - margin
	        upper_ci = mean_diff + margin
	
	        # Calculate p-value (two-sided)
	        t_statistic = mean_diff / std_error
	        p_value = 2 * (1 - cdf(TDist(df_residual1), abs(t_statistic)))
	        
	        # Append results
	        push!(results, (level1, level2, mean_diff, lower_ci, upper_ci, p_value))
	    end
	end
	
	# Display pairwise comparisons
	println(results)
end

# ╔═╡ c4368266-a77f-4b4a-8eb8-e6726015fd29
md"## MANOVA"

# ╔═╡ 41b9535d-17a0-4967-b064-bff06259548b
md"## Water Hardness"

# ╔═╡ 92308194-eaea-4c2f-80da-5e4a3064acbc
water = CSV.read("/Users/VSR/Desktop/Capstone/ANOVA/water.csv", DataFrame)

# ╔═╡ 131f4fe5-fd45-495b-b62f-aeefd7a7252c
begin
	unique_locations = unique(water.location)
	psymb = [findfirst(==(loc), unique_locations) for loc in water.location]
	
end

# ╔═╡ b9c559c0-1f1e-4819-9722-7a10fe38d66c
water.location = CategoricalArray(water.location)


# ╔═╡ 103bb0b4-7f8f-4eb3-970b-a61d56f69632
p1 = @df water scatter(:hardness, :mortality, group=:location,
    markershape=[:circle :square], label=["North" "South"], 
    color=[:blue :green], # Distinct colors for groups
    title="Mortality vs Hardness", legend=:topright)


# ╔═╡ 52fb5d9d-3a01-40e8-bda3-a4eb73f16435
# Fit a linear model (lm)
lm_model = lm(@formula(mortality ~ hardness), water)


# ╔═╡ 4a71b95c-6a00-4b35-a91f-a242122a729d
# Generate predictions from the linear model
predicted_values = predict(lm_model)


# ╔═╡ b980b430-cc5a-4f16-8e6d-2eed48a714c6
# Plot the regression line based on the fitted model
@df water plot!(water.hardness, predicted_values, label="Regression Line", lw=2)


# ╔═╡ 744ae223-b026-40eb-9c63-17a04712239d
histogram(water.hardness, 
    title="Histogram of Hardness", 
    xlabel="Hardness", 
    ylabel="Frequency", 
    bins=10, # You can adjust the number of bins
    color=:blue, 
    legend=false)

# ╔═╡ c5572c60-58e7-47e7-b214-095fea7e0d43
boxplot(water.mortality, 
    title="Boxplot of Mortality", 
    ylabel="Mortality", 
    color=:lightblue, 
    legend=false)

# ╔═╡ dc981b52-b0e8-4696-b3b6-9703ef308b91
begin
	correlation_coefficient = cor(water.mortality, water.hardness)
	
	
	correlation_test = CorrelationTest(water.mortality, water.hardness)
	
	
	
end

# ╔═╡ b98bce10-7237-47e5-8bcd-b7356a3afb56
begin
	Y = Matrix(water[:, [:hardness, :mortality]])
	
	# Encode location as dummy variables
	water.location = categorical(water.location)
	x = hcat(ones(nrow(water)), Matrix(water[:, [:location]]))
	
end

# ╔═╡ d0cc5752-31be-40eb-98ab-395fd1ba4536
begin
	Y_bar = mean(Y, dims=1)
	
	# Compute the total SSCP matrix
	T = (Y .- Y_bar)' * (Y .- Y_bar)
	
	# Compute the within-group SSCP matrix
	groups = groupby(water, :location)
	WT = zeros(2, 2)
	for g in groups
	    Y_g = Matrix(g[:, [:hardness, :mortality]])
	    Y_g_bar = mean(Y_g, dims=1)
	    WT += (Y_g .- Y_g_bar)' * (Y_g .- Y_g_bar)
	end
	
	# Between-group SSCP matrix
	B = T - WT
	
end

# ╔═╡ 3d447f1a-8f03-4c93-8d61-972e3005edc5
begin
	# use LinearAlgebra's eigvals function
	eigvals_ = LinearAlgebra.eigvals
	
	# Compute the Hotelling-Lawley trace
	W_inv = inv(WT)
	eigvals_result = eigvals_(W_inv * B)  # Use the alias eigvals_
	hotelling_lawley_trace = sum(eigvals_result)
	
	println("Hotelling-Lawley Trace: ", hotelling_lawley_trace)
	
end

# ╔═╡ 6649a4e3-516f-43ae-ba28-1a7e2d42e218
md"The Hotelling-Lawley Trace in Julia is 0.90021, which aligns with the Hotelling-Lawley approximation F-value of 26.106 reported in R."

# ╔═╡ fe446dd4-2632-4e6b-ac43-a48da6be747d


# ╔═╡ efddab82-1394-4bf8-bf47-788076ce0985
md"### Skulls Dataset(MANOVA)"

# ╔═╡ 21002f9f-a82c-4483-9648-48337bd987ce
skull_data = CSV.read("/Users/VSR/Desktop/Capstone/ANOVA/skulls.csv", DataFrame)


# ╔═╡ 64ba3a5c-9e91-424c-a65c-938b54a628d1
begin
	# Response matrix Y (dependent variables: mb, bh, bl, nh)
	response_matrix = Matrix(skull_data[:, [:mb, :bh, :bl, :nh]])
	
	# Encode epoch as categorical
	skull_data.epoch = categorical(skull_data.epoch)
	design_matrix = hcat(ones(nrow(skull_data)), Matrix(skull_data[:, [:epoch]]))
end

# ╔═╡ f7a6aeb5-56a4-42a2-8f68-4c43e5597d56
begin
    # Mean vector of Y
    mean_vector = mean(response_matrix, dims=1)
    
    # Compute the total SSCP matrix
    total_sscp = (response_matrix .- mean_vector)' * (response_matrix .- mean_vector)
    
    # Compute the within-group SSCP matrix
    groups1 = groupby(skull_data, :epoch)
    within_group_sscp = zeros(4, 4)  # There are 4 measurements: mb, bh, bl, nh
    for group in groups1
        group_data = Matrix(group[:, [:mb, :bh, :bl, :nh]])
        group_mean = mean(group_data, dims=1)
        within_group_sscp += (group_data .- group_mean)' * (group_data .- group_mean)
    end
    
    # Between-group SSCP matrix
    between_group_sscp = total_sscp - within_group_sscp
end

# ╔═╡ 1099b39b-8b72-4fc2-82b6-1094b13d65e9
begin
    # Inverse of the within-group SSCP matrix
    within_group_inv = inv(within_group_sscp)
    
    # Compute eigenvalues of W⁻¹B
    eigenvalues_result = LinearAlgebra.eigvals(within_group_inv * between_group_sscp)
    
    # Hotelling-Lawley trace is the sum of the eigenvalues
    hotelling_lawley_trace1 = sum(eigenvalues_result)
    
    println("Hotelling-Lawley Trace: ", hotelling_lawley_trace1)
end

# ╔═╡ 25d5d567-18e9-4693-b647-c95a95b9dd7f
begin
    # Define trace function (sum of diagonal elements)
    trace_matrix(M) = sum(diag(M))

    # Compute the matrix (within_group + between_group)
    combined_matrix = within_group_sscp + between_group_sscp

    # Invert the (within_group + between_group) matrix
    combined_matrix_inv = inv(combined_matrix)

    # Compute the product (W + B)⁻¹ * B
    product_matrix = combined_matrix_inv * between_group_sscp

    # Compute Pillai's trace (sum of the diagonal elements)
    pillai_trace = trace_matrix(product_matrix)

    println("Pillai's Trace: ", pillai_trace)
end


# ╔═╡ 02f5240c-dea3-421d-b5e6-029fc5d448e4
begin
    # Compute the determinant of within-group and (W + B)
    det_within = det(within_group_sscp)
    det_combined = det(combined_matrix)

    # Compute Wilks' Lambda
    wilks_lambda = det_within / det_combined

    println("Wilks' Lambda: ", wilks_lambda)
end

# ╔═╡ c9191a5e-edd8-456f-96a9-076210d5a262
begin
    # Compute Roy's Largest Root
    roy_largest_root = maximum(eigen(product_matrix).values)

    println("Roy's Largest Root: ", roy_largest_root)
end

# ╔═╡ cd953ac2-6779-4d15-bf19-98a713d5355f
begin
	anova_mb = anova(skull_data, :mb, [:epoch])
	anova_bh = anova(skull_data, :bh, [:epoch])
	anova_bl = anova(skull_data, :bl, [:epoch])
	anova_nh = anova(skull_data, :nh, [:epoch])
	
	# Display results
	println("ANOVA for mb:")
	println(anova_mb)
	println("\nANOVA for bh:")
	println(anova_bh)
	println("\nANOVA for bl:")
	println(anova_bl)
	println("\nANOVA for nh:")
	println(anova_nh)
end

# ╔═╡ 2ca819bd-763b-42d1-a1b9-938b0f572e68
md" the results from Julia and R are consistent across all metrics"

# ╔═╡ 3f172d18-9719-42c8-b3a1-0948fd465d6c


# ╔═╡ f4c35c69-9068-422b-a1c8-793d1d923d21


# ╔═╡ 7b8ceade-d907-467f-b2cd-5f70b785d7ab


# ╔═╡ 97c26e17-0f2a-40c9-a31e-0dc8b323e9fc


# ╔═╡ b3027a00-0739-406e-8a50-e28a1ddd0b41


# ╔═╡ 2df7671e-aca9-4b33-8f80-06332ed4ec04


# ╔═╡ e7ccc49f-3ec7-4f5d-a8fc-59955c039e3e


# ╔═╡ 27354a1b-556d-4179-a4df-1232b78f2dcc


# ╔═╡ 4fefb7ad-99e7-4a5e-ac42-f64d40597101


# ╔═╡ 680a6074-6a15-4815-aa9e-8dbaa045d9fb


# ╔═╡ 667edbf5-651a-43a7-a0f3-64cefd6711e2


# ╔═╡ f1f5b90b-192c-4015-85b1-0ccb095c8101


# ╔═╡ 5095626f-3f19-488f-92ee-28d524a72c05


# ╔═╡ 71ea19ff-6246-40f0-8903-62d32100d11c


# ╔═╡ f31c6ad0-eb5f-4078-a395-c5de7c71828c


# ╔═╡ e6e281f2-0a6e-45f6-b663-2ba216644fc8


# ╔═╡ a229308c-5778-4e08-bf01-8aeccf4ce3f3


# ╔═╡ 39c35d65-7dec-4e4c-85f6-29320195402b


# ╔═╡ 57b1ac3d-c2f2-48be-99e1-9f771282af4c


# ╔═╡ 1923ffad-32ef-4b6e-9345-0e5acb08ffff


# ╔═╡ 13b7e4d2-7115-4d9c-8e64-48476fa2bbf7


# ╔═╡ fbf5b367-d81d-4fe0-821a-6b33a84468a7


# ╔═╡ c107f7c0-49fe-4de7-922e-beb9fb88328a


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
HypothesisTests = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
SimpleANOVA = "fff527a3-8410-504e-9ca3-60d5e79bb1e4"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.14"
CategoricalArrays = "~0.10.8"
DataFrames = "~1.7.0"
Distributions = "~0.25.112"
GLM = "~1.9.0"
HypothesisTests = "~0.11.3"
MultivariateStats = "~0.10.3"
Plots = "~1.40.8"
PrettyTables = "~2.4.0"
SimpleANOVA = "~0.8.2"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "fcd5bba47812ac8292b21a6a826d3103fad3ba7d"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "b392ede862e506d451fc1616e79aa6f4c673dab8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.38"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "6c834533dc1fabd820c1db03c839bf97e45a3fab"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.14"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "d7477ecdafb813ddee2ae727afa94e9dcb5f3fb0"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.112"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "629693584cef594c3f6f99e76e7a7ad17e60e8d5"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a8863b69c2a0859f2c2c87ebdc4c6712e88bdf0d"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.7+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "7c4195be1649ae622304031ed46a2f4df989f1eb"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.24"

[[deps.HypothesisTests]]
deps = ["Combinatorics", "Distributions", "LinearAlgebra", "Printf", "Random", "Rmath", "Roots", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "6c3ce99fdbaf680aa6716f4b919c19e902d67c9c"
uuid = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
version = "0.11.3"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "39d64b09147620f5ffbf6b2d3255be3c901bec63"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.8"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "f389674c99bfcde17dc57454011aa44d5a260a40"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "3cebfc94a0754cc329ebc3bab1e6c89621e791ad"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "6e55c6841ce3411ccb3457ee52fc48cb698d6fb0"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.2.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "45470145863035bb124ca51b320ed35d071cc6c2"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.8"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "3a7c7e5c3f015415637f5debdf8a674aa2c979c4"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.1"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ff11acffdb082493657550959d4feb4b6149e73a"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.5"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleANOVA]]
deps = ["CategoricalArrays", "Combinatorics", "Distributions", "InvertedIndices", "Requires", "Statistics"]
git-tree-sha1 = "77915e352267a4ea37b47ee329d37a82854c046d"
uuid = "fff527a3-8410-504e-9ca3-60d5e79bb1e4"
version = "0.8.2"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "9022bcaa2fc1d484f1326eaa4db8db543ca8c66d"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.4"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "1165b0443d0eca63ac1e32b8c0eb69ed2f4f8127"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "936081b536ae4aa65415d869287d43ef3cb576b2"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.53.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═ac1a58b5-ef21-4cad-bc48-225a83a5eb3f
# ╠═e67bbc24-85ad-11ef-3b1a-3196ad739f27
# ╟─5d804c63-60ae-47e1-92d5-98fd8815dcc8
# ╠═5477ed79-0911-41b2-8b6f-93dde49a22c3
# ╠═0a7c07c1-c0f4-4b1d-865b-ec9b9c506006
# ╠═870c766b-c0fc-459e-bd83-d7f1e438a4b6
# ╠═702adce5-bebe-4662-8b73-f91da402522f
# ╠═c6d2ba6f-0b82-41e7-8431-c62774ccaae7
# ╠═1e49fc14-de7e-40c0-9145-9f175c063bae
# ╠═9c49a264-ce78-4cf0-b21b-cbf8b5b04ed9
# ╠═f4924357-e978-4f31-8474-652c3ad73019
# ╠═8c44db6c-9556-46c0-b60a-b270700025b5
# ╠═cd8d00c2-354b-44c3-b1de-721edf41a41d
# ╠═64143f68-9eb9-4e14-855b-123c2813dfb3
# ╠═6f1b8760-c928-4c40-bb90-cc93daad5577
# ╠═8195f5ae-4625-4547-91e4-e3537dfb0338
# ╠═b39c4573-77e6-42b8-85fa-b7664231d67d
# ╠═21a7b431-b82d-4bfa-b7c2-13646265281a
# ╠═cb8aefb1-bb13-4f4f-ac59-fe3c41e7db62
# ╠═c3164907-8584-4d65-b98a-3855b9fd299e
# ╠═3e8df5f8-bb44-40e1-a111-8a6455b36817
# ╠═c16999a7-a93b-4df6-99cf-e7367c06d4b6
# ╠═5c1e3029-1a31-4b15-8357-d2f800cd6535
# ╠═c99e9560-8a22-4026-998f-454f6f18af2c
# ╠═9228ffb0-e188-4dd0-a88b-dafaf7274a9d
# ╠═a0772d4c-2426-4a1c-b62d-569a5b6eee69
# ╠═20bbedda-2d40-4e7f-8b54-f0fa15282a2b
# ╠═1cefa728-c820-4303-820a-ee15ef55e4b1
# ╟─320cffaf-12ab-4ef9-8449-014231ae0c8e
# ╠═615ac17b-a752-4dac-85ae-19bb61bd2818
# ╟─82d0e06b-1338-4dc5-be13-eac736de6fd4
# ╠═3719a7f5-9d24-48c0-becf-6ebde74647b0
# ╠═a4036c49-d98f-408c-81e7-5b3d1579b9a2
# ╠═5f2b372f-c800-4507-b754-8136b16901ed
# ╠═b3f1927e-06aa-413f-bfe1-eed3e9ed08f8
# ╠═a590b1f0-33dd-46ad-869a-1ad4e971c8ab
# ╠═df963c9a-6866-4826-ac9a-460cfdbe3b2b
# ╠═effab8c8-b628-42de-adcb-6d53a52d8161
# ╠═f626b8f7-a493-49c1-8fae-6a795bf7b84f
# ╠═1851053b-b5c4-4639-b448-b3c95cc03353
# ╠═f091fbe0-7ff9-4855-90da-43c483581c78
# ╠═6e9b327e-81f4-4fac-a6d3-523d1e97f7e1
# ╠═8797aed7-1262-4af9-815f-433916f4acc1
# ╠═24046ba5-4261-4d97-b86f-e61e062853eb
# ╠═187afb8c-27d8-460e-9ca0-7269a73eeb1d
# ╠═e7dafb0e-5f00-4078-81fc-3210ae21295c
# ╠═70e3909f-7f7b-44f8-a413-ff243576e7bd
# ╠═19102cc7-8b46-432e-956e-996aa107c718
# ╠═1d836aa4-4231-4e80-825a-6271340e8d4f
# ╠═67ae4170-513f-40db-91db-b6707337c230
# ╠═20f97841-2e6d-47fc-a922-847d4e3f1241
# ╠═12623e3a-5b2c-42bd-a409-9fb5b9693283
# ╠═047cc7e7-a107-4c3a-b6d8-5199e80db41d
# ╠═45dbdd86-a05e-48ee-b706-cf4a25a925f0
# ╠═cf5383b7-233f-4c6f-95d9-783b4997d3e8
# ╠═b99d984d-2213-4f4e-a6e6-96784b61a3e9
# ╠═ae59c06b-4c65-42a2-8b89-a095456f3726
# ╠═1c398664-5909-4764-9624-f2782171a624
# ╠═2b415bb9-5836-4fee-9f4b-7b96a8151482
# ╠═f2e09ce7-b4d8-4e17-9649-d7531a274014
# ╠═6c50940a-0b8e-4468-8df1-52ee63f1dded
# ╠═9209ef19-8381-4fe4-8410-3362d80fa7cf
# ╠═1f1e2362-ce8f-498a-8929-5441ad0d3433
# ╠═2a13f0e2-d877-4fbc-a90e-998cf98a014f
# ╠═89e27878-4d50-4c35-a9be-9ba9e5dda66c
# ╠═7e32ff8c-0eef-45af-8df9-4258e64ebac6
# ╟─d5a96796-3ce4-419d-a7d1-0d99037706d9
# ╠═e85178af-0684-40ed-9a7f-c39571c2168b
# ╠═2de6f0f2-2b92-43eb-a73d-9fd45c387981
# ╠═ec67fb3f-a749-43eb-bed3-4f37e8ca5509
# ╠═899cdb7a-4acd-4a24-aeb7-bde129e72574
# ╠═ea3006b1-fb0e-43ed-bcb4-859e3b755be2
# ╠═e1a67a30-ac33-4dec-bd7c-ffa400d2a311
# ╠═3e1d39eb-581b-477e-94ba-3e596799296e
# ╠═3120cf59-0987-406c-b1ce-aa766e148264
# ╠═fe104779-9dd3-4119-afaa-20f1fd421425
# ╠═391ebd54-6a27-4b0f-9ce0-85dc9351d056
# ╠═e6e9df9f-b50f-4506-8918-dc8a3d5734d9
# ╠═e35237cf-a8a8-4910-ac7f-fa6b6b17e551
# ╟─bc277181-4705-4eb6-b05e-a8c00afc977c
# ╠═816d3763-b06e-4f99-84d2-63228f7a0f51
# ╟─3d86fef6-3cbe-4add-8cb8-c25f110f71a9
# ╠═5adde739-5dcb-4ac7-98b7-580c00ccbaea
# ╠═a1c33c88-02c0-4619-830d-aad0ae3d1615
# ╠═76ecb0e6-28f8-4e2c-9e58-4817769506b5
# ╠═a9b3e916-0a48-4ca7-b9b5-f09c245b6e62
# ╠═054aaba5-a36c-43e4-8666-fe411dba62c2
# ╠═440954b0-f92f-4493-a965-6160e2a10249
# ╟─2ad9c510-053d-436c-a7ac-f092b4b4db9a
# ╠═8e6b1681-7702-41dd-9beb-63d909b708b2
# ╠═6d8421e9-2208-441b-bbe7-731215dca23c
# ╠═32ae55a9-41c5-4f77-8b25-051b65eeb894
# ╟─0e8cf9c1-184d-439d-84dd-821f6e7e994e
# ╟─d9f1c109-c4a3-4060-b488-ab8c68e9d09d
# ╠═d5ce1c12-0755-47f9-8e0a-e70d40650bca
# ╠═6577d7cc-2570-451a-9756-5e439b1d7f0f
# ╠═25aca87e-ec1a-4980-9bd8-f6eb65fce683
# ╠═131f4fe5-fd45-495b-b62f-aeefd7a7252c
# ╠═9116ea87-ff8d-4cf0-b603-6105a55178e1
# ╠═88a1c2bf-5d1c-40e2-97fd-c3cefcc048e5
# ╠═b9c559c0-1f1e-4819-9722-7a10fe38d66c
# ╠═103bb0b4-7f8f-4eb3-970b-a61d56f69632
# ╠═52fb5d9d-3a01-40e8-bda3-a4eb73f16435
# ╠═4a71b95c-6a00-4b35-a91f-a242122a729d
# ╠═b980b430-cc5a-4f16-8e6d-2eed48a714c6
# ╠═744ae223-b026-40eb-9c63-17a04712239d
# ╠═c5572c60-58e7-47e7-b214-095fea7e0d43
# ╠═dc981b52-b0e8-4696-b3b6-9703ef308b91
# ╠═56277a4c-f567-4398-b633-1d8088916e82
# ╟─023f80bd-c317-4379-82c1-e7267edea220
# ╟─9134a25a-82de-49d5-8386-3e4231dad1f6
# ╠═3ae400f6-607e-47cb-995a-5ccacc2c03c9
# ╠═152799e2-59a4-4ad0-9916-94b9f996bd92
# ╠═ac9ce404-fb41-47c0-b3db-4ff97dba729d
# ╠═3344c894-773e-4c01-af0c-1f1ed8353e3d
# ╠═e1ad8092-0e6d-490a-bca2-24de12a5c524
# ╟─26161c4d-ff5e-4013-ad28-2e3cb25c1a84
# ╟─e81e5d26-38d7-4339-af39-d39b74f9bb29
# ╠═bf4b07ef-a021-4ce5-9191-dbd98466ca3b
# ╠═99ec059d-b109-44ce-9c84-21432576c79a
# ╠═b70e519d-50ba-4d2a-9495-35ff950bdbfd
# ╠═507c1ed9-4827-4638-8a11-2c2e264d5701
# ╠═7cd9b362-ebf3-4f00-8f8a-7249eaab7781
# ╠═f14cc1be-b3fb-4af3-86a9-4589e84b2da6
# ╠═1355f643-926d-44fa-8334-7f79f247d0d1
# ╠═b7c88aec-89e9-413b-9cac-82ffb92f38ad
# ╟─c4368266-a77f-4b4a-8eb8-e6726015fd29
# ╠═41b9535d-17a0-4967-b064-bff06259548b
# ╠═92308194-eaea-4c2f-80da-5e4a3064acbc
# ╠═b98bce10-7237-47e5-8bcd-b7356a3afb56
# ╠═d0cc5752-31be-40eb-98ab-395fd1ba4536
# ╠═3d447f1a-8f03-4c93-8d61-972e3005edc5
# ╟─6649a4e3-516f-43ae-ba28-1a7e2d42e218
# ╠═fe446dd4-2632-4e6b-ac43-a48da6be747d
# ╟─efddab82-1394-4bf8-bf47-788076ce0985
# ╠═21002f9f-a82c-4483-9648-48337bd987ce
# ╠═64ba3a5c-9e91-424c-a65c-938b54a628d1
# ╠═f7a6aeb5-56a4-42a2-8f68-4c43e5597d56
# ╠═1099b39b-8b72-4fc2-82b6-1094b13d65e9
# ╠═25d5d567-18e9-4693-b647-c95a95b9dd7f
# ╠═02f5240c-dea3-421d-b5e6-029fc5d448e4
# ╠═c9191a5e-edd8-456f-96a9-076210d5a262
# ╠═cd953ac2-6779-4d15-bf19-98a713d5355f
# ╟─2ca819bd-763b-42d1-a1b9-938b0f572e68
# ╠═3f172d18-9719-42c8-b3a1-0948fd465d6c
# ╠═f4c35c69-9068-422b-a1c8-793d1d923d21
# ╠═7b8ceade-d907-467f-b2cd-5f70b785d7ab
# ╠═97c26e17-0f2a-40c9-a31e-0dc8b323e9fc
# ╠═b3027a00-0739-406e-8a50-e28a1ddd0b41
# ╠═2df7671e-aca9-4b33-8f80-06332ed4ec04
# ╠═e7ccc49f-3ec7-4f5d-a8fc-59955c039e3e
# ╠═27354a1b-556d-4179-a4df-1232b78f2dcc
# ╠═4fefb7ad-99e7-4a5e-ac42-f64d40597101
# ╠═680a6074-6a15-4815-aa9e-8dbaa045d9fb
# ╠═667edbf5-651a-43a7-a0f3-64cefd6711e2
# ╠═f1f5b90b-192c-4015-85b1-0ccb095c8101
# ╠═5095626f-3f19-488f-92ee-28d524a72c05
# ╠═71ea19ff-6246-40f0-8903-62d32100d11c
# ╠═f31c6ad0-eb5f-4078-a395-c5de7c71828c
# ╠═e6e281f2-0a6e-45f6-b663-2ba216644fc8
# ╠═a229308c-5778-4e08-bf01-8aeccf4ce3f3
# ╠═39c35d65-7dec-4e4c-85f6-29320195402b
# ╠═57b1ac3d-c2f2-48be-99e1-9f771282af4c
# ╠═1923ffad-32ef-4b6e-9345-0e5acb08ffff
# ╠═13b7e4d2-7115-4d9c-8e64-48476fa2bbf7
# ╠═fbf5b367-d81d-4fe0-821a-6b33a84468a7
# ╠═c107f7c0-49fe-4de7-922e-beb9fb88328a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
