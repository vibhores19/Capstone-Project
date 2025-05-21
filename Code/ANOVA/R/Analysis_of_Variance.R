# Load the required package and dataset
library(HSAUR)
data("weightgain", package = "HSAUR")

# Summary statistics: Mean and standard deviation
# Mean of weight gain by source and type
mean_stats <- tapply(weightgain$weightgain, list(weightgain$source, weightgain$type), mean)
print(mean_stats)

# Standard deviation of weight gain by source and type
sd_stats <- tapply(weightgain$weightgain, list(weightgain$source, weightgain$type), sd)
print(sd_stats)

# Two-way ANOVA with interaction terms
wg_aov <- aov(weightgain ~ source * type, data = weightgain)
summary(wg_aov)

# Extracting the coefficients from the ANOVA model
coef(wg_aov)

# Display the contrasts used
options("contrasts")

# Adjust plot margins to avoid figure margins too large error
par(mar = c(5, 4, 4, 2) + 0.1)

# Plot of mean weight gain for each factor level
plot.design(weightgain)

# Interaction plot for weightgain data
interaction.plot(weightgain$type, weightgain$source, weightgain$weightgain)

# save the plots as PNG files in case of plot size issues
# Save plot as PNG
# png("/Users/mozumdertushar/Desktop/weightgain_design.png", width = 800, height = 600)
# plot.design(weightgain)
# dev.off()

# Save interaction plot as PNG
# png("/Users/mozumdertushar/Desktop/weightgain_interaction.png", width = 800, height = 600)
# interaction.plot(weightgain$type, weightgain$source, weightgain$weightgain)
# dev.off()

# Re-run ANOVA with Pi γi = 0 constraint
wg_aov_constr <- aov(weightgain ~ source + type + source:type, data = weightgain, contrasts = list(source = contr.sum))

# Extracting the coefficients with the new contrast
coef(wg_aov_constr)

