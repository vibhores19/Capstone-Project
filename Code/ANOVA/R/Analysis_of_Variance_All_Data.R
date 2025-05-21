###################################################
### Weight Gain in Rats (Two-Way ANOVA)
###################################################

# Load the required package and dataset
library(HSAUR)

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Weight Gain in Rats Analysis
time_weightgain <- system.time({
  data("weightgain", package = "HSAUR")
  
  # Summary statistics: Mean and standard deviation
  print(tapply(weightgain$weightgain, list(weightgain$source, weightgain$type), mean))
  print(tapply(weightgain$weightgain, list(weightgain$source, weightgain$type), sd))
  
  # Two-way ANOVA with interaction terms
  wg_aov <- aov(weightgain ~ source * type, data = weightgain)
  print(summary(wg_aov))
  
  # Extracting the coefficients
  print(coef(wg_aov))
  
  # Adjust plot margins to avoid figure margins too large error
  par(mar = c(5, 4, 4, 2) + 0.1)
  
  # Plot of mean weight gain for each factor level
  plot.design(weightgain)
  
  # Interaction plot for weightgain data
  interaction.plot(weightgain$type, weightgain$source, weightgain$weightgain)
})

# Print runtime for Weight Gain in Rats Analysis
print(time_weightgain)

###################################################
### Foster Feeding of Rats (Two-Way ANOVA and Tukey HSD)
###################################################

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Foster Feeding of Rats Analysis
time_foster <- system.time({
  # Load the foster dataset
  data("foster", package = "HSAUR")
  
  # Two-way ANOVA for litter and mother genotypes
  foster_aov <- aov(weight ~ litgen * motgen, data = foster)
  print(summary(foster_aov))
  
  # Multiple comparisons with Tukey's Honest Significant Differences (HSD)
  foster_hsd <- TukeyHSD(foster_aov, "motgen")
  print(foster_hsd)
  
  # Adjust margins to avoid figure margins too large error
  par(mar = c(5, 4, 4, 2) + 0.1)
  
  # Plot the Tukey HSD results
  plot(foster_hsd)
  
  # Plot of mean litter weight for each factor level
  plot.design(foster)
})

# Print runtime for Foster Feeding of Rats Analysis
print(time_foster)

###################################################
### Water Hardness and Mortality (MANOVA)
###################################################

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Water Hardness and Mortality Analysis
time_water_hardness <- system.time({
  # Load the water dataset
  data("water", package = "HSAUR")
  
  # MANOVA: Hotelling-Lawley test
  print(summary(manova(cbind(hardness, mortality) ~ location, data = water), test = "Hotelling-Lawley"))
  
  # Summary of means for water hardness and mortality by location
  print(tapply(water$hardness, water$location, mean))
  print(tapply(water$mortality, water$location, mean))
})

# Print runtime for Water Hardness and Mortality Analysis
print(time_water_hardness)

###################################################
### Male Egyptian Skulls (MANOVA)
###################################################

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Male Egyptian Skulls Analysis
time_skulls <- system.time({
  # Load the skulls dataset
  data("skulls", package = "HSAUR")
  
  # MANOVA on skull measurements across epochs
  skulls_manova <- manova(cbind(mb, bh, bl, nh) ~ epoch, data = skulls)
  
  # Perform MANOVA with different test criteria
  print(summary(skulls_manova, test = "Pillai"))
  print(summary(skulls_manova, test = "Wilks"))
  print(summary(skulls_manova, test = "Hotelling-Lawley"))
  print(summary(skulls_manova, test = "Roy"))
  
  # Univariate ANOVA for each skull measurement
  print(summary.aov(skulls_manova))
})

# Print runtime for Male Egyptian Skulls Analysis
print(time_skulls)
