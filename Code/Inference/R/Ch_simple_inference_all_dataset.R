###################################################
### Room Width Data Analysis
###################################################

# Load the required package
library(HSAUR)

# Reset graphics device
while (!is.null(dev.list())) dev.off()

# Measure runtime for Room Width Analysis
time_roomwidth <- system.time({
  # Load the roomwidth dataset
  data("roomwidth", package = "HSAUR")
  
  # Convert measurements: 1 for feet, 3.28 for metres
  convert <- ifelse(roomwidth$unit == "feet", 1, 3.28)
  
  # Summary statistics
  print(tapply(roomwidth$width * convert, roomwidth$unit, summary))
  
  # Standard deviations
  print(tapply(roomwidth$width * convert, roomwidth$unit, sd))
  
  # Adjust margins to avoid "figure margins too large" error
  par(mar = c(5, 4, 4, 2) + 0.1)
  
  # Plot: Boxplot and QQ plots for roomwidth data
  layout(matrix(c(1,2,1,3), nrow = 2, ncol = 2, byrow = FALSE))
  boxplot(I(width * convert) ~ unit, data = roomwidth,
          ylab = "Estimated width (feet)",
          varwidth = TRUE, names = c("Estimates in feet", "Estimates in metres (converted to feet)"))
  feet <- roomwidth$unit == "feet"
  qqnorm(roomwidth$width[feet], ylab = "Estimated width (feet)")
  qqline(roomwidth$width[feet])
  qqnorm(roomwidth$width[!feet], ylab = "Estimated width (metres)")
  qqline(roomwidth$width[!feet])
  
  # Two-sample t-test assuming equal variances
  print(t.test(I(width * convert) ~ unit, data = roomwidth, var.equal = TRUE))
  
  # Two-sample t-test assuming unequal variances
  print(t.test(I(width * convert) ~ unit, data = roomwidth, var.equal = FALSE))
  
  # Wilcoxon rank sum test
  print(wilcox.test(I(width * convert) ~ unit, data = roomwidth, conf.int = TRUE))
})

# Print runtime for Room Width Analysis
print(time_roomwidth)

###################################################
### Wave Energy Device Mooring Data Analysis
###################################################

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Wave Energy Device Mooring Analysis
time_waves <- system.time({
  # Load the waves dataset
  data("waves", package = "HSAUR")
  
  # Differences between two mooring methods
  mooringdiff <- waves$method1 - waves$method2
  
  # Adjust margins to avoid "figure margins too large" error
  par(mar = c(5, 4, 4, 2) + 0.1)
  
  # Plot: Boxplot and QQ plot of differences
  layout(matrix(1:2, ncol = 2))
  boxplot(mooringdiff, ylab = "Differences (Newton metres)", main = "Boxplot")
  abline(h = 0, lty = 2)
  qqnorm(mooringdiff, ylab = "Differences (Newton metres)")
  qqline(mooringdiff)
  
  # Two-sample t-test for mooring methods
  print(t.test(mooringdiff))
  
  # Wilcoxon signed rank test
  print(wilcox.test(mooringdiff))
})

# Print runtime for Wave Energy Device Mooring Analysis
print(time_waves)

###################################################
### Mortality and Water Hardness Data Analysis
###################################################

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Mortality and Water Hardness Analysis
time_water <- system.time({
  # Load the water dataset
  data("water", package = "HSAUR")
  
  # Adjust margins to avoid "figure margins too large" error
  # Setting all margins to a minimal value (left, bottom, right, top)
  par(mar = c(2, 2, 2, 2) + 0.1)
  
  # Enhanced scatterplot with marginal histograms
  # Reset layout parameters to ensure there's enough space
  layout(matrix(c(1, 2, 3, 4), 2, 2, byrow = TRUE), widths = c(3, 1), heights = c(1, 3))
  
  psymb <- as.numeric(water$location)
  
  # Scatter plot with regression line
  plot(mortality ~ hardness, data = water, pch = psymb, main = "Scatterplot: Mortality vs Hardness",
       xlab = "Water Hardness", ylab = "Mortality")
  abline(lm(mortality ~ hardness, data = water))
  
  # Adding legend
  legend("topright", legend = levels(water$location), pch = c(1, 2), bty = "n")
  
  # Marginal histogram for hardness
  hist(water$hardness, main = "Histogram: Water Hardness", xlab = "Water Hardness", ylab = "Frequency", col = "lightblue")
  
  # Boxplot for mortality
  boxplot(water$mortality, main = "Boxplot: Mortality", ylab = "Mortality", horizontal = TRUE, col = "lightgreen")
  
  # Pearson's correlation test between mortality and hardness
  print(cor.test(~ mortality + hardness, data = water))
})

# Print runtime for Mortality and Water Hardness Analysis
print(time_water)


###################################################
### Piston-ring Failures Data Analysis
###################################################

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Piston-ring Failures Analysis
time_pistonrings <- system.time({
  # Load the pistonrings dataset
  data("pistonrings", package = "HSAUR")
  
  # Chi-squared test for independence
  chi_test <- chisq.test(pistonrings)
  print(chi_test)
  
  # Residuals from chi-squared test
  print(chi_test$residuals)
  
  # Association plot of residuals
  library("vcd")
  # Reset graphics state and open a new device before plotting
  if (is.null(dev.list())) dev.new()
  assoc(pistonrings)
})

# Print runtime for Piston-ring Failures Analysis
print(time_pistonrings)

###################################################
### Rearrests of Juveniles Data Analysis
###################################################

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Rearrests of Juveniles Analysis
time_rearrests <- system.time({
  # Load the rearrests dataset
  data("rearrests", package = "HSAUR")
  print(rearrests)
  
  # McNemar's test for matched pairs
  print(mcnemar.test(rearrests, correct = FALSE))
  
  # Exact binomial test
  print(binom.test(rearrests[2], n = sum(rearrests[c(2,3)])))
})

# Print runtime for Rearrests of Juveniles Analysis
print(time_rearrests)
