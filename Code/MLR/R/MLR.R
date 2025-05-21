###################################################
### Cloud Seeding Data Analysis
###################################################

# Load the required package and dataset
library(HSAUR)

# Reset graphics device before plotting
while (!is.null(dev.list())) dev.off()

# Measure runtime for Cloud Seeding Data Analysis
time_clouds <- system.time({
  # Load the clouds dataset
  data("clouds", package = "HSAUR")
  
  # Identify extreme observations in clouds data.frame
  bxpseeding <- boxplot(rainfall ~ seeding, data = clouds, plot = FALSE)
  bxpecho <- boxplot(rainfall ~ echomotion, data = clouds, plot = FALSE)
  
  extreme_obs <- rownames(clouds)[clouds$rainfall %in% c(bxpseeding$out, bxpecho$out)]
  print(paste("Extreme Observations:", paste(extreme_obs, collapse = ", ")))
  
  # Fitting a Linear Model
  clouds_formula <- rainfall ~ seeding * (sne + cloudcover + prewetness + echomotion) + time
  Xstar <- model.matrix(clouds_formula, data = clouds)
  print(attr(Xstar, "contrasts"))
  
  # Boxplots for rainfall vs. seeding and echo motion
  par(mar = c(2, 2, 2, 2))  # Set minimal margins to avoid figure margin issues
  dev.new()  # Create new plot window for each boxplot
  boxplot(rainfall ~ seeding, data = clouds, ylab = "Rainfall", xlab = "Seeding", main = "Rainfall vs Seeding")
  
  dev.new()  # Create new plot window for next boxplot
  boxplot(rainfall ~ echomotion, data = clouds, ylab = "Rainfall", xlab = "Echo Motion", main = "Rainfall vs Echo Motion")
  
  # Scatterplots of rainfall against covariates
  par(mfrow = c(2, 2))  # Create a 2x2 layout for scatterplots
  par(mar = c(2, 2, 2, 2))  # Set minimal margins
  plot(rainfall ~ time, data = clouds, main = "Rainfall vs Time")
  plot(rainfall ~ cloudcover, data = clouds, main = "Rainfall vs Cloud Cover")
  plot(rainfall ~ sne, data = clouds, xlab = "S-Ne Criterion", main = "Rainfall vs S-Ne Criterion")
  plot(rainfall ~ prewetness, data = clouds, main = "Rainfall vs Prewetness")
  
  # Fit the linear model
  clouds_lm <- lm(clouds_formula, data = clouds)
  print(summary(clouds_lm))
  
  # Extract coefficients and covariance matrix
  betastar <- coef(clouds_lm)
  Vbetastar <- vcov(clouds_lm)
  print(betastar)
  print(sqrt(diag(Vbetastar)))
  
  # Residuals and Fitted Values
  clouds_resid <- residuals(clouds_lm)
  clouds_fitted <- fitted(clouds_lm)
  
  # Diagnostic plots for the linear model
  dev.new()  # Create new plot window for diagnostic plots
  par(mfrow = c(2, 2))  # 2x2 layout for diagnostic plots
  par(mar = c(2, 2, 2, 2))  # Minimal margins to ensure enough plotting space
  
  # Residuals vs. S-Ne Criterion
  psymb <- as.numeric(clouds$seeding)
  plot(rainfall ~ sne, data = clouds, pch = psymb, xlab = "S-Ne criterion", main = "Rainfall vs S-Ne Criterion")
  abline(lm(rainfall ~ sne, data = clouds, subset = seeding == "no"))
  abline(lm(rainfall ~ sne, data = clouds, subset = seeding == "yes"), lty = 2)
  legend("topright", legend = c("No seeding", "Seeding"), pch = 1:2, lty = 1:2, bty = "n")
  
  # Residuals vs. Fitted Values
  plot(clouds_fitted, clouds_resid, xlab = "Fitted values", ylab = "Residuals", type = "n",
       ylim = max(abs(clouds_resid)) * c(-1, 1), main = "Residuals vs Fitted Values")
  abline(h = 0, lty = 2)
  text(clouds_fitted, clouds_resid, labels = rownames(clouds), cex = 0.7)
  
  # Normal probability plot of residuals
  qqnorm(clouds_resid, ylab = "Residuals", main = "Normal Q-Q Plot of Residuals")
  qqline(clouds_resid)
  
  # Cook's distance plot
  plot(clouds_lm, which = 4, main = "Cook's Distance")
})

# Print runtime for Cloud Seeding Data Analysis
print(time_clouds)
