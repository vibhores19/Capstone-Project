# Load required packages
# install.packages("HSAUR")
# install.packages("ggplot2") # For violin plots

library(HSAUR)
library(ggplot2)

# Load the roomwidth dataset
data("roomwidth", package = "HSAUR")

# Convert metres to feet (1 for feet, 3.28 for metres)
convert <- ifelse(roomwidth$unit == "feet", 1, 3.28)

# Summary statistics for both feet and metre estimates (converted)
summary_stats <- tapply(roomwidth$width * convert, roomwidth$unit, summary)
print(summary_stats)

# Standard deviations for both feet and metre estimates (converted)
std_devs <- tapply(roomwidth$width * convert, roomwidth$unit, sd)
print(std_devs)

# Plot layout for boxplot and QQ plots
layout(matrix(c(1, 2, 1, 3), nrow = 2, ncol = 2, byrow = FALSE))

# Boxplot comparing estimates in feet and metres (converted)
boxplot(I(width * convert) ~ unit, data = roomwidth,
        ylab = "Estimated width (feet)",
        varwidth = TRUE, 
        names = c("Estimates in feet", "Estimates in metres (converted to feet)"))

# QQ plot for feet estimates
feet <- roomwidth$unit == "feet"
qqnorm(roomwidth$width[feet], ylab = "Estimated width (feet)")
qqline(roomwidth$width[feet])

# QQ plot for metre estimates (converted to feet)
qqnorm(roomwidth$width[!feet], ylab = "Estimated width (metres)")
qqline(roomwidth$width[!feet])

# Two-sample t-test assuming equal variances
t_test_equal <- t.test(I(width * convert) ~ unit, data = roomwidth, var.equal = TRUE)
print(t_test_equal)

# Welch Two-sample t-test assuming unequal variances
t_test_unequal <- t.test(I(width * convert) ~ unit, data = roomwidth, var.equal = FALSE)
print(t_test_unequal)

# Wilcoxon rank sum test
wilcox_test <- wilcox.test(I(width * convert) ~ unit, data = roomwidth, conf.int = TRUE)
print(wilcox_test)

# Histogram for feet estimates
hist(roomwidth$width[feet], main = "Histogram of Estimates in Feet", 
     xlab = "Width (Feet)", col = "lightblue")

# Histogram for metre estimates (converted to feet)
hist(roomwidth$width[!feet] * 3.28, main = "Histogram of Estimates in Metres (Converted to Feet)", 
     xlab = "Width (Metres Converted to Feet)", col = "lightgreen")

# Density plot for feet estimates
plot(density(roomwidth$width[feet]), main = "Density of Estimates in Feet", 
     xlab = "Width (Feet)", col = "blue")

# Density plot for metre estimates (converted to feet)
plot(density(roomwidth$width[!feet] * 3.28), main = "Density of Estimates in Metres (Converted to Feet)", 
     xlab = "Width (Metres Converted to Feet)", col = "green")

# Violin plot using ggplot2 (combining density and boxplot characteristics)
roomwidth$converted_width <- roomwidth$width * convert
ggplot(roomwidth, aes(x = unit, y = converted_width)) +
  geom_violin(fill = "lightblue") +
  labs(title = "Violin Plot of Room Width Estimates", x = "Unit", y = "Converted Width (Feet)") +
  theme_minimal()

### Save the dataset as a CSV and RDS file ###

# Save the roomwidth dataset as a CSV file to the Desktop
write.csv(roomwidth, "/Users/mozumdertushar/Desktop/roomwidth.csv", row.names = FALSE)
cat("roomwidth dataset saved as 'roomwidth.csv' on Desktop\n")


