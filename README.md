# Capstone-Project
Comparative Analysis of Machine Learning Models Across Programming Languages


# 📊 Comparative Study of ML Models and Statistical Metrics Across Languages

This repository hosts the source code, documentation, and deliverables for our capstone project titled **"A Comparative Study of Machine Learning Models and Statistical Metrics Across Different Languages."** The objective is to evaluate and compare the effectiveness, performance, and development experience of machine learning and statistical models across five programming languages: **Python, R, Julia, C++, and Rust**.

---

## 🧠 Project Overview

As machine learning (ML) becomes increasingly integral to data-driven applications, the choice of programming language plays a crucial role in performance, scalability, and development efficiency. Each language offers unique strengths, but little work has been done to evaluate them side-by-side across both classification and statistical tasks.

This project systematically implements and evaluates ML models and statistical techniques across multiple languages to help developers, researchers, and analysts make informed language choices.

---
---

## 📊 Techniques Implemented

### Classification Models
- Logistic Regression
- Support Vector Machines (SVM)

### Statistical Analysis
- Multiple Linear Regression (MLR)
- Two-sample t-tests (equal & unequal variance)
- Wilcoxon Rank-Sum and Signed-Rank Tests
- Pearson's Correlation
- Chi-Square Test
- Binomial Test
- Two-Way ANOVA and Tukey's HSD
- MANOVA (Pillai’s Trace, Wilks’ Lambda, etc.)

---

## 📐 Metrics Used

| Metric               | Purpose                                        |
|----------------------|------------------------------------------------|
| Accuracy             | Overall correctness of the model               |
| Precision            | True positives out of predicted positives      |
| Recall               | True positives out of actual positives         |
| F1 Score             | Harmonic mean of Precision and Recall          |
| Specificity          | True negatives out of actual negatives         |
| Jaccard Coefficient  | Similarity between predicted & actual labels   |
| Adjusted Rand Index  | Clustering similarity                          |
| Purity               | Cluster homogeneity                            |
| Entropy              | Prediction uncertainty                         |
| Error Rate           | Misclassification rate                         |
| Confidence Interval  | Range of accuracy certainty                    |
| Training Error       | Model error during training                    |
| Generalization Error | Error on unseen data                           |

---

## 🧪 Tools and Libraries

| Language | Libraries Used |
|----------|----------------|
| Python   | scikit-learn, statsmodels, seaborn, matplotlib |
| R        | caret, glm, e1071, ggplot2, stats, HSAUR |
| Julia    | GLM.jl, LIBSVM.jl, HypothesisTests.jl, Plots.jl |
| C++      | Eigen, STL, custom implementations |
| Rust     | smartcore, ndarray, csv |

---

## 💻 Platforms and IDEs

- Python: Jupyter Notebook (macOS)
- R: RStudio (macOS)
- Julia: Pluto & Jupyter (macOS)
- C++: Eclipse (Windows)
- Rust: Eclipse (Windows)

---

## ⏱️ Execution Time Comparison

| Language | Execution Time | Notes |
|----------|----------------|-------|
| C++      | 1.7 sec        | Fastest but most complex to develop |
| Julia    | 2.2 sec        | Best balance between speed and usability |
| Python   | 2.87 sec       | Easiest and most intuitive |
| Rust     | 2.8 sec        | Strong performance, poor library support |
| R        | 4.2 sec        | Great for statistics, but slower |

---

## ✅ Key Results

- **Python**: Most beginner-friendly with rich library support.
- **R**: Best suited for statistical analysis and visualizations.
- **Julia**: Balanced, powerful, and expressive for scientific computing.
- **C++**: Fastest execution time, steepest learning curve.
- **Rust**: Safe memory handling and decent performance but not mature for data science.

---

## ⚠️ Limitations

- Incomplete results for C++ and Rust in statistical models
- Rust and C++ lack mature libraries for metrics like entropy or ANOVA
- Visualization support limited in Rust and C++
- No GPU utilization
- Language-specific development overhead

---
