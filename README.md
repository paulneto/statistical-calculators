# Statistical Calculators Collection

A comprehensive collection of web-based statistical calculators designed for marketing research, data analysis, and educational purposes. All calculators are self-contained HTML files built with React and Tailwind CSS, with mathematically rigorous implementations using the jStat statistical library.

**Live Site:** [https://paulneto.github.io/statistical-calculators/](https://paulneto.github.io/statistical-calculators/)

## Overview

This repository contains 18 different statistical calculators and reference guides, each designed to handle specific types of statistical analyses commonly used in marketing research, A/B testing, and data science. All calculators have been mathematically verified and include effect sizes, confidence intervals, and exact p-values.

## Calculators

### 1. ANOVA Calculator
**Location:** `anova/`

**Description:** Performs both One-Way and Two-Way Analysis of Variance (ANOVA) tests to compare means across multiple groups.

**Features:**
- One-Way ANOVA for comparing 2+ groups
- Two-Way ANOVA for analyzing two factors and their interaction
- Interactive data entry for multiple groups
- **F-statistic calculation with proper F-distribution (jStat)**
- **Eta-squared (η²) effect sizes** - shows proportion of variance explained
- **Exact p-values** for all tests
- Significance testing at α = 0.05 and 0.01
- Detailed explanations and interpretation guides

**Use Cases:**
- Comparing marketing channel effectiveness
- Testing multiple campaign variations
- Analyzing factor interactions (e.g., channel × time of day)

---

### 2. Chi-Square Calculator
**Location:** `chi/`

**Description:** Performs Chi-Square tests for categorical data analysis with two test types.

**Features:**
- Goodness of Fit test (comparing observed vs expected frequencies)
- Test of Independence (analyzing relationship between two categorical variables)
- Contingency table support
- **Cramér's V effect size** - shows strength of association (0-1 scale)
- **Chi-square critical values for any degrees of freedom (jStat)**
- **Exact p-values** for all tests
- Detailed contribution analysis per category/cell

**Use Cases:**
- Testing if data matches expected distributions
- Analyzing relationship between demographics and behavior
- Survey response analysis

---

### 3. T-Test Calculator
**Location:** `t-test/`

**Description:** Comprehensive t-test calculator supporting three types of t-tests for comparing means.

**Features:**
- Independent Samples T-Test (comparing two separate groups)
- Paired Samples T-Test (before/after comparisons)
- One-Sample T-Test (comparing sample to known benchmark)
- **Cohen's d effect size** - shows magnitude of difference (small/medium/large)
- **95% confidence intervals** for all test types
- **T-critical values using proper t-distribution (jStat)**
- **Exact p-values** for all tests
- Pre-loaded example datasets

**Use Cases:**
- A/B testing (email campaigns, landing pages)
- Before/after intervention comparisons
- Comparing against industry benchmarks

---

### 4. Z-Test Significance Calculator
**Location:** `z test sig/`

**Description:** Pairwise significance testing for comparing proportions between two groups across multiple items.

**Features:**
- Bulk comparison of multiple percentages
- Multiple confidence levels (99%, 95%, 90%, 80%)
- Z-score calculation for each comparison
- Visual indicators for significant differences
- Support for different sample sizes per group

**Use Cases:**
- Survey response comparisons between segments
- Multi-item A/B test analysis
- Campaign performance across multiple metrics

---

### 5. Correlation Calculator
**Location:** `correlation/`

**Description:** Calculates Pearson correlation coefficient to measure linear relationships between two variables.

**Features:**
- Pearson's r calculation
- Coefficient of determination (r²)
- Interactive scatter plot visualization
- Strength and direction interpretation
- Pre-loaded marketing example data

**Use Cases:**
- Analyzing relationship between video length and engagement
- Correlating ad spend with conversions
- Understanding relationships between metrics

---

### 6. Mann-Whitney U Test Calculator
**Location:** `mann-whitney/`

**Description:** Non-parametric test for comparing two independent groups when data doesn't follow normal distribution.

**Features:**
- U-statistic calculation with **tie correction**
- Z-score for larger samples
- **Exact p-values** using normal approximation
- Median comparison between groups
- Multiple confidence levels
- Pre-loaded example datasets

**Use Cases:**
- Comparing session durations between user segments
- Analyzing engagement scores with outliers
- Small sample size comparisons
- Skewed distribution data

---

### 7. Normal Distribution Dashboard
**Location:** `normal distribution/`

**Description:** Interactive visualization tool for exploring normal distributions and probability calculations.

**Features:**
- Interactive distribution visualization using D3.js
- Adjustable mean and standard deviation
- Probability calculations
- Visual area-under-curve highlighting
- Significance testing cheat sheet included

**Use Cases:**
- Understanding sampling distributions
- Probability calculations
- Educational demonstrations
- Z-score visualization

---

### 8. Power Analysis Calculator
**Location:** `power analysis/`

**Description:** Calculate required sample sizes and statistical power for designing experiments.

**Features:**
- Sample size calculation
- Power analysis
- Effect size estimation
- Multiple test type support
- Visual power curves

**Use Cases:**
- A/B test planning
- Determining required sample sizes
- Understanding detection capability
- Budget planning for research studies

---

### 9. Regression Calculator
**Location:** `regression/`

**Description:** Linear regression analysis with correlation and prediction capabilities.

**Features:**
- Simple linear regression
- Correlation analysis
- Prediction functionality
- Visual scatter plot with regression line
- R-squared and coefficients

**Use Cases:**
- Sales forecasting
- Predicting outcomes from variables
- Understanding variable relationships
- Trend analysis

---

### 10. Proportion Significance Calculator
**Location:** `proportion sig/`

**Description:** Z-test for proportions to determine if differences in percentages are statistically significant.

**Features:**
- Multiple proportion comparisons
- Label and value pairing
- Confidence level selection
- Z-statistic calculation
- Marketing-focused examples

**Use Cases:**
- Email campaign click-through rate comparison
- Conversion rate testing
- Survey response analysis
- Engagement metric comparisons

---

### 11. Bayesian Statistics Interactive Demo ✨ NEW
**Location:** `bayesian/`

**Description:** Comprehensive interactive learning tool for Bayesian inference using Beta-Binomial conjugate priors.

**Features:**
- **5 Interactive Demos:**
  1. Basic Bayesian Inference - proportion estimation
  2. Email Campaign - conversion rate analysis
  3. A/B Testing - Bayesian comparison of treatments
  4. Customer Lifetime Value - simplified Bayesian updating
  5. Real-time Updating - sequential data incorporation
- **Monte Carlo sampling** for P(B > A) calculations (10,000 samples)
- **95% credible intervals** using exact Beta quantiles
- Educational content on conjugate priors
- Comparison between Bayesian and frequentist approaches
- Marketing-focused examples throughout

**Use Cases:**
- Learning Bayesian statistics fundamentals
- A/B testing with small sample sizes
- Sequential decision-making
- Incorporating prior knowledge
- Understanding probability of superiority

---

### 12. Stats Cheatsheet 📚
**Location:** `cheatsheet.html`

**Description:** Professional guide to all statistical tests with accessible language for marketing research and data analysis.

**Features:**
- Quick reference for 8 major statistical tests
- **"What is it?"** - Plain English explanations
- **"When to use it"** - Decision guides
- **Real marketing examples** for each test
- **How to interpret** - Results in business context
- **Watch out for** - Common pitfalls
- Sticky navigation for easy browsing
- Visual quick reference table

**Covers:**
- T-Test, ANOVA, Chi-Square, Correlation, Regression, Mann-Whitney, Proportions, Bayesian

**Use Cases:**
- Exam preparation
- Quick test selection reference
- Learning statistical concepts
- Understanding when to use each test

---

### 13. Python Cheatsheet 🐍 NEW
**Location:** `python_cheatsheet.html`

**Description:** Copy-paste ready Python code examples for all statistical tests with complete imports and marketing-focused examples.

**Features:**
- **8 complete Python examples** with all imports
- Copy-to-clipboard functionality for every code block
- Syntax highlighting using Prism.js
- Real marketing research examples (email campaigns, A/B tests, customer segmentation)
- Uses standard libraries: numpy, scipy, pandas, statsmodels
- Detailed comments throughout code
- Self-contained, ready-to-run examples

**Covers:**
- T-Test, ANOVA, Chi-Square, Correlation, Linear Regression, Mann-Whitney, Proportions, Logistic Regression

**Use Cases:**
- Quick Python reference
- Learning statistical programming
- Copy-paste code for analysis
- Understanding Python statistical libraries

---

### 14. Multiple Regression Calculator ✨ NEW
**Location:** `multiple-regression/`

**Description:** Predict outcomes using multiple predictor variables with comprehensive coefficient analysis and multicollinearity diagnostics.

**Features:**
- **Matrix algebra implementation** (β = (X'X)⁻¹X'y)
- Multiple predictor variables support
- Full coefficient table with:
  - Coefficients and standard errors
  - t-statistics and p-values
  - Significance indicators
- **VIF (Variance Inflation Factor)** for multicollinearity detection
- R-squared and Adjusted R-squared
- F-statistic for overall model significance
- Regression equation display
- Pre-loaded marketing example (ad spend analysis)

**Use Cases:**
- Sales forecasting with multiple factors
- Marketing mix modeling
- Understanding driver contribution
- Multivariate prediction models

---

### 15. Post-Hoc Tests Calculator ✨ NEW
**Location:** `posthoc-tests/`

**Description:** Pairwise comparison tests for identifying specific group differences after significant ANOVA results.

**Features:**
- **Tukey HSD (Honestly Significant Difference)** test
- **Bonferroni correction** for multiple comparisons
- Complete pairwise comparison table with:
  - Mean differences
  - Test statistics (q or t)
  - P-values (adjusted for Bonferroni)
  - Significance indicators
- Group summary statistics
- MSE and degrees of freedom calculations
- Pre-loaded marketing channel example

**Use Cases:**
- Follow-up analysis after ANOVA
- Identifying which specific groups differ
- Marketing channel performance comparisons
- Campaign variation analysis

---

### 16. Logistic Regression Calculator ✨ NEW
**Location:** `logistic-regression/`

**Description:** Binary outcome prediction with odds ratios, confusion matrix, and comprehensive classification metrics.

**Features:**
- **Newton-Raphson iterative method** for maximum likelihood estimation
- Binary outcome modeling (0/1, Yes/No)
- Coefficient table with:
  - Coefficients and standard errors
  - z-statistics and p-values
  - **Odds ratios** with interpretations
- **Confusion matrix** (True Positives, False Positives, etc.)
- Classification metrics:
  - Accuracy, Precision, Recall, F1 Score
- **McFadden's pseudo R-squared**
- Pre-loaded customer conversion example

**Use Cases:**
- Customer conversion prediction
- Churn modeling
- Click-through rate prediction
- Binary classification problems

---

### 17. Sample Size Calculator ✨ NEW
**Location:** `sample-size/`

**Description:** Determine required sample sizes for adequate statistical power across multiple test types.

**Features:**
- **4 test types supported:**
  1. T-Test (two-sample means)
  2. Proportions (two-sample proportions)
  3. Correlation (Pearson's r)
  4. ANOVA (multiple groups)
- Effect size guidance (small/medium/large for each test)
- Power level specification (default: 0.80)
- Significance level specification (default: 0.05)
- Power curve visualization
- Detailed sample size recommendations

**Use Cases:**
- A/B test planning
- Experiment design
- Budget planning for research
- Ensuring adequate statistical power

---

### 18. Cluster Analysis Calculator ✨ NEW
**Location:** `cluster-analysis/`

**Description:** K-means clustering for customer segmentation with elbow plot and comprehensive cluster analysis.

**Features:**
- **K-means algorithm** implementation (Lloyd's method)
- Multiple cluster support (2-10 clusters)
- Multi-feature analysis
- Cluster assignment table with:
  - Customer IDs
  - Cluster assignments
  - Feature values
- **WCSS (Within-Cluster Sum of Squares)** analysis
- Elbow plot data for optimal k selection
- Cluster centroids display
- Marketing interpretation guide
- Pre-loaded customer data example

**Use Cases:**
- Customer segmentation
- Market segmentation
- Behavioral grouping
- Targeting strategy development

---

## Technology Stack

All calculators are built with:
- **HTML5** - Structure and content
- **React 18** - UI components and state management (loaded via CDN)
- **Tailwind CSS 2** - Styling and responsive design (loaded via CDN)
- **jStat** - JavaScript statistical library for distributions (F, t, Chi-square, Beta, Normal)
- **D3.js** - Data visualization (where applicable)
- **Vanilla JavaScript** - Core calculations and logic

## Features

- **No Installation Required:** All calculators run directly in the browser
- **Mathematically Rigorous:** All critical values calculated using proper statistical distributions (jStat)
- **Effect Sizes Included:** Cohen's d, Eta-squared, Cramér's V - understand practical significance
- **Confidence Intervals:** 95% CIs for point estimates (T-Test, Bayesian demos)
- **Exact P-Values:** Continuous probability measures, not just significant/not significant
- **Responsive Design:** Works on desktop, tablet, and mobile devices
- **Self-Contained:** Each calculator is a single HTML file (requires CDN access)
- **Educational:** Includes interpretation guides and explanations
- **Example Data:** Pre-loaded datasets for testing and learning
- **Copy & Paste Friendly:** Easy data entry from spreadsheets
- **Student-Friendly:** Cheatsheet provides accessible guide for test selection

## Usage

1. Open any HTML file in a modern web browser (Chrome, Firefox, Safari, Edge)
2. Enter your data or load example datasets
3. Click calculate to see results
4. Review interpretations and statistical significance

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Requires internet connection for CDN-loaded libraries (React, Tailwind CSS, jStat, D3.js).

## File Organization

```
calculators/
├── index.html                          # Main dashboard
├── cheatsheet.html                     # Stats reference guide
├── python_cheatsheet.html              # Python code examples
├── anova/
│   ├── index.html
│   └── anova-calculator.html
├── chi/
│   ├── index.html
│   └── chi-square.html
├── t-test/
│   ├── index.html
│   └── ttest-calculator.html
├── z test sig/
│   ├── index.html
│   └── [additional variations]
├── correlation/
│   └── correlation-calculator.html
├── mann-whitney/
│   └── mann-whitney-html.html
├── normal distribution/
│   ├── dashboard-simulator v2.1.html
│   └── Significance_Testing_Cheat_Sheet.html
├── power analysis/
│   └── power-analysis-calculator.html
├── regression/
│   └── simple-linear-regression.html
├── proportion sig/
│   └── z proportion-calculator %.html
├── bayesian/
│   └── bayesian_interactive.html
├── multiple-regression/               # NEW
│   └── index.html
├── posthoc-tests/                     # NEW
│   └── index.html
├── logistic-regression/               # NEW
│   └── index.html
├── sample-size/                       # NEW
│   └── index.html
└── cluster-analysis/                  # NEW
    └── index.html
```

## Common Statistical Terms

**p-value:** Probability that results occurred by chance. Lower values indicate stronger evidence against the null hypothesis.

**Significance Level (α):** Threshold for determining statistical significance (commonly 0.05 or 0.01).

**Confidence Level:** The probability that the true value lies within the confidence interval (commonly 95% or 99%).

**Effect Size:** The magnitude of difference between groups.

**Statistical Power:** The probability of detecting an effect when it exists.

## Best Practices

1. **Check Assumptions:** Each test has specific assumptions (normality, sample size, etc.)
2. **Consider Effect Size:** Statistical significance doesn't always mean practical importance
3. **Multiple Comparisons:** Be cautious when running many tests simultaneously
4. **Sample Size:** Larger samples provide more reliable results
5. **Data Quality:** Ensure data is clean and properly formatted

## Future Enhancements

Potential improvements for future versions:
- Offline capability with bundled libraries
- Data export functionality (CSV, JSON)
- Advanced visualizations
- Multi-language support
- Batch processing capabilities
- Integration with APIs

## Contributing

This is a personal collection of calculators. If you find issues or have suggestions, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Free to use and modify for personal and commercial purposes.

## Author

Paul - Statistical Calculators Collection

## Acknowledgments

Built with modern web technologies to make statistical analysis accessible and user-friendly for marketing researchers, data analysts, and students.

**Mathematical Verification:** All calculators have been reviewed for mathematical accuracy. See internal documentation for detailed verification.

---

**Last Updated:** January 2025
