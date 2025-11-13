# Statistical Calculators Collection

A comprehensive collection of web-based statistical calculators designed for marketing research, data analysis, and educational purposes. All calculators are self-contained HTML files built with React and Tailwind CSS.

## Overview

This repository contains 10 different statistical calculators, each designed to handle specific types of statistical analyses commonly used in marketing research, A/B testing, and data science.

## Calculators

### 1. ANOVA Calculator
**Location:** `anova/`

**Description:** Performs both One-Way and Two-Way Analysis of Variance (ANOVA) tests to compare means across multiple groups.

**Features:**
- One-Way ANOVA for comparing 2+ groups
- Two-Way ANOVA for analyzing two factors and their interaction
- Interactive data entry for multiple groups
- F-statistic calculation with critical values
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
- Detailed contribution analysis per category/cell
- Critical values for multiple significance levels

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
- Pre-loaded example datasets
- T-statistic and degrees of freedom calculations
- Significance testing at p < 0.05

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
- U-statistic calculation
- Z-score for larger samples
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

## Technology Stack

All calculators are built with:
- **HTML5** - Structure and content
- **React 18** - UI components and state management (loaded via CDN)
- **Tailwind CSS 2** - Styling and responsive design (loaded via CDN)
- **D3.js** - Data visualization (where applicable)
- **Vanilla JavaScript** - Statistical calculations

## Features

- **No Installation Required:** All calculators run directly in the browser
- **Responsive Design:** Works on desktop, tablet, and mobile devices
- **Self-Contained:** Each calculator is a single HTML file with no dependencies
- **Educational:** Includes interpretation guides and explanations
- **Example Data:** Pre-loaded datasets for testing and learning
- **Copy & Paste Friendly:** Easy data entry from spreadsheets

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

Requires internet connection for CDN-loaded libraries (React, Tailwind CSS, D3.js).

## File Organization

```
calculators/
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
│   └── correlation-regression-html2.html
└── proportion sig/
    └── z proportion-calculator %.html
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

---

**Last Updated:** November 2025
