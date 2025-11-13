# Python Examples for Statistical Calculators

Complete, copy-paste ready Python implementations for all statistical calculators in this repository. Each example includes all necessary imports and can be run directly in a Python terminal or script.

**Requirements:**
```bash
pip install numpy scipy pandas matplotlib statsmodels
```

---

## Table of Contents

1. [ANOVA (One-Way and Two-Way)](#1-anova-one-way-and-two-way)
2. [Chi-Square Tests](#2-chi-square-tests)
3. [T-Tests (One-Sample, Paired, Independent)](#3-t-tests)
4. [Z-Test for Significance](#4-z-test-for-significance)
5. [Correlation (Pearson's r)](#5-correlation-pearsons-r)
6. [Mann-Whitney U Test](#6-mann-whitney-u-test)
7. [Normal Distribution](#7-normal-distribution)
8. [Power Analysis](#8-power-analysis)
9. [Simple Linear Regression](#9-simple-linear-regression)
10. [Proportion Significance (Z-Test)](#10-proportion-significance-z-test)
11. [Bayesian Statistics (Beta-Binomial)](#11-bayesian-statistics-beta-binomial)

---

## 1. ANOVA (One-Way and Two-Way)

### One-Way ANOVA

```python
"""
One-Way ANOVA: Compare means across 3+ independent groups
Use case: Comparing email open rates across 3 different subject line types
"""

import numpy as np
from scipy import stats

# Sample data: Email open rates (%) for three different subject line types
group_a = [12.5, 15.3, 13.8, 14.2, 16.1]  # Question subject lines
group_b = [18.2, 19.5, 17.8, 20.1, 18.9]  # Urgency subject lines
group_c = [14.5, 15.8, 13.9, 16.2, 14.7]  # Personalized subject lines

# Perform One-Way ANOVA
f_statistic, p_value = stats.f_oneway(group_a, group_b, group_c)

# Calculate effect size (eta-squared)
all_data = np.concatenate([group_a, group_b, group_c])
grand_mean = np.mean(all_data)

# Sum of Squares Between (SSB)
ssb = len(group_a) * (np.mean(group_a) - grand_mean)**2 + \
      len(group_b) * (np.mean(group_b) - grand_mean)**2 + \
      len(group_c) * (np.mean(group_c) - grand_mean)**2

# Sum of Squares Total (SST)
sst = np.sum((all_data - grand_mean)**2)

# Eta-squared (proportion of variance explained)
eta_squared = ssb / sst

# Print results
print("=" * 60)
print("ONE-WAY ANOVA RESULTS")
print("=" * 60)
print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Eta-squared (effect size): {eta_squared:.4f}")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")
print("\nGroup Means:")
print(f"  Group A (Questions): {np.mean(group_a):.2f}%")
print(f"  Group B (Urgency): {np.mean(group_b):.2f}%")
print(f"  Group C (Personalized): {np.mean(group_c):.2f}%")
print("\nInterpretation:")
if p_value < 0.05:
    print("  There is a statistically significant difference between groups.")
    print(f"  {eta_squared*100:.1f}% of variance in open rates is explained by subject line type.")
else:
    print("  No statistically significant difference between groups.")
```

### Two-Way ANOVA

```python
"""
Two-Way ANOVA: Analyze effects of two factors and their interaction
Use case: Email open rates by subject line type AND time of day
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Sample data: Email open rates (%)
# Factor A: Subject line type (Question, Urgency, Personalized)
# Factor B: Time of day (Morning, Afternoon)
data = pd.DataFrame({
    'open_rate': [12.5, 15.3, 13.8, 18.2, 19.5, 17.8, 14.5, 15.8, 13.9,
                  16.1, 14.2, 15.5, 20.1, 18.9, 19.3, 16.2, 14.7, 15.1],
    'subject_type': ['Question', 'Question', 'Question', 'Urgency', 'Urgency', 'Urgency',
                     'Personalized', 'Personalized', 'Personalized'] * 2,
    'time_of_day': ['Morning']*9 + ['Afternoon']*9
})

# Perform Two-Way ANOVA
model = ols('open_rate ~ C(subject_type) + C(time_of_day) + C(subject_type):C(time_of_day)',
            data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Calculate partial eta-squared for each factor
ss_total = anova_table['sum_sq'].sum()
anova_table['eta_sq'] = anova_table['sum_sq'] / ss_total

# Print results
print("=" * 60)
print("TWO-WAY ANOVA RESULTS")
print("=" * 60)
print("\nANOVA Table:")
print(anova_table.round(4))

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("\nMain Effect - Subject Type:")
if anova_table.loc['C(subject_type)', 'PR(>F)'] < 0.05:
    print(f"  Significant (p={anova_table.loc['C(subject_type)', 'PR(>F)']:.4f})")
    print(f"  Eta-squared: {anova_table.loc['C(subject_type)', 'eta_sq']:.4f}")
else:
    print(f"  Not significant (p={anova_table.loc['C(subject_type)', 'PR(>F)']:.4f})")

print("\nMain Effect - Time of Day:")
if anova_table.loc['C(time_of_day)', 'PR(>F)'] < 0.05:
    print(f"  Significant (p={anova_table.loc['C(time_of_day)', 'PR(>F)']:.4f})")
    print(f"  Eta-squared: {anova_table.loc['C(time_of_day)', 'eta_sq']:.4f}")
else:
    print(f"  Not significant (p={anova_table.loc['C(time_of_day)', 'PR(>F)']:.4f})")

print("\nInteraction Effect:")
if anova_table.loc['C(subject_type):C(time_of_day)', 'PR(>F)'] < 0.05:
    print(f"  Significant (p={anova_table.loc['C(subject_type):C(time_of_day)', 'PR(>F)']:.4f})")
    print("  The effect of subject type depends on time of day (or vice versa).")
else:
    print(f"  Not significant (p={anova_table.loc['C(subject_type):C(time_of_day)', 'PR(>F)']:.4f})")
```

---

## 2. Chi-Square Tests

### Goodness of Fit Test

```python
"""
Chi-Square Goodness of Fit: Test if observed frequencies match expected distribution
Use case: Are website visits evenly distributed across days of the week?
"""

import numpy as np
from scipy import stats

# Observed frequencies: Website visits per day
observed = np.array([150, 135, 142, 148, 155, 95, 85])  # Mon-Sun
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Expected frequencies: Equal distribution (uniform)
total_visits = np.sum(observed)
expected = np.array([total_visits / 7] * 7)

# Perform Chi-Square Goodness of Fit test
chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

# Degrees of freedom
df = len(observed) - 1

# Print results
print("=" * 60)
print("CHI-SQUARE GOODNESS OF FIT TEST")
print("=" * 60)
print("\nObserved vs Expected Frequencies:")
for day, obs, exp in zip(days, observed, expected):
    print(f"  {day}: Observed={obs:>3}, Expected={exp:>6.2f}, Diff={obs-exp:>6.2f}")

print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
print(f"Degrees of Freedom: {df}")
print(f"P-value: {p_value:.4f}")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

print("\nInterpretation:")
if p_value < 0.05:
    print("  The distribution differs significantly from uniform.")
    print("  Website traffic is NOT evenly distributed across days.")
else:
    print("  The distribution does not differ significantly from uniform.")
    print("  Website traffic is roughly evenly distributed.")
```

### Test of Independence

```python
"""
Chi-Square Test of Independence: Test if two categorical variables are related
Use case: Is there a relationship between age group and product preference?
"""

import numpy as np
from scipy import stats

# Contingency table: Age group (rows) vs Product preference (columns)
# Rows: 18-29, 30-44, 45-60
# Columns: Product A, Product B, Product C
contingency_table = np.array([
    [45, 30, 25],  # 18-29
    [35, 50, 30],  # 30-44
    [20, 40, 55]   # 45-60
])

# Perform Chi-Square Test of Independence
chi2_stat, p_value, df, expected = stats.chi2_contingency(contingency_table)

# Calculate Cramér's V (effect size)
n = np.sum(contingency_table)
min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
cramers_v = np.sqrt(chi2_stat / (n * min_dim))

# Print results
print("=" * 60)
print("CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 60)
print("\nContingency Table (Observed):")
print("Age Group    Product A  Product B  Product C")
print(f"18-29             {contingency_table[0,0]:>3}       {contingency_table[0,1]:>3}       {contingency_table[0,2]:>3}")
print(f"30-44             {contingency_table[1,0]:>3}       {contingency_table[1,1]:>3}       {contingency_table[1,2]:>3}")
print(f"45-60             {contingency_table[2,0]:>3}       {contingency_table[2,1]:>3}       {contingency_table[2,2]:>3}")

print("\nExpected Frequencies:")
print("Age Group    Product A  Product B  Product C")
print(f"18-29          {expected[0,0]:>6.2f}    {expected[0,1]:>6.2f}    {expected[0,2]:>6.2f}")
print(f"30-44          {expected[1,0]:>6.2f}    {expected[1,1]:>6.2f}    {expected[1,2]:>6.2f}")
print(f"45-60          {expected[2,0]:>6.2f}    {expected[2,1]:>6.2f}    {expected[2,2]:>6.2f}")

print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
print(f"Degrees of Freedom: {df}")
print(f"P-value: {p_value:.4f}")
print(f"Cramér's V (effect size): {cramers_v:.4f}")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

print("\nEffect Size Interpretation:")
if cramers_v < 0.1:
    print("  Weak association")
elif cramers_v < 0.3:
    print("  Moderate association")
else:
    print("  Strong association")

print("\nInterpretation:")
if p_value < 0.05:
    print("  There IS a significant relationship between age and product preference.")
else:
    print("  There is NO significant relationship between age and product preference.")
```

---

## 3. T-Tests

### One-Sample T-Test

```python
"""
One-Sample T-Test: Compare sample mean to a known benchmark
Use case: Is our average conversion rate different from the industry benchmark of 3.5%?
"""

import numpy as np
from scipy import stats

# Sample data: Conversion rates (%) from 10 campaigns
conversion_rates = np.array([3.8, 4.2, 3.5, 4.0, 3.9, 4.1, 3.7, 3.6, 4.3, 3.8])

# Industry benchmark
benchmark = 3.5

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(conversion_rates, benchmark)

# Calculate Cohen's d (effect size)
mean_diff = np.mean(conversion_rates) - benchmark
std_dev = np.std(conversion_rates, ddof=1)
cohens_d = mean_diff / std_dev

# Calculate 95% confidence interval
n = len(conversion_rates)
df = n - 1
se = stats.sem(conversion_rates)
ci_margin = stats.t.ppf(0.975, df) * se
ci_lower = np.mean(conversion_rates) - ci_margin
ci_upper = np.mean(conversion_rates) + ci_margin

# Print results
print("=" * 60)
print("ONE-SAMPLE T-TEST")
print("=" * 60)
print(f"Sample Mean: {np.mean(conversion_rates):.2f}%")
print(f"Benchmark: {benchmark:.2f}%")
print(f"Difference: {mean_diff:.2f}%")
print(f"Sample Size: {n}")
print(f"Standard Deviation: {std_dev:.4f}")
print(f"\nT-statistic: {t_stat:.4f}")
print(f"Degrees of Freedom: {df}")
print(f"P-value (two-tailed): {p_value:.4f}")
print(f"Cohen's d (effect size): {cohens_d:.4f}")
print(f"\n95% Confidence Interval: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

print("\nEffect Size Interpretation:")
if abs(cohens_d) < 0.2:
    print("  Small effect")
elif abs(cohens_d) < 0.5:
    print("  Medium effect")
else:
    print("  Large effect")

print("\nInterpretation:")
if p_value < 0.05:
    if mean_diff > 0:
        print(f"  Our conversion rate ({np.mean(conversion_rates):.2f}%) is significantly HIGHER than the benchmark.")
    else:
        print(f"  Our conversion rate ({np.mean(conversion_rates):.2f}%) is significantly LOWER than the benchmark.")
else:
    print(f"  Our conversion rate is not significantly different from the benchmark.")
```

### Paired T-Test

```python
"""
Paired T-Test: Compare two related samples (before/after, matched pairs)
Use case: Did our website redesign improve time on site?
"""

import numpy as np
from scipy import stats

# Sample data: Time on site (minutes) before and after redesign
before = np.array([3.2, 2.8, 3.5, 3.0, 2.9, 3.3, 3.1, 2.7, 3.4, 3.0])
after = np.array([3.8, 3.5, 4.0, 3.6, 3.4, 3.9, 3.7, 3.2, 4.1, 3.5])

# Calculate differences
differences = after - before

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(after, before)

# Calculate Cohen's d (effect size) for paired samples
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)
cohens_d = mean_diff / std_diff

# Calculate 95% confidence interval for the difference
n = len(differences)
df = n - 1
se = stats.sem(differences)
ci_margin = stats.t.ppf(0.975, df) * se
ci_lower = mean_diff - ci_margin
ci_upper = mean_diff + ci_margin

# Print results
print("=" * 60)
print("PAIRED T-TEST")
print("=" * 60)
print(f"Before Mean: {np.mean(before):.2f} minutes")
print(f"After Mean: {np.mean(after):.2f} minutes")
print(f"Mean Difference: {mean_diff:.2f} minutes")
print(f"Sample Size (pairs): {n}")
print(f"\nT-statistic: {t_stat:.4f}")
print(f"Degrees of Freedom: {df}")
print(f"P-value (two-tailed): {p_value:.4f}")
print(f"Cohen's d (effect size): {cohens_d:.4f}")
print(f"\n95% Confidence Interval for Difference: [{ci_lower:.2f}, {ci_upper:.2f}] minutes")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

print("\nEffect Size Interpretation:")
if abs(cohens_d) < 0.2:
    print("  Small effect")
elif abs(cohens_d) < 0.5:
    print("  Medium effect")
else:
    print("  Large effect")

print("\nInterpretation:")
if p_value < 0.05:
    if mean_diff > 0:
        print(f"  Time on site significantly INCREASED by {mean_diff:.2f} minutes after redesign.")
    else:
        print(f"  Time on site significantly DECREASED by {abs(mean_diff):.2f} minutes after redesign.")
else:
    print("  No significant change in time on site after redesign.")
```

### Independent T-Test

```python
"""
Independent T-Test: Compare means of two independent groups
Use case: Do email campaign A and B have different click-through rates?
"""

import numpy as np
from scipy import stats

# Sample data: Click-through rates (%) for two campaigns
campaign_a = np.array([5.2, 6.1, 5.8, 6.3, 5.5, 6.0, 5.7, 6.2, 5.9, 6.4])
campaign_b = np.array([4.8, 5.2, 5.0, 5.5, 4.9, 5.1, 5.3, 4.7, 5.4, 5.0])

# Perform independent t-test (assuming equal variances)
t_stat, p_value = stats.ttest_ind(campaign_a, campaign_b)

# Calculate pooled standard deviation and Cohen's d
n1 = len(campaign_a)
n2 = len(campaign_b)
var1 = np.var(campaign_a, ddof=1)
var2 = np.var(campaign_b, ddof=1)
pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
pooled_std = np.sqrt(pooled_var)
cohens_d = (np.mean(campaign_a) - np.mean(campaign_b)) / pooled_std

# Calculate 95% confidence interval for difference in means
df = n1 + n2 - 2
se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
ci_margin = stats.t.ppf(0.975, df) * se_diff
mean_diff = np.mean(campaign_a) - np.mean(campaign_b)
ci_lower = mean_diff - ci_margin
ci_upper = mean_diff + ci_margin

# Print results
print("=" * 60)
print("INDEPENDENT T-TEST")
print("=" * 60)
print(f"Campaign A Mean: {np.mean(campaign_a):.2f}%")
print(f"Campaign B Mean: {np.mean(campaign_b):.2f}%")
print(f"Difference: {mean_diff:.2f}%")
print(f"Sample Size A: {n1}")
print(f"Sample Size B: {n2}")
print(f"\nT-statistic: {t_stat:.4f}")
print(f"Degrees of Freedom: {df}")
print(f"P-value (two-tailed): {p_value:.4f}")
print(f"Cohen's d (effect size): {cohens_d:.4f}")
print(f"\n95% Confidence Interval for Difference: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

print("\nEffect Size Interpretation:")
if abs(cohens_d) < 0.2:
    print("  Small effect")
elif abs(cohens_d) < 0.5:
    print("  Medium effect")
else:
    print("  Large effect")

print("\nInterpretation:")
if p_value < 0.05:
    if mean_diff > 0:
        print(f"  Campaign A has a significantly HIGHER CTR than Campaign B.")
    else:
        print(f"  Campaign A has a significantly LOWER CTR than Campaign B.")
else:
    print("  No significant difference in CTR between campaigns.")
```

---

## 4. Z-Test for Significance

```python
"""
Z-Test for Proportions: Compare two proportions
Use case: Is there a significant difference in conversion rates between two landing pages?
"""

import numpy as np
from scipy import stats

# Sample data
n1 = 1000  # Sample size for landing page A
p1 = 0.052  # Conversion rate for A (5.2%)

n2 = 950   # Sample size for landing page B
p2 = 0.048  # Conversion rate for B (4.8%)

# Calculate pooled proportion
pooled_p = (n1 * p1 + n2 * p2) / (n1 + n2)

# Calculate standard error
se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))

# Calculate z-statistic
z_stat = (p1 - p2) / se

# Calculate p-value (two-tailed)
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Critical z-values for different confidence levels
z_critical_95 = 1.96
z_critical_99 = 2.576

# Print results
print("=" * 60)
print("Z-TEST FOR PROPORTIONS")
print("=" * 60)
print(f"Landing Page A: {p1*100:.2f}% conversion (n={n1})")
print(f"Landing Page B: {p2*100:.2f}% conversion (n={n2})")
print(f"Difference: {(p1-p2)*100:.2f} percentage points")
print(f"\nPooled Proportion: {pooled_p:.4f}")
print(f"Standard Error: {se:.6f}")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value (two-tailed): {p_value:.4f}")
print(f"\nCritical values:")
print(f"  95% confidence: ±{z_critical_95:.3f}")
print(f"  99% confidence: ±{z_critical_99:.3f}")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

print("\nInterpretation:")
if p_value < 0.05:
    if p1 > p2:
        print(f"  Landing Page A converts significantly BETTER than B.")
    else:
        print(f"  Landing Page B converts significantly BETTER than A.")
    print(f"  You can be 95% confident this difference is real.")
else:
    print("  No significant difference in conversion rates.")
    print("  The observed difference could be due to random chance.")
```

---

## 5. Correlation (Pearson's r)

```python
"""
Pearson Correlation: Measure linear relationship between two continuous variables
Use case: Is there a relationship between ad spend and website traffic?
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Sample data: Ad spend ($1000s) and website visits (1000s)
ad_spend = np.array([10, 15, 12, 18, 20, 14, 16, 22, 19, 17])
website_visits = np.array([25, 32, 28, 38, 42, 30, 34, 45, 40, 36])

# Calculate Pearson correlation
r, p_value = stats.pearsonr(ad_spend, website_visits)

# Calculate coefficient of determination
r_squared = r ** 2

# Calculate 95% confidence interval for r using Fisher's z-transformation
n = len(ad_spend)
z = np.arctanh(r)  # Fisher's z-transformation
se = 1 / np.sqrt(n - 3)
z_critical = 1.96
ci_lower_z = z - z_critical * se
ci_upper_z = z + z_critical * se
ci_lower_r = np.tanh(ci_lower_z)
ci_upper_r = np.tanh(ci_upper_z)

# Print results
print("=" * 60)
print("PEARSON CORRELATION ANALYSIS")
print("=" * 60)
print(f"Sample Size: {n}")
print(f"Correlation Coefficient (r): {r:.4f}")
print(f"Coefficient of Determination (r²): {r_squared:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"95% Confidence Interval: [{ci_lower_r:.4f}, {ci_upper_r:.4f}]")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

# Interpret correlation strength
print("\nCorrelation Strength:")
if abs(r) < 0.3:
    print("  Weak correlation")
elif abs(r) < 0.7:
    print("  Moderate correlation")
else:
    print("  Strong correlation")

print("\nCorrelation Direction:")
if r > 0:
    print("  Positive: As ad spend increases, website visits increase")
else:
    print("  Negative: As ad spend increases, website visits decrease")

print("\nInterpretation:")
if p_value < 0.05:
    print(f"  There is a significant {'positive' if r > 0 else 'negative'} correlation.")
    print(f"  {r_squared*100:.1f}% of variance in website visits is explained by ad spend.")
else:
    print("  No significant correlation detected.")

# Optional: Create scatter plot with regression line
print("\nGenerating scatter plot...")
plt.figure(figsize=(10, 6))
plt.scatter(ad_spend, website_visits, s=100, alpha=0.6, edgecolors='black')
plt.xlabel('Ad Spend ($1000s)', fontsize=12)
plt.ylabel('Website Visits (1000s)', fontsize=12)
plt.title(f'Correlation: Ad Spend vs Website Visits\n(r = {r:.3f}, p = {p_value:.4f})', fontsize=14)

# Add regression line
z = np.polyfit(ad_spend, website_visits, 1)
p = np.poly1d(z)
plt.plot(ad_spend, p(ad_spend), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('correlation_plot.png', dpi=300)
print("Plot saved as 'correlation_plot.png'")
# plt.show()  # Uncomment to display plot
```

---

## 6. Mann-Whitney U Test

```python
"""
Mann-Whitney U Test: Non-parametric test for comparing two independent groups
Use case: Compare session durations between mobile and desktop users (non-normal data)
"""

import numpy as np
from scipy import stats

# Sample data: Session duration (minutes) - note the skewed/outlier-prone nature
mobile_users = np.array([2.3, 3.1, 2.8, 3.5, 2.5, 3.0, 2.9, 4.2, 2.7, 3.3, 15.5])  # outlier: 15.5
desktop_users = np.array([4.5, 5.2, 4.8, 5.0, 4.3, 5.5, 4.7, 4.9, 5.1, 4.6])

# Perform Mann-Whitney U test
u_statistic, p_value = stats.mannwhitneyu(mobile_users, desktop_users, alternative='two-sided')

# Calculate medians (more robust than means for skewed data)
median_mobile = np.median(mobile_users)
median_desktop = np.median(desktop_users)

# Calculate rank-biserial correlation (effect size for Mann-Whitney)
n1 = len(mobile_users)
n2 = len(desktop_users)
rank_biserial = 1 - (2*u_statistic) / (n1 * n2)

# Print results
print("=" * 60)
print("MANN-WHITNEY U TEST")
print("=" * 60)
print(f"Mobile Users (n={n1}):")
print(f"  Median: {median_mobile:.2f} minutes")
print(f"  Mean: {np.mean(mobile_users):.2f} minutes (note: affected by outliers)")
print(f"  Data: {mobile_users}")

print(f"\nDesktop Users (n={n2}):")
print(f"  Median: {median_desktop:.2f} minutes")
print(f"  Mean: {np.mean(desktop_users):.2f} minutes")
print(f"  Data: {desktop_users}")

print(f"\nU-statistic: {u_statistic:.4f}")
print(f"P-value (two-tailed): {p_value:.4f}")
print(f"Rank-biserial correlation (effect size): {rank_biserial:.4f}")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

print("\nEffect Size Interpretation:")
if abs(rank_biserial) < 0.1:
    print("  Negligible effect")
elif abs(rank_biserial) < 0.3:
    print("  Small effect")
elif abs(rank_biserial) < 0.5:
    print("  Medium effect")
else:
    print("  Large effect")

print("\nInterpretation:")
if p_value < 0.05:
    if median_mobile < median_desktop:
        print(f"  Desktop users have significantly LONGER session durations than mobile users.")
    else:
        print(f"  Mobile users have significantly LONGER session durations than desktop users.")
else:
    print("  No significant difference in session durations between mobile and desktop users.")

print("\nWhy use Mann-Whitney instead of t-test?")
print("  - Mobile data has an outlier (15.5 minutes)")
print("  - Mann-Whitney is robust to outliers and doesn't assume normal distribution")
print("  - It compares medians and ranks rather than means")
```

---

## 7. Normal Distribution

```python
"""
Normal Distribution: Calculate probabilities and z-scores
Use case: Understanding probability of different conversion rates
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Distribution parameters
mean = 5.0  # Mean conversion rate (%)
std_dev = 1.2  # Standard deviation

# Example: What's the probability of getting conversion rate > 6.5%?
threshold = 6.5

# Calculate z-score
z_score = (threshold - mean) / std_dev

# Calculate probability (area under curve)
prob_above = 1 - stats.norm.cdf(z_score)
prob_below = stats.norm.cdf(z_score)

# Calculate specific percentiles
percentile_95 = stats.norm.ppf(0.95, loc=mean, scale=std_dev)
percentile_99 = stats.norm.ppf(0.99, loc=mean, scale=std_dev)

# Print results
print("=" * 60)
print("NORMAL DISTRIBUTION ANALYSIS")
print("=" * 60)
print(f"Distribution: N(μ={mean}, σ={std_dev})")
print(f"\nQuestion: What's the probability of conversion rate > {threshold}%?")
print(f"\nZ-score: {z_score:.4f}")
print(f"Probability (> {threshold}%): {prob_above:.4f} ({prob_above*100:.2f}%)")
print(f"Probability (≤ {threshold}%): {prob_below:.4f} ({prob_below*100:.2f}%)")

print(f"\nKey Percentiles:")
print(f"  95th percentile: {percentile_95:.2f}%")
print(f"  99th percentile: {percentile_99:.2f}%")

print("\nInterpretation:")
print(f"  If conversion rates follow N({mean}, {std_dev}), then:")
print(f"  - {prob_above*100:.1f}% of the time, conversion rate will exceed {threshold}%")
print(f"  - 95% of conversion rates will be below {percentile_95:.2f}%")

# Calculate confidence intervals
ci_68 = (mean - std_dev, mean + std_dev)  # 68% CI (±1 SD)
ci_95 = (mean - 1.96*std_dev, mean + 1.96*std_dev)  # 95% CI (±1.96 SD)
ci_99 = (mean - 2.576*std_dev, mean + 2.576*std_dev)  # 99% CI (±2.576 SD)

print(f"\nConfidence Intervals:")
print(f"  68% of data: [{ci_68[0]:.2f}%, {ci_68[1]:.2f}%]")
print(f"  95% of data: [{ci_95[0]:.2f}%, {ci_95[1]:.2f}%]")
print(f"  99% of data: [{ci_99[0]:.2f}%, {ci_99[1]:.2f}%]")

# Optional: Visualize
print("\nGenerating visualization...")
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = stats.norm.pdf(x, mean, std_dev)

plt.figure(figsize=(12, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Normal Distribution')
plt.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Mean = {mean}')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')

# Shade area above threshold
x_fill = x[x >= threshold]
y_fill = stats.norm.pdf(x_fill, mean, std_dev)
plt.fill_between(x_fill, y_fill, alpha=0.3, color='red', label=f'P(X > {threshold}) = {prob_above:.4f}')

plt.xlabel('Conversion Rate (%)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title(f'Normal Distribution: N({mean}, {std_dev}²)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('normal_distribution.png', dpi=300)
print("Plot saved as 'normal_distribution.png'")
# plt.show()  # Uncomment to display
```

---

## 8. Power Analysis

```python
"""
Power Analysis: Calculate required sample size for detecting an effect
Use case: How many subjects do we need for our A/B test?
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power

# Parameters for power analysis
effect_size = 0.5  # Cohen's d (0.2=small, 0.5=medium, 0.8=large)
alpha = 0.05  # Significance level (Type I error rate)
power = 0.80  # Statistical power (1 - Type II error rate)

print("=" * 60)
print("POWER ANALYSIS - SAMPLE SIZE CALCULATION")
print("=" * 60)

# 1. T-Test (comparing two means)
print("\n1. INDEPENDENT T-TEST")
print("-" * 60)
sample_size_per_group = tt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    ratio=1.0,  # Equal group sizes
    alternative='two-sided'
)
total_sample_t = int(np.ceil(sample_size_per_group)) * 2

print(f"Parameters:")
print(f"  Effect Size (Cohen's d): {effect_size}")
print(f"  Significance Level (α): {alpha}")
print(f"  Desired Power (1-β): {power}")
print(f"\nRequired Sample Size:")
print(f"  Per Group: {int(np.ceil(sample_size_per_group))} subjects")
print(f"  Total: {total_sample_t} subjects")

# 2. Z-Test for Proportions
print("\n2. Z-TEST FOR PROPORTIONS")
print("-" * 60)
p1 = 0.10  # Control group conversion rate (10%)
p2 = 0.15  # Treatment group conversion rate (15%)
effect_size_prop = (p2 - p1) / np.sqrt(p1 * (1 - p1))  # Effect size for proportions

sample_size_prop = zt_ind_solve_power(
    effect_size=effect_size_prop,
    alpha=alpha,
    power=power,
    ratio=1.0,
    alternative='two-sided'
)
total_sample_prop = int(np.ceil(sample_size_prop)) * 2

print(f"Parameters:")
print(f"  Control Conversion Rate: {p1*100:.1f}%")
print(f"  Treatment Conversion Rate: {p2*100:.1f}%")
print(f"  Absolute Difference: {(p2-p1)*100:.1f} percentage points")
print(f"  Effect Size: {effect_size_prop:.4f}")
print(f"  Significance Level (α): {alpha}")
print(f"  Desired Power (1-β): {power}")
print(f"\nRequired Sample Size:")
print(f"  Per Group: {int(np.ceil(sample_size_prop))} subjects")
print(f"  Total: {total_sample_prop} subjects")

# 3. Power curve - show relationship between sample size and power
print("\n3. POWER CURVE ANALYSIS")
print("-" * 60)
sample_sizes = np.arange(10, 200, 10)
powers = [tt_ind_solve_power(effect_size=effect_size, alpha=alpha, nobs1=n, ratio=1.0, alternative='two-sided')
          for n in sample_sizes]

print("Sample Size per Group -> Statistical Power:")
for n, p in zip(sample_sizes[::3], powers[::3]):  # Show every 3rd value
    print(f"  n = {n:3d} -> Power = {p:.4f} ({p*100:.1f}%)")

# Interpretation
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("\nWhat is Statistical Power?")
print("  Power is the probability of detecting a true effect when it exists.")
print(f"  With {power*100:.0f}% power, we have an {power*100:.0f}% chance of detecting")
print(f"  an effect size of {effect_size} if it truly exists.")

print("\nWhat happens if we use fewer subjects?")
if sample_size_per_group > 20:
    small_sample = int(sample_size_per_group / 2)
    small_power = tt_ind_solve_power(effect_size=effect_size, alpha=alpha, nobs1=small_sample, ratio=1.0, alternative='two-sided')
    print(f"  With only {small_sample} per group:")
    print(f"    Power drops to {small_power:.4f} ({small_power*100:.1f}%)")
    print(f"    {(1-small_power)*100:.1f}% chance of missing a real effect (Type II error)")

print("\nRecommendation:")
print(f"  Use at least {int(np.ceil(sample_size_per_group))} subjects per group")
print(f"  Total minimum sample size: {total_sample_t} subjects")

# Optional: Plot power curve
print("\nGenerating power curve...")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, powers, 'b-', linewidth=2)
plt.axhline(y=0.80, color='r', linestyle='--', linewidth=2, label='Target Power = 0.80')
plt.axvline(x=sample_size_per_group, color='g', linestyle='--', linewidth=2,
            label=f'Required n = {int(np.ceil(sample_size_per_group))}')
plt.xlabel('Sample Size per Group', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.title(f'Power Analysis: Effect Size = {effect_size}, α = {alpha}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('power_analysis.png', dpi=300)
print("Plot saved as 'power_analysis.png'")
# plt.show()  # Uncomment to display
```

---

## 9. Simple Linear Regression

```python
"""
Simple Linear Regression: Predict outcome from one predictor
Use case: Predict sales from advertising spend
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Sample data: Ad spend ($1000s) and Sales ($1000s)
ad_spend = np.array([10, 15, 12, 18, 20, 14, 16, 22, 19, 17, 13, 21, 11, 16, 19])
sales = np.array([50, 75, 60, 90, 100, 70, 80, 110, 95, 85, 65, 105, 55, 82, 98])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(ad_spend, sales)

# Calculate predictions
predictions = slope * ad_spend + intercept

# Calculate residuals
residuals = sales - predictions

# Calculate R-squared
r_squared = r_value ** 2

# Calculate standard error of the estimate
n = len(ad_spend)
degrees_freedom = n - 2
ss_residual = np.sum(residuals ** 2)
se_estimate = np.sqrt(ss_residual / degrees_freedom)

# Calculate 95% confidence intervals for slope and intercept
se_slope = std_err
t_critical = stats.t.ppf(0.975, degrees_freedom)
slope_ci_lower = slope - t_critical * se_slope
slope_ci_upper = slope + t_critical * se_slope

# Print results
print("=" * 60)
print("SIMPLE LINEAR REGRESSION")
print("=" * 60)
print(f"Sample Size: {n}")
print(f"\nRegression Equation:")
print(f"  Sales = {intercept:.2f} + {slope:.2f} × Ad Spend")
print(f"\nCoefficients:")
print(f"  Intercept (b₀): {intercept:.4f}")
print(f"  Slope (b₁): {slope:.4f}")
print(f"  95% CI for Slope: [{slope_ci_lower:.4f}, {slope_ci_upper:.4f}]")
print(f"\nModel Fit:")
print(f"  R (correlation): {r_value:.4f}")
print(f"  R² (coefficient of determination): {r_squared:.4f}")
print(f"  Standard Error of Estimate: {se_estimate:.4f}")
print(f"  P-value for slope: {p_value:.4f}")
print(f"\nSignificant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

print("\nInterpretation:")
print(f"  - For every $1,000 increase in ad spend, sales increase by ${slope:.2f}k")
print(f"  - {r_squared*100:.1f}% of variance in sales is explained by ad spend")
print(f"  - When ad spend is $0, predicted sales are ${intercept:.2f}k (intercept)")

# Example predictions
print("\nExample Predictions:")
new_ad_spends = [15, 20, 25]
for new_spend in new_ad_spends:
    predicted_sales = slope * new_spend + intercept
    print(f"  Ad Spend = ${new_spend}k -> Predicted Sales = ${predicted_sales:.2f}k")

# Residual analysis
print("\nResidual Analysis:")
print(f"  Mean Residual: {np.mean(residuals):.4f} (should be ~0)")
print(f"  Residual Std Dev: {np.std(residuals):.4f}")
print(f"  Max Positive Residual: {np.max(residuals):.2f}")
print(f"  Max Negative Residual: {np.min(residuals):.2f}")

# Optional: Create visualization
print("\nGenerating regression plot...")
plt.figure(figsize=(12, 5))

# Plot 1: Scatter plot with regression line
plt.subplot(1, 2, 1)
plt.scatter(ad_spend, sales, s=100, alpha=0.6, edgecolors='black', label='Actual Data')
plt.plot(ad_spend, predictions, 'r-', linewidth=2, label=f'Regression Line\n(R² = {r_squared:.3f})')
plt.xlabel('Ad Spend ($1000s)', fontsize=12)
plt.ylabel('Sales ($1000s)', fontsize=12)
plt.title('Linear Regression: Sales vs Ad Spend', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residual plot
plt.subplot(1, 2, 2)
plt.scatter(predictions, residuals, s=100, alpha=0.6, edgecolors='black')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Sales ($1000s)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_analysis.png', dpi=300)
print("Plot saved as 'regression_analysis.png'")
# plt.show()  # Uncomment to display
```

---

## 10. Proportion Significance (Z-Test)

```python
"""
Z-Test for Proportions: Compare multiple proportions pairwise
Use case: Compare click-through rates across different email segments
"""

import numpy as np
from scipy import stats

# Sample data: CTR for different email segments
segments = ['18-25', '26-35', '36-45', '46-55', '55+']
sample_sizes = [500, 650, 720, 580, 450]
click_rates = [0.068, 0.082, 0.075, 0.061, 0.055]  # CTR as proportions

print("=" * 60)
print("PAIRWISE Z-TEST FOR PROPORTIONS")
print("=" * 60)
print("\nSegment Data:")
for seg, n, ctr in zip(segments, sample_sizes, click_rates):
    print(f"  {seg:>10}: n={n:>4}, CTR={ctr*100:>5.2f}%")

# Perform pairwise comparisons
print("\n" + "=" * 60)
print("PAIRWISE COMPARISONS (α = 0.05)")
print("=" * 60)

comparisons = []
for i in range(len(segments)):
    for j in range(i+1, len(segments)):
        # Extract data for this comparison
        n1, n2 = sample_sizes[i], sample_sizes[j]
        p1, p2 = click_rates[i], click_rates[j]

        # Calculate pooled proportion
        pooled_p = (n1 * p1 + n2 * p2) / (n1 + n2)

        # Calculate standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))

        # Calculate z-statistic
        z_stat = (p1 - p2) / se

        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Determine significance
        is_significant = p_value < 0.05

        # Store results
        comparisons.append({
            'seg1': segments[i],
            'seg2': segments[j],
            'p1': p1,
            'p2': p2,
            'z_stat': z_stat,
            'p_value': p_value,
            'significant': is_significant
        })

        # Print result
        diff = (p1 - p2) * 100
        print(f"\n{segments[i]} vs {segments[j]}:")
        print(f"  Difference: {diff:+.2f} percentage points")
        print(f"  Z-statistic: {z_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant? {'✓ YES' if is_significant else '✗ No'}")

# Summary of significant differences
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
significant_comps = [c for c in comparisons if c['significant']]
print(f"\nSignificant Differences Found: {len(significant_comps)} out of {len(comparisons)}")

if significant_comps:
    print("\nSegments with Significant Differences:")
    for comp in significant_comps:
        direction = "higher" if comp['p1'] > comp['p2'] else "lower"
        print(f"  - {comp['seg1']} has {direction} CTR than {comp['seg2']} (p={comp['p_value']:.4f})")
else:
    print("\nNo significant differences detected between any segments.")

# Bonferroni correction for multiple comparisons
bonferroni_alpha = 0.05 / len(comparisons)
print(f"\nMultiple Comparison Correction:")
print(f"  Original α: 0.05")
print(f"  Bonferroni-corrected α: {bonferroni_alpha:.4f}")
print(f"  (Dividing by {len(comparisons)} comparisons)")

bonferroni_significant = [c for c in comparisons if c['p_value'] < bonferroni_alpha]
if bonferroni_significant:
    print(f"\n  Significant after correction: {len(bonferroni_significant)}")
    for comp in bonferroni_significant:
        print(f"    - {comp['seg1']} vs {comp['seg2']} (p={comp['p_value']:.4f})")
else:
    print(f"\n  No comparisons remain significant after Bonferroni correction.")
```

---

## 11. Bayesian Statistics (Beta-Binomial)

```python
"""
Bayesian A/B Testing using Beta-Binomial Conjugate Prior
Use case: Compare two variations with proper Bayesian inference
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Prior beliefs (Beta distribution parameters)
# Using uniform prior: Beta(1, 1) = no prior knowledge
alpha_prior = 1
beta_prior = 1

# Observed data
# Variation A
n_a = 1000  # Number of visitors
conversions_a = 52  # Number of conversions

# Variation B
n_b = 950
conversions_b = 68

# Update posterior using Beta-Binomial conjugate prior
# Posterior = Beta(α_prior + successes, β_prior + failures)
alpha_posterior_a = alpha_prior + conversions_a
beta_posterior_a = beta_prior + (n_a - conversions_a)

alpha_posterior_b = alpha_prior + conversions_b
beta_posterior_b = beta_prior + (n_b - conversions_b)

# Calculate posterior means (point estimates)
mean_a = alpha_posterior_a / (alpha_posterior_a + beta_posterior_a)
mean_b = alpha_posterior_b / (alpha_posterior_b + beta_posterior_b)

# Calculate 95% credible intervals
ci_a_lower = stats.beta.ppf(0.025, alpha_posterior_a, beta_posterior_a)
ci_a_upper = stats.beta.ppf(0.975, alpha_posterior_a, beta_posterior_a)

ci_b_lower = stats.beta.ppf(0.025, alpha_posterior_b, beta_posterior_b)
ci_b_upper = stats.beta.ppf(0.975, alpha_posterior_b, beta_posterior_b)

# Calculate P(B > A) using Monte Carlo sampling
np.random.seed(42)
n_samples = 100000
samples_a = stats.beta.rvs(alpha_posterior_a, beta_posterior_a, size=n_samples)
samples_b = stats.beta.rvs(alpha_posterior_b, beta_posterior_b, size=n_samples)
prob_b_better = np.mean(samples_b > samples_a)

# Calculate expected loss (how much we lose if we choose wrong variant)
expected_loss_a = np.mean(np.maximum(samples_b - samples_a, 0))
expected_loss_b = np.mean(np.maximum(samples_a - samples_b, 0))

# Print results
print("=" * 60)
print("BAYESIAN A/B TEST ANALYSIS")
print("=" * 60)

print("\nPrior Beliefs:")
print(f"  Beta({alpha_prior}, {beta_prior}) - Uniform prior (no prior knowledge)")

print("\nObserved Data:")
print(f"  Variation A: {conversions_a}/{n_a} conversions ({conversions_a/n_a*100:.2f}%)")
print(f"  Variation B: {conversions_b}/{n_b} conversions ({conversions_b/n_b*100:.2f}%)")

print("\nPosterior Distributions:")
print(f"  Variation A: Beta({alpha_posterior_a}, {beta_posterior_a})")
print(f"    Mean: {mean_a:.4f} ({mean_a*100:.2f}%)")
print(f"    95% Credible Interval: [{ci_a_lower:.4f}, {ci_a_upper:.4f}]")
print(f"                           [{ci_a_lower*100:.2f}%, {ci_a_upper*100:.2f}%]")

print(f"\n  Variation B: Beta({alpha_posterior_b}, {beta_posterior_b})")
print(f"    Mean: {mean_b:.4f} ({mean_b*100:.2f}%)")
print(f"    95% Credible Interval: [{ci_b_lower:.4f}, {ci_b_upper:.4f}]")
print(f"                           [{ci_b_lower*100:.2f}%, {ci_b_upper*100:.2f}%]")

print("\n" + "=" * 60)
print("DECISION ANALYSIS")
print("=" * 60)
print(f"\nP(B > A) = {prob_b_better:.4f} ({prob_b_better*100:.1f}%)")
print(f"P(A > B) = {1-prob_b_better:.4f} ({(1-prob_b_better)*100:.1f}%)")

print(f"\nExpected Loss:")
print(f"  If we choose A: {expected_loss_a:.6f} ({expected_loss_a*100:.4f}%)")
print(f"  If we choose B: {expected_loss_b:.6f} ({expected_loss_b*100:.4f}%)")

print("\nRecommendation:")
if prob_b_better > 0.95:
    print(f"  ✓ Strong evidence for B (P(B>A) = {prob_b_better*100:.1f}%)")
    print(f"  You can confidently choose Variation B.")
elif prob_b_better > 0.90:
    print(f"  ✓ Moderate evidence for B (P(B>A) = {prob_b_better*100:.1f}%)")
    print(f"  Consider choosing Variation B.")
elif prob_b_better < 0.05:
    print(f"  ✓ Strong evidence for A (P(A>B) = {(1-prob_b_better)*100:.1f}%)")
    print(f"  You can confidently choose Variation A.")
elif prob_b_better < 0.10:
    print(f"  ✓ Moderate evidence for A (P(A>B) = {(1-prob_b_better)*100:.1f}%)")
    print(f"  Consider choosing Variation A.")
else:
    print(f"  ⚠ Inconclusive (P(B>A) = {prob_b_better*100:.1f}%)")
    print(f"  Collect more data before making a decision.")

print("\nInterpretation:")
print(f"  There is a {prob_b_better*100:.1f}% probability that Variation B")
print(f"  has a higher conversion rate than Variation A.")
print(f"\n  The 95% credible interval tells us:")
print(f"  - We're 95% confident A's true rate is between {ci_a_lower*100:.2f}% and {ci_a_upper*100:.2f}%")
print(f"  - We're 95% confident B's true rate is between {ci_b_lower*100:.2f}% and {ci_b_upper*100:.2f}%")

# Optional: Visualize posterior distributions
print("\nGenerating visualization...")
x = np.linspace(0, 0.12, 1000)
pdf_a = stats.beta.pdf(x, alpha_posterior_a, beta_posterior_a)
pdf_b = stats.beta.pdf(x, alpha_posterior_b, beta_posterior_b)

plt.figure(figsize=(12, 6))

# Plot 1: Posterior distributions
plt.subplot(1, 2, 1)
plt.plot(x, pdf_a, 'b-', linewidth=2, label=f'Variation A (Mean={mean_a:.4f})')
plt.plot(x, pdf_b, 'r-', linewidth=2, label=f'Variation B (Mean={mean_b:.4f})')
plt.axvline(mean_a, color='blue', linestyle='--', alpha=0.7)
plt.axvline(mean_b, color='red', linestyle='--', alpha=0.7)
plt.fill_between(x, pdf_a, alpha=0.2, color='blue')
plt.fill_between(x, pdf_b, alpha=0.2, color='red')
plt.xlabel('Conversion Rate', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Posterior Distributions', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Histogram of differences (B - A)
plt.subplot(1, 2, 2)
differences = samples_b - samples_a
plt.hist(differences, bins=100, alpha=0.7, color='green', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
plt.axvline(np.mean(differences), color='blue', linestyle='--', linewidth=2,
            label=f'Mean Diff = {np.mean(differences):.6f}')
plt.xlabel('Difference in Conversion Rate (B - A)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Distribution of Differences\nP(B > A) = {prob_b_better:.4f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_ab_test.png', dpi=300)
print("Plot saved as 'bayesian_ab_test.png'")
# plt.show()  # Uncomment to display
```

---

## Installation Requirements

To run all examples, install the required packages:

```bash
pip install numpy scipy pandas matplotlib statsmodels
```

Individual package versions (tested):
- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- statsmodels >= 0.13.0

---

## Quick Reference: When to Use Each Test

| Test | Use When | Data Type | Example |
|------|----------|-----------|---------|
| **ANOVA** | Comparing 3+ groups | Continuous outcome, categorical predictor | Email open rates across 3 subject lines |
| **Chi-Square** | Testing categorical relationships | Categorical variables | Age group vs product preference |
| **T-Test** | Comparing 2 groups or to benchmark | Continuous outcome | A/B test with conversion rates |
| **Z-Test** | Comparing proportions (large n) | Proportions/percentages | CTR comparison with n>30 |
| **Correlation** | Measuring linear relationship | Two continuous variables | Ad spend vs website traffic |
| **Mann-Whitney** | Comparing 2 groups (non-normal) | Ordinal or skewed continuous | Session duration with outliers |
| **Regression** | Predicting outcome | Continuous outcome & predictor | Predict sales from ad spend |
| **Power Analysis** | Planning study sample size | Any | How many subjects for A/B test? |
| **Bayesian** | A/B testing with uncertainty | Binary outcomes | Conversion rate comparison |

---

## Additional Tips

1. **Always check assumptions** before running parametric tests (normality, equal variances)
2. **Visualize your data** first - plots reveal patterns tests might miss
3. **Report effect sizes** along with p-values - statistical ≠ practical significance
4. **Consider multiple comparison corrections** when doing many tests (Bonferroni)
5. **Use Bayesian methods** for small samples or sequential testing

---

**License:** MIT - Free to use and modify
**Last Updated:** January 2025
