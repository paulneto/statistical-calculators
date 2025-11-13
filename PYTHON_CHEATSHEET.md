# 🐍 Python Stats Cheatsheet

Your friendly guide to running statistical tests in Python. Each test includes a complete, copy-paste ready example.

**Install once:** `pip install numpy scipy pandas statsmodels matplotlib`

---

## 📊 Quick Navigation
- [T-Test](#-t-test) • [ANOVA](#-anova) • [Chi-Square](#-chi-square) • [Correlation](#-correlation)
- [Regression](#-regression) • [Mann-Whitney](#-mann-whitney) • [Proportions](#-proportions-z-test) • [Bayesian](#-bayesian-ab-testing)

---

## 📌 T-Test

**What is it?** Compares averages between two groups or against a benchmark.

**When to use it:** A/B testing conversion rates, before/after comparisons, checking if you beat a benchmark.

### Copy-Paste Example: Independent T-Test (A/B Test)

```python
import numpy as np
from scipy import stats

# Your data: conversion rates for two campaigns
campaign_a = np.array([5.2, 6.1, 5.8, 6.3, 5.5, 6.0, 5.7])
campaign_b = np.array([4.8, 5.2, 5.0, 5.5, 4.9, 5.1, 5.3])

# Run the test
t_stat, p_value = stats.ttest_ind(campaign_a, campaign_b)

# Calculate Cohen's d (effect size)
mean_diff = np.mean(campaign_a) - np.mean(campaign_b)
pooled_std = np.sqrt((np.var(campaign_a, ddof=1) + np.var(campaign_b, ddof=1)) / 2)
cohens_d = mean_diff / pooled_std

# Results
print(f"Campaign A mean: {np.mean(campaign_a):.2f}%")
print(f"Campaign B mean: {np.mean(campaign_b):.2f}%")
print(f"P-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.3f}")
print(f"Significant? {'YES ✓' if p_value < 0.05 else 'No ✗'}")
```

**How to interpret:**
- **P-value < 0.05?** Campaigns are significantly different
- **Cohen's d:** 0.2=small, 0.5=medium, 0.8=large effect
- **Means:** Which campaign actually performed better?

**Watch out for:**
- Need at least ~5-10 samples per group
- Data should be roughly bell-shaped (use Mann-Whitney if super skewed)
- Large sample sizes can make tiny differences "significant"

---

## 📊 ANOVA

**What is it?** Like a t-test but for comparing 3+ groups at once.

**When to use it:** Testing multiple email subject lines, comparing performance across regions, analyzing 3+ campaign variations.

### Copy-Paste Example: One-Way ANOVA

```python
import numpy as np
from scipy import stats

# Your data: open rates (%) for three subject line types
questions = np.array([12.5, 15.3, 13.8, 14.2, 16.1])
urgency = np.array([18.2, 19.5, 17.8, 20.1, 18.9])
personalized = np.array([14.5, 15.8, 13.9, 16.2, 14.7])

# Run ANOVA
f_stat, p_value = stats.f_oneway(questions, urgency, personalized)

# Calculate eta-squared (effect size)
all_data = np.concatenate([questions, urgency, personalized])
grand_mean = np.mean(all_data)
ssb = len(questions) * (np.mean(questions) - grand_mean)**2 + \
      len(urgency) * (np.mean(urgency) - grand_mean)**2 + \
      len(personalized) * (np.mean(personalized) - grand_mean)**2
sst = np.sum((all_data - grand_mean)**2)
eta_squared = ssb / sst

# Results
print(f"Questions: {np.mean(questions):.1f}%")
print(f"Urgency: {np.mean(urgency):.1f}%")
print(f"Personalized: {np.mean(personalized):.1f}%")
print(f"\nF-statistic: {f_stat:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Eta-squared: {eta_squared:.3f} ({eta_squared*100:.1f}% of variance explained)")
print(f"Significant? {'YES ✓' if p_value < 0.05 else 'No ✗'}")
```

**How to interpret:**
- **P-value < 0.05?** At least one group is different (but doesn't say which!)
- **Eta-squared:** % of variance explained by your groups
- **Means:** Look at which group has highest/lowest average

**Watch out for:**
- ANOVA only tells you groups differ - not which specific pairs
- Need follow-up tests (Tukey HSD) to compare specific groups
- All groups should have similar spread (variance)

---

## 🎲 Chi-Square

**What is it?** Tests if categories are related or if frequencies match expectations.

**When to use it:** Testing if age affects product preference, checking if days of week affect sales, analyzing survey responses.

### Copy-Paste Example: Test of Independence

```python
import numpy as np
from scipy import stats

# Your data: Age group vs Product preference (contingency table)
# Rows: 18-29, 30-44, 45-60  |  Columns: Product A, B, C
observed = np.array([
    [45, 30, 25],  # 18-29
    [35, 50, 30],  # 30-44
    [20, 40, 55]   # 45-60
])

# Run Chi-Square test
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

# Calculate Cramér's V (effect size)
n = np.sum(observed)
min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim))

# Results
print("Observed frequencies:")
print(observed)
print(f"\nChi-square: {chi2:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Cramér's V: {cramers_v:.3f}")
print(f"Significant? {'YES ✓' if p_value < 0.05 else 'No ✗'}")

# Interpretation
if cramers_v < 0.1:
    strength = "weak"
elif cramers_v < 0.3:
    strength = "moderate"
else:
    strength = "strong"
print(f"Association strength: {strength}")
```

**How to interpret:**
- **P-value < 0.05?** The two variables are related
- **Cramér's V:** 0=no relationship, 1=perfect relationship
- **Look at table:** Which cells are way higher/lower than expected?

**Watch out for:**
- All expected cell counts should be ≥5 (check the `expected` array)
- Can't determine cause and effect - just association
- Works with categories, not continuous numbers

---

## 📈 Correlation

**What is it?** Measures how two continuous variables move together.

**When to use it:** Checking if ad spend relates to sales, seeing if email length affects open rate.

### Copy-Paste Example: Pearson Correlation

```python
import numpy as np
from scipy import stats

# Your data
ad_spend = np.array([10, 15, 12, 18, 20, 14, 16, 22, 19, 17])  # $1000s
sales = np.array([50, 75, 60, 90, 100, 70, 80, 110, 95, 85])  # $1000s

# Calculate correlation
r, p_value = stats.pearsonr(ad_spend, sales)
r_squared = r ** 2

# Results
print(f"Correlation (r): {r:.3f}")
print(f"R-squared: {r_squared:.3f} ({r_squared*100:.1f}% of variance explained)")
print(f"P-value: {p_value:.4f}")
print(f"Significant? {'YES ✓' if p_value < 0.05 else 'No ✗'}")

# Interpretation
if abs(r) < 0.3:
    strength = "weak"
elif abs(r) < 0.7:
    strength = "moderate"
else:
    strength = "strong"
direction = "positive" if r > 0 else "negative"
print(f"\n{strength.capitalize()} {direction} correlation")
```

**How to interpret:**
- **r = 1:** Perfect positive relationship
- **r = 0:** No relationship
- **r = -1:** Perfect negative relationship
- **P-value < 0.05?** Relationship is statistically significant

**Watch out for:**
- Correlation ≠ causation! Just because they move together doesn't mean one causes the other
- Only measures LINEAR relationships
- Very sensitive to outliers

---

## 📉 Regression

**What is it?** Predicts one variable from another and gives you an equation.

**When to use it:** Forecasting sales from ad spend, predicting clicks from email length.

### Copy-Paste Example: Simple Linear Regression

```python
import numpy as np
from scipy import stats

# Your data
ad_spend = np.array([10, 15, 12, 18, 20, 14, 16, 22, 19, 17])
sales = np.array([50, 75, 60, 90, 100, 70, 80, 110, 95, 85])

# Run regression
slope, intercept, r_value, p_value, std_err = stats.linregress(ad_spend, sales)

# Results
print(f"Equation: Sales = {intercept:.2f} + {slope:.2f} × Ad Spend")
print(f"R-squared: {r_value**2:.3f} ({r_value**2*100:.1f}% variance explained)")
print(f"P-value: {p_value:.4f}")
print(f"Significant? {'YES ✓' if p_value < 0.05 else 'No ✗'}")

# Make predictions
new_ad_spend = 25
predicted_sales = slope * new_ad_spend + intercept
print(f"\nPrediction: ${new_ad_spend}k ad spend → ${predicted_sales:.2f}k sales")
```

**How to interpret:**
- **Slope:** For every $1k more in ad spend, sales increase by $[slope]k
- **Intercept:** Predicted sales when ad spend is zero
- **R-squared:** How well the line fits the data

**Watch out for:**
- Don't predict way beyond your data range (extrapolation)
- Check for outliers - they can mess up your line
- Just because you can predict doesn't mean causation

---

## 🎯 Mann-Whitney

**What is it?** Non-parametric test for comparing two groups (doesn't need normal distribution).

**When to use it:** Small samples, super skewed data, or data with outliers. Same situations as t-test but when assumptions are violated.

### Copy-Paste Example: Mann-Whitney U Test

```python
import numpy as np
from scipy import stats

# Your data: session duration (minutes) - note the outlier!
mobile = np.array([2.3, 3.1, 2.8, 3.5, 2.5, 3.0, 15.5])  # outlier: 15.5
desktop = np.array([4.5, 5.2, 4.8, 5.0, 4.3, 5.5, 4.7])

# Run Mann-Whitney test
u_stat, p_value = stats.mannwhitneyu(mobile, desktop, alternative='two-sided')

# Calculate medians (better than means for skewed data)
median_mobile = np.median(mobile)
median_desktop = np.median(desktop)

# Results
print(f"Mobile median: {median_mobile:.2f} min")
print(f"Desktop median: {median_desktop:.2f} min")
print(f"U-statistic: {u_stat:.1f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant? {'YES ✓' if p_value < 0.05 else 'No ✗'}")
```

**How to interpret:**
- **P-value < 0.05?** Groups have different distributions
- **Compare medians:** Which group is typically higher?
- Focus on median, not mean (outliers don't affect median)

**Watch out for:**
- Compares entire distributions, not just averages
- Less powerful than t-test if data IS normal
- Good choice when you have outliers or small samples

---

## 🎲 Proportions (Z-Test)

**What is it?** Compares percentages/conversion rates between groups.

**When to use it:** Comparing CTR, conversion rates, success rates with large samples (n>30).

### Copy-Paste Example: Compare Two Proportions

```python
import numpy as np
from scipy import stats

# Your data
n1 = 1000        # Landing page A visitors
conversions1 = 52   # Conversions from A
p1 = conversions1 / n1

n2 = 950         # Landing page B visitors
conversions2 = 68   # Conversions from B
p2 = conversions2 / n2

# Calculate pooled proportion
pooled_p = (conversions1 + conversions2) / (n1 + n2)

# Calculate standard error and z-statistic
se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
z_stat = (p1 - p2) / se

# Calculate p-value (two-tailed)
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Results
print(f"Landing Page A: {p1*100:.2f}% conversion")
print(f"Landing Page B: {p2*100:.2f}% conversion")
print(f"Difference: {(p1-p2)*100:+.2f} percentage points")
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant? {'YES ✓' if p_value < 0.05 else 'No ✗'}")
```

**How to interpret:**
- **P-value < 0.05?** Conversion rates are significantly different
- **Look at difference:** How many percentage points better/worse?
- **Direction matters:** Which page converted better?

**Watch out for:**
- Need decent sample sizes (at least 30 per group, ideally 100+)
- Watch for multiple comparisons (if testing many pairs, use Bonferroni correction)
- Small differences can be "significant" with huge samples

---

## 🎰 Bayesian A/B Testing

**What is it?** Gives you the probability that one variation is better than another.

**When to use it:** A/B testing when you want to know "how likely is B better than A?" instead of just yes/no.

### Copy-Paste Example: Bayesian A/B Test

```python
import numpy as np
from scipy import stats

# Your data
n_a = 1000
conversions_a = 52

n_b = 950
conversions_b = 68

# Bayesian update: Beta(1,1) prior + data → Beta(α,β) posterior
alpha_a = 1 + conversions_a
beta_a = 1 + (n_a - conversions_a)

alpha_b = 1 + conversions_b
beta_b = 1 + (n_b - conversions_b)

# Calculate posterior means
mean_a = alpha_a / (alpha_a + beta_a)
mean_b = alpha_b / (alpha_b + beta_b)

# 95% credible intervals
ci_a = (stats.beta.ppf(0.025, alpha_a, beta_a),
        stats.beta.ppf(0.975, alpha_a, beta_a))
ci_b = (stats.beta.ppf(0.025, alpha_b, beta_b),
        stats.beta.ppf(0.975, alpha_b, beta_b))

# Monte Carlo: P(B > A)
np.random.seed(42)
samples_a = stats.beta.rvs(alpha_a, beta_a, size=100000)
samples_b = stats.beta.rvs(alpha_b, beta_b, size=100000)
prob_b_better = np.mean(samples_b > samples_a)

# Results
print(f"Variation A: {mean_a*100:.2f}% (95% CI: {ci_a[0]*100:.2f}%-{ci_a[1]*100:.2f}%)")
print(f"Variation B: {mean_b*100:.2f}% (95% CI: {ci_b[0]*100:.2f}%-{ci_b[1]*100:.2f}%)")
print(f"\nP(B > A) = {prob_b_better*100:.1f}%")
print(f"P(A > B) = {(1-prob_b_better)*100:.1f}%")

# Decision
if prob_b_better > 0.95:
    print(f"\n✓ Strong evidence for B - go with it!")
elif prob_b_better < 0.05:
    print(f"\n✓ Strong evidence for A - go with it!")
else:
    print(f"\n⚠ Inconclusive - need more data")
```

**How to interpret:**
- **P(B > A) > 95%?** Very confident B is better
- **Credible interval:** Range where true conversion rate likely lies
- **Direction is clear:** You get direct probability of which is better

**Watch out for:**
- Credible intervals ≠ confidence intervals (similar but different interpretation)
- Prior matters (we used uniform prior here = no prior knowledge)
- Can keep monitoring as data comes in (unlike frequentist tests)

---

## 🔍 Quick Comparison Table

| Test | Data Type | Use When | Sample Size | Python Function |
|------|-----------|----------|-------------|----------------|
| **T-Test** | Continuous | 2 groups or vs benchmark | Small-Medium | `stats.ttest_ind()` |
| **ANOVA** | Continuous | 3+ groups | Small-Medium | `stats.f_oneway()` |
| **Chi-Square** | Categorical | Categories related? | Any | `stats.chi2_contingency()` |
| **Correlation** | Both continuous | Linear relationship? | Medium-Large | `stats.pearsonr()` |
| **Regression** | Continuous | Predict one from other | Medium-Large | `stats.linregress()` |
| **Mann-Whitney** | Continuous/Ordinal | 2 groups, skewed data | Small-Medium | `stats.mannwhitneyu()` |
| **Z-Test Props** | Proportions | Compare %s, rates | Large (>30) | Manual calc |
| **Bayesian** | Binary outcomes | A/B test probability | Any | `stats.beta` |

---

## 💡 General Tips

### Before You Run Any Test

1. **Plot your data first** - histograms, scatter plots, box plots
2. **Check for outliers** - one weird value can ruin everything
3. **Look at sample sizes** - tiny samples = unreliable results
4. **Consider effect size** - statistical ≠ practical significance

### After You Get Results

1. **Always report effect size** - not just p-values
2. **Include confidence intervals** - show uncertainty
3. **Think about practical meaning** - is the difference meaningful for business?
4. **Be honest about limitations** - sample size, assumptions, etc.

### Common Pitfalls

- ❌ **P-hacking:** Running tests until you find p<0.05
- ❌ **Multiple comparisons:** Testing many things without correction
- ❌ **Assuming causation:** Correlation doesn't mean X causes Y
- ❌ **Ignoring assumptions:** Using wrong test for your data type
- ❌ **Stopping too early:** Ending A/B test as soon as p<0.05

---

## 📦 One-Time Setup

```bash
# Install all packages you need
pip install numpy scipy pandas statsmodels matplotlib

# Or if using conda
conda install numpy scipy pandas statsmodels matplotlib
```

---

## 🎓 Need More Help?

**For each test, remember:**
1. What's the business question?
2. What type of data do I have?
3. How many groups am I comparing?
4. Is my data normal or skewed?
5. What's my sample size?

**Answer these 5 questions → Pick the right test!**

---

**Last Updated:** January 2025
**License:** MIT - Free to use and modify

Made with ❤️ for marketing students and data analysts who want to get stuff done.
