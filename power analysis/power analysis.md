# Power Analysis Guide for Marketing Research

## Introduction

This guide accompanies the Power Analysis Calculator designed specifically for marketing research applications, with a focus on TV campaign effectiveness studies. Power analysis is a crucial step in research design that helps determine the appropriate sample size needed to detect meaningful effects.

## Table of Contents

1. [Key Concepts in Power Analysis](#key-concepts-in-power-analysis)
2. [Using the Power Analysis Calculator](#using-the-power-analysis-calculator)
3. [Statistical Tests and When to Use Them](#statistical-tests-and-when-to-use-them)
4. [Practical Examples in TV Campaign Research](#practical-examples-in-tv-campaign-research)
5. [Interpreting Results](#interpreting-results)
6. [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)
7. [Technical Implementation](#technical-implementation)

## Key Concepts in Power Analysis

### What is Statistical Power?

Statistical power is the probability that a study will detect an effect when there is an effect to be detected. In other words, it's the likelihood of avoiding a Type II error (false negative).

### The Four Interconnected Components

Power analysis involves four key components, where setting any three allows you to calculate the fourth:

1. **Sample Size (n)**: The number of participants or observations in your study
2. **Effect Size**: The magnitude of the difference or relationship you're trying to detect
3. **Significance Level (α)**: The probability of making a Type I error (finding an effect when none exists)
4. **Statistical Power (1-β)**: The probability of detecting an effect when it truly exists

### Types of Effect Sizes

Different statistical tests use different measures of effect size:

| Test Type | Effect Size Measure | Small | Medium | Large |
|-----------|---------------------|-------|--------|-------|
| t-test | Cohen's d | 0.2 | 0.5 | 0.8 |
| Proportion test | Proportion difference | 0.1 (10%) | 0.3 (30%) | 0.5 (50%) |
| Correlation | Pearson's r | 0.1 | 0.3 | 0.5 |

### The Relationship Between Components

- Increasing sample size → Increases power
- Increasing effect size → Increases power
- Increasing significance level (e.g., from 0.01 to 0.05) → Increases power
- Increasing desired power → Requires larger sample size

## Using the Power Analysis Calculator

### Step 1: Select Analysis Type

Choose the appropriate statistical test for your research question:

- **Two-Sample t-test**: For comparing means between exposed and control groups
  - Example: Comparing brand awareness scores between viewers and non-viewers
  - Input required: Cohen's d effect size

- **Proportion Test**: For comparing conversion rates or percentages
  - Example: Comparing purchase rates between exposed and control groups
  - Input required: Expected proportions for both groups

- **Correlation Analysis**: For examining relationships between variables
  - Example: Relationship between ad exposure frequency and brand recall
  - Input required: Expected correlation coefficient (r)

### Step 2: Enter Parameters

1. **Effect Size**: Enter the expected magnitude of effect based on previous research or pilot studies
2. **Significance Level (α)**: Typically 0.05 (5%)
3. **Desired Power**: Typically 0.80 (80%) or higher
4. **Test Direction**: Two-tailed (checking for effects in both directions) or one-tailed (only interested in effects in one direction)
5. **Allocation Ratio**: Ratio of participants in exposed vs. control groups

### Step 3: Advanced Settings (Optional)

1. **Dropout Rate**: Adjust for expected attrition in longitudinal studies
2. **Multiple Comparison Adjustments**: Apply corrections when conducting multiple tests

### Step 4: Interpret Results

The calculator will display:
- Required total sample size
- Group sizes (if applicable)
- Practical interpretation of results
- Power vs. sample size chart

## Statistical Tests and When to Use Them

### Two-Sample t-test

**When to use**: Compare means between two groups (exposed vs. non-exposed to TV campaign)

**Examples**:
- Comparing brand awareness scores between viewers and non-viewers
- Measuring differences in website visit duration between exposed and control groups
- Evaluating changes in purchase intent before and after campaign exposure

**Sample size formula**:
```
n = (zα + zβ)² × (1 + 1/r) / d²
```
Where:
- n = sample size per group
- zα = z-score for significance level
- zβ = z-score for power
- r = allocation ratio
- d = Cohen's d effect size

### Proportion Test

**When to use**: Compare conversion rates, click-through rates, or other percentage-based metrics

**Examples**:
- Comparing conversion rates between exposed and control groups
- Measuring difference in coupon redemption rates
- Analyzing click-through rates between campaign variants

**Sample size formula**:
```
n = (zα + zβ)² × p(1-p) × (1 + 1/r) / Δp²
```
Where:
- p = pooled proportion
- Δp = absolute difference between proportions

### Correlation Analysis

**When to use**: Examine relationships between continuous variables

**Examples**:
- Relationship between ad exposure frequency and brand recall
- Correlation between ad viewing duration and purchase amount
- Association between sentiment ratings and social sharing behavior

**Sample size formula**:
```
n = ((zα + zβ)/FisherZ)² + 3
```
Where:
- FisherZ = 0.5 × log((1+r)/(1-r)) (Fisher's transformation)
- r = expected correlation coefficient

## Practical Examples in TV Campaign Research

### Example 1: Website Traffic Impact Study

**Research Question**: Does our national TV campaign increase website traffic?

**Study Design**:
- Two-sample comparison (exposed vs. non-exposed)
- Outcome: Daily website visits
- Previous data suggests exposed users generate 25% more visits
- Standard deviation estimated at 85% of the mean

**Power Analysis Parameters**:
- Effect size (Cohen's d): 0.30
- Significance level (α): 0.05
- Desired power: 0.80 (80%)
- Two-tailed test

**Result**: Required sample size of approximately 175 participants per group (350 total)

**Application**: With 350 total participants (175 exposed to the TV campaign and 175 not exposed), researchers would have an 80% chance of detecting a 25% difference in website traffic if such a difference exists.

### Example 2: Conversion Rate Study

**Research Question**: Does TV ad exposure increase online purchase conversion rates?

**Study Design**:
- Proportion test (comparing conversion rates)
- Baseline conversion rate: 5% (control group)
- Expected conversion with TV exposure: 8%
- 3 percentage point absolute increase (60% relative increase)

**Power Analysis Parameters**:
- Control proportion: 0.05 (5%)
- Exposed proportion: 0.08 (8%)
- Significance level (α): 0.05
- Desired power: 0.90 (90%)

**Result**: Required sample size of approximately 966 participants per group (1,932 total)

**Application**: Testing for a 3 percentage point increase in conversion rate requires a much larger sample than detecting differences in continuous variables. This illustrates why conversion studies often need larger panels.

### Example 3: Brand Recall Correlation

**Research Question**: Is there a correlation between TV ad exposure frequency and brand recall?

**Study Design**:
- Correlation analysis
- Variables: Number of ad exposures and brand recall score
- Expected moderate correlation

**Power Analysis Parameters**:
- Expected correlation (r): 0.25
- Significance level (α): 0.05
- Desired power: 0.80 (80%)
- Two-tailed test

**Result**: Required sample size of approximately 123 participants

**Application**: To detect a correlation of 0.25 between ad exposure and brand recall, researchers would need data from at least 123 participants with varying levels of exposure.

## Interpreting Results

### Understanding Sample Size Requirements

The calculated sample size represents the minimum number of participants needed to have the specified probability (power) of detecting the effect size you've entered, if such an effect exists.

For TV campaign studies, consider:

1. **Practical significance vs. statistical significance**: Just because you can detect a tiny effect with a large enough sample doesn't mean the effect is meaningful for marketing decisions.

2. **Cost-benefit tradeoff**: Larger samples provide more power but increase research costs. The calculator helps find the optimal balance.

3. **Group allocation**: Sometimes unequal group sizes are necessary (e.g., when TV exposure can't be perfectly controlled). The calculator accounts for this through the allocation ratio.

### Power vs. Sample Size Curves

The power curve shows how power increases with sample size. Key insights:

1. **Diminishing returns**: Power increases rapidly at first, then levels off. Going from 80% to 90% power requires a larger sample increase than going from 70% to 80%.

2. **Effect size impact**: Smaller effects require much larger samples to detect with the same power.

3. **Practical constraints**: Use the slider to see how adjusting power affects required sample size, helping to make informed tradeoffs.

## Common Pitfalls and Best Practices

### Pitfalls to Avoid

1. **Overestimating effect sizes**: Marketing effects, especially from TV campaigns, are often smaller than anticipated. Using excessively optimistic effect sizes leads to underpowered studies.

2. **Ignoring dropout**: Longitudinal studies tracking TV campaign effects over time need to account for participant attrition.

3. **Overlooking multiple comparisons**: When testing multiple outcomes, adjust for increased Type I error risk.

4. **Neglecting practical significance**: A statistically significant tiny effect may not justify campaign costs.

5. **Using inappropriate tests**: Match the test to your research question and data type.

### Best Practices

1. **Base effect sizes on prior research**: Use pilot studies, previous campaigns, or industry benchmarks to inform effect size estimates.

2. **Consider one-tailed tests when appropriate**: If you're only interested in positive effects (e.g., increased sales), a one-tailed test provides more power.

3. **Adjust for real-world constraints**: Use the dropout rate adjustment for longitudinal studies.

4. **Report power analysis details**: Include sample size calculations and assumptions in research reports for transparency.

5. **Conduct sensitivity analyses**: Try different effect sizes to understand how robust your study design is.

## Technical Implementation

The Power Analysis Calculator uses standard statistical formulas to calculate sample sizes for different tests. Key components include:

1. **Z-value approximation**: The calculator uses an algorithm to approximate z-values for different probabilities (for significance level and power).

2. **Interactive visualization**: Chart.js is used to create dynamic power curves showing the relationship between sample size and power.

3. **Real-time calculations**: Sample size updates instantly when parameters change, including when using the power slider.

4. **Marketing-specific interpretations**: Results are presented in terms relevant to TV campaign research, making statistical concepts accessible to marketing researchers.

---

By understanding these concepts and using the Power Analysis Calculator, marketing researchers can design more effective studies to measure TV campaign effectiveness, ensuring they have sufficient statistical power to detect meaningful effects while optimizing research resources.