# Mathematical Review of Statistical Calculators
## Critical Analysis for 3rd Year University Marketing Research & Statistics Class

**Review Date:** November 2025
**Reviewer:** Claude (Anthropic)
**Context:** These calculators are intended for educational use in a 3rd-year university marketing research and statistics course.

---

## Executive Summary

After a thorough mathematical review of all 10 calculators, I identified **3 CRITICAL ERRORS** and **4 MODERATE ISSUES** that must be addressed before use in an academic setting. Additionally, 1 calculator is non-functional (educational visualization only).

### Critical Issues (Must Fix):
1. **ANOVA Calculator** - Hardcoded F-critical values (incorrect for varying df)
2. **T-Test Calculator** - Hardcoded t-critical value of 2.0 (incorrect for varying df)
3. **ANOVA Two-Way** - Completely non-functional (returns fake hardcoded values)

### Moderate Issues (Should Fix):
4. **Mann-Whitney** - Missing tie correction in variance calculation
5. **Mann-Whitney** - No continuity correction for z-approximation
6. **Regression** - Not actually a calculator, just a static example
7. **Chi-Square** - Limited critical value table (only goes to df=20)

---

## Detailed Calculator Reviews

### 1. ANOVA Calculator ⚠️ **CRITICAL ISSUES**

**File:** `anova/index.html`

#### One-Way ANOVA - Mathematical Analysis

**✅ CORRECT Calculations:**
- **Sum of Squares Between (SSB):** `SSB = Σ[n_i(x̄_i - x̄)²]` - Lines 76-77 ✓
- **Sum of Squares Within (SSW):** `SSW = ΣΣ(x_ij - x̄_i)²` - Lines 79-81 ✓
- **Degrees of Freedom Between:** `df_b = k - 1` - Line 83 ✓
- **Degrees of Freedom Within:** `df_w = N - k` - Line 84 ✓
- **Mean Squares:** `MSB = SSB/df_b, MSW = SSW/df_w` - Lines 85-86 ✓
- **F-Statistic:** `F = MSB/MSW` - Line 87 ✓

**❌ CRITICAL ERROR - Lines 53-55:**
```javascript
const getFCriticalValue = (alpha) => {
    return alpha === 0.05 ? 3.89 : 6.93;
};
```

**Problem:** F-critical values depend on BOTH df_between AND df_within, not just α!

**Examples showing why this is wrong:**
- F(2, 12, 0.05) = 3.89 ✓ (correct by coincidence)
- F(5, 30, 0.05) = 2.53 (NOT 3.89!) ❌
- F(3, 20, 0.05) = 3.10 (NOT 3.89!) ❌
- F(4, 100, 0.05) = 2.46 (NOT 3.89!) ❌

**Impact:** This calculator will give **incorrect significance determinations** for most datasets. Students will learn incorrect statistical inference.

**Recommended Fix:**
1. Implement F-distribution CDF or use a comprehensive lookup table
2. Use an F-distribution library (e.g., jStat)
3. At minimum, create a 2D lookup table for common df combinations

---

#### Two-Way ANOVA - **COMPLETELY NON-FUNCTIONAL**

**❌ CRITICAL ERROR - Lines 143-175:**

The Two-Way ANOVA **does not calculate anything**. It returns hardcoded dummy values:

```javascript
const calculateTwoWayAnova = () => {
    try {
        // For demonstration, we'll create sample results
        setResultsTwoWay({
            factorA: { df: 2, ss: 245.67, ms: 122.84, f: 15.23, p: 0.001 },
            factorB: { df: 1, ss: 156.78, ms: 156.78, f: 8.45, p: 0.015 },
            interaction: { df: 2, ss: 89.34, ms: 44.67, f: 3.67, p: 0.048 }
        });
```

**Problem:** No matter what data you enter, it returns the same results!

**Impact:** Students cannot learn Two-Way ANOVA from this calculator. It's completely fake.

**Recommended Fix:**
1. Implement actual Two-Way ANOVA calculations:
   - `SS_A = bn Σ(ȳ_i.. - ȳ...)²`
   - `SS_B = an Σ(ȳ_.j. - ȳ...)²`
   - `SS_AB = n ΣΣ(ȳ_ij. - ȳ_i.. - ȳ_.j. + ȳ...)²`
   - `SS_E = ΣΣΣ(y_ijk - ȳ_ij.)²`
2. Calculate proper F-statistics for each effect
3. Or remove Two-Way ANOVA entirely until it can be properly implemented

---

### 2. Chi-Square Calculator ✅ **MATHEMATICALLY CORRECT** (with minor limitation)

**File:** `chi/index.html`

#### Goodness of Fit Test

**✅ All calculations correct:**
- **Chi-Square Statistic:** `χ² = Σ[(O_i - E_i)²/E_i]` - Line 163 ✓
- **Degrees of Freedom:** `df = k - 1` - Line 173 ✓
- **Critical Values:** Lookup table - Lines 117-131 ✓

#### Test of Independence

**✅ All calculations correct:**
- **Expected Frequencies:** `E_ij = (row_i × col_j)/N` - Line 203 ✓
- **Chi-Square Statistic:** Same formula - Line 204 ✓
- **Degrees of Freedom:** `df = (r-1)(c-1)` - Line 215 ✓

**⚠️ MINOR LIMITATION:**
Critical value table only includes df = {1,2,3,4,5,6,7,8,9,10,12,15,20}. For df > 20, the calculator returns `undefined` and fails.

**Recommendation:** Extend table to df=30 or add a note about limitations.

---

### 3. T-Test Calculator ⚠️ **CRITICAL ERROR**

**File:** `t-test/index.html`

#### One-Sample T-Test

**✅ CORRECT Calculations:**
- **Sample Mean:** `x̄ = Σx/n` - Line 34 ✓
- **Sample Variance:** `s² = Σ(x - x̄)²/(n-1)` - Line 36 ✓
- **Standard Error:** `SE = s/√n` - Line 37 ✓
- **T-Statistic:** `t = (x̄ - μ₀)/SE` - Line 38 ✓
- **Degrees of Freedom:** `df = n - 1` - Line 39 ✓

**❌ CRITICAL ERROR - Line 44:**
```javascript
significant: Math.abs(tStat) > 2.0
```

**Problem:** t-critical values depend on degrees of freedom, not a universal 2.0!

**Examples showing why this is wrong:**
- t(5, 0.05, two-tailed) = 2.571 (NOT 2.0) ❌
- t(10, 0.05, two-tailed) = 2.228 (NOT 2.0) ❌
- t(30, 0.05, two-tailed) = 2.042 (close to 2.0) ≈
- t(100, 0.05, two-tailed) = 1.984 (NOT 2.0) ❌

**Impact:**
- **Type I errors** (false positives) for small samples (t > 2.0 but < t_critical)
- **Type II errors** (false negatives) for large samples (t < 2.0 but > t_critical)

#### Paired T-Test

**✅ Calculations correct** (Lines 48-56) but **❌ Same critical value error** (Line 60)

#### Independent T-Test

**✅ CORRECT Calculations:**
- **Pooled Variance:** `s²_p = [(n₁-1)s₁² + (n₂-1)s₂²]/(n₁+n₂-2)` - Lines 75-76 ✓
- **T-Statistic:** `t = (x̄₁ - x̄₂)/√[s²_p(1/n₁ + 1/n₂)]` - Lines 78-79 ✓
- **Degrees of Freedom:** `df = n₁ + n₂ - 2` - Line 81 ✓

**❌ Same critical value error** (Line 89)

**Recommended Fix:**
1. Implement t-distribution CDF
2. Use jStat library: `jStat.studentt.inv(1 - alpha/2, df)`
3. Or create lookup table for common df values

---

### 4. Z-Test Significance Calculator ✅ **MATHEMATICALLY CORRECT**

**File:** `z test sig/index.html`

**✅ All calculations correct:**
- **Standard Error for proportion:** `SE_p = √[p(1-p)/n]` - Lines 73-74 ✓
- **SE for difference:** `SE_diff = √(SE₁² + SE₂²)` - Line 75 ✓
- **Z-Statistic:** `z = (p₁ - p₂)/SE_diff` - Line 77 ✓
- **Critical Z-values:** Correct for two-tailed tests - Lines 58-64 ✓
  - 99%: 2.576 ✓
  - 95%: 1.96 ✓
  - 90%: 1.645 ✓
  - 80%: 1.28 ✓

**Assessment:** No mathematical errors found. Suitable for educational use.

---

### 5. Correlation Calculator ✅ **MATHEMATICALLY CORRECT**

**File:** `correlation/correlation-calculator.html`

**✅ All calculations correct:**
- **Pearson's r (computational formula):**
  ```
  r = [nΣXY - (ΣX)(ΣY)] / √[(nΣX² - (ΣX)²)(nΣY² - (ΣY)²)]
  ```
  - Lines 119-120 ✓
- **Coefficient of determination:** `r² = r²` - Line 128 ✓
- **Division by zero check** - Line 122 ✓

**Assessment:** Mathematically sound. Good educational tool.

**Minor note:** Correlation strength interpretations (lines 138-145) are somewhat arbitrary but within acceptable ranges for introductory statistics.

---

### 6. Mann-Whitney U Test Calculator ✅ **MOSTLY CORRECT** (moderate issues)

**File:** `mann-whitney/mann-whitney-html.html`

**✅ CORRECT Calculations:**
- **Ranking procedure** - Lines 73-94 ✓
- **Tie handling (midrank method)** - Lines 87-93 ✓
- **U-statistics:**
  - `U₁ = R₁ - n₁(n₁+1)/2` - Line 104 ✓
  - `U₂ = n₁n₂ - U₁` - Line 105 ✓
- **Using min(U₁, U₂)** - Line 108 ✓ (correct for two-tailed)
- **Mean of U:** `μᵤ = n₁n₂/2` - Line 111 ✓

**⚠️ MODERATE ISSUE #1 - Line 112:**
```javascript
const stdDev = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);
```

**Problem:** This is the formula **without tie correction**.

**Correct formula with ties:**
```
σᵤ = √[(n₁n₂/12)((N+1) - ΣT/(N(N-1)))]
```
where `T = t³ - t` for each group of t tied values.

**Impact:** When ties exist, the test is slightly **conservative** (less likely to detect true differences). For moderate ties, this is acceptable. For extensive ties, it reduces statistical power.

**⚠️ MODERATE ISSUE #2 - Line 113:**
```javascript
const zScore = (U - mean) / stdDev;
```

**Problem:** Should use **continuity correction** for better normal approximation:
```javascript
const zScore = (U - mean + 0.5) / stdDev;  // +0.5 or -0.5 depending on tail
```

**Impact:** Minor - makes p-values slightly more accurate for small to moderate sample sizes.

**Recommendation:**
1. Add tie correction for σᵤ calculation
2. Add continuity correction (+0.5) to z-score
3. Add warning when ties constitute >10% of observations

---

### 7. Normal Distribution Dashboard ✅ **MATHEMATICALLY CORRECT**

**File:** `normal distribution/dashboard-simulator v2.1.html`

**✅ CORRECT Implementation:**
- **Box-Muller transform** for generating normal random variables - Lines 1072-1080 ✓
- **Mean calculation:** `μ = Σx/n` - Line 1134 ✓
- **Variance calculation:** `σ² = Σ(x-μ)²/n` - Lines 1138-1139 ✓

**Assessment:** This is primarily a visualization tool. Mathematical implementations are correct.

---

### 8. Power Analysis Calculator ✅ **MOSTLY CORRECT**

**File:** `power analysis/power-analysis-calculator.html`

**✅ CORRECT Calculations:**

#### For T-Tests (Lines 937-940):
```javascript
const numerator = Math.pow(zAlpha + zBeta, 2) * (1 + 1/ratio);
const denominator = Math.pow(effectSize, 2);
sampleSize = Math.ceil(numerator / denominator);
```
Formula: `n = (z_α + z_β)²(1 + 1/r)/d²` ✓

This is the **approximate formula** using z-distribution. More accurate would use non-central t-distribution, but for planning purposes this is acceptable.

#### For Proportions (Lines 951-954):
```javascript
const p = (p1 + p2) / 2; // Pooled proportion
const numerator = Math.pow(zAlpha + zBeta, 2) * p * (1 - p) * (1 + 1/ratio);
const denominator = Math.pow(proportionDiff, 2);
```
Formula: `n = (z_α + z_β)² × p̄(1-p̄)(1 + 1/r)/(p₁-p₂)²` ✓

#### For Correlations (Lines 963-967):
```javascript
const fisherZ = 0.5 * Math.log((1 + r) / (1 - r));
const numerator = Math.pow(zAlpha + zBeta, 2);
const denominator = Math.pow(fisherZ, 2);
sampleSize = Math.ceil(numerator / denominator + 3);
```
Uses **Fisher's z-transformation** ✓ - appropriate for correlation testing.

**Assessment:** Mathematically sound for power analysis. The z-approximation for t-tests is standard practice in sample size calculations.

---

### 9. Regression Calculator ❌ **NOT A CALCULATOR**

**File:** `regression/correlation-regression-html2.html`

**Problem:** This is a **static educational example**, not an interactive calculator.

Lines 290-298 show hardcoded values:
```javascript
<p><strong>Correlation Coefficient (r):</strong> 0.98</p>
<p><strong>Regression Equation:</strong> Sales = 4.68 × Ad Spend + 130.54</p>
```

**Impact:** This cannot be used to calculate regression for student data. It's purely illustrative.

**Recommendation:**
1. Either label this clearly as "Educational Example Only" in the README
2. Or implement actual simple linear regression:
   - `b₁ = Σ[(x-x̄)(y-ȳ)]/Σ(x-x̄)²`
   - `b₀ = ȳ - b₁x̄`

---

### 10. Proportion Significance Calculator ✅ **MATHEMATICALLY CORRECT**

**File:** `proportion sig/z proportion-calculator %.html`

**✅ CORRECT Calculations (Lines 299-302):**
- **Pooled proportion:** `p̄ = (p₁ + p₂)/2` - Line 300 ✓
- **Standard error:** `SE = √[p̄(1-p̄)(2/n)]` - Line 301 ✓
  - Correctly simplifies to `√[2p̄(1-p̄)/n]` when n₁ = n₂
- **Z-statistic:** `z = |p₁ - p₂|/SE` - Line 302 ✓

**Note:** This assumes **same sample size** for both proportions. This is appropriate for the pairwise comparisons shown.

**Assessment:** Mathematically correct for the intended use case.

---

## Summary Table

| Calculator | Status | Critical Issues | Moderate Issues | Notes |
|------------|--------|----------------|-----------------|-------|
| **ANOVA** | ❌ | Hardcoded F-critical; Two-Way is fake | None | **Must fix before use** |
| **Chi-Square** | ✅ | None | Limited df table | Acceptable with caveat |
| **T-Test** | ❌ | Hardcoded t-critical (2.0) | None | **Must fix before use** |
| **Z-Test** | ✅ | None | None | Ready for use |
| **Correlation** | ✅ | None | None | Ready for use |
| **Mann-Whitney** | ⚠️ | None | No tie correction, no continuity correction | Acceptable but should improve |
| **Normal Dist** | ✅ | None | None | Ready for use |
| **Power Analysis** | ✅ | None | None | Ready for use |
| **Regression** | ⚠️ | Not functional (static example) | None | Label as example only |
| **Proportions** | ✅ | None | None | Ready for use |

---

## Recommendations for Educational Use

### **URGENT (Must Fix Before Using):**

1. **Fix ANOVA F-critical values** - Either implement F-distribution or comprehensive lookup table
2. **Fix T-Test critical values** - Either implement t-distribution or comprehensive lookup table
3. **Remove or Fix Two-Way ANOVA** - Current version is completely non-functional

### **Important (Should Fix):**

4. **Add tie correction to Mann-Whitney** - Improves accuracy when ties exist
5. **Label Regression as "Example Only"** - Or implement actual regression calculator
6. **Extend Chi-Square table** - Add df values up to 30

### **For Academic Rigor:**

7. **Add assumption checking** - Normality tests, homogeneity of variance
8. **Include confidence intervals** - Not just hypothesis testing
9. **Add p-value calculations** - Students should see exact p-values, not just significant/not significant
10. **Document limitations** - Each calculator should state its assumptions clearly

---

## Pedagogical Considerations

### **Strengths:**
- Calculators use correct formulas (where functional)
- Good visual design makes them approachable
- Example data helps students learn
- Interpretations are generally accurate

### **Weaknesses for Education:**
- Critical errors will teach students incorrect statistical inference
- Binary significant/not significant encourages dichotomous thinking
- No discussion of effect sizes
- No confidence intervals
- Limited error checking for assumption violations

### **Recommendation:**
Fix the 3 critical issues before deploying in classroom. The other calculators are suitable for educational use with appropriate instructor guidance on their limitations.

---

**End of Mathematical Review**
