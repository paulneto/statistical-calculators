# Testing Guide for Statistical Calculators
## Verification of Mathematical Fixes

This guide provides specific test cases with known correct answers to verify all fixes are working properly.

---

## 1. ANOVA Calculator - F-Critical Values

### Test Case 1.1: Basic One-Way ANOVA (3 groups)
**Data:**
- Group 1: `5, 7, 8, 6, 7`
- Group 2: `9, 11, 10, 12, 11`
- Group 3: `13, 15, 14, 16, 15`

**Expected Results:**
- df_between = 2
- df_within = 12
- F-critical (α=0.05) ≈ **3.885** (NOT 3.89 hardcoded!)
- F-statistic should be > 50 (very significant)
- Result: **Significant**

**How to Test:**
1. Open `anova/index.html`
2. Enter the three groups of data
3. Set significance level to 0.05
4. Check that F-critical shows approximately 3.885
5. Verify F-statistic is very high and marked significant

---

### Test Case 1.2: Five Groups (Tests Different df)
**Data:**
- Group 1: `20, 22, 21`
- Group 2: `25, 27, 26`
- Group 3: `30, 32, 31`
- Group 4: `35, 37, 36`
- Group 5: `40, 42, 41`

**Expected Results:**
- df_between = 4
- df_within = 10
- F-critical (α=0.05) ≈ **3.478** (Different from 3.89!)
- This tests that F-critical is correctly calculated based on df

**Verification:** The F-critical should be ~3.478, NOT 3.89

---

## 2. Two-Way ANOVA - Real Calculations

### Test Case 2.1: 2×2 Design (Simplest Case)

**Setup:**
- Factor A: Design (2 levels: "Old", "New")
- Factor B: Time (2 levels: "Morning", "Evening")
- n = 3 per cell (balanced design)

**Data:**
```
Old-Morning: 10, 12, 11
Old-Evening: 15, 17, 16
New-Morning: 20, 22, 21
New-Evening: 25, 27, 26
```

**Expected Results:**
- Factor A (Design) should be significant (big difference between Old and New)
- Factor B (Time) should be significant (big difference between Morning and Evening)
- Interaction should be very small/non-significant (parallel lines)
- All p-values should be calculated (not hardcoded!)

**Key Test:** Enter the data TWICE and verify results are IDENTICAL. Previously, it would return the same fake values regardless of input.

**Second Test:** Change just one value and verify results CHANGE. This proves it's calculating, not returning hardcoded values.

---

## 3. T-Test Calculator - Proper Critical Values

### Test Case 3.1: Small Sample (df=5)
**Data (One-Sample T-Test):**
- Sample: `10, 12, 14, 16, 18, 20`
- Population mean: 12
- n = 6, df = 5

**Expected Results:**
- Sample mean = 15
- t-statistic ≈ 2.74
- t-critical (df=5, α=0.05, two-tailed) = **2.571** (NOT 2.0!)
- Result: **Significant** (because 2.74 > 2.571)

**Key Test:**
- With old hardcoded t-critical = 2.0, this would be significant (2.74 > 2.0) ✓
- With new correct t-critical = 2.571, this is STILL significant (2.74 > 2.571) ✓
- Verify calculator shows t-critical = 2.571, not 2.0

---

### Test Case 3.2: Large Sample (df=100)
**Data (Independent T-Test):**
- Group 1: 50 values averaging 100 with SD ≈ 10
- Group 2: 52 values averaging 103 with SD ≈ 10
- Use example data or generate random data

**Expected Results:**
- df = 100
- t-critical (df=100, α=0.05) ≈ **1.984** (NOT 2.0!)
- Small difference in means might NOT be significant

**Key Test:**
- Verify t-critical shows ~1.984 instead of 2.0
- This proves the calculator adjusts for large samples

---

### Test Case 3.3: Paired T-Test (Before/After)
**Data:**
- Before: `70, 75, 68, 72, 74`
- After: `78, 82, 75, 80, 81`

**Expected Results:**
- Mean difference ≈ 8
- df = 4
- t-critical (df=4) ≈ **2.776** (NOT 2.0!)
- t-statistic should be around 5-7 (very significant)

---

## 4. Mann-Whitney U Test - Tie Correction

### Test Case 4.1: Data WITHOUT Ties
**Data:**
- Group A: `1, 3, 5, 7, 9`
- Group B: `2, 4, 6, 8, 10`

**Expected Results:**
- No ties, so tie correction should = 0
- U-statistic = 12.5
- Results should be very close to old calculator

---

### Test Case 4.2: Data WITH Many Ties
**Data:**
- Group A: `5, 5, 5, 7, 7, 9`
- Group B: `5, 6, 6, 8, 8, 8`

**Expected Results:**
- Tie correction should be > 0
- Standard error should be SMALLER than without correction
- z-score should be larger (more conservative test)

**How to Verify:**
- Compare results with/without ties
- With many ties, the new calculator should give different (more accurate) results

---

## 5. Regression Calculator - Full Functionality

### Test Case 5.1: Perfect Linear Relationship
**Data:**
- X: `1, 2, 3, 4, 5`
- Y: `2, 4, 6, 8, 10` (exactly Y = 2X)

**Expected Results:**
- Slope (b₁) = **2.0** exactly
- Intercept (b₀) = **0.0** exactly
- r = **1.0** exactly (perfect correlation)
- r² = **1.0** exactly
- Standard error ≈ **0** (perfect fit)

**Key Test:** This proves the calculator calculates, not shows static values

---

### Test Case 5.2: Example Data (Ad Spend vs Sales)
**Data (provided in calculator):**
- X: `10, 15, 20, 25, 30, 35, 40, 45, 50, 55`
- Y: `170, 180, 220, 250, 270, 305, 325, 360, 390, 400`

**Expected Results (approximately):**
- Slope ≈ 4.68
- Intercept ≈ 130.54
- r ≈ 0.98
- r² ≈ 0.96

**Prediction Test:**
- Enter X = 60
- Expected Y ≈ 411 (130.54 + 4.68 × 60)

---

### Test Case 5.3: Negative Relationship
**Data:**
- X: `1, 2, 3, 4, 5`
- Y: `10, 8, 6, 4, 2` (Y = 12 - 2X)

**Expected Results:**
- Slope = **-2.0** (negative!)
- Intercept = **12.0**
- r = **-1.0** (perfect negative correlation)

---

## Quick Verification Checklist

### ✅ ANOVA
- [ ] F-critical changes when df changes (not always 3.89)
- [ ] Three different group sizes give three different F-critical values
- [ ] Two-Way ANOVA gives different results for different data inputs

### ✅ T-Test
- [ ] t-critical shows 2.571 for df=5 (not 2.0)
- [ ] t-critical shows 1.984 for df=100 (not 2.0)
- [ ] Results marked as "t-critical = X.XXX" in output

### ✅ Mann-Whitney
- [ ] Works with no ties (baseline test)
- [ ] Gives reasonable results with many ties
- [ ] Doesn't crash or give errors

### ✅ Regression
- [ ] Perfect line (Y=2X) gives slope=2, intercept=0, r=1
- [ ] Changing data changes all results
- [ ] Prediction works and gives sensible values
- [ ] Scatter plot displays correctly

---

## Testing Against Known Statistical Software

For rigorous verification, you can compare results against:

### Option 1: R (Free Statistical Software)
```R
# ANOVA
group1 <- c(5, 7, 8, 6, 7)
group2 <- c(9, 11, 10, 12, 11)
group3 <- c(13, 15, 14, 16, 15)
data <- data.frame(
  value = c(group1, group2, group3),
  group = factor(rep(1:3, each=5))
)
summary(aov(value ~ group, data=data))

# T-Test
t.test(c(10, 12, 14, 16, 18, 20), mu=12)

# Regression
x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 6, 8, 10)
summary(lm(y ~ x))
```

### Option 2: Online Calculators
- **GraphPad**: https://www.graphpad.com/quickcalcs/
- **Social Science Statistics**: https://www.socscistatistics.com/
- **Stat Trek**: https://stattrek.com/online-calculator/

### Option 3: Excel
- Use Excel's built-in functions:
  - `=T.INV.2T(0.05, df)` for t-critical
  - `=F.INV.RT(0.05, df1, df2)` for F-critical
  - Data Analysis ToolPak for ANOVA and Regression

---

## Browser Testing

Test in multiple browsers to ensure jStat library loads:
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari

**Check browser console for errors:**
1. Open calculator
2. Press F12 (Developer Tools)
3. Go to Console tab
4. Look for any red error messages
5. Particularly check that jStat loads successfully

---

## Edge Cases to Test

### 1. Very Small Samples
- n = 2 or 3 in each group
- Should still calculate correctly
- May show warnings about low power

### 2. Very Large Samples
- n > 100
- Should handle without crashing
- Critical values should be close to z-values

### 3. Equal Groups
- All group means exactly the same
- F or t-statistic should be ≈ 0
- Should NOT be significant

### 4. Missing/Invalid Data
- Empty cells
- Non-numeric values
- Should show clear error messages

---

## Recommended Testing Sequence

### Day 1: Quick Smoke Tests
1. Open each calculator
2. Use the pre-loaded example data
3. Click "Calculate"
4. Verify it works and shows results

### Day 2: Known Value Tests
1. ANOVA Test Case 1.1 and 1.2
2. T-Test Case 3.1
3. Regression Test Case 5.1 (perfect line)

### Day 3: Cross-Verification
1. Pick 2-3 test cases
2. Run in R or Excel
3. Compare results (should match to 2-3 decimal places)

### Day 4: Student Testing (If Possible)
1. Give to a student or colleague
2. Ask them to use it for real homework
3. Check if they find any issues

---

## Success Criteria

All calculators pass testing if:
- ✅ No JavaScript errors in browser console
- ✅ Results match known values within ±0.01
- ✅ Critical values are NOT hardcoded (change with df)
- ✅ Results change when input data changes
- ✅ Clear error messages for invalid input
- ✅ Scatter plots and visualizations render correctly

---

## If You Find Issues

Document:
1. Which calculator
2. Which test case
3. Expected result
4. Actual result
5. Screenshot if helpful

Then we can debug and fix!

---

**Ready to Test?** Start with the Quick Verification Checklist, then work through the specific test cases for the calculators you'll use most in your class.
