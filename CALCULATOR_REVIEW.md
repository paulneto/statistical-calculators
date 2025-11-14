# Mathematical Review of 5 New Calculators
**Date:** January 14, 2025
**Reviewer:** Claude Code
**Status:** REVIEW COMPLETE - ISSUES FOUND

## 1. Multiple Regression Calculator ✅ APPROVED

### Mathematical Components Reviewed:
- **OLS Matrix Algebra**: β = (X'X)⁻¹X'y ✅ CORRECT
- **R-squared**: SSR/SST ✅ CORRECT
- **Adjusted R-squared**: 1 - ((1-R²)(n-1)/(n-k-1)) ✅ CORRECT
- **Standard Errors**: SE(β) = sqrt(MSE * (X'X)⁻¹[i,i]) ✅ CORRECT
- **T-statistics**: t = β/SE(β) with df = n-k-1 ✅ CORRECT
- **F-statistic**: F = (SSR/k) / (SSE/(n-k-1)) ✅ CORRECT
- **VIF Calculation**: VIF_j = 1/(1-R²_j) ✅ CORRECT

### Verdict: **MATHEMATICALLY SOUND - NO ISSUES**

---

## 2. Post-Hoc Tests Calculator ⚠️ ISSUE FOUND

### Mathematical Components Reviewed:
- **MSE Calculation**: Pooled variance from ANOVA ✅ CORRECT
- **Bonferroni Correction**: α_adjusted = α/m, using t-distribution ✅ CORRECT
- **Tukey HSD SE**: sqrt(MSE * (1/n1 + 1/n2) / 2) ✅ CORRECT (Tukey-Kramer)
- **Tukey HSD q-statistic**: q = |M1-M2|/SE ✅ CORRECT
- **Tukey HSD critical value**: jStat.tukey.inv() ✅ CORRECT

### ISSUE IDENTIFIED:
**Location:** Lines 87-88
```javascript
const tStat = q / Math.sqrt(2);
const pValue = 2 * (1 - jStat.studentt.cdf(Math.abs(tStat), dfWithin));
```

**Problem:**
- Tukey HSD p-values should use the **studentized range distribution**, not the t-distribution
- This code approximates by converting q to t, which is **mathematically incorrect**
- P-values will be inaccurate

**Impact:**
- P-values for Tukey HSD comparisons may be incorrect
- Significance decisions may be wrong
- Bonferroni is unaffected (that implementation is correct)

**Established Practice:**
- Tukey HSD uses the studentized range distribution (Q-distribution)
- Most software (R, SPSS, SAS) uses exact Q-distribution p-values or accurate approximations
- The t-distribution approximation used here is not standard practice

### Verdict: **NEEDS CORRECTION** - Tukey HSD p-values are incorrect

---

## 3. Logistic Regression Calculator ⚠️ POTENTIAL ISSUE

### Mathematical Components Reviewed:
- **Logistic Function**: p = 1/(1+e^(-Xβ)) ✅ CORRECT
- **Newton-Raphson**: β_new = β_old - H⁻¹∇L ✅ CORRECT
- **Gradient**: ∇L = Σ X_i(y_i - p_i) ✅ CORRECT
- **Hessian**: H = -Σ X_i X_i' p_i(1-p_i) ✅ CORRECT (negative)
- **Odds Ratios**: OR = e^β ✅ CORRECT
- **Confusion Matrix**: TP, TN, FP, FN ✅ CORRECT
- **Metrics**: Accuracy, Precision, Recall, F1 ✅ CORRECT
- **McFadden's R²**: 1 - LL_full/LL_null ✅ CORRECT

### POTENTIAL ISSUE IDENTIFIED:
**Location:** Line 142
```javascript
const covMatrix = matrixInverse(hessian).map(row => row.map(val => -val));
```

**Analysis:**
- Hessian is already negative: H = -Σ X_i X_i' p_i(1-p_i)
- Covariance matrix should be: Cov = -H⁻¹
- If H is already negative, then: Cov = -H⁻¹ = -(-actual_H)⁻¹ = actual_H⁻¹
- The code computes: -H⁻¹ which would give the correct result IF H is negative
- Then it negates again: `map(val => -val)` which would make it wrong

**However**, checking line 102:
```javascript
hessian[j1][j2] -= X_with_intercept[i][j1] * X_with_intercept[i][j2] * p * (1 - p);
```
The `-=` operator means it's subtracting, so hessian IS negative.

**So the issue is**:
- hessian = negative values
- hessianInv = inverse of negative values
- covMatrix = negating again makes it positive when it should stay negative for Cov = -H⁻¹

**Actually wait**, let me reconsider:
- The observed information matrix is I = -H (where H is the Hessian)
- The covariance matrix is Cov = I⁻¹ = (-H)⁻¹
- In the code, hessian is already -H
- So covMatrix should be inverse(hessian) without additional negation
- But line 142 negates it again, which would be wrong

**Impact:** Standard errors, z-statistics, and p-values may all be incorrect

### Verdict: **NEEDS VERIFICATION** - Check if standard errors are being calculated correctly

---

## 4. Sample Size Calculator ⚠️ MIXED

### Mathematical Components Reviewed:

#### T-Test Sample Size ✅ MOSTLY CORRECT
- **Basic Formula**: n = (z_α + z_β)²(1 + 1/r)/d² ✅ CORRECT
- **Iterative Refinement**: Using non-central t-distribution ✅ EXCELLENT (if jStat supports noncentralt)
- **Non-centrality Parameter**: ncp = d√(n·r/(1+r)) ✅ CORRECT

**Concern:** Code assumes `jStat.noncentralt` exists. Need to verify jStat library has this function.

#### Proportions Sample Size ✅ CORRECT
- **Pooled Proportion**: Used for variance calculation ✅ CORRECT
- **Formula**: Standard two-proportion z-test formula ✅ CORRECT

#### Correlation Sample Size ✅ CORRECT
- **Fisher's z-transformation**: z_r = 0.5·ln((1+r)/(1-r)) ✅ CORRECT
- **Formula**: n = ((z_α + z_β)/z_r)² + 3 ✅ CORRECT
- **Bias Correction**: +3 accounts for Fisher's transformation ✅ CORRECT

#### ANOVA Sample Size ⚠️ APPROXIMATION
**Location:** Lines 122-161

**Issues:**
1. **Non-centrality Parameter** (line 138):
   ```javascript
   const lambda = totalN * (effectSizeF ** 2);
   ```
   - For Cohen's f, should be: λ = n_per_group × k × f²
   - Code uses total N × f² which may be acceptable depending on f definition

2. **Power Calculation** (lines 145-149):
   ```javascript
   const dfAdjusted = dfBetween + lambda;
   actualPower = 1 - jStat.centralF.cdf(fCrit, dfBetween, dfWithin);
   const approximatePower = 1 - jStat.centralF.cdf(fCrit / (1 + lambda / dfWithin), dfBetween, dfWithin);
   ```
   - This is a **rough approximation**, not using proper non-central F distribution
   - **Not standard practice** - should use non-central F or accurate approximations (like Ury & Wiggins)
   - May give inaccurate sample sizes

**Established Practice:**
- Professional software (G*Power, R pwr package) uses non-central F distribution
- Approximations exist but should be well-documented and accurate

### Verdict: **ANOVA PORTION USES ROUGH APPROXIMATION** - May not be accurate

---

## 5. Cluster Analysis Calculator ✅ APPROVED

### Mathematical Components Reviewed:
- **Euclidean Distance**: √(Σ(x_i - y_i)²) ✅ CORRECT
- **K-means (Lloyd's Algorithm)**:
  - Random initialization ✅ CORRECT
  - Assignment step ✅ CORRECT
  - Update step (centroids = mean of clusters) ✅ CORRECT
  - Convergence check ✅ CORRECT
  - Empty cluster handling ✅ REASONABLE
- **WCSS**: Σ distance²(point, centroid) ✅ CORRECT
- **Elbow Method**: Running k-means for multiple k values ✅ CORRECT

### Verdict: **MATHEMATICALLY SOUND - NO ISSUES**

---

## SUMMARY

| Calculator | Status | Issues |
|------------|--------|---------|
| Multiple Regression | ✅ APPROVED | None |
| Post-Hoc Tests | ⚠️ NEEDS FIX | Tukey HSD p-values incorrect |
| Logistic Regression | ⚠️ NEEDS VERIFICATION | Possible covariance matrix error |
| Sample Size | ⚠️ PARTIAL | ANOVA uses rough approximation; jStat dependency |
| Cluster Analysis | ✅ APPROVED | None |

## RECOMMENDATIONS

### Critical (Must Fix):
1. **Post-Hoc Tests**: Either remove p-values for Tukey HSD or note they are approximations, OR find proper studentized range p-value calculation

### High Priority (Should Fix):
2. **Logistic Regression**: Verify covariance matrix calculation at line 142
3. **Sample Size - ANOVA**: Add disclaimer that ANOVA sample sizes are approximations, or improve calculation

### Medium Priority (Should Verify):
4. **Sample Size**: Verify jStat has `noncentralt` functions; add fallback if not

## DECISION

**Can commit as-is?** NO - Critical issues need attention

**Recommended Action:** Fix Post-Hoc Tests Tukey p-values and verify Logistic Regression before committing.
