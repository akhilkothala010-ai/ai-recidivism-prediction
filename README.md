# CSET485: AI and Society — Assignment #1
## Predictive Modeling, Adversarial Stress-Testing, and Fairness Trade-Offs in COMPAS Recidivism Scoring

This repository contains the complete, fully reproducible implementation for **Assignment #1 (CSET485: AI and Society)**. The project analyzes the **COMPAS Recidivism Risk Score Dataset** (assigned to student roll number last digit `0`).

---

### Project Structure

```text
├── CSET485_Assignment1_Report.pdf   # Formal publication-quality submission PDF report
├── compas_recidivism_analysis.ipynb # Fully documented & pre-executed Jupyter Notebook
├── run_all.py                       # One-click master execution script
├── pipeline.py                      # Core modeling, stress-testing, and metrics engine
├── build_full_artifacts.py          # PDF compiler (ReportLab) & Notebook generator (nbformat)
├── figures/                         # High-resolution generated plots (DPI=300)
│   ├── fig1_model_comparison.png
│   ├── fig2_subgroup_heterogeneity.png
│   ├── fig3_omitted_variable_bias.png
│   ├── fig4_adversarial_subset.png
│   └── fig5_fairness_tradeoff_curve.png
├── compas-scores-two-years.csv      # ProPublica benchmark COMPAS dataset
└── requirements.txt                 # Python dependencies
```

---

### Requirements & Installation

The project uses Python 3.10+ and standard data science libraries:
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn reportlab pypdf nbformat nbconvert
```

---

### How to Run the Project

#### Option 1: Master Script (Recommended — One Command)
To run the entire pipeline, regenerate all figures, compile the PDF report, and pre-populate the Jupyter Notebook:
```bash
python run_all.py
```
To run with your specific roll number seed (last 3 digits):
```bash
python run_all.py --seed <YOUR_ROLL_NUMBER_LAST_3_DIGITS>
```
*Example:* `python run_all.py --seed 100`

#### Option 2: Interactive Jupyter Notebook
1. Open `compas_recidivism_analysis.ipynb` in **VS Code** or **JupyterLab**.
2. Select your Python kernel.
3. Click **"Run All"** (or step through cell-by-cell). All computations, tables, and visualization plots will render inline.

---

### How to Push to GitHub (Submission Requirement)

The assignment requires submitting your notebook via a GitHub link:
```bash
# 1. Initialize git (if not already done)
git init

# 2. Add files
git add compas_recidivism_analysis.ipynb CSET485_Assignment1_Report.pdf run_all.py pipeline.py build_full_artifacts.py README.md figures/

# 3. Commit
git commit -m "Complete CSET485 Assignment 1: COMPAS Analysis and Stress-Testing"

# 4. Link your remote repository and push
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

---

### Key Findings Summary

1. **Part 1 (Model A vs Model B Mismatch):** 
   - Linear regression on `decile_score` (median split = 4.4404) vs. Logistic regression on `is_recid` (0.50 threshold) disagreed on **9.77% of test cases (181 individuals)**. 
   - Proves percentile ranking and absolute rearrest hazard calibration are not mathematically interchangeable.
2. **Part 2.1 (Simpson's Paradox / Age Moderation):**
   - The marginal impact of prior arrests is **3.13× stronger** for youth under 25 ($\beta = +0.4577$, 95% CI: $[0.366, 0.549]$) than for defendants over 45 ($\beta = +0.1461$, 95% CI: $[0.120, 0.172]$), with zero overlap in confidence intervals ($p < 10^{-20}$).
3. **Part 2.2 (Omitted-Variable Bias):**
   - Omitting `priors_count` caused the coefficient for `is_African_American` to balloon by **+229.56%** (from $+0.1228$ to $+0.4046$), demonstrating how race indicators absorb omitted systemic policing variance.
4. **Part 2.3 (Adversarial Subsets):**
   - Identified 76 high-confidence failures (4.10% of test set). Confident False Positives averaged **12.71 prior arrests**, revealing rigid linear log-odds extrapolation failure on desisting individuals.
5. **Part 2.4 (Fairness Frontier):**
   - Empirical sweep across $t \in [0.05, 0.95]$ proved the Kleinberg–Chouldechova impossibility theorem. Improving Demographic Parity from $t=0.45$ to $t=0.35$ (+22.14% gain toward parity) degrades Equalized Odds by 17.80% (trade-off cost: 0.80).
