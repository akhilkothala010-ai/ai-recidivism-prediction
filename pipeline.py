"""
COMPAS Recidivism Risk Score Predictive Modeling & Stress-Testing Pipeline
CSET485 - AI and Society (Assignment #1)
Dataset: COMPAS Recidivism Risk Score Dataset
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm

# Set style for academic charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 13

def run_pipeline(seed=100, figures_dir='figures'):
    os.makedirs(figures_dir, exist_ok=True)
    results = {'seed': seed}
    
    # -------------------------------------------------------------
    # Step 1: Data Ingestion and Preprocessing
    # -------------------------------------------------------------
    data_path = 'compas-scores-two-years.csv'
    if not os.path.exists(data_path):
        import urllib.request
        url = 'https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv'
        urllib.request.urlretrieve(url, data_path)
        
    df_raw = pd.read_csv(data_path)
    results['raw_shape'] = df_raw.shape
    
    # Standard ProPublica data cleaning filters:
    # 1. days_b_screening_arrest between -30 and 30 days
    # 2. is_recid != -1 (unobserved recidivism)
    # 3. c_charge_degree != 'O' (traffic infractions / ordinary violations)
    # 4. score_text != 'N/A'
    df_clean = df_raw[
        (df_raw['days_b_screening_arrest'] <= 30) &
        (df_raw['days_b_screening_arrest'] >= -30) &
        (df_raw['is_recid'] != -1) &
        (df_raw['c_charge_degree'] != 'O') &
        (df_raw['score_text'] != 'N/A')
    ].copy()
    
    results['clean_shape'] = df_clean.shape
    results['base_recid_rate'] = float(df_clean['is_recid'].mean())
    results['recid_rate_aa'] = float(df_clean[df_clean['race'] == 'African-American']['is_recid'].mean())
    results['recid_rate_caucasian'] = float(df_clean[df_clean['race'] == 'Caucasian']['is_recid'].mean())
    
    # Features engineered for models:
    df_clean['is_African_American'] = (df_clean['race'] == 'African-American').astype(float)
    df_clean['is_Hispanic'] = (df_clean['race'] == 'Hispanic').astype(float)
    df_clean['is_Other'] = (df_clean['race'].isin(['Other', 'Asian', 'Native American'])).astype(float)
    df_clean['is_Female'] = (df_clean['sex'] == 'Female').astype(float)
    df_clean['is_Felony'] = (df_clean['c_charge_degree'] == 'F').astype(float)
    
    feature_cols = [
        'age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'is_African_American', 'is_Hispanic', 'is_Other', 'is_Female', 'is_Felony'
    ]
    
    X = df_clean[feature_cols].copy()
    y_linear = df_clean['decile_score'].values # Continuous risk score 1-10
    y_logistic = df_clean['is_recid'].values  # Binary outcome 0/1
    
    # Train-test split (70% train, 30% test) with stratified sampling on is_recid
    X_train, X_test, y_lin_train, y_lin_test, y_log_train, y_log_test, df_train, df_test = train_test_split(
        X, y_linear, y_logistic, df_clean, test_size=0.30, random_state=seed, stratify=y_logistic
    )
    
    results['train_size'] = len(X_train)
    results['test_size'] = len(X_test)
    
    # Standardize continuous variables
    scaler = StandardScaler()
    continuous_cols = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    # -------------------------------------------------------------
    # Part 1: Fit Both Models & Threshold Comparison
    # -------------------------------------------------------------
    # Model A: Linear Regression predicting decile_score
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_lin_train)
    pred_lin_test = lin_reg.predict(X_test_scaled)
    median_lin_pred = float(np.median(pred_lin_test))
    cls_lin = (pred_lin_test >= median_lin_pred).astype(int)
    
    # Model B: Logistic Regression predicting is_recid
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_log_train)
    pred_log_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
    cls_log = (pred_log_prob >= 0.5).astype(int)
    
    # Confusion matrix between Model A and Model B classifications:
    # Rows: Model A (Low Risk=0, High Risk=1), Cols: Model B (Low Risk=0, High Risk=1)
    cm = confusion_matrix(cls_lin, cls_log)
    both_low = int(cm[0, 0])
    lin_low_log_high = int(cm[0, 1])
    lin_high_log_low = int(cm[1, 0])
    both_high = int(cm[1, 1])
    mismatches = lin_low_log_high + lin_high_log_low
    pct_mismatch = (mismatches / len(X_test)) * 100.0
    
    results['part1'] = {
        'median_lin_pred': median_lin_pred,
        'cm': cm,
        'both_low': both_low,
        'lin_low_log_high': lin_low_log_high,
        'lin_high_log_low': lin_high_log_low,
        'both_high': both_high,
        'mismatches': mismatches,
        'pct_mismatch': pct_mismatch,
        'lin_high_count': int(np.sum(cls_lin == 1)),
        'log_high_count': int(np.sum(cls_log == 1)),
        'lin_r2': float(lin_reg.score(X_test_scaled, y_lin_test)),
        'log_accuracy': float(accuracy_score(y_log_test, cls_log))
    }
    
    # Plot Part 1: Threshold and Mismatch Visualization
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Subplot 1: Distribution of predictions
    sns.kdeplot(pred_lin_test, ax=axes[0], color='#1f77b4', fill=True, alpha=0.3, label='Model A (Linear Prediction)')
    axes[0].axvline(median_lin_pred, color='red', linestyle='--', linewidth=2, label=f'Model A Median = {median_lin_pred:.2f}')
    axes[0].set_title('Model A Decile Score Predictions & Median Threshold', fontweight='bold')
    axes[0].set_xlabel('Predicted Decile Score')
    axes[0].set_ylabel('Density')
    axes[0].legend(loc='upper right')
    
    # Subplot 2: Confusion Matrix Heatmap
    cm_display = pd.DataFrame(
        [[both_low, lin_low_log_high], [lin_high_log_low, both_high]],
        index=['Model A: Low Risk (<Med)', 'Model A: High Risk (>=Med)'],
        columns=['Model B: Low Risk (<0.5)', 'Model B: High Risk (>=0.5)']
    )
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1], annot_kws={'size': 12, 'weight': 'bold'})
    axes[1].set_title(f'Classification Agreement: {pct_mismatch:.2f}% Mismatch ({mismatches}/{len(X_test)})', fontweight='bold')
    plt.tight_layout()
    fig_path_p1 = os.path.join(figures_dir, 'fig1_model_comparison.png')
    plt.savefig(fig_path_p1, dpi=300)
    plt.close()
    results['fig_p1'] = fig_path_p1
    
    # -------------------------------------------------------------
    # Part 2.1: Simpson's Paradox / Subgroup Analysis
    # -------------------------------------------------------------
    # Key predictor: priors_count and age
    # Splitting by age_cat: 'Less than 25' (Subgroup A) vs 'Greater than 45' (Subgroup B)
    X_sm_full = sm.add_constant(df_clean[['priors_count', 'age']])
    m_sm_full = sm.Logit(df_clean['is_recid'], X_sm_full).fit(disp=False)
    
    sub_A = df_clean[df_clean['age_cat'] == 'Less than 25']
    X_sm_subA = sm.add_constant(sub_A[['priors_count', 'age']])
    m_sm_subA = sm.Logit(sub_A['is_recid'], X_sm_subA).fit(disp=False)
    
    sub_B = df_clean[df_clean['age_cat'] == 'Greater than 45']
    X_sm_subB = sm.add_constant(sub_B[['priors_count', 'age']])
    m_sm_subB = sm.Logit(sub_B['is_recid'], X_sm_subB).fit(disp=False)
    
    results['part2_1'] = {
        'full_n': len(df_clean),
        'full_priors_coef': float(m_sm_full.params['priors_count']),
        'full_priors_ci': [float(m_sm_full.conf_int().loc['priors_count', 0]), float(m_sm_full.conf_int().loc['priors_count', 1])],
        'full_priors_p': float(m_sm_full.pvalues['priors_count']),
        'full_age_coef': float(m_sm_full.params['age']),
        'full_age_ci': [float(m_sm_full.conf_int().loc['age', 0]), float(m_sm_full.conf_int().loc['age', 1])],
        
        'subA_name': 'Age < 25 (Young)',
        'subA_n': len(sub_A),
        'subA_priors_coef': float(m_sm_subA.params['priors_count']),
        'subA_priors_ci': [float(m_sm_subA.conf_int().loc['priors_count', 0]), float(m_sm_subA.conf_int().loc['priors_count', 1])],
        'subA_priors_p': float(m_sm_subA.pvalues['priors_count']),
        'subA_age_coef': float(m_sm_subA.params['age']),
        'subA_age_ci': [float(m_sm_subA.conf_int().loc['age', 0]), float(m_sm_subA.conf_int().loc['age', 1])],
        
        'subB_name': 'Age > 45 (Older)',
        'subB_n': len(sub_B),
        'subB_priors_coef': float(m_sm_subB.params['priors_count']),
        'subB_priors_ci': [float(m_sm_subB.conf_int().loc['priors_count', 0]), float(m_sm_subB.conf_int().loc['priors_count', 1])],
        'subB_priors_p': float(m_sm_subB.pvalues['priors_count']),
        'subB_age_coef': float(m_sm_subB.params['age']),
        'subB_age_ci': [float(m_sm_subB.conf_int().loc['age', 0]), float(m_sm_subB.conf_int().loc['age', 1])]
    }
    
    # Plot Part 2.1: Subgroup Effect Heterogeneity
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = ['Full Dataset\n(All Ages)', 'Subgroup A\n(Age < 25)', 'Subgroup B\n(Age > 45)']
    coefs = [results['part2_1']['full_priors_coef'], results['part2_1']['subA_priors_coef'], results['part2_1']['subB_priors_coef']]
    yerr = [
        [coefs[0] - results['part2_1']['full_priors_ci'][0], results['part2_1']['full_priors_ci'][1] - coefs[0]],
        [coefs[1] - results['part2_1']['subA_priors_ci'][0], results['part2_1']['subA_priors_ci'][1] - coefs[1]],
        [coefs[2] - results['part2_1']['subB_priors_ci'][0], results['part2_1']['subB_priors_ci'][1] - coefs[2]],
    ]
    yerr = np.array(yerr).T
    bars = ax.bar(groups, coefs, yerr=yerr, capsize=8, color=['#4a7bb0', '#e74c3c', '#2ecc71'], alpha=0.85, width=0.5)
    for bar, coef in zip(bars, coefs):
        ax.text(bar.get_x() + bar.get_width()/2, coef/2, f'β = {coef:.4f}', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    ax.set_ylabel('Logistic Regression Coefficient (Log-Odds of Recidivism)')
    ax.set_title('Subgroup Effect Heterogeneity: Prior Convictions Impact on Recidivism\n(95% Confidence Intervals)', fontweight='bold')
    plt.tight_layout()
    fig_path_p2_1 = os.path.join(figures_dir, 'fig2_subgroup_heterogeneity.png')
    plt.savefig(fig_path_p2_1, dpi=300)
    plt.close()
    results['fig_p2_1'] = fig_path_p2_1
    
    # -------------------------------------------------------------
    # Part 2.2: Omitted-Variable Bias
    # -------------------------------------------------------------
    # Full Model vs Reduced Model (omitting priors_count)
    X_full_df = df_clean[['age', 'priors_count', 'juv_fel_count', 'is_African_American', 'is_Hispanic', 'is_Female', 'is_Felony']]
    m_full_ovb = sm.Logit(df_clean['is_recid'], sm.add_constant(X_full_df)).fit(disp=False)
    
    X_red_df = X_full_df.drop(columns=['priors_count'])
    m_red_ovb = sm.Logit(df_clean['is_recid'], sm.add_constant(X_red_df)).fit(disp=False)
    
    ovb_list = []
    for col in X_red_df.columns:
        b_f = float(m_full_ovb.params[col])
        b_r = float(m_red_ovb.params[col])
        abs_diff = b_r - b_f
        pct_chg = ((b_r - b_f) / abs(b_f)) * 100.0
        ovb_list.append({
            'predictor': col,
            'full_coef': b_f,
            'red_coef': b_r,
            'abs_change': abs_diff,
            'pct_change': pct_chg
        })
    ovb_table = pd.DataFrame(ovb_list)
    results['part2_2'] = {
        'table': ovb_table.to_dict('records'),
        'max_shift_predictor': ovb_table.loc[ovb_table['pct_change'].abs().idxmax()]['predictor'],
        'max_shift_pct': float(ovb_table.loc[ovb_table['pct_change'].abs().idxmax()]['pct_change'])
    }
    
    # Plot Part 2.2: Omitted-Variable Bias Shift
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x_pos = np.arange(len(ovb_table))
    width = 0.35
    ax.bar(x_pos - width/2, ovb_table['full_coef'], width, label='Full Model (With Priors)', color='#34495e', alpha=0.85)
    ax.bar(x_pos + width/2, ovb_table['red_coef'], width, label='Reduced Model (Priors Omitted)', color='#e67e22', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ovb_table['predictor'], rotation=15, ha='right', fontweight='bold')
    ax.set_ylabel('Coefficient Value (Log-Odds)')
    ax.set_title('Omitted-Variable Bias: Predictor Coefficient Shifts When Priors Count is Dropped', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig_path_p2_2 = os.path.join(figures_dir, 'fig3_omitted_variable_bias.png')
    plt.savefig(fig_path_p2_2, dpi=300)
    plt.close()
    results['fig_p2_2'] = fig_path_p2_2
    
    # -------------------------------------------------------------
    # Part 2.3: Adversarial Subset Construction
    # -------------------------------------------------------------
    df_test_eval = df_test.copy()
    df_test_eval['prob'] = pred_log_prob
    df_test_eval['y_true'] = y_log_test
    
    fp_high = df_test_eval[(df_test_eval['prob'] >= 0.75) & (df_test_eval['y_true'] == 0)]
    fn_high = df_test_eval[(df_test_eval['prob'] <= 0.25) & (df_test_eval['y_true'] == 1)]
    adv_all = pd.concat([fp_high, fn_high])
    
    results['part2_3'] = {
        'test_count': len(df_test_eval),
        'adv_total_count': len(adv_all),
        'adv_pct': float(len(adv_all) / len(df_test_eval) * 100.0),
        'fp_count': len(fp_high),
        'fn_count': len(fn_high),
        'fp_mean_priors': float(fp_high['priors_count'].mean()) if len(fp_high) > 0 else 0,
        'fn_mean_priors': float(fn_high['priors_count'].mean()) if len(fn_high) > 0 else 0,
        'full_mean_priors': float(df_test_eval['priors_count'].mean()),
        'full_mean_age': float(df_test_eval['age'].mean()),
        'full_male_pct': float((df_test_eval['sex'] == 'Male').mean() * 100.0),
        'full_aa_pct': float((df_test_eval['race'] == 'African-American').mean() * 100.0),
        'full_cauc_pct': float((df_test_eval['race'] == 'Caucasian').mean() * 100.0)
    }
    
    # Sample 50 adversarial rows
    sample_50 = adv_all.sample(n=min(50, len(adv_all)), random_state=seed)
    results['part2_3']['sample_50_mean_age'] = float(sample_50['age'].mean())
    results['part2_3']['sample_50_mean_priors'] = float(sample_50['priors_count'].mean())
    results['part2_3']['sample_50_male_pct'] = float((sample_50['sex'] == 'Male').mean() * 100.0)
    results['part2_3']['sample_50_aa_pct'] = float((sample_50['race'] == 'African-American').mean() * 100.0)
    results['part2_3']['sample_50_cauc_pct'] = float((sample_50['race'] == 'Caucasian').mean() * 100.0)
    results['part2_3']['sample_50_priors_extreme_pct'] = float(((sample_50['priors_count'] >= 5) | (sample_50['priors_count'] <= 1)).mean() * 100.0)
    
    # Plot Part 2.3: Adversarial Subset Characteristics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.kdeplot(df_test_eval['age'], ax=axes[0], color='gray', label='Full Test Set', fill=True, alpha=0.3)
    sns.kdeplot(sample_50['age'], ax=axes[0], color='red', label='Adversarial Subset (N=50)', fill=True, alpha=0.3)
    axes[0].set_title('Age Distribution: Full Test Set vs Adversarial Subset', fontweight='bold')
    axes[0].set_xlabel('Age (Years)')
    axes[0].legend()
    
    prior_df = pd.DataFrame({
        'Dataset': ['Full Test Set', 'Sample 50 Adversarial', 'Confident FP (P>=0.75)', 'Confident FN (P<=0.25)'],
        'Mean Priors': [df_test_eval['priors_count'].mean(), sample_50['priors_count'].mean(), fp_high['priors_count'].mean(), fn_high['priors_count'].mean()]
    })
    sns.barplot(x='Dataset', y='Mean Priors', hue='Dataset', data=prior_df, ax=axes[1], palette=['#34495e', '#e74c3c', '#9b59b6', '#3498db'], legend=False)
    axes[1].set_title('Mean Prior Arrests Breakdown', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    fig_path_p2_3 = os.path.join(figures_dir, 'fig4_adversarial_subset.png')
    plt.savefig(fig_path_p2_3, dpi=300)
    plt.close()
    results['fig_p2_3'] = fig_path_p2_3
    
    # -------------------------------------------------------------
    # Part 2.4: Fairness Trade-Off Analysis
    # -------------------------------------------------------------
    grp_A = df_test_eval[df_test_eval['race'] == 'African-American']
    grp_B = df_test_eval[df_test_eval['race'] == 'Caucasian']
    
    thresholds = np.arange(0.05, 1.0, 0.1)
    fairness_rows = []
    
    for t in thresholds:
        y_hat_A = (grp_A['prob'] >= t).astype(int)
        y_hat_B = (grp_B['prob'] >= t).astype(int)
        
        # Demographic Parity: P(Y_hat=1 | Group A) / P(Y_hat=1 | Group B)
        p_hat_A = float(y_hat_A.mean())
        p_hat_B = float(y_hat_B.mean())
        dp = p_hat_A / p_hat_B if p_hat_B > 0 else np.nan
        
        # TPR and FPR for Group A
        tp_A = np.sum((y_hat_A == 1) & (grp_A['y_true'] == 1))
        fn_A = np.sum((y_hat_A == 0) & (grp_A['y_true'] == 1))
        fp_A = np.sum((y_hat_A == 1) & (grp_A['y_true'] == 0))
        tn_A = np.sum((y_hat_A == 0) & (grp_A['y_true'] == 0))
        tpr_A = float(tp_A / (tp_A + fn_A)) if (tp_A + fn_A) > 0 else 0.0
        fpr_A = float(fp_A / (fp_A + tn_A)) if (fp_A + tn_A) > 0 else 0.0
        
        # TPR and FPR for Group B
        tp_B = np.sum((y_hat_B == 1) & (grp_B['y_true'] == 1))
        fn_B = np.sum((y_hat_B == 0) & (grp_B['y_true'] == 1))
        fp_B = np.sum((y_hat_B == 1) & (grp_B['y_true'] == 0))
        tn_B = np.sum((y_hat_B == 0) & (grp_B['y_true'] == 0))
        tpr_B = float(tp_B / (tp_B + fn_B)) if (tp_B + fn_B) > 0 else 0.0
        fpr_B = float(fp_B / (fp_B + tn_B)) if (fp_B + tn_B) > 0 else 0.0
        
        # Equalized Odds: min(TPR_A / TPR_B, FPR_A / FPR_B)
        tpr_ratio = tpr_A / tpr_B if tpr_B > 0 else np.nan
        fpr_ratio = fpr_A / fpr_B if fpr_B > 0 else np.nan
        eo = min(tpr_ratio, fpr_ratio) if not (np.isnan(tpr_ratio) or np.isnan(fpr_ratio)) else np.nan
        
        # Predictive Parity: P(Y=1 | Y_hat=1, Group A) / P(Y=1 | Y_hat=1, Group B) = Prec_A / Prec_B
        prec_A = float(tp_A / (tp_A + fp_A)) if (tp_A + fp_A) > 0 else np.nan
        prec_B = float(tp_B / (tp_B + fp_B)) if (tp_B + fp_B) > 0 else np.nan
        pp = prec_A / prec_B if (not np.isnan(prec_A) and not np.isnan(prec_B) and prec_B > 0) else np.nan
        
        # Overall accuracy
        overall_y_hat = (df_test_eval['prob'] >= t).astype(int)
        acc = float(accuracy_score(df_test_eval['y_true'], overall_y_hat))
        
        fairness_rows.append({
            'threshold': round(float(t), 2),
            'p_hat_A': round(p_hat_A, 3),
            'p_hat_B': round(p_hat_B, 3),
            'dp': round(float(dp), 3) if not np.isnan(dp) else np.nan,
            'tpr_A': round(tpr_A, 3),
            'tpr_B': round(tpr_B, 3),
            'fpr_A': round(fpr_A, 3),
            'fpr_B': round(fpr_B, 3),
            'eo': round(float(eo), 3) if not np.isnan(eo) else np.nan,
            'prec_A': round(prec_A, 3) if not np.isnan(prec_A) else np.nan,
            'prec_B': round(prec_B, 3) if not np.isnan(prec_B) else np.nan,
            'pp': round(float(pp), 3) if not np.isnan(pp) else np.nan,
            'accuracy': round(acc, 3)
        })
        
    fairness_df = pd.DataFrame(fairness_rows)
    results['part2_4'] = {
        'table': fairness_df.to_dict('records')
    }
    
    # Trade-off quantification calculation:
    # Compare t = 0.35 to t = 0.45 (or t=0.25 to t=0.45)
    row_t35 = fairness_df[fairness_df['threshold'] == 0.35].iloc[0]
    row_t45 = fairness_df[fairness_df['threshold'] == 0.45].iloc[0]
    
    dp_35, dp_45 = row_t35['dp'], row_t45['dp']
    eo_35, eo_45 = row_t35['eo'], row_t45['eo']
    acc_35, acc_45 = row_t35['accuracy'], row_t45['accuracy']
    
    # Improvement in DP toward ideal 1.0:
    # At t=0.35, DP is 1.340 (closer to 1.0) compared to t=0.45 where DP is 1.721 (worse disparity)
    # Moving from t=0.45 to t=0.35:
    dp_gain = ((dp_45 - dp_35) / dp_45) * 100.0
    eo_drop = ((eo_45 - eo_35) / eo_45) * 100.0
    cost_ratio = eo_drop / dp_gain if dp_gain > 0 else 0.0
    
    results['part2_4']['tradeoff_quant'] = {
        't_base': 0.45,
        't_target': 0.35,
        'dp_base': float(dp_45),
        'dp_target': float(dp_35),
        'dp_gain_pct': float(dp_gain),
        'eo_base': float(eo_45),
        'eo_target': float(eo_35),
        'eo_drop_pct': float(eo_drop),
        'tradeoff_cost': float(cost_ratio),
        'acc_35': float(acc_35),
        'acc_45': float(acc_45)
    }
    
    # Plot Part 2.4: Trade-off Chart (X: DP, Y: EO)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    valid_pts = fairness_df.dropna(subset=['dp', 'eo'])
    
    # Filter points where DP and EO are within reasonable plotting boundaries
    plot_pts = valid_pts[(valid_pts['dp'] <= 4.0) & (valid_pts['eo'] <= 3.0)]
    
    ax.plot(plot_pts['dp'], plot_pts['eo'], marker='o', color='#2980b9', linewidth=2, markersize=8)
    
    for _, row in plot_pts.iterrows():
        offset_x = 0.05
        offset_y = 0.02
        if row['threshold'] == 0.35:
            ax.scatter(row['dp'], row['eo'], color='red', s=130, zorder=5, label='Recommended t=0.35')
            ax.annotate(f"t={row['threshold']:.2f} (Recommended)", (row['dp'], row['eo']),
                        textcoords="offset points", xytext=(-20, 12), fontweight='bold', color='red', fontsize=10)
        else:
            ax.annotate(f"t={row['threshold']:.2f}", (row['dp'], row['eo']),
                        textcoords="offset points", xytext=(8, -4), fontsize=9)
            
    ax.axvline(1.0, color='gray', linestyle=':', label='Ideal DP = 1.0')
    ax.axhline(1.0, color='gray', linestyle='--', label='Ideal EO = 1.0')
    ax.set_xlim(0.8, 4.0)
    ax.set_ylim(0.8, 3.0)
    ax.set_xlabel('Demographic Parity Ratio: P(Ŷ=1 | African-American) / P(Ŷ=1 | Caucasian)', fontweight='bold')
    ax.set_ylabel('Equalized Odds Ratio: min(TPR_A/TPR_B, FPR_A/FPR_B)', fontweight='bold')
    ax.set_title('Fairness Trade-Off Frontier Across Decision Thresholds (t in [0.05, 0.75])', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig_path_p2_4 = os.path.join(figures_dir, 'fig5_fairness_tradeoff_curve.png')
    plt.savefig(fig_path_p2_4, dpi=300)
    plt.close()
    results['fig_p2_4'] = fig_path_p2_4
    
    return results

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    res = run_pipeline(seed=seed)
    print("Pipeline executed successfully for seed:", seed)
    print("Mismatch rate:", res['part1']['pct_mismatch'])
    print("OVB Max Shift:", res['part2_2']['max_shift_predictor'], res['part2_2']['max_shift_pct'])
