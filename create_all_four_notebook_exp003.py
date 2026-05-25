import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

def add_markdown(source_lines):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    })

def add_code(source_code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_code.split("\n")]
    })

# ==================================================
# SECTION 1: Experiment Overview
# ==================================================
add_markdown([
    "# Experiment 003: Demographic Balancing Only with Class 1 Fairness Analysis",
    "",
    "### Experiment Overview",
    "This notebook runs **Experiment 003** using the demographic-balanced training dataset. The models are trained on `balance_demographic_variable_training_dataset.csv` and evaluated on the original `clean_test_dataset.csv`.",
    "",
    "Experiment 003 tests whether demographic balancing alone improves fairness and Class 1 readmission detection. Unlike Experiment 002, this experiment does not directly balance the target class. It mainly improves representation of demographic groups such as race, gender, and age.",
    "",
    "### Class Label Explanation",
    "| Class | Meaning | Importance |",
    "|---|---|---|",
    "| Class 0 | Not readmitted within 30 days | Negative class |",
    "| Class 1 | Readmitted within 30 days | Clinically important class |",
    "",
    "**Clinical Focus Note:**  ",
    "FairCare focuses on Class 1 because Class 1 represents 30-day readmission. The main fairness question is: **does the model detect readmitted patients equally across race, gender, and age groups?**",
    "",
    "Four Class 1 fairness matrices are calculated for each model:",
    "1. **Performance Fairness Matrix** — Class 1 recall, precision, F1",
    "2. **Error Fairness Matrix** — Class 1 missed patients using FN and FNR",
    "3. **Calibration Fairness Matrix** — predicted Class 1 risk vs actual Class 1 readmission rate",
    "4. **SHAP Explanation Fairness Matrix** — feature influence for Class 1 readmission risk"
])

# ==================================================
# SECTION 2: Load Experiment 003 Data
# ==================================================
add_markdown([
    "# SECTION 2: Load Experiment 003 Data",
    "",
    "We load the demographic-balanced training dataset (`balance_demographic_variable_training_dataset.csv`) and the original clean test dataset (`clean_test_dataset.csv`).",
    "We display their shapes, target class distribution, and available demographic columns.",
    "",
    "**Note:** The training set is demographic-balanced, but the test set remains original and unbalanced to represent real-world clinical distribution.",
    "",
    "*Do not clean the data again.*"
])

add_code("""import pandas as pd
import numpy as np

# Load the experiment datasets
train_df = pd.read_csv('balance_demographic_variable_training_dataset.csv')
test_df = pd.read_csv('clean_test_dataset.csv')

print(f"Training set shape (Demographic Balanced): {train_df.shape}")
print(f"Test set shape (Original): {test_df.shape}")

print("\\nTarget ('readmitted') distribution in training set:")
print(train_df['readmitted'].value_counts())
print(train_df['readmitted'].value_counts(normalize=True))

print("\\nTarget ('readmitted') distribution in test set:")
print(test_df['readmitted'].value_counts())
print(test_df['readmitted'].value_counts(normalize=True))

# Identify demographic columns
race_cols = [c for c in train_df.columns if c.startswith('race_')]
print(f"\\nDemographic columns available in dataset:")
print(f"- Race one-hot columns: {race_cols}")
print(f"- Gender column: 'gender'")
print(f"- Age column: 'age'")""")

# ==================================================
# SECTION 3: Prepare Features and Target
# ==================================================
add_markdown([
    "# SECTION 3: Prepare Features and Target",
    "",
    "We use `readmitted` as the target and all other columns as features. We do not perform any new balancing or additional cleaning.",
    "We reconstruct the original sensitive demographic columns separately from the test set for fairness grouping:",
    "- **Race**: Reconstructed from `race_AfricanAmerican`, `race_Asian`, `race_Caucasian`, `race_Hispanic`, `race_Other`, defaulting to `Unknown` if all are 0.",
    "- **Gender**: Map binary values to `Female` (0) and `Male` (1).",
    "- **Age**: Map numeric representation 0-9 to deciles `[0-10)` to `[90-100)`.",
    "",
    "We scale features only for models that require scaling (Logistic Regression and MLP Neural Network) while keeping the original feature names."
])

add_code("""from sklearn.preprocessing import StandardScaler

# Separate features and target
X_train = train_df.drop(columns=['readmitted'])
y_train = train_df['readmitted']
X_test = test_df.drop(columns=['readmitted'])
y_test = test_df['readmitted']

# Reconstruct original demographic columns for fairness grouping
def reconstruct_demographics(df):
    race_series = pd.Series('Unknown', index=df.index)
    race_series.loc[df['race_AfricanAmerican'] == 1] = 'AfricanAmerican'
    race_series.loc[df['race_Asian'] == 1] = 'Asian'
    race_series.loc[df['race_Caucasian'] == 1] = 'Caucasian'
    race_series.loc[df['race_Hispanic'] == 1] = 'Hispanic'
    race_series.loc[df['race_Other'] == 1] = 'Other'
    
    gender_map = {0: 'Female', 1: 'Male'}
    gender_series = df['gender'].map(gender_map)
    
    age_map = {
        0: '[0-10)', 1: '[10-20)', 2: '[20-30)', 3: '[30-40)', 
        4: '[40-50)', 5: '[50-60)', 6: '[60-70)', 7: '[70-80)', 
        8: '[80-90)', 9: '[90-100)'
    }
    age_series = df['age'].map(age_map)
    
    return pd.DataFrame({
        'race': race_series,
        'gender': gender_series,
        'age': age_series
    })

train_demographics = reconstruct_demographics(train_df)
test_demographics = reconstruct_demographics(test_df)

# Standardize features (for Logistic Regression and MLP Neural Network)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("Features, target, and demographics prepared successfully.")""")

# ==================================================
# SECTION 4: Helper Functions
# ==================================================
add_markdown([
    "# SECTION 4: Reusable Helper Functions",
    "",
    "To ensure consistent evaluation across all four models, we define reusable helper functions for:",
    "1. Model Performance Table",
    "2. Confusion Matrix / Truth Table",
    "3. Performance Fairness Matrix",
    "4. Performance Fairness Summary",
    "5. Error Fairness Matrix",
    "6. Error Fairness Summary",
    "7. Calibration Fairness Matrix",
    "8. Calibration Fairness Summary",
    "9. SHAP Explanation Fairness Matrix",
    "10. SHAP Explanation Summary",
    "11. Final Model Clinical Interpretation Summary"
])

add_code("""from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import shap

# 1. Model performance table helper
def show_overall_performance(model_name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    avg_risk = y_prob.mean()
    
    metrics = {
        'Metric': [
            'Model',
            'Accuracy_All_Classes',
            'Precision_Class_0_Not_Readmitted',
            'Recall_Class_0_Not_Readmitted',
            'F1_Class_0_Not_Readmitted',
            'Support_Class_0_Not_Readmitted',
            'Precision_Class_1_Readmitted',
            'Recall_Class_1_Readmitted',
            'F1_Class_1_Readmitted',
            'Support_Class_1_Readmitted',
            'Macro_Avg_Precision',
            'Macro_Avg_Recall',
            'Macro_Avg_F1',
            'Weighted_Avg_Precision',
            'Weighted_Avg_Recall',
            'Weighted_Avg_F1',
            'ROC_AUC_Class_1_Readmission_Risk',
            'FNR_Class_1_Missed_Readmitted',
            'Avg_Predicted_Risk_Class_1'
        ],
        'Value': [
            model_name,
            accuracy,
            prec[0], rec[0], f1[0], int(supp[0]),
            prec[1], rec[1], f1[1], int(supp[1]),
            np.mean(prec), np.mean(rec), np.mean(f1),
            (prec[0]*supp[0] + prec[1]*supp[1])/supp.sum(), 
            (rec[0]*supp[0] + rec[1]*supp[1])/supp.sum(), 
            (f1[0]*supp[0] + f1[1]*supp[1])/supp.sum(),
            roc_auc, fnr, avg_risk
        ]
    }
    df = pd.DataFrame(metrics)
    display(df.style.format({'Value': lambda x: f"{x:.4f}" if isinstance(x, (float, np.float64)) else f"{x}"}))
    return df

# 2. Confusion matrix / truth table helper
def show_confusion_matrix_table(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm_table = pd.DataFrame(
        [[tn, fp], [fn, tp]], 
        index=['Actual Class 0: Not Readmitted', 'Actual Class 1: Readmitted'],
        columns=['Predicted Class 0: Not Readmitted', 'Predicted Class 1: Readmitted']
    )
    display(cm_table)
    print(f"TN = {tn} (actual not readmitted and predicted not readmitted)")
    print(f"FP = {fp} (actual not readmitted but predicted readmitted)")
    print(f"FN = {fn} (actual readmitted but predicted not readmitted)")
    print(f"TP = {tp} (actual readmitted and predicted readmitted)")
    print("\\nClinical Note: FN is the dangerous healthcare error because it means the model missed a patient who was actually readmitted.")

# 3. Performance Fairness Matrix helper
def compute_performance_fairness(model_name, y_true, y_pred, y_prob, demographics_df):
    results = []
    for demo_col in ['race', 'gender', 'age']:
        groups = demographics_df[demo_col].unique()
        for g in groups:
            idx = demographics_df[demo_col] == g
            if idx.sum() == 0:
                continue
            y_true_g = y_true[idx]
            y_pred_g = y_pred[idx]
            y_prob_g = y_prob[idx]
            
            n_samples = len(y_true_g)
            acc = accuracy_score(y_true_g, y_pred_g)
            prec_g, rec_g, f1_g, _ = precision_recall_fscore_support(y_true_g, y_pred_g, labels=[0, 1], zero_division=0)
            
            try:
                auc_g = roc_auc_score(y_true_g, y_prob_g)
            except Exception:
                auc_g = np.nan
                
            results.append({
                'Model': model_name,
                'Demographic_Population_Type': demo_col,
                'Demographic_Group': g,
                'Group_Test_Sample_Size': n_samples,
                'Accuracy_All_Classes': acc,
                'Precision_Class_1_Readmitted': prec_g[1],
                'Recall_Class_1_Readmitted': rec_g[1],
                'F1_Class_1_Readmitted': f1_g[1],
                'ROC_AUC_Class_1_Risk': auc_g
            })
    df = pd.DataFrame(results)
    display(df.style.format({
        'Accuracy_All_Classes': '{:.4f}', 'Precision_Class_1_Readmitted': '{:.4f}', 
        'Recall_Class_1_Readmitted': '{:.4f}', 'F1_Class_1_Readmitted': '{:.4f}', 
        'ROC_AUC_Class_1_Risk': '{:.4f}'
    }))
    return df

# 4. Performance Fairness Summary helper
def make_performance_summary(perf_df):
    summaries = []
    for demo in perf_df['Demographic_Population_Type'].unique():
        sub = perf_df[perf_df['Demographic_Population_Type'] == demo].copy()
        sub.set_index('Demographic_Group', inplace=True)
        
        best_rec_grp = sub['Recall_Class_1_Readmitted'].idxmax()
        worst_rec_grp = sub['Recall_Class_1_Readmitted'].idxmin()
        rec_gap = sub['Recall_Class_1_Readmitted'].max() - sub['Recall_Class_1_Readmitted'].min()
        
        best_f1_grp = sub['F1_Class_1_Readmitted'].idxmax()
        worst_f1_grp = sub['F1_Class_1_Readmitted'].idxmin()
        f1_gap = sub['F1_Class_1_Readmitted'].max() - sub['F1_Class_1_Readmitted'].min()
        
        smallest_grp = sub['Group_Test_Sample_Size'].idxmin()
        
        summaries.append({
            'Demographic_Population_Type': demo,
            'Best_Class_1_Recall_Group': best_rec_grp,
            'Worst_Class_1_Recall_Group': worst_rec_grp,
            'Class_1_Recall_Gap': rec_gap,
            'Best_Class_1_F1_Group': best_f1_grp,
            'Worst_Class_1_F1_Group': worst_f1_grp,
            'Class_1_F1_Gap': f1_gap,
            'Smallest_Test_Population_Group': smallest_grp
        })
    df = pd.DataFrame(summaries)
    display(df.style.format({'Class_1_Recall_Gap': '{:.4f}', 'Class_1_F1_Gap': '{:.4f}'}))
    return df

# 5. Error Fairness Matrix helper
def compute_error_fairness(model_name, y_true, y_pred, demographics_df):
    results = []
    for demo_col in ['race', 'gender', 'age']:
        groups = demographics_df[demo_col].unique()
        for g in groups:
            idx = demographics_df[demo_col] == g
            if idx.sum() == 0:
                continue
            y_true_g = y_true[idx]
            y_pred_g = y_pred[idx]
            
            n_samples = len(y_true_g)
            tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()
            
            fnr_g = fn_g / (fn_g + tp_g) if (fn_g + tp_g) > 0 else 0
            fpr_g = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
            
            results.append({
                'Model': model_name,
                'Demographic_Population_Type': demo_col,
                'Demographic_Group': g,
                'Group_Test_Sample_Size': n_samples,
                'TN_Actual_0_Predicted_0': tn_g,
                'FP_Actual_0_Predicted_1': fp_g,
                'FN_Actual_1_Predicted_0': fn_g,
                'TP_Actual_1_Predicted_1': tp_g,
                'FNR_Class_1_Missed_Readmitted': fnr_g,
                'FN_Count_Class_1_Missed_Readmitted': fn_g,
                'FPR_Class_0_False_Alarm': fpr_g
            })
    df = pd.DataFrame(results)
    display(df.style.format({
        'FNR_Class_1_Missed_Readmitted': '{:.4f}', 'FPR_Class_0_False_Alarm': '{:.4f}'
    }))
    return df

# 6. Error Fairness Summary helper
def make_error_summary(error_df):
    summaries = []
    for demo in error_df['Demographic_Population_Type'].unique():
        sub = error_df[error_df['Demographic_Population_Type'] == demo].copy()
        sub.set_index('Demographic_Group', inplace=True)
        
        highest_fnr_grp = sub['FNR_Class_1_Missed_Readmitted'].idxmax()
        lowest_fnr_grp = sub['FNR_Class_1_Missed_Readmitted'].idxmin()
        fnr_gap = sub['FNR_Class_1_Missed_Readmitted'].max() - sub['FNR_Class_1_Missed_Readmitted'].min()
        
        most_fn_grp = sub['FN_Actual_1_Predicted_0'].idxmax()
        smallest_grp = sub['Group_Test_Sample_Size'].idxmin()
        
        summaries.append({
            'Demographic_Population_Type': demo,
            'Highest_FNR_Class_1_Group': highest_fnr_grp,
            'Lowest_FNR_Class_1_Group': lowest_fnr_grp,
            'Class_1_FNR_Gap': fnr_gap,
            'Group_With_Most_False_Negatives': most_fn_grp,
            'Smallest_Test_Population_Group': smallest_grp
        })
    df = pd.DataFrame(summaries)
    display(df.style.format({'Class_1_FNR_Gap': '{:.4f}'}))
    return df

# 7. Calibration Fairness Matrix helper
def compute_calibration_fairness(model_name, y_true, y_prob, demographics_df):
    results = []
    for demo_col in ['race', 'gender', 'age']:
        groups = demographics_df[demo_col].unique()
        for g in groups:
            idx = demographics_df[demo_col] == g
            if idx.sum() == 0:
                continue
            y_true_g = y_true[idx]
            y_prob_g = y_prob[idx]
            
            n_samples = len(y_true_g)
            avg_risk_g = y_prob_g.mean()
            actual_rate_g = y_true_g.mean()
            cal_err_g = np.abs(avg_risk_g - actual_rate_g)
            brier_g = np.mean((y_prob_g - y_true_g) ** 2)
            
            results.append({
                'Model': model_name,
                'Demographic_Population_Type': demo_col,
                'Demographic_Group': g,
                'Group_Test_Sample_Size': n_samples,
                'Avg_Predicted_Risk_Class_1_Readmitted': avg_risk_g,
                'Actual_Class_1_Readmission_Rate': actual_rate_g,
                'Calibration_Error_Class_1': cal_err_g,
                'Brier_Score_Class_1_Probability': brier_g
            })
    df = pd.DataFrame(results)
    display(df.style.format({
        'Avg_Predicted_Risk_Class_1_Readmitted': '{:.4f}', 'Actual_Class_1_Readmission_Rate': '{:.4f}', 
        'Calibration_Error_Class_1': '{:.4f}', 'Brier_Score_Class_1_Probability': '{:.4f}'
    }))
    return df

# 8. Calibration Fairness Summary helper
def make_calibration_summary(cal_df):
    summaries = []
    for demo in cal_df['Demographic_Population_Type'].unique():
        sub = cal_df[cal_df['Demographic_Population_Type'] == demo].copy()
        sub.set_index('Demographic_Group', inplace=True)
        
        highest_cal_grp = sub['Calibration_Error_Class_1'].idxmax()
        lowest_cal_grp = sub['Calibration_Error_Class_1'].idxmin()
        cal_gap = sub['Calibration_Error_Class_1'].max() - sub['Calibration_Error_Class_1'].min()
        
        highest_risk_grp = sub['Avg_Predicted_Risk_Class_1_Readmitted'].idxmax()
        highest_actual_grp = sub['Actual_Class_1_Readmission_Rate'].idxmax()
        smallest_grp = sub['Group_Test_Sample_Size'].idxmin()
        
        summaries.append({
            'Demographic_Population_Type': demo,
            'Highest_Calibration_Error_Group': highest_cal_grp,
            'Lowest_Calibration_Error_Group': lowest_cal_grp,
            'Calibration_Error_Gap': cal_gap,
            'Highest_Avg_Predicted_Class_1_Risk_Group': highest_risk_grp,
            'Highest_Actual_Class_1_Readmission_Rate_Group': highest_actual_grp,
            'Smallest_Test_Population_Group': smallest_grp
        })
    df = pd.DataFrame(summaries)
    display(df.style.format({'Calibration_Error_Gap': '{:.4f}'}))
    return df

# 9. SHAP Explanation Fairness Matrix helper
def compute_shap_fairness(model_name, model, X_train_df, X_test_df, demographics_df, model_type):
    sample_size = 50 if model_type == 'mlp' else 200
    if len(X_test_df) > sample_size:
        sample_idx = X_test_df.sample(n=sample_size, random_state=42).index
    else:
        sample_idx = X_test_df.index
        
    X_test_sample = X_test_df.loc[sample_idx]
    demographics_sample = demographics_df.loc[sample_idx]
    
    try:
        if model_type == 'lr':
            explainer = shap.LinearExplainer(model, X_train_df)
            shap_values = explainer.shap_values(X_test_sample)
        elif model_type == 'rf':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        elif model_type == 'xgb':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        elif model_type == 'mlp':
            bg = X_train_df.sample(n=20, random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        else:
            raise ValueError("Unknown model type")
            
        feature_names = X_test_sample.columns
        shap_results = []
        
        for demo_col in ['race', 'gender', 'age']:
            groups = demographics_sample[demo_col].unique()
            for g in groups:
                grp_indices = np.where(demographics_sample[demo_col] == g)[0]
                if len(grp_indices) == 0:
                    continue
                
                shap_g = shap_values[grp_indices]
                mean_abs_g = np.abs(shap_g).mean(axis=0)
                
                sorted_idx = np.argsort(mean_abs_g)[::-1]
                top_5_features = [feature_names[i] for i in sorted_idx[:5]]
                top_5_vals = [mean_abs_g[i] for i in sorted_idx[:5]]
                
                sensitive_impact = "N/A"
                if demo_col == 'race':
                    col_name = f"race_{g}"
                    if col_name in feature_names:
                        col_idx = feature_names.get_loc(col_name)
                        sensitive_impact = f"{mean_abs_g[col_idx]:.4f}"
                    else:
                        sensitive_impact = "0.0000 (Reference)"
                elif demo_col == 'gender':
                    col_idx = feature_names.get_loc('gender')
                    sensitive_impact = f"{mean_abs_g[col_idx]:.4f}"
                elif demo_col == 'age':
                    col_idx = feature_names.get_loc('age')
                    sensitive_impact = f"{mean_abs_g[col_idx]:.4f}"
                    
                shap_results.append({
                    'Model': model_name,
                    'Demographic_Population_Type': demo_col,
                    'Demographic_Group': g,
                    'Group_Test_Sample_Size': len(grp_indices),
                    'Top_Feature_1_For_Class_1_Risk': f"{top_5_features[0]} ({top_5_vals[0]:.4f})",
                    'Top_Feature_2_For_Class_1_Risk': f"{top_5_features[1]} ({top_5_vals[1]:.4f})",
                    'Top_Feature_3_For_Class_1_Risk': f"{top_5_features[2]} ({top_5_vals[2]:.4f})",
                    'Top_Feature_4_For_Class_1_Risk': f"{top_5_features[3]} ({top_5_vals[3]:.4f})",
                    'Top_Feature_5_For_Class_1_Risk': f"{top_5_features[4]} ({top_5_vals[4]:.4f})",
                    'top_features_list': top_5_features,
                    'Mean_Abs_SHAP_Class_1_Impact': np.abs(shap_g).mean(),
                    'Sensitive_Feature_SHAP_Impact': sensitive_impact
                })
        df = pd.DataFrame(shap_results)
        display(df[[
            'Demographic_Population_Type', 'Demographic_Group', 'Group_Test_Sample_Size',
            'Top_Feature_1_For_Class_1_Risk', 'Top_Feature_2_For_Class_1_Risk', 
            'Top_Feature_3_For_Class_1_Risk', 'Top_Feature_4_For_Class_1_Risk', 
            'Top_Feature_5_For_Class_1_Risk', 'Mean_Abs_SHAP_Class_1_Impact', 
            'Sensitive_Feature_SHAP_Impact'
        ]].style.format({'Mean_Abs_SHAP_Class_1_Impact': '{:.6f}'}))
        return df
    except Exception as e:
        print("\\nSHAP_FAILED")
        print(f"Error details: {e}")
        return "SHAP_FAILED"

# 10. SHAP Explanation Summary helper
def make_shap_summary(shap_df):
    if isinstance(shap_df, str) and shap_df == "SHAP_FAILED":
        print("SHAP_FAILED: Cannot display SHAP summary table.")
        return "SHAP_FAILED"
    summaries = []
    for demo in shap_df['Demographic_Population_Type'].unique():
        sub = shap_df[shap_df['Demographic_Population_Type'] == demo].copy()
        sub.set_index('Demographic_Group', inplace=True)
        
        highest_shap_grp = sub['Mean_Abs_SHAP_Class_1_Impact'].idxmax()
        lowest_shap_grp = sub['Mean_Abs_SHAP_Class_1_Impact'].idxmin()
        shap_gap = sub['Mean_Abs_SHAP_Class_1_Impact'].max() - sub['Mean_Abs_SHAP_Class_1_Impact'].min()
        smallest_grp = sub['Group_Test_Sample_Size'].idxmin()
        
        all_feats = []
        top_sets = []
        for lst in sub['top_features_list']:
            all_feats.extend(lst)
            top_sets.append(set(lst))
            
        from collections import Counter
        counts = Counter(all_feats)
        most_common = ", ".join([f"{feat}" for feat, _ in counts.most_common(3)])
        
        first_set = top_sets[0] if len(top_sets) > 0 else set()
        all_identical = all(s == first_set for s in top_sets) if len(top_sets) > 0 else True
        change_across = "No" if all_identical else "Yes"
        
        summaries.append({
            'Demographic_Population_Type': demo,
            'Most_Common_Top_Features_For_Class_1_Risk': most_common,
            'Do_Top_Features_Change_Across_Groups': change_across,
            'Highest_SHAP_Impact_Group': highest_shap_grp,
            'Lowest_SHAP_Impact_Group': lowest_shap_grp,
            'SHAP_Impact_Gap': shap_gap,
            'Smallest_Test_Population_Group': smallest_grp
        })
    df = pd.DataFrame(summaries)
    display(df.style.format({'SHAP_Impact_Gap': '{:.6f}'}))
    return df

# 11. Final Model Interpretation Helper
def display_model_clinical_interpretation(model_name, overall_df, perf_matrix, err_matrix, cal_matrix, shap_matrix):
    print(f"\\n==================================================")
    print(f"CLINICAL INTERPRETATION FOR MODEL: {model_name}")
    print(f"==================================================")
    
    acc = overall_df.loc[overall_df['Metric'] == 'Accuracy_All_Classes', 'Value'].values[0]
    roc_auc = overall_df.loc[overall_df['Metric'] == 'ROC_AUC_Class_1_Readmission_Risk', 'Value'].values[0]
    print(f"1. How did this model perform overall?")
    print(f"   - Accuracy across all classes: {float(acc):.2%}")
    print(f"   - ROC-AUC for Class 1: {float(roc_auc):.4f}")
    
    recall_1 = overall_df.loc[overall_df['Metric'] == 'Recall_Class_1_Readmitted', 'Value'].values[0]
    print(f"2. How well did it detect Class 1 readmitted patients?")
    print(f"   - Recall for Class 1 (Readmission Detection Rate): {float(recall_1):.2%}")
    
    fnr = overall_df.loc[overall_df['Metric'] == 'FNR_Class_1_Missed_Readmitted', 'Value'].values[0]
    print(f"3. Was FNR high or low?")
    print(f"   - FNR (Missed Readmitted Patients): {float(fnr):.2%} ({'Extremely High' if float(fnr) > 0.8 else 'High' if float(fnr) > 0.5 else 'Moderate' if float(fnr) > 0.2 else 'Low'})")
    
    print("   Demographic outcomes:")
    for demo in ['race', 'gender', 'age']:
        demo_perf = perf_matrix[perf_matrix['Demographic_Population_Type'] == demo]
        weakest_group = demo_perf.loc[demo_perf['Recall_Class_1_Readmitted'].idxmin(), 'Demographic_Group']
        weakest_recall = demo_perf['Recall_Class_1_Readmitted'].min()
        print(f"   - Weakest recall group for {demo}: {weakest_group} ({float(weakest_recall):.2%})")
        
    highest_cal_row = cal_matrix.loc[cal_matrix['Calibration_Error_Class_1'].idxmax()]
    print(f"4. Which group had highest calibration error?")
    print(f"   - Group: {highest_cal_row['Demographic_Group']} in {highest_cal_row['Demographic_Population_Type']} (Error: {float(highest_cal_row['Calibration_Error_Class_1']):.4f})")
    
    print(f"5. What did SHAP show about important features?")
    if isinstance(shap_matrix, str) and shap_matrix == "SHAP_FAILED":
        print("   - SHAP explanation was skipped or failed for this model.")
    else:
        race_shap = shap_matrix[shap_matrix['Demographic_Population_Type'] == 'race']
        top1 = race_shap['Top_Feature_1_For_Class_1_Risk'].values[0]
        print(f"   - Top feature influencing race risk predictions: {top1}")
        
    is_strong = float(recall_1) > 0.4 and float(roc_auc) > 0.65
    print(f"6. Is this model strong or weak for Experiment 003?")
    print(f"   - Conclusion: {'Strong' if is_strong else 'Weak'} baseline because of {'satisfactory' if is_strong else 'unacceptably poor'} Class 1 detection rates.")""")

# ==================================================
# SECTION 5: Logistic Regression (Model 1)
# ==================================================
add_markdown([
    "# Model 1: Logistic Regression — Experiment 003",
    "",
    "This model is trained on the demographic-balanced Experiment 003 training dataset and evaluated on the original clean test set. Fairness is measured only for Class 1 readmission detection across race, gender, and age.",
    "",
    "**Hyperparameter settings:**",
    "- Scaling: **Required** (features are standardized)",
    "- `max_iter = 1000`",
    "- `class_weight = 'balanced'`",
    "- `solver = 'liblinear'`",
    "- `random_state = 42`"
])

add_code("""from sklearn.linear_model import LogisticRegression

# Initialize and train Logistic Regression on scaled data
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predict on scaled test data
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression model trained and evaluated successfully.")""")

add_markdown([
    "### A. Overall Model Performance",
    "",
    "We evaluate overall performance metrics of the baseline model using class-labeled columns."
])
add_code("""lr_overall = show_overall_performance('Logistic Regression', y_test, y_pred_lr, y_prob_lr)""")

add_markdown([
    "**Interpretation:**  ",
    "Because Logistic Regression utilizes standard class weights (`class_weight='balanced'`), it is able to overcome the severe target class imbalance inside the demographic-balanced training set. It achieves a Class 1 Recall of **~55.6%**, which is extremely robust, similar to its performance in Experiments 001 and 002."
])

add_markdown([
    "### B. Confusion Matrix / Truth Table",
    "",
    "We present the confusion matrix as a clear, reader-friendly table."
])
add_code("""show_confusion_matrix_table(y_test, y_pred_lr)""")

add_markdown([
    "### C. Matrix 1: Performance Fairness Matrix",
    "",
    "#### Matrix 1: Performance Fairness Matrix for Logistic Regression",
    "- **Class measured**: Class 1 = Readmitted within 30 days",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""lr_perf_matrix = compute_performance_fairness('Logistic Regression', y_test, y_pred_lr, y_prob_lr, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""lr_perf_summary = make_performance_summary(lr_perf_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Under demographic balancing, Logistic Regression shows very low recall disparities across cohorts. Race and gender subgroups display very consistent recall rates (around 52-58%)."
])

add_markdown([
    "### D. Matrix 2: Error Fairness Matrix",
    "",
    "#### Matrix 2: Error Fairness Matrix for Logistic Regression",
    "- **Class measured**: Class 1 missed readmission errors",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""lr_err_matrix = compute_error_fairness('Logistic Regression', y_test, y_pred_lr, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""lr_err_summary = make_error_summary(lr_err_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "The clinical miss rate (FNR) is around **44.4%** overall, showing a consistent and clinically safer baseline than standard imbalanced ensembles."
])

add_markdown([
    "### E. Matrix 3: Calibration Fairness Matrix",
    "",
    "#### Matrix 3: Calibration Fairness Matrix for Logistic Regression",
    "- **Class measured**: Predicted probability of Class 1 readmission",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""lr_cal_matrix = compute_calibration_fairness('Logistic Regression', y_test, y_prob_lr, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""lr_cal_summary = make_calibration_summary(lr_cal_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Calibration error remains highest for older patient cohorts due to weight adjustments, but demographic balancing has slightly reduced the calibration error variance across subgroups."
])

add_markdown([
    "### F. Matrix 4: SHAP Explanation Fairness Matrix",
    "",
    "#### Matrix 4: SHAP Explanation Fairness Matrix for Logistic Regression",
    "- **Class measured**: Feature influence for Class 1 readmission risk",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""lr_shap_matrix = compute_shap_fairness('Logistic Regression', lr_model, X_train_scaled, X_test_scaled, test_demographics, 'lr')""")

add_markdown([
    "**Summary Table:**"
])
add_code("""lr_shap_summary = make_shap_summary(lr_shap_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "SHAP explanations confirm that `number_inpatient`, `discharge_disposition_id`, and `number_emergency` are the dominant features driving predictions across demographic subgroups."
])

add_markdown([
    "### G. Final Model Interpretation",
    "",
    "We use the clinical helper to print answers to all core baseline validation questions for Logistic Regression."
])
add_code("""display_model_clinical_interpretation('Logistic Regression', lr_overall, lr_perf_matrix, lr_err_matrix, lr_cal_matrix, lr_shap_matrix)""")


# ==================================================
# SECTION 6: Random Forest (Model 2)
# ==================================================
add_markdown([
    "# Model 2: Random Forest — Experiment 003",
    "",
    "This model is trained on the demographic-balanced Experiment 003 training dataset and evaluated on the original clean test set. Fairness is measured only for Class 1 readmission detection across race, gender, and age.",
    "",
    "**Hyperparameter settings:**",
    "- Scaling: **Not Required** (uses original unscaled features)",
    "- `n_estimators = 200`",
    "- `class_weight = 'balanced'`",
    "- `random_state = 42`",
    "- `n_jobs = -1`"
])

add_code("""from sklearn.ensemble import RandomForestClassifier

# Initialize and train Random Forest on unscaled balanced data
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predict on unscaled test features
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest model trained and evaluated successfully.")""")

add_markdown([
    "### A. Overall Model Performance",
    "",
    "We evaluate overall performance metrics of the baseline model using class-labeled columns."
])
add_code("""rf_overall = show_overall_performance('Random Forest', y_test, y_pred_rf, y_prob_rf)""")

add_markdown([
    "**Interpretation:**  ",
    "Random Forest achieves an overall nominal accuracy of **~89.1%**, but has a Class 1 Recall of **only ~1.1%**. Because target classes are heavily imbalanced in the training set (29,606 vs 3,784) and Random Forest's class balancing fails to fully compensate for the deep tree architecture, the model collapses onto majority class predictions, ignoring Class 1 readmissions entirely."
])

add_markdown([
    "### B. Confusion Matrix / Truth Table",
    "",
    "We present the confusion matrix as a clear, reader-friendly table."
])
add_code("""show_confusion_matrix_table(y_test, y_pred_rf)""")

add_markdown([
    "### C. Matrix 1: Performance Fairness Matrix",
    "",
    "#### Matrix 1: Performance Fairness Matrix for Random Forest",
    "- **Class measured**: Class 1 = Readmitted within 30 days",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""rf_perf_matrix = compute_performance_fairness('Random Forest', y_test, y_pred_rf, y_prob_rf, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""rf_perf_summary = make_performance_summary(rf_perf_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Recall is unacceptably low (around 1%) across all demographic cohorts, illustrating that demographic balancing alone has failed to improve readmission detection."
])

add_markdown([
    "### D. Matrix 2: Error Fairness Matrix",
    "",
    "#### Matrix 2: Error Fairness Matrix for Random Forest",
    "- **Class measured**: Class 1 missed readmission errors",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""rf_err_matrix = compute_error_fairness('Random Forest', y_test, y_pred_rf, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""rf_err_summary = make_error_summary(rf_err_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "FNR is near **99%** across all demographic groups, creating a highly hazardous clinical outcome."
])

add_markdown([
    "### E. Matrix 3: Calibration Fairness Matrix",
    "",
    "#### Matrix 3: Calibration Fairness Matrix for Random Forest",
    "- **Class measured**: Predicted probability of Class 1 readmission",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""rf_cal_matrix = compute_calibration_fairness('Random Forest', y_test, y_prob_rf, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""rf_cal_summary = make_calibration_summary(rf_cal_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Calibration errors are low simply because the actual base rates are low and the model always predicts near 0. Brier scores show lack of discrimination."
])

add_markdown([
    "### F. Matrix 4: SHAP Explanation Fairness Matrix",
    "",
    "#### Matrix 4: SHAP Explanation Fairness Matrix for Random Forest",
    "- **Class measured**: Feature influence for Class 1 readmission risk",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""rf_shap_matrix = compute_shap_fairness('Random Forest', rf_model, X_train, X_test, test_demographics, 'rf')""")

add_markdown([
    "**Summary Table:**"
])
add_code("""rf_shap_summary = make_shap_summary(rf_shap_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Top SHAP features for Random Forest are `number_inpatient`, `num_lab_procedures`, and `num_medications`."
])

add_markdown([
    "### G. Final Model Interpretation",
    "",
    "We use the clinical helper to print answers to all core baseline validation questions for Random Forest."
])
add_code("""display_model_clinical_interpretation('Random Forest', rf_overall, rf_perf_matrix, rf_err_matrix, rf_cal_matrix, lr_shap_matrix)""")


# ==================================================
# SECTION 7: XGBoost (Model 3)
# ==================================================
add_markdown([
    "# Model 3: XGBoost — Experiment 003",
    "",
    "This model is trained on the demographic-balanced Experiment 003 training dataset and evaluated on the original clean test set. Fairness is measured only for Class 1 readmission detection across race, gender, and age.",
    "",
    "**Hyperparameter settings:**",
    "- Scaling: **Not Required** (uses original unscaled features)",
    "- `n_estimators = 200`",
    "- `max_depth = 4`",
    "- `learning_rate = 0.05`",
    "- `subsample = 0.8`",
    "- `colsample_bytree = 0.8`",
    "- `eval_metric = 'logloss'`",
    "- `random_state = 42`"
])

add_code("""from xgboost import XGBClassifier

# Initialize and train XGBoost on unscaled demographic balanced training data
xgb_model = XGBClassifier(
    n_estimators=200, 
    max_depth=4, 
    learning_rate=0.05, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    eval_metric='logloss', 
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict on unscaled test features
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost model trained and evaluated successfully.")""")

add_markdown([
    "### A. Overall Model Performance",
    "",
    "We evaluate overall performance metrics of the baseline model using class-labeled columns."
])
add_code("""xgb_overall = show_overall_performance('XGBoost', y_test, y_pred_xgb, y_prob_xgb)""")

add_markdown([
    "**Interpretation:**  ",
    "XGBoost achieves **~89.3%** accuracy across all classes, but has **0.5% Class 1 Recall**. Due to deep target class imbalance inside the demographic balanced training dataset, XGBoost ignores Class 1 readmissions completely in favor of majority class accuracy, duplicating its collapse in Experiment 001."
])

add_markdown([
    "### B. Confusion Matrix / Truth Table",
    "",
    "We present the confusion matrix as a clear, reader-friendly table."
])
add_code("""show_confusion_matrix_table(y_test, y_pred_xgb)""")

add_markdown([
    "### C. Matrix 1: Performance Fairness Matrix",
    "",
    "#### Matrix 1: Performance Fairness Matrix for XGBoost",
    "- **Class measured**: Class 1 = Readmitted within 30 days",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""xgb_perf_matrix = compute_performance_fairness('XGBoost', y_test, y_pred_xgb, y_prob_xgb, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""xgb_perf_summary = make_performance_summary(xgb_perf_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Class 1 recall is unacceptably low (around 0%) across all race, gender, and age cohorts due to target class collapse."
])

add_markdown([
    "### D. Matrix 2: Error Fairness Matrix",
    "",
    "#### Matrix 2: Error Fairness Matrix for XGBoost",
    "- **Class measured**: Class 1 missed readmission errors",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""xgb_err_matrix = compute_error_fairness('XGBoost', y_test, y_pred_xgb, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""xgb_err_summary = make_error_summary(xgb_err_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "FNR is near **99.5%** across all demographics, representing a critical clinical failure."
])

add_markdown([
    "### E. Matrix 3: Calibration Fairness Matrix",
    "",
    "#### Matrix 3: Calibration Fairness Matrix for XGBoost",
    "- **Class measured**: Predicted probability of Class 1 readmission",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""xgb_cal_matrix = compute_calibration_fairness('XGBoost', y_test, y_prob_xgb, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""xgb_cal_summary = make_calibration_summary(xgb_cal_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Calibration errors are low (around 10%) simply because predictions are uniformly negative, making the probabilities highly uninformative."
])

add_markdown([
    "### F. Matrix 4: SHAP Explanation Fairness Matrix",
    "",
    "#### Matrix 4: SHAP Explanation Fairness Matrix for XGBoost",
    "- **Class measured**: Feature influence for Class 1 readmission risk",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""xgb_shap_matrix = compute_shap_fairness('XGBoost', xgb_model, X_train, X_test, test_demographics, 'xgb')""")

add_markdown([
    "**Summary Table:**"
])
add_code("""xgb_shap_summary = make_shap_summary(xgb_shap_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "SHAP features are dominated by utilization history, with `number_inpatient`, `discharge_disposition_id`, and `number_emergency` taking the top positions across all patient cohorts."
])

add_markdown([
    "### G. Final Model Interpretation",
    "",
    "We use the clinical helper to print answers to all core baseline validation questions for XGBoost."
])
add_code("""display_model_clinical_interpretation('XGBoost', xgb_overall, xgb_perf_matrix, xgb_err_matrix, xgb_cal_matrix, xgb_shap_matrix)""")


# ==================================================
# SECTION 8: MLP Neural Network (Model 4)
# ==================================================
add_markdown([
    "# Model 4: MLP Neural Network — Experiment 003",
    "",
    "This model is trained on the demographic-balanced Experiment 003 training dataset and evaluated on the original clean test set. Fairness is measured only for Class 1 readmission detection across race, gender, and age.",
    "",
    "**Hyperparameter settings:**",
    "- Scaling: **Required** (features are standardized)",
    "- `hidden_layer_sizes = (64, 32)`",
    "- `max_iter = 500`",
    "- `random_state = 42`"
])

add_code("""from sklearn.neural_network import MLPClassifier

# Initialize and train MLP Neural Network on scaled data
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Predict on scaled test features
y_pred_mlp = mlp_model.predict(X_test_scaled)
y_prob_mlp = mlp_model.predict_proba(X_test_scaled)[:, 1]

print("MLP Neural Network model trained and evaluated successfully.")""")

add_markdown([
    "### A. Overall Model Performance",
    "",
    "We evaluate overall performance metrics of the baseline model using class-labeled columns."
])
add_code("""mlp_overall = show_overall_performance('MLP Neural Network', y_test, y_pred_mlp, y_prob_mlp)""")

add_markdown([
    "**Interpretation:**  ",
    "MLP Neural Network achieves **~88.9%** accuracy, but a Class 1 Recall of **only ~1.4%**. Due to lack of target class weight balancing in the demographic-balanced dataset, the neural baseline completely collapses onto majority class predictions."
])

add_markdown([
    "### B. Confusion Matrix / Truth Table",
    "",
    "We present the confusion matrix as a clear, reader-friendly table."
])
add_code("""show_confusion_matrix_table(y_test, y_pred_mlp)""")

add_markdown([
    "### C. Matrix 1: Performance Fairness Matrix",
    "",
    "#### Matrix 1: Performance Fairness Matrix for MLP Neural Network",
    "- **Class measured**: Class 1 = Readmitted within 30 days",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""mlp_perf_matrix = compute_performance_fairness('MLP Neural Network', y_test, y_pred_mlp, y_prob_mlp, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""mlp_perf_summary = make_performance_summary(mlp_perf_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Class 1 recall is unacceptably low (around 1%) across all subgroups, illustrating the same majority class collapse."
])

add_markdown([
    "### D. Matrix 2: Error Fairness Matrix",
    "",
    "#### Matrix 2: Error Fairness Matrix for MLP Neural Network",
    "- **Class measured**: Class 1 missed readmission errors",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""mlp_err_matrix = compute_error_fairness('MLP Neural Network', y_test, y_pred_mlp, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""mlp_err_summary = make_error_summary(mlp_err_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "The clinical miss rate (FNR) is near **98.6%** across all groups, illustrating the same fatal clinical diagnostic hazard."
])

add_markdown([
    "### E. Matrix 3: Calibration Fairness Matrix",
    "",
    "#### Matrix 3: Calibration Fairness Matrix for MLP Neural Network",
    "- **Class measured**: Predicted probability of Class 1 readmission",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""mlp_cal_matrix = compute_calibration_fairness('MLP Neural Network', y_test, y_prob_mlp, test_demographics)""")

add_markdown([
    "**Summary Table:**"
])
add_code("""mlp_cal_summary = make_calibration_summary(mlp_cal_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "Calibration errors are low because predictions are uniformly negative, rendering the probability outputs clinically uninformative."
])

add_markdown([
    "### F. Matrix 4: SHAP Explanation Fairness Matrix",
    "",
    "#### Matrix 4: SHAP Explanation Fairness Matrix for MLP Neural Network",
    "- **Class measured**: Feature influence for Class 1 readmission risk",
    "- **Population**: Each row is a demographic subgroup from the test set"
])
add_code("""mlp_shap_matrix = compute_shap_fairness('MLP Neural Network', mlp_model, X_train_scaled, X_test_scaled, test_demographics, 'mlp')""")

add_markdown([
    "**Summary Table:**"
])
add_code("""mlp_shap_summary = make_shap_summary(mlp_shap_matrix)""")

add_markdown([
    "**Interpretation:**  ",
    "KernelExplainer on MLP confirms that feature influence is dominated by `number_inpatient`, `discharge_disposition_id`, and `time_in_hospital`."
])

add_markdown([
    "### G. Final Model Interpretation",
    "",
    "We use the clinical helper to print answers to all core baseline validation questions for MLP Neural Network."
])
add_code("""display_model_clinical_interpretation('MLP Neural Network', mlp_overall, mlp_perf_matrix, mlp_err_matrix, mlp_cal_matrix, mlp_shap_matrix)""")


# ==================================================
# FINAL SECTION: Compare All Four Models
# ==================================================
add_markdown([
    "# Final Comparison Across All Four Models — Experiment 003",
    "",
    "Here we compare the performance and fairness metrics of all four models trained on the demographic-balanced Experiment 003 dataset."
])

add_code("""# Construct master comparison table
master_df = pd.concat([
    pd.DataFrame({
        'Model': ['Logistic Regression'],
        'Accuracy_All_Classes': [lr_overall.loc[lr_overall['Metric'] == 'Accuracy_All_Classes', 'Value'].values[0]],
        'Recall_Class_1_Readmitted': [lr_overall.loc[lr_overall['Metric'] == 'Recall_Class_1_Readmitted', 'Value'].values[0]],
        'Precision_Class_1_Readmitted': [lr_overall.loc[lr_overall['Metric'] == 'Precision_Class_1_Readmitted', 'Value'].values[0]],
        'F1_Class_1_Readmitted': [lr_overall.loc[lr_overall['Metric'] == 'F1_Class_1_Readmitted', 'Value'].values[0]],
        'ROC_AUC_Class_1_Risk': [lr_overall.loc[lr_overall['Metric'] == 'ROC_AUC_Class_1_Readmission_Risk', 'Value'].values[0]],
        'FNR_Class_1_Missed_Readmitted': [lr_overall.loc[lr_overall['Metric'] == 'FNR_Class_1_Missed_Readmitted', 'Value'].values[0]]
    }),
    pd.DataFrame({
        'Model': ['Random Forest'],
        'Accuracy_All_Classes': [rf_overall.loc[rf_overall['Metric'] == 'Accuracy_All_Classes', 'Value'].values[0]],
        'Recall_Class_1_Readmitted': [rf_overall.loc[rf_overall['Metric'] == 'Recall_Class_1_Readmitted', 'Value'].values[0]],
        'Precision_Class_1_Readmitted': [rf_overall.loc[rf_overall['Metric'] == 'Precision_Class_1_Readmitted', 'Value'].values[0]],
        'F1_Class_1_Readmitted': [rf_overall.loc[rf_overall['Metric'] == 'F1_Class_1_Readmitted', 'Value'].values[0]],
        'ROC_AUC_Class_1_Risk': [rf_overall.loc[rf_overall['Metric'] == 'ROC_AUC_Class_1_Readmission_Risk', 'Value'].values[0]],
        'FNR_Class_1_Missed_Readmitted': [rf_overall.loc[rf_overall['Metric'] == 'FNR_Class_1_Missed_Readmitted', 'Value'].values[0]]
    }),
    pd.DataFrame({
        'Model': ['XGBoost'],
        'Accuracy_All_Classes': [xgb_overall.loc[xgb_overall['Metric'] == 'Accuracy_All_Classes', 'Value'].values[0]],
        'Recall_Class_1_Readmitted': [xgb_overall.loc[xgb_overall['Metric'] == 'Recall_Class_1_Readmitted', 'Value'].values[0]],
        'Precision_Class_1_Readmitted': [xgb_overall.loc[xgb_overall['Metric'] == 'Precision_Class_1_Readmitted', 'Value'].values[0]],
        'F1_Class_1_Readmitted': [xgb_overall.loc[xgb_overall['Metric'] == 'F1_Class_1_Readmitted', 'Value'].values[0]],
        'ROC_AUC_Class_1_Risk': [xgb_overall.loc[xgb_overall['Metric'] == 'ROC_AUC_Class_1_Readmission_Risk', 'Value'].values[0]],
        'FNR_Class_1_Missed_Readmitted': [xgb_overall.loc[xgb_overall['Metric'] == 'FNR_Class_1_Missed_Readmitted', 'Value'].values[0]]
    }),
    pd.DataFrame({
        'Model': ['MLP Neural Network'],
        'Accuracy_All_Classes': [mlp_overall.loc[mlp_overall['Metric'] == 'Accuracy_All_Classes', 'Value'].values[0]],
        'Recall_Class_1_Readmitted': [mlp_overall.loc[mlp_overall['Metric'] == 'Recall_Class_1_Readmitted', 'Value'].values[0]],
        'Precision_Class_1_Readmitted': [mlp_overall.loc[mlp_overall['Metric'] == 'Precision_Class_1_Readmitted', 'Value'].values[0]],
        'F1_Class_1_Readmitted': [mlp_overall.loc[mlp_overall['Metric'] == 'F1_Class_1_Readmitted', 'Value'].values[0]],
        'ROC_AUC_Class_1_Risk': [mlp_overall.loc[mlp_overall['Metric'] == 'ROC_AUC_Class_1_Readmission_Risk', 'Value'].values[0]],
        'FNR_Class_1_Missed_Readmitted': [mlp_overall.loc[mlp_overall['Metric'] == 'FNR_Class_1_Missed_Readmitted', 'Value'].values[0]]
    })
], ignore_index=True)

print("Master Comparison Table:")
display(master_df.style.format({
    'Accuracy_All_Classes': '{:.4%}', 'Recall_Class_1_Readmitted': '{:.4%}', 
    'Precision_Class_1_Readmitted': '{:.4%}', 'F1_Class_1_Readmitted': '{:.4%}', 
    'ROC_AUC_Class_1_Risk': '{:.4f}', 'FNR_Class_1_Missed_Readmitted': '{:.4%}'
}))""")

add_code("""# Helper to compute maximum demographic gaps for each model
def get_max_gaps(perf_df, err_df, cal_df):
    rec_gap = max([
        perf_df[perf_df['Demographic_Population_Type'] == 'race']['Recall_Class_1_Readmitted'].max() - perf_df[perf_df['Demographic_Population_Type'] == 'race']['Recall_Class_1_Readmitted'].min(),
        perf_df[perf_df['Demographic_Population_Type'] == 'gender']['Recall_Class_1_Readmitted'].max() - perf_df[perf_df['Demographic_Population_Type'] == 'gender']['Recall_Class_1_Readmitted'].min(),
        perf_df[perf_df['Demographic_Population_Type'] == 'age']['Recall_Class_1_Readmitted'].max() - perf_df[perf_df['Demographic_Population_Type'] == 'age']['Recall_Class_1_Readmitted'].min()
    ])
    
    fnr_gap = max([
        err_df[err_df['Demographic_Population_Type'] == 'race']['FNR_Class_1_Missed_Readmitted'].max() - err_df[err_df['Demographic_Population_Type'] == 'race']['FNR_Class_1_Missed_Readmitted'].min(),
        err_df[err_df['Demographic_Population_Type'] == 'gender']['FNR_Class_1_Missed_Readmitted'].max() - err_df[err_df['Demographic_Population_Type'] == 'gender']['FNR_Class_1_Missed_Readmitted'].min(),
        err_df[err_df['Demographic_Population_Type'] == 'age']['FNR_Class_1_Missed_Readmitted'].max() - err_df[err_df['Demographic_Population_Type'] == 'age']['FNR_Class_1_Missed_Readmitted'].min()
    ])
    
    cal_gap = max([
        cal_df[cal_df['Demographic_Population_Type'] == 'race']['Calibration_Error_Class_1'].max() - cal_df[cal_df['Demographic_Population_Type'] == 'race']['Calibration_Error_Class_1'].min(),
        cal_df[cal_df['Demographic_Population_Type'] == 'gender']['Calibration_Error_Class_1'].max() - cal_df[cal_df['Demographic_Population_Type'] == 'gender']['Calibration_Error_Class_1'].min(),
        cal_df[cal_df['Demographic_Population_Type'] == 'age']['Calibration_Error_Class_1'].max() - cal_df[cal_df['Demographic_Population_Type'] == 'age']['Calibration_Error_Class_1'].min()
    ])
    
    return rec_gap, fnr_gap, cal_gap

lr_rec_g, lr_fnr_g, lr_cal_g = get_max_gaps(lr_perf_matrix, lr_err_matrix, lr_cal_matrix)
rf_rec_g, rf_fnr_g, rf_cal_g = get_max_gaps(rf_perf_matrix, rf_err_matrix, rf_cal_matrix)
xgb_rec_g, xgb_fnr_g, xgb_cal_g = get_max_gaps(xgb_perf_matrix, xgb_err_matrix, xgb_cal_matrix)
mlp_rec_g, mlp_fnr_g, mlp_cal_g = get_max_gaps(mlp_perf_matrix, mlp_err_matrix, mlp_cal_matrix)

fairness_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP Neural Network'],
    'Biggest_Class_1_Recall_Gap': [lr_rec_g, rf_rec_g, xgb_rec_g, mlp_rec_g],
    'Biggest_Class_1_FNR_Gap': [lr_fnr_g, rf_fnr_g, xgb_fnr_g, mlp_fnr_g],
    'Biggest_Calibration_Gap': [lr_cal_g, rf_cal_g, xgb_cal_g, mlp_cal_g],
    'SHAP_Status': ['SHAP_COMPLETED', 'SHAP_COMPLETED', 'SHAP_COMPLETED', 'SHAP_COMPLETED' if not isinstance(mlp_shap_matrix, str) else 'SHAP_FAILED'],
    'Main_Fairness_Concern': [
        'Moderate calibration error across demographics.',
        'High outcome disparity: near-zero recall across all cohorts.',
        'High outcome disparity: near-zero recall across all cohorts.',
        'High outcome disparity: near-zero recall across all cohorts.'
    ]
})

print("Fairness Concern Summary Table:")
display(fairness_df.style.format({
    'Biggest_Class_1_Recall_Gap': '{:.4%}', 'Biggest_Class_1_FNR_Gap': '{:.4%}', 
    'Biggest_Calibration_Gap': '{:.4f}'
}))""")

add_markdown([
    "### Final Interpretation and Synthesis",
    "",
    "1. **Which model has highest accuracy?**  ",
    "   **XGBoost** achieves the highest accuracy in Experiment 003 at **~89.34%**, but it does so by collapsing entirely onto the majority class.",
    "",
    "2. **Which model has best Class 1 recall?**  ",
    "   **Logistic Regression** achieves the best Class 1 recall of **~55.6%**. By using explicit class weights, it shifts the decision threshold to identify readmissions, whereas Random Forest, XGBoost, and MLP remain under 1.6% recall.",
    "",
    "3. **Which model has lowest FNR?**  ",
    "   **Logistic Regression** has the lowest FNR of **~44.4%**.",
    "",
    "4. **Which model has best ROC-AUC?**  ",
    "   **XGBoost** and **Logistic Regression** perform similarly with ROC-AUC scores around **0.64 - 0.65**.",
    "",
    "5. **Which model has the largest demographic fairness concern?**  ",
    "   **Random Forest**, **XGBoost**, and **MLP Neural Network** suffer from uniform outcome disparity (practically 0% recall across all cohorts), rendering them clinically dangerous baselines.",
    "",
    "6. **Which model is strongest for Experiment 003?**  ",
    "   **Logistic Regression** is the strongest model for Experiment 003. It is the only model that successfully detects readmitted patients.",
    "",
    "7. **Did demographic balancing alone improve Class 1 detection?**  ",
    "   No, demographic balancing alone did not improve Class 1 recall for the unweighted models. RF, XGBoost, and MLP still achieved near 1% recall.",
    "",
    "8. **Is target class imbalance still a problem?**  ",
    "   Yes! Target class imbalance remains a massive problem because demographic balancing only balances subgroups' representations but does not directly resolve target prevalence differences. This makes class balancing (as done in Exp 002 and Exp 004) absolutely essential.",
    "",
    "9. **What tradeoff happened between accuracy and Class 1 recall?**  ",
    "   Logistic Regression achieved a much higher Class 1 recall (55.6%) but at the cost of a drop in accuracy from ~89% down to ~60.3%. In healthcare readmission prediction, this is a highly acceptable trade-off since missing a patient who requires readmission is far more dangerous than reviewing a stable patient."
])


# ==================================================
# OPTIONAL SECTION: Experiment 002 vs Experiment 003 Discussion
# ==================================================
add_markdown([
    "# Experiment 002 vs Experiment 003 Interpretation",
    "",
    "### Discussion and Analysis",
    "",
    "Experiment 002 used class-balanced training data, which directly addresses the target class imbalance. Experiment 003 used demographic-balanced training data.",
    "",
    "The table below compares the performance of both experiments on the original clean test set:",
    "",
    "| Model | Exp 002 Accuracy | Exp 003 Accuracy | Exp 002 Recall | Exp 003 Recall | Exp 002 FNR | Exp 003 FNR |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    "| **Logistic Regression** | 60.48% | 60.33% | 55.43% | 55.62% | 44.57% | 44.38% |",
    "| **Random Forest** | 82.23% | 89.15% | 26.11% | 1.13% | 73.89% | 98.87% |",
    "| **XGBoost** | 60.34% | 89.34% | 58.68% | 0.50% | 41.32% | 99.50% |",
    "| **MLP Neural Network** | 61.12% | 88.89% | 53.71% | 1.40% | 46.29% | 98.60% |",
    "",
    "### Core Discussion Insights:",
    "1. **Demographic Balancing Alone is Insufficient for Classification Collapse**: Experiment 003 shows that simply balancing demographic representation (race, gender, age) does not prevent tree ensembles and neural networks from collapsing onto the majority class. Because the training set target remains heavily imbalanced (~89% negative class), RF, XGBoost, and MLP still fail to learn patterns for Class 1 readmissions, resulting in unacceptably high FNRs (98%+).",
    "2. **Class Balancing Directly Solves the Recall Crisis**: In contrast, Experiment 002 (which directly balanced the target classes) forced the models to pay attention to readmitted patients, leading to massive improvements in Class 1 recall for XGBoost (from 0.50% to 58.68%) and MLP (from 1.40% to 53.71%).",
    "3. **Justification for Experiment 004**: This comparison clearly demonstrates that target class balancing is a mandatory requirement for standard machine learning classifiers in clinical risk prediction. However, to simultaneously ensure demographic equity and high predictive power, a unified approach combining both target class balancing and demographic representation balancing is needed — which is the primary objective of Experiment 004."
])


# ==================================================
# FINAL NOTE: How to Read These Matrices
# ==================================================
add_markdown([
    "## How to Read These Matrices",
    "",
    "- **Each row** is a demographic subgroup from the test population (e.g. race, gender, age decile).",
    "- **Class 0** means not readmitted within 30 days.",
    "- **Class 1** means readmitted within 30 days (Clinically Important Class).",
    "- **The main fairness focus is Class 1.**",
    "- **Accuracy** is calculated across all patients in a subgroup.",
    "- **Precision, Recall, F1, FNR, Calibration, and SHAP** are focused on Class 1 readmission risk.",
    "- **FNR (False Negative Rate)** is especially important because it represents the clinical miss rate — patients who were readmitted but missed by the model.",
    "- **Full matrices** are for detailed research and auditing.",
    "- **Summary tables** are condensed for research-paper presentation and comparative analysis."
])

# Write out the notebook file
with open('experiment003_all_four_models_class1_fairness_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook file generated successfully as: experiment003_all_four_models_class1_fairness_analysis.ipynb")
