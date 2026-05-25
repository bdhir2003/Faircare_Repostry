#!/usr/bin/env python3
"""
generate_dashboard.py — FairCare Research-Ready Dashboard Builder v3

Restructures navigation into 10 clean sections, adds Research Results &
Fairness Interpretation section, and uses collapsible matrices in the
experiment viewer.
"""
import json, os

# ─── Load data ───
def load_experiment_data():
    with open('faircare_dashboard/experiment_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_fairness_gaps():
    path = 'faircare_dashboard/fairness_gaps.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def load_group_findings():
    path = 'faircare_dashboard/group_findings.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

# ─── Also extract raw HTML tables from notebooks ───
def get_html_tables(path):
    tables = []
    if not os.path.exists(path):
        return tables
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            for out in cell.get('outputs', []):
                if out.get('output_type') == 'display_data':
                    data = out.get('data', {})
                    if 'text/html' in data:
                        h = "".join(data['text/html']).replace('<table', '<table class="data-table"')
                        tables.append(h)
    return tables

# ─── Experiment metadata ───
EXP_META = {
    1: {'title': 'Raw Cleaned Baseline', 'dataset': 'clean_train_dataset.csv', 'rows': '81,474', 'c0': '72,286 (88.72%)', 'c1': '9,188 (11.28%)', 'strategy': 'No balancing', 'question': 'What happens if we train directly on the original cleaned data?', 'why': 'Establishes the baseline. Shows how models behave on real-world imbalanced healthcare data.'},
    2: {'title': 'Class-Balanced Training', 'dataset': 'balance_classes_training_dataset.csv', 'rows': '18,376', 'c0': '9,188 (50.00%)', 'c1': '9,188 (50.00%)', 'strategy': 'Downsample Class 0 to match Class 1', 'question': 'Does balancing the target class improve detection of readmitted patients?', 'why': 'Tests whether equal class exposure forces models to learn readmission patterns.'},
    3: {'title': 'Demographic-Balanced Training', 'dataset': 'balance_demographic_variable_training_dataset.csv', 'rows': '33,390', 'c0': '29,606 (88.67%)', 'c1': '3,784 (11.33%)', 'strategy': 'Soft cap demographics, class imbalance kept', 'question': 'Is demographic balancing alone enough to improve Class 1 fairness?', 'why': 'Isolates demographic representation without addressing target imbalance.'},
    4: {'title': 'Class + Soft Demographic Balancing', 'dataset': 'Balanceclasses+softbalanced_demographic_groups.csv', 'rows': '7,568', 'c0': '3,784 (50.00%)', 'c1': '3,784 (50.00%)', 'strategy': 'Class balanced + soft demographic balancing', 'question': 'Can combining class and demographic balancing improve both detection and fairness?', 'why': 'Most comprehensive data preparation strategy.'},
}
MODEL_NAMES = ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP Neural Network']
MODEL_IDS = ['lr', 'rf', 'xgb', 'mlp']
NB_FILES = [
    "experiment001_all_four_models_class1_fairness_analysis.ipynb",
    "experiment002_all_four_models_class1_fairness_analysis.ipynb",
    "experiment003_all_four_models_class1_fairness_analysis.ipynb",
    "experiment004_all_four_models_class1_fairness_analysis.ipynb",
]

def interpret_short(acc, recall, fnr):
    if recall < 0.02:
        if acc > 0.85: return "Accuracy misleading; near-zero Class 1 detection."
        return "Very weak Class 1 detection."
    elif recall < 0.10:
        if acc > 0.85: return "Accuracy misleading; very weak Class 1 detection."
        return "Very weak Class 1 detection."
    elif recall < 0.40:
        return "Limited Class 1 detection."
    elif recall < 0.55:
        if acc < 0.65: return "Moderate detection; recall improved with accuracy tradeoff."
        return "Moderate Class 1 detection."
    elif recall < 0.65:
        if acc < 0.65: return "Recall improved with accuracy tradeoff."
        return "Moderate-to-strong Class 1 detection."
    else:
        return "Strong Class 1 detection; check precision tradeoff."

def val_class(v, thresholds):
    """Return CSS class for a value: val-bad, val-mid, val-good."""
    lo, hi = thresholds
    if v < lo: return 'val-bad'
    if v < hi: return 'val-mid'
    return 'val-good'

def pct(v):
    return f"{v*100:.2f}%"

def interpret_model_long(name, kv):
    r = kv['recall_class1']; a = kv['accuracy']; fnr = kv['fnr']
    lines = []
    if r < 0.02:
        lines.append(f"{name} catches almost no readmitted patients (Recall: {pct(r)}). Despite {pct(a)} accuracy, it essentially predicts only Class 0.")
    elif r < 0.10:
        lines.append(f"{name} catches very few readmissions (Recall: {pct(r)}). FNR of {pct(fnr)} means nearly all are missed.")
    elif r < 0.55:
        lines.append(f"{name} catches about half of readmissions (Recall: {pct(r)}). FNR of {pct(fnr)} means roughly half are missed.")
    else:
        lines.append(f"{name} catches a majority of readmissions (Recall: {pct(r)}). FNR of {pct(fnr)} shows meaningful improvement.")
    if a > 0.85 and r < 0.10:
        lines.append(f"High accuracy ({pct(a)}) is misleading — achieved by predicting mostly Class 0.")
    if a < 0.65 and r > 0.50:
        lines.append(f"Lower accuracy ({pct(a)}) reflects the standard tradeoff for better Class 1 detection.")
    return " ".join(lines)

# ═══════════════════════ BUILD HTML ═══════════════════════

def build_model_viewer(exp_num, model_idx, model_name, tables, kv):
    """Build model section with collapsible matrices."""
    t_off = model_idx * 10
    interp = interpret_model_long(model_name, kv)
    if len(tables) <= t_off + 9:
        return '<p>Output not available in saved notebook state.</p>'

    def collapsible(title, badge_class, badge_text, content_html):
        return f'''
        <button class="collapsible-trigger"><span><span class="badge {badge_class}">{badge_text}</span> {title}</span><span class="chevron">▼</span></button>
        <div class="collapsible-content">{content_html}</div>'''

    perf_block = f'''
        <div class="explain-box"><h5>What This Table Shows</h5><p>Overall model performance. Key columns: <span class="badge badge-cyan">Class 1 Recall</span> and <span class="badge badge-rose">FNR</span>.</p></div>
        <div class="table-wrapper">{tables[t_off]}</div>
        <div class="interpret-box"><h5>Interpretation</h5><p>{interp}</p></div>'''

    cm_content = f'''
        <div class="explain-box"><h5>How to Read</h5><p>TN = correct not-readmitted, FP = false alarm, <strong>FN = missed readmission (most critical)</strong>, TP = caught readmission.</p></div>
        <div class="table-wrapper">{tables[t_off+1]}</div>'''

    def matrix_block(idx, title, before_text, after_text):
        return f'''
        <div class="explain-box"><h5>What This Shows</h5><p>{before_text}</p></div>
        <div class="table-wrapper scroll-x">{tables[t_off+idx]}</div>
        <div class="interpret-box"><h5>Summary</h5><p>{after_text}</p></div>
        <div class="table-wrapper scroll-x">{tables[t_off+idx+1]}</div>'''

    return f'''
    <div class="model-results-container">
        <h3 class="model-title">{model_name}</h3>
        {perf_block}
        {collapsible("Confusion Matrix", "badge-amber", "CM", cm_content)}
        {collapsible("Performance Fairness Matrix", "badge-cyan", "M1", matrix_block(2, "Performance Fairness", "Compares Class 1 Recall, Precision, F1 across race, gender, age. Lower recall = fewer readmissions caught for that group.", "The summary below shows best/worst groups. A large recall gap indicates uneven detection."))}
        {collapsible("Error Fairness Matrix", "badge-rose", "M2", matrix_block(4, "Error Fairness", "Shows FNR and FN counts per group. High FNR = more readmissions missed.", "Groups with high FNR need attention — missed patients may lack follow-up care."))}
        {collapsible("Calibration Fairness Matrix", "badge-amber", "M3", matrix_block(6, "Calibration Fairness", "Checks if predicted risk scores match actual readmission rates per group.", "High calibration error means predicted risk scores are unreliable for that group."))}
        {collapsible("SHAP Explanation Fairness Matrix", "badge-purple", "M4", matrix_block(8, "SHAP Explanation", "Shows which features drive Class 1 prediction per group. If sensitive features dominate, it raises fairness concerns.", "Different top features across groups may indicate the model uses different reasoning patterns."))}
    </div>'''

def build_experiment_viewer(exp_num, nb_path, exp_data):
    """Section 8: Full Experiment Viewer."""
    exp = EXP_META[exp_num]
    tables = get_html_tables(nb_path)

    tabs_html = '<div class="tab-container">\n<div class="tab-buttons">\n'
    panes_html = ''
    for i, (mname, mid) in enumerate(zip(MODEL_NAMES, MODEL_IDS)):
        active = ' active' if i == 0 else ''
        tabs_html += f'<button class="tab-btn{active}" data-tab="ev{exp_num}-{mid}">{mname}</button>\n'
        kv = exp_data['models'][mname].get('key_values', {})
        panes_html += f'<div class="tab-pane{active}" id="ev{exp_num}-{mid}">\n{build_model_viewer(exp_num, i, mname, tables, kv)}\n</div>\n'
    tabs_html += '</div>\n'

    return f'''
    <div class="research-card mt-2">
        <h3>Experiment 00{exp_num}: {exp['title']}</h3>
        <div class="exp-setup mt-1">
            <div><div class="exp-setup-label">Training Data</div><div class="exp-setup-value">{exp['dataset']}</div></div>
            <div><div class="exp-setup-label">Rows</div><div class="exp-setup-value">{exp['rows']}</div></div>
            <div><div class="exp-setup-label">Class 0 / Class 1</div><div class="exp-setup-value">{exp['c0']} / {exp['c1']}</div></div>
            <div><div class="exp-setup-label">Strategy</div><div class="exp-setup-value">{exp['strategy']}</div></div>
        </div>
        <div class="explain-box"><h5>Research Question</h5><p>{exp['question']}</p></div>
        {tabs_html}{panes_html}</div>
    </div>'''

def perf_meaning(recall, fnr):
    if recall < 0.02: return 'Misses nearly all readmitted patients.'
    if recall < 0.10: return 'Misses most readmitted patients.'
    if recall < 0.40: return 'Catches some, but still misses many.'
    if recall < 0.55: return 'Catches about half of readmitted patients.'
    if recall < 0.65: return 'Catches more readmitted patients; check false alarms.'
    return 'Strong detection; check precision tradeoff.'

def gap_meaning(r_gap, c_gap):
    parts = []
    if r_gap is None: return 'Gaps need notebook review.'
    if r_gap > 0.60: parts.append('Large recall gap across groups.')
    elif r_gap > 0.30: parts.append('Moderate recall gap.')
    else: parts.append('Smaller recall gap.')
    if c_gap is not None and c_gap > 0.15: parts.append('Calibration varies across groups.')
    parts.append('Review subgroup matrices.')
    return ' '.join(parts)

def generate():
    data = load_experiment_data()
    gaps = load_fairness_gaps()
    gf = load_group_findings()

    # Build 16-model results table rows
    results_rows = ''
    for exp_num in [1,2,3,4]:
        exp_key = f'exp00{exp_num}'
        exp = EXP_META[exp_num]
        for mname in MODEL_NAMES:
            kv = data[exp_key]['models'][mname].get('key_values', {})
            r = kv.get('recall_class1', 0)
            fnr_v = kv.get('fnr', 0)
            acc = kv.get('accuracy', 0)
            rcls = val_class(r, (0.10, 0.50))
            fcls = val_class(1-fnr_v, (0.10, 0.50))  # invert for color
            interp = interpret_short(acc, r, fnr_v)
            results_rows += f'''<tr>
                <td>00{exp_num}</td><td>{exp['strategy']}</td><td>{mname}</td>
                <td class="text-right">{pct(acc)}</td>
                <td class="text-right">{pct(kv.get('precision_class1',0))}</td>
                <td class="text-right {rcls}">{pct(r)}</td>
                <td class="text-right">{pct(kv.get('f1_class1',0))}</td>
                <td class="text-right">{kv.get('roc_auc','N/A')}</td>
                <td class="text-right {fcls}">{pct(fnr_v)}</td>
                <td class="interp-cell">{interp}</td>
            </tr>\n'''

    # Build experiment viewer sections
    exp_viewers = ''
    for i in range(4):
        exp_viewers += build_experiment_viewer(i+1, NB_FILES[i], data[f'exp00{i+1}'])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FairCare: Research Results Viewer</title>
    <link rel="stylesheet" href="style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
</head>
<body>

<header class="header-nav">
    <div class="nav-container">
        <a href="#hero" class="nav-brand"><span class="nav-brand-dot"></span><span>FairCare</span></a>
        <ul class="nav-menu">
            <li><a href="#hero" class="nav-link active">Overview</a></li>
            <li><a href="#dataset" class="nav-link">Dataset & Cleaning</a></li>
            <li><a href="#features" class="nav-link">Features</a></li>
            <li><a href="#files" class="nav-link">Files</a></li>
            <li><a href="#design" class="nav-link">Experiment Design</a></li>
            <li><a href="#architecture" class="nav-link">Architecture</a></li>
            <li><a href="#results" class="nav-link">Results</a></li>
            <li><a href="#viewer" class="nav-link">Experiment Viewer</a></li>
            <li><a href="#fairness-measures" class="nav-link">Fairness Findings</a></li>
            <li><a href="#matrices-explained" class="nav-link">Matrices Explained</a></li>
            <li><a href="#guide" class="nav-link">Guide</a></li>
            <li><a href="#next-steps" class="nav-link">Next Steps</a></li>
        </ul>
    </div>
</header>

<!-- ═══════ 1. OVERVIEW ═══════ -->
<section id="hero" class="hero-section">
    <div class="container">
        <h1 class="gradient-text">FairCare</h1>
        <div class="hero-subtitle">Fairness-Aware Healthcare Readmission Prediction</div>
        <p class="hero-description">FairCare is a <strong>fairness auditing project</strong>. It checks whether ML models detect 30-day diabetic readmissions fairly across race, gender, and age groups.</p>
        <div class="summary-cards">
            <div class="summary-card"><div class="summary-card-label"><i class="fa-solid fa-circle-question"></i> Research Question</div><div class="summary-card-value">Can we predict readmission within 30 days and ensure the model treats demographic groups fairly?</div></div>
            <div class="summary-card"><div class="summary-card-label"><i class="fa-solid fa-heart-pulse"></i> Clinical Focus</div><div class="summary-card-value">Class 1 = Readmitted within 30 days</div></div>
            <div class="summary-card"><div class="summary-card-label"><i class="fa-solid fa-scale-balanced"></i> Fairness Question</div><div class="summary-card-value">Does the model catch Class 1 patients equally across race, gender, and age?</div></div>
        </div>
        <div class="table-wrapper mt-2"><table><thead><tr><th>Class</th><th>Meaning</th><th>Role</th></tr></thead><tbody>
            <tr><td class="val-bad">Class 0</td><td>Not readmitted within 30 days</td><td>Negative class</td></tr>
            <tr><td class="val-good">Class 1</td><td>Readmitted within 30 days</td><td>Primary clinical risk class — focus of all fairness analyses</td></tr>
        </tbody></table></div>
        <div class="summary-cards mt-2">
            <div class="summary-card"><div class="summary-card-label">Target</div><div class="summary-card-value">readmitted</div></div>
            <div class="summary-card"><div class="summary-card-label">Fairness Groups</div><div class="summary-card-value">race, gender, age</div></div>
            <div class="summary-card"><div class="summary-card-label">Models × Experiments</div><div class="summary-card-value">4 models × 4 experiments = 16 trained models</div></div>
            <div class="summary-card"><div class="summary-card-label">Fairness Framework</div><div class="summary-card-value">Performance · Error · Calibration · SHAP</div></div>
        </div>
        <div class="speaker-notes mt-2"><h5>Speaker Notes</h5><p>Emphasize FairCare is not only about accuracy. The main question is whether readmitted patients are caught fairly across demographic groups.</p></div>
    </div>
</section>

<!-- ═══════ 2. DATASET CLEANING PIPELINE ═══════ -->
<section id="dataset">
    <div class="container">
        <h2 class="section-title">Dataset Cleaning &amp; Feature Engineering Pipeline</h2>
        <p class="section-lead">How the raw diabetes dataset was transformed into model-ready experiment datasets.</p>
        <p>This section explains how FairCare transformed the raw UCI diabetes dataset into cleaned, encoded, and experiment-ready CSV files. The goal was to create a consistent modeling dataset for predicting 30-day readmission while preserving race, gender, and age for fairness analysis.</p>

        <!-- PART 1: RAW DATASET STARTING POINT -->
        <h3 class="mt-3">1. Raw Dataset Used</h3>
        <div class="research-card">
            <p>The raw dataset contains diabetic patient hospital encounters. Each row represents one hospital encounter. The raw dataset contains many original columns, but FairCare only uses a selected set of clinically meaningful and demographic features.</p>
            <div class="table-wrapper mt-2"><table class="table-compact">
                <thead><tr><th>Item</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>Raw file</td><td>diabetic_data.csv</td></tr>
                    <tr><td>Raw rows</td><td>101,766</td></tr>
                    <tr><td>Raw columns</td><td>50</td></tr>
                    <tr><td>Target column</td><td>readmitted</td></tr>
                    <tr><td>Original target values</td><td>NO (54,864), &gt;30 (35,545), &lt;30 (11,357)</td></tr>
                    <tr><td>Notebook used for explanation</td><td>faircare_raw_dataset_explanation.ipynb</td></tr>
                </tbody>
            </table></div>
        </div>

        <!-- PART 2: TARGET VARIABLE PREPARATION -->
        <h3 class="mt-3">2. Target Variable Preparation</h3>
        <div class="research-card">
            <p>The original target column is readmitted. It has three original values: NO, &gt;30, and &lt;30. FairCare converts this into a binary classification problem:</p>
            <div class="table-wrapper mt-2"><table class="table-compact">
                <thead><tr><th>Original readmitted value</th><th class="text-right">New class</th><th>Meaning</th></tr></thead>
                <tbody>
                    <tr><td>&lt;30</td><td class="text-right val-good">1</td><td>Readmitted within 30 days</td></tr>
                    <tr><td>&gt;30</td><td class="text-right val-bad">0</td><td>Readmitted after 30 days, treated as not 30-day readmission</td></tr>
                    <tr><td>NO</td><td class="text-right val-bad">0</td><td>Not readmitted</td></tr>
                </tbody>
            </table></div>
            <p class="mt-2">Class 1 is the main healthcare outcome because it means the patient came back to the hospital within 30 days.</p>
            <div class="table-wrapper mt-2"><table class="table-compact">
                <thead><tr><th>Dataset</th><th class="text-right">Class 0 Count</th><th class="text-right">Class 1 Count</th><th class="text-right">Class 0 %</th><th class="text-right">Class 1 %</th></tr></thead>
                <tbody>
                    <tr><td>diabetic_data.csv (binary)</td><td class="text-right">90,409</td><td class="text-right">11,357</td><td class="text-right">88.84%</td><td class="text-right">11.16%</td></tr>
                    <tr><td>clean_train_dataset.csv</td><td class="text-right">72,286</td><td class="text-right">9,188</td><td class="text-right">88.72%</td><td class="text-right">11.28%</td></tr>
                    <tr><td>clean_test_dataset.csv</td><td class="text-right">18,120</td><td class="text-right">2,169</td><td class="text-right">89.31%</td><td class="text-right">10.69%</td></tr>
                </tbody>
            </table></div>
        </div>

        <!-- PART 3: SELECTED 18 FEATURES -->
        <h3 class="mt-3">3. Selected 18 Features Before Encoding</h3>
        <p>FairCare selected 18 features from the original dataset before final encoding.</p>
        <div class="table-wrapper scroll-x"><table class="table-compact">
            <thead><tr><th>Original Selected Feature</th><th>Feature Group</th><th>Why It Was Used</th></tr></thead>
            <tbody>
                <tr><td>number_inpatient</td><td>Prior visit features</td><td>Indicates chronic instability</td></tr>
                <tr><td>number_emergency</td><td>Prior visit features</td><td>Unstable health conditions</td></tr>
                <tr><td>number_outpatient</td><td>Prior visit features</td><td>Ongoing care needs</td></tr>
                <tr><td>time_in_hospital</td><td>Hospital stay / admission features</td><td>Longer stays reflect severity</td></tr>
                <tr><td>admission_type_id</td><td>Hospital stay / admission features</td><td>Urgency of admission</td></tr>
                <tr><td>discharge_disposition_id</td><td>Hospital stay / admission features</td><td>Discharge destination</td></tr>
                <tr><td>num_medications</td><td>Medication and lab features</td><td>Clinical complexity</td></tr>
                <tr><td>num_lab_procedures</td><td>Medication and lab features</td><td>Medical complexity</td></tr>
                <tr><td>diabetesMed</td><td>Medication and lab features</td><td>Diabetes treatment indicator</td></tr>
                <tr><td>insulin</td><td>Medication and lab features</td><td>Treatment intensity</td></tr>
                <tr><td>metformin</td><td>Medication and lab features</td><td>Treatment adjustment</td></tr>
                <tr><td>max_glu_serum</td><td>Medication and lab features</td><td>Blood sugar levels</td></tr>
                <tr><td>A1Cresult</td><td>Medication and lab features</td><td>Long-term blood sugar control</td></tr>
                <tr><td>diag_1</td><td>Diagnosis features</td><td>Primary diagnosis</td></tr>
                <tr><td>diag_2</td><td>Diagnosis features</td><td>Secondary diagnosis</td></tr>
                <tr><td style="color:var(--accent-cyan);">race</td><td>Demographic features</td><td>Input + fairness slicing variable</td></tr>
                <tr><td style="color:var(--accent-cyan);">age</td><td>Demographic features</td><td>Input + fairness slicing variable</td></tr>
                <tr><td style="color:var(--accent-cyan);">gender</td><td>Demographic features</td><td>Input + fairness slicing variable</td></tr>
            </tbody>
        </table></div>
        <p class="mt-1">These 18 features are the human-readable features selected before model-ready encoding. After encoding, some categorical features expand into multiple columns, so the final cleaned CSV has more than 18 columns.</p>

        <!-- PART 4: CLEANING STEPS -->
        <h3 class="mt-3">4. Cleaning Steps Applied</h3>
        <div class="table-wrapper scroll-x"><table class="table-compact">
            <thead><tr><th>Step</th><th>What Happened</th><th>Source Notebook</th></tr></thead>
            <tbody>
                <tr><td>Target Binarization</td><td>Mapped '&gt;30' and 'NO' to 0, '&lt;30' to 1</td><td>Data_cleaning.ipynb</td></tr>
                <tr><td>Feature Filtering</td><td>Dropped 32 unused columns, keeping 18 selected features</td><td>Data_cleaning.ipynb</td></tr>
                <tr><td>Unknown Gender Drop</td><td>Dropped 3 rows with 'Unknown/Invalid' gender</td><td>Data_cleaning.ipynb</td></tr>
                <tr><td>Unknown Race Drop</td><td>Dropped the 'race_?' column after one-hot encoding</td><td>Data_cleaning.ipynb</td></tr>
                <tr><td>Categorical Encoding</td><td>Mapped string values to numeric representations</td><td>Data_cleaning.ipynb</td></tr>
                <tr><td>Diagnosis Grouping</td><td>Grouped ~700 ICD-9 codes into 9 categories</td><td>Data_cleaning.ipynb</td></tr>
                <tr><td>Train/Test Split</td><td>Split into train_data.csv (80%) and test_data.csv (20%)</td><td>Data_cleaning.ipynb</td></tr>
            </tbody>
        </table></div>

        <!-- PART 5: MISSING/UNKNOWN VALUES -->
        <h3 class="mt-3">5. Missing and Unknown Value Handling</h3>
        <p>Missing and unknown values matter because they can affect both prediction performance and fairness analysis.</p>
        <div class="table-wrapper scroll-x"><table class="table-compact">
            <thead><tr><th>Feature</th><th>Issue Found</th><th>Handling Method</th><th>Source</th></tr></thead>
            <tbody>
                <tr><td>gender</td><td>'Unknown/Invalid' values (3 rows)</td><td>Rows were dropped from the dataset</td><td>Data_cleaning.ipynb</td></tr>
                <tr><td>race</td><td>'?' missing values</td><td>Became 'race_?' during one-hot encoding, then the column was dropped</td><td>Data_cleaning.ipynb</td></tr>
            </tbody>
        </table></div>

        <!-- PART 6: ENCODING -->
        <h3 class="mt-3">6. Encoding and Feature Engineering</h3>
        <p>Machine learning models need numeric input. Therefore, categorical features were transformed into model-ready numeric columns.</p>
        <div class="table-wrapper scroll-x"><table class="table-compact">
            <thead><tr><th>Original Feature</th><th>Original Type</th><th>Encoding / Transformation</th><th>Final Column(s) Created</th></tr></thead>
            <tbody>
                <tr><td>race</td><td>String</td><td>One-hot encoding</td><td>race_AfricanAmerican, race_Asian, race_Caucasian, race_Hispanic, race_Other</td></tr>
                <tr><td>diag_1</td><td>ICD-9 Code</td><td>Grouped into 9 categories, then One-hot encoded</td><td>diag_1_Circulatory, diag_1_Diabetes, etc.</td></tr>
                <tr><td>diag_2</td><td>ICD-9 Code</td><td>Grouped into 9 categories, then One-hot encoded</td><td>diag_2_Circulatory, diag_2_Diabetes, etc.</td></tr>
                <tr><td>gender</td><td>String</td><td>Binary mapping (Female=0, Male=1)</td><td>gender</td></tr>
                <tr><td>age</td><td>Ordinal String</td><td>Ordinal mapping ([0-10)=0 ... [90-100)=9)</td><td>age</td></tr>
                <tr><td>diabetesMed</td><td>String</td><td>Binary mapping (No=0, Yes=1)</td><td>diabetesMed</td></tr>
                <tr><td>insulin</td><td>String</td><td>Ordinal mapping (No=0, Down=1, Steady=2, Up=3)</td><td>insulin</td></tr>
                <tr><td>metformin</td><td>String</td><td>Ordinal mapping (No=0, Down=1, Steady=2, Up=3)</td><td>metformin</td></tr>
                <tr><td>max_glu_serum</td><td>String</td><td>Ordinal mapping (None=0, Norm=1, &gt;200=2, &gt;300=3)</td><td>max_glu_serum</td></tr>
                <tr><td>A1Cresult</td><td>String</td><td>Ordinal mapping (None=0, Norm=1, &gt;7=2, &gt;8=3)</td><td>A1Cresult</td></tr>
            </tbody>
        </table></div>

        <!-- PART 7: FINAL COLUMNS -->
        <h3 class="mt-3">7. Final Model-Ready Columns After Encoding</h3>
        <p>The project starts with 18 selected features, but after encoding categorical and diagnosis variables, the final clean dataset contains more model-ready columns.</p>
        <p>The cleaned files have <strong>39 total columns</strong> (38 features + 1 target).<br/>
        clean_train_dataset.csv shape: <strong>81,474 rows × 39 columns</strong><br/>
        clean_test_dataset.csv shape: <strong>20,289 rows × 39 columns</strong></p>
        
        <button class="collapsible-trigger mt-2"><span>Show All 39 Final Columns</span><span class="chevron">▼</span></button>
        <div class="collapsible-content">
            <div class="table-wrapper scroll-x"><table class="table-compact">
                <thead><tr><th>Final Column Name</th><th>Came From Original Feature</th><th>Type</th></tr></thead>
                <tbody>
                    <tr><td>readmitted</td><td>readmitted</td><td>Target</td></tr>
                    <tr><td>number_inpatient, time_in_hospital, number_emergency, number_outpatient, num_medications, num_lab_procedures</td><td>(Same)</td><td>Numeric</td></tr>
                    <tr><td>admission_type_id, discharge_disposition_id</td><td>(Same)</td><td>Numeric ID</td></tr>
                    <tr><td>diabetesMed, insulin, metformin, max_glu_serum, A1Cresult</td><td>(Same)</td><td>Ordinal/Binary</td></tr>
                    <tr><td>age, gender</td><td>(Same)</td><td>Ordinal/Binary</td></tr>
                    <tr><td>race_AfricanAmerican, race_Asian, race_Caucasian, race_Hispanic, race_Other</td><td>race</td><td>One-hot Binary</td></tr>
                    <tr><td>diag_1_Circulatory, diag_1_Diabetes, diag_1_Digestive, diag_1_Genitourinary, diag_1_Injury, diag_1_Musculoskeletal, diag_1_Neoplasms, diag_1_Other, diag_1_Respiratory</td><td>diag_1</td><td>One-hot Binary</td></tr>
                    <tr><td>diag_2_Circulatory, diag_2_Diabetes, diag_2_Digestive, diag_2_Genitourinary, diag_2_Injury, diag_2_Musculoskeletal, diag_2_Neoplasms, diag_2_Other, diag_2_Respiratory</td><td>diag_2</td><td>One-hot Binary</td></tr>
                </tbody>
            </table></div>
        </div>

        <!-- PART 8: CSV FILES CREATED -->
        <h3 class="mt-3">8. CSV Files Created by the Data Pipeline</h3>
        <div class="table-wrapper scroll-x"><table class="table-compact">
            <thead><tr><th>CSV File</th><th>Created By Notebook</th><th class="text-right">Rows</th><th class="text-right">Columns</th><th>Purpose</th></tr></thead>
            <tbody>
                <tr><td>train_data.csv</td><td>Data_cleaning.ipynb</td><td class="text-right">81,477</td><td class="text-right">50</td><td>Initial training split (before encoding)</td></tr>
                <tr><td>test_data.csv</td><td>Data_cleaning.ipynb</td><td class="text-right">20,289</td><td class="text-right">50</td><td>Initial test split (before encoding)</td></tr>
                <tr><td>clean_train_dataset.csv</td><td>Data_cleaning.ipynb</td><td class="text-right">81,474</td><td class="text-right">39</td><td>Experiment 001 Training Set</td></tr>
                <tr><td>clean_test_dataset.csv</td><td>Data_cleaning.ipynb</td><td class="text-right">20,289</td><td class="text-right">39</td><td>Test Set for All Experiments</td></tr>
                <tr><td>balance_classes_training_dataset.csv</td><td>balance_classes_dataset.ipynb</td><td class="text-right">18,376</td><td class="text-right">39</td><td>Experiment 002 Training Set</td></tr>
                <tr><td>balance_demographic_variable_training_dataset.csv</td><td>balance_demographihc.ipynb</td><td class="text-right">33,390</td><td class="text-right">39</td><td>Experiment 003 Training Set</td></tr>
                <tr><td>Balanceclasses+softbalanced_demographic_groups.csv</td><td>balanceclasses+softbalancedemographic groups.ipynb</td><td class="text-right">7,568</td><td class="text-right">39</td><td>Experiment 004 Training Set</td></tr>
            </tbody>
        </table></div>

        <!-- PART 9: BALANCED DATASETS -->
        <h3 class="mt-3">9. How Balanced Training Datasets Were Created</h3>
        <div class="research-card-grid">
            <div class="research-card"><h4>Experiment 001 Dataset</h4><p><strong>Source:</strong> clean_train_dataset.csv</p><p><strong>Meaning:</strong> raw cleaned training dataset.</p><p><strong>Created by:</strong> Data_cleaning.ipynb</p></div>
            <div class="research-card"><h4>Experiment 002 Dataset</h4><p><strong>Source:</strong> balance_classes_dataset.ipynb</p><p><strong>Output:</strong> balance_classes_training_dataset.csv</p><p><strong>Meaning:</strong> class-balanced training dataset.</p><p><strong>Purpose:</strong> balance Class 0 and Class 1.</p></div>
            <div class="research-card"><h4>Experiment 003 Dataset</h4><p><strong>Source:</strong> balance_demographihc.ipynb</p><p><strong>Output:</strong> balance_demographic_variable_training_dataset.csv</p><p><strong>Meaning:</strong> demographic-balanced training dataset.</p><p><strong>Purpose:</strong> adjust race/gender/age representation.</p></div>
            <div class="research-card"><h4>Experiment 004 Dataset</h4><p><strong>Source:</strong> balanceclasses+softbalancedemographic groups.ipynb</p><p><strong>Output:</strong> Balanceclasses+softbalanced_demographic_groups.csv</p><p><strong>Meaning:</strong> class + soft demographic-balanced training dataset.</p><p><strong>Purpose:</strong> combine target balancing and demographic balancing.</p></div>
        </div>

        <!-- PART 10: POPULATION COMPARISON -->
        <h3 class="mt-3">10. Dataset Population Comparison</h3>
        <p>This table shows how the training population changes across experiments.</p>
        <div class="table-wrapper scroll-x"><table class="table-compact">
            <thead><tr><th>Dataset</th><th class="text-right">Rows</th><th class="text-right">Columns</th><th class="text-right">Class 0 Count</th><th class="text-right">Class 1 Count</th><th class="text-right">Class 0 %</th><th class="text-right">Class 1 %</th><th>Used In</th></tr></thead>
            <tbody>
                <tr><td>clean_train_dataset.csv</td><td class="text-right">81,474</td><td class="text-right">39</td><td class="text-right">72,286</td><td class="text-right">9,188</td><td class="text-right">88.72%</td><td class="text-right">11.28%</td><td>Exp 001</td></tr>
                <tr><td>clean_test_dataset.csv</td><td class="text-right">20,289</td><td class="text-right">39</td><td class="text-right">18,120</td><td class="text-right">2,169</td><td class="text-right">89.31%</td><td class="text-right">10.69%</td><td>Testing</td></tr>
                <tr><td>balance_classes_training_dataset.csv</td><td class="text-right">18,376</td><td class="text-right">39</td><td class="text-right">9,188</td><td class="text-right">9,188</td><td class="text-right">50.00%</td><td class="text-right">50.00%</td><td>Exp 002</td></tr>
                <tr><td>balance_demographic_variable_training_dataset.csv</td><td class="text-right">33,390</td><td class="text-right">39</td><td class="text-right">29,606</td><td class="text-right">3,784</td><td class="text-right">88.67%</td><td class="text-right">11.33%</td><td>Exp 003</td></tr>
                <tr><td>Balanceclasses+softbalanced_demographic_groups.csv</td><td class="text-right">7,568</td><td class="text-right">39</td><td class="text-right">3,784</td><td class="text-right">3,784</td><td class="text-right">50.00%</td><td class="text-right">50.00%</td><td>Exp 004</td></tr>
            </tbody>
        </table></div>
        
        <div class="research-card-grid mt-2">
            <div class="research-card"><h4>Experiment 001</h4><p>Preserves the original class imbalance (~11% Class 1).</p></div>
            <div class="research-card"><h4>Experiment 002</h4><p>Balances the target classes exactly (50% Class 1, 50% Class 0).</p></div>
            <div class="research-card"><h4>Experiment 003</h4><p>Changes demographic representation but retains the original ~11% class imbalance.</p></div>
            <div class="research-card"><h4>Experiment 004</h4><p>Combines class balancing (50% Class 1) with soft demographic balancing.</p></div>
        </div>

        <!-- PART 11: CURATION DIAGRAM -->
        <h3 class="mt-3">11. Data Curation Diagram</h3>
        <div class="diagram-container"><div class="mermaid">
        flowchart TD
            A["diabetic_data.csv<br/>Raw UCI Diabetes Dataset"] --> B["faircare_raw_dataset_explanation.ipynb<br/>Understand raw dataset"]
            A --> C["Data_cleaning.ipynb<br/>Select 18 features, clean values, encode categories, binarize target"]

            C --> D["train_data.csv<br/>Initial training split"]
            C --> E["test_data.csv<br/>Initial test split"]
            C --> F["clean_train_dataset.csv<br/>Experiment 001 training dataset"]
            C --> G["clean_test_dataset.csv<br/>Common test dataset"]

            F --> H["balance_classes_dataset.ipynb"]
            H --> I["balance_classes_training_dataset.csv<br/>Experiment 002 training"]

            F --> J["balance_demographihc.ipynb"]
            J --> K["balance_demographic_variable_training_dataset.csv<br/>Experiment 003 training"]

            F --> L["balanceclasses+softbalancedemographic groups.ipynb"]
            L --> M["Balanceclasses+softbalanced_demographic_groups.csv<br/>Experiment 004 training"]

            G --> N["Used as common test set<br/>for all four experiments"]
        </div></div>

    </div>
</section>

<!-- ═══════ 5. EXPERIMENT DESIGN ═══════ -->
<section id="design">
    <div class="container">
        <h2 class="section-title">Experiment Design</h2>
        <p class="section-lead">Four experiments test how different training data strategies affect Class 1 detection and fairness.</p>
        <div class="table-wrapper scroll-x"><table><thead><tr><th>Exp</th><th>Training Strategy</th><th class="text-right">Train Rows</th><th class="text-right">Class 1 %</th><th>Research Question</th></tr></thead><tbody>
            <tr><td>001</td><td>Raw cleaned baseline</td><td class="text-right">81,474</td><td class="text-right val-bad">11.28%</td><td>What happens without balancing?</td></tr>
            <tr><td>002</td><td>Class balanced (50/50)</td><td class="text-right">18,376</td><td class="text-right val-good">50.00%</td><td>Does class balancing improve detection?</td></tr>
            <tr><td>003</td><td>Demographic balanced only</td><td class="text-right">33,390</td><td class="text-right val-bad">11.33%</td><td>Is demographic balancing alone enough?</td></tr>
            <tr><td>004</td><td>Class + demographic balanced</td><td class="text-right">7,568</td><td class="text-right val-good">50.00%</td><td>Can both improve detection and fairness?</td></tr>
        </tbody></table></div>
        <div class="interpret-box"><h5>Key Design Point</h5><p>All experiments share the same test set (20,289 rows). Differences in results are due solely to training data preparation.</p></div>
    </div>
</section>

<!-- ═══════ 6. ARCHITECTURE ═══════ -->
<section id="architecture">
    <div class="container">
        <h2 class="section-title">Architecture</h2>
        <div class="diagram-container"><div class="mermaid">
        flowchart TD
            A["Raw Dataset<br/>101,766 × 50"] --> B["Feature Selection + Encoding"]
            B --> E1["Exp 001: Raw Baseline<br/>81,474 rows"]
            B --> E2["Exp 002: Class Balanced<br/>18,376 rows"]
            B --> E3["Exp 003: Demo Balanced<br/>33,390 rows"]
            B --> E4["Exp 004: Combined<br/>7,568 rows"]
            E1 & E2 & E3 & E4 --> T["Common Test Set: 20,289 rows"]
            T --> M["4 Models: LR · RF · XGB · MLP"]
            M --> P["Predictions: y_pred + y_prob"]
            P --> F["4 Fairness Matrices per model"]
            F --> G["Race · Gender · Age Analysis"]
        </div></div>
    </div>
</section>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- ═══════ 7. RESEARCH RESULTS & FAIRNESS INTERPRETATION ═══════ -->
<!-- ═══════════════════════════════════════════════════════ -->
<section id="results">
    <div class="container">
        <h2 class="section-title">Research Results &amp; Fairness Interpretation</h2>
        <p class="section-lead">A research-style summary of all 16 trained models and the four Class 1 fairness matrices across all experiments.</p>

        <div class="summary-cards">
            <div class="summary-card"><div class="summary-card-label"><i class="fa-solid fa-flask"></i> Research Question</div><div class="summary-card-value">How do different training data strategies affect Class 1 readmission detection and fairness across race, gender, and age?</div></div>
            <div class="summary-card"><div class="summary-card-label"><i class="fa-solid fa-chart-simple"></i> Key Metrics</div><div class="summary-card-value"><span class="badge badge-cyan">Recall</span> <span class="badge badge-rose">FNR</span> <span class="badge badge-emerald">F1</span> <span class="badge badge-amber">Calibration</span> <span class="badge badge-purple">SHAP</span></div></div>
        </div>

        <!-- PART A: 16-Model Table -->
        <h3 class="mt-3">A. All 16 Model Results</h3>
        <div class="explain-box"><h5>What This Table Shows</h5><p>Every model trained in the project. The most important columns are <span class="badge badge-cyan">Class 1 Recall</span> (patients caught) and <span class="badge badge-rose">FNR</span> (patients missed).</p></div>
        <div class="table-wrapper scroll-x"><table class="table-compact"><thead><tr>
            <th>Exp</th><th>Training Strategy</th><th>Model</th><th class="text-right">Accuracy</th><th class="text-right">Precision</th><th class="text-right">Recall</th><th class="text-right">F1</th><th class="text-right">ROC-AUC</th><th class="text-right">FNR</th><th>Interpretation</th>
        </tr></thead><tbody>
{results_rows}
        </tbody></table></div>
        <div class="interpret-box"><h5>How to Read</h5><p>A good healthcare model should not only have high accuracy. It should catch Class 1 patients (high recall) and keep the false negative rate low. Green values indicate stronger Class 1 detection; red values indicate weak detection.</p></div>
        <div class="speaker-notes"><h5>Speaker Notes</h5><p>When presenting, explain that the project compares four training strategies. The key question is not just accuracy — it is whether each model catches Class 1 readmitted patients.</p></div>

        <!-- PART B: Experiment Summary Cards -->
        <h3 class="mt-3">B. Experiment-Level Interpretation</h3>
        <div class="research-card-grid">
            <div class="research-card"><h4>Experiment 001: Raw Baseline</h4><p>The available results suggest that without class balancing, tree-based models (RF, XGBoost) prioritize the majority class and achieve near-zero Class 1 recall despite high accuracy. Logistic Regression, as a simpler linear model, catches ~53% of readmissions regardless of imbalance.</p></div>
            <div class="research-card"><h4>Experiment 002: Class Balanced</h4><p>The available results suggest that class balancing significantly increases Class 1 Recall for tree-based models (RF: ~62%, XGB: ~61%). The accuracy decrease is an expected tradeoff when the model shifts from majority-class bias toward detecting the minority class.</p></div>
            <div class="research-card"><h4>Experiment 003: Demographic Balanced</h4><p>The available results suggest that demographic balancing alone does not resolve the target class imbalance problem. RF achieves 0% Class 1 recall, confirming that target-level balancing is the primary driver of recall improvement.</p></div>
            <div class="research-card"><h4>Experiment 004: Combined Balanced</h4><p>The available results suggest that combining class and demographic balancing maintains the Class 1 detection gains while also adjusting demographic representation. This pattern should be interpreted alongside the fairness matrices.</p></div>
        </div>

        <!-- PART C: Performance Fairness -->
        <h3 class="mt-3">C. Performance Fairness Interpretation</h3>
        <div class="explain-box"><h5>What This Matrix Measures</h5><p>Class 1 Recall, Precision, F1 across race, gender, and age groups. A large recall gap means the model catches fewer readmissions in some groups.</p><p><strong>Why it matters:</strong> Unequal recall = unequal clinical protection across demographic groups.</p></div>
        <div class="interpret-box"><h5>How to Interpret</h5><p>Look at Recall_Class_1_Readmitted per group. If one group has significantly lower recall, readmitted patients in that group are more likely to be missed. The full per-model matrices are available in the Experiment Viewer section below.</p></div>

        <!-- PART D: Error Fairness -->
        <h3 class="mt-3">D. Error Fairness Interpretation</h3>
        <div class="explain-box"><h5>What This Matrix Measures</h5><p>FN count and FNR per demographic group. High FNR = more readmitted patients missed in that group.</p><p><strong>Why it matters:</strong> False negatives are the most clinically important error. Missed patients may not receive follow-up care.</p></div>
        <div class="interpret-box"><h5>How to Interpret</h5><p>Groups with high FNR need attention. If FNR varies strongly across race or age groups, the model creates a healthcare disparity in readmission detection.</p></div>

        <!-- PART E: Calibration Fairness -->
        <h3 class="mt-3">E. Calibration Fairness Interpretation</h3>
        <div class="explain-box"><h5>What This Matrix Measures</h5><p>Whether predicted Class 1 risk scores match actual readmission rates per group. High calibration error = unreliable risk scores.</p><p><strong>Why it matters:</strong> Poor calibration can lead clinicians to make wrong treatment decisions for specific populations.</p></div>
        <div class="interpret-box"><h5>How to Interpret</h5><p>Compare Avg_Predicted_Risk with Actual_Readmission_Rate. If the model predicts 40% risk but true rate is 10%, it is poorly calibrated for that group.</p></div>

        <!-- PART F: SHAP -->
        <h3 class="mt-3">F. SHAP Explanation Fairness Interpretation</h3>
        <div class="explain-box"><h5>What This Matrix Measures</h5><p>Which features drive Class 1 predictions per group using SHAP values. If sensitive features dominate, it raises proxy concerns.</p><p><strong>Why it matters:</strong> Different prediction patterns across groups may indicate the model reasons differently about different populations.</p></div>
        <div class="interpret-box"><h5>How to Interpret</h5><p>If the same clinical features (e.g., number_inpatient) appear as top drivers across all groups, the model uses consistent clinical reasoning. If top features vary strongly, the model may be using different patterns.</p></div>

        <!-- PART G: Model-by-Model -->
        <h3 class="mt-3">G. Model-by-Model Interpretation</h3>
        <div class="research-card-grid">
            <div class="research-card"><h4>Logistic Regression</h4><p>Recall is remarkably stable across all experiments (~53%). Class balancing does not significantly change its behavior, suggesting LR captures a linear readmission signal that is independent of class distribution. As the simplest model, it serves as an interpretable baseline.</p></div>
            <div class="research-card"><h4>Random Forest</h4><p>Highly sensitive to class imbalance. Without balancing (Exp 001, 003), RF achieves ≤0.14% recall. With balancing (Exp 002, 004), recall jumps to ~60%. This indicates RF defaults to majority-class prediction under imbalance. Fairness matrices should be reviewed for the balanced experiments.</p></div>
            <div class="research-card"><h4>XGBoost</h4><p>Similar sensitivity pattern to RF but with slightly higher ROC-AUC. Without balancing, recall is ≤1.11%. With balancing, recall reaches ~60%. XGBoost achieves the highest ROC-AUC in most experiments, suggesting strong ranking ability.</p></div>
            <div class="research-card"><h4>MLP Neural Network</h4><p>Shows moderate improvement with class balancing (6.5% → 54%). More variable than tree-based models. Lower ROC-AUC suggests weaker ranking ability. The fairness matrices should be checked to assess whether MLP distributes errors evenly.</p></div>
        </div>

        <!-- PART H: Research Summary -->
        <h3 class="mt-3">H. Research Interpretation Summary</h3>
        <div class="research-card">
            <p><strong>1. What changed across experiments?</strong> The training data class distribution was the primary variable. Experiments with 50/50 class balance (002, 004) showed dramatically different model behavior from imbalanced experiments (001, 003).</p>
            <p><strong>2. How did Class 1 recall respond?</strong> For tree-based models (RF, XGBoost), recall went from near-zero to ~60% with class balancing. LR remained stable at ~53% regardless of strategy.</p>
            <p><strong>3. What happened to FNR?</strong> FNR decreased from ~99% to ~39% for RF and XGBoost with class balancing. This means ~61% more readmissions are now detected.</p>
            <p><strong>4. Did class balancing help?</strong> The available results strongly suggest yes. Class balancing is the primary driver of improved Class 1 detection for tree-based models.</p>
            <p><strong>5. Did demographic balancing alone help?</strong> The available results suggest no. Experiment 003 shows that demographic balancing without class balancing does not improve Class 1 recall.</p>
            <p><strong>6. Why do fairness matrices matter?</strong> Overall recall improvement does not guarantee equal improvement across demographic groups. The fairness matrices check whether detection gains are distributed evenly across race, gender, and age.</p>
            <p><strong>7. What still needs verification?</strong> Further analysis is needed to quantify fairness gaps across demographic groups and determine whether the combined strategy (Exp 004) reduces those gaps compared to class-only balancing (Exp 002).</p>
        </div>
        <div class="speaker-notes"><h5>Speaker Notes</h5><p>When presenting fairness matrices, explain that each row is a demographic subgroup. The project asks whether recall, FNR, calibration, and feature influence are similar across race, gender, and age.</p></div>
    </div>
</section>

<!-- ═══════ 8. FULL EXPERIMENT VIEWER ═══════ -->
<section id="viewer">
    <div class="container">
        <h2 class="section-title">Full Experiment Viewer</h2>
        <p class="section-lead">Explore each experiment's complete model results and fairness matrices. Click model tabs to switch models. Click matrix buttons to expand full tables.</p>

        <div class="tab-container">
            <div class="tab-buttons">
                <button class="tab-btn active" data-tab="ev-exp1">Exp 001</button>
                <button class="tab-btn" data-tab="ev-exp2">Exp 002</button>
                <button class="tab-btn" data-tab="ev-exp3">Exp 003</button>
                <button class="tab-btn" data-tab="ev-exp4">Exp 004</button>
            </div>
            <div class="tab-pane active" id="ev-exp1">{exp_viewers.split('<!-- EXP_SPLIT -->')[0] if '<!-- EXP_SPLIT -->' in exp_viewers else ''}</div>
            <div class="tab-pane" id="ev-exp2"></div>
            <div class="tab-pane" id="ev-exp3"></div>
            <div class="tab-pane" id="ev-exp4"></div>
        </div>
    </div>
</section>

<!-- ═══════ FAIRNESS MEASURES ═══════ -->
<section id="fairness-measures">
    <div class="container">
        <h2 class="section-title">Fairness Measures Across All 16 Models</h2>
        <p class="section-lead">This section breaks down the 16 trained models into two clear views: first performance, then fairness gaps.</p>

        <!-- HOW TO READ THIS SECTION -->
        <div class="research-card" style="border-left: 3px solid var(--accent-cyan);">
            <h4><i class="fa-solid fa-book-open"></i> How to Read This Section</h4>
            <ol style="padding-left:1.2rem;">
                <li style="margin-bottom:0.5rem;"><strong>First look at Class 1 Recall.</strong> This tells how many readmitted patients were caught.</li>
                <li style="margin-bottom:0.5rem;"><strong>Then look at Class 1 FNR.</strong> This tells how many readmitted patients were missed.</li>
                <li style="margin-bottom:0.5rem;"><strong>Then look at fairness gaps.</strong> This tells whether some race, gender, or age groups are treated worse.</li>
                <li style="margin-bottom:0.5rem;"><strong>Then look at SHAP.</strong> This tells which features influenced the predictions.</li>
            </ol>
            <p style="margin-top:0.75rem;"><em>If Recall is low and FNR is high, the model is not useful for catching readmitted patients.</em></p>
        </div>

        <!-- COLOR LEGEND -->
        <div class="table-wrapper mt-2" style="max-width:400px;"><table><thead><tr><th>Color</th><th>Meaning</th></tr></thead><tbody>
            <tr><td><span class="val-good">● Green</span></td><td>Better result</td></tr>
            <tr><td><span class="val-mid">● Orange</span></td><td>Needs review</td></tr>
            <tr><td><span class="val-bad">● Red</span></td><td>Concern</td></tr>
        </tbody></table></div>

        <!-- PART 1: MODEL PERFORMANCE -->
        <h3 class="mt-3">1. Model Performance Across All 16 Models</h3>
        <div class="explain-box"><h5>What This Table Shows</h5><p>Basic predictive performance of all 16 trained models. Each row is one model trained under one experiment. The most important column is <span class="badge badge-cyan">Class 1 Recall</span> because Class 1 = readmitted within 30 days.</p></div>
        <div class="table-wrapper scroll-x"><table class="table-compact"><thead><tr>
            <th>Exp</th><th>Training Strategy</th><th>Model</th><th class="text-right">Accuracy</th><th class="text-right">Class 1 Recall</th><th class="text-right">Class 1 FNR</th><th class="text-right">Class 1 F1</th><th class="text-right">ROC-AUC</th><th>Simple Meaning</th>
        </tr></thead><tbody>
''' + '{perf_rows}' + '''
        </tbody></table></div>
        <div class="interpret-box"><h5>How to Read</h5><p><strong>Accuracy</strong> = total correct predictions. <strong>Recall</strong> = of patients actually readmitted, how many were caught? <strong>FNR</strong> = of patients actually readmitted, how many were missed? <strong>F1</strong> = balance between recall and precision. <strong>ROC-AUC</strong> = how well the model ranks risk levels.</p></div>

        <!-- PART 2: FAIRNESS RESULTS BY DEMOGRAPHIC GROUP -->
        <h3 class="mt-3">2. Fairness Results by Race, Gender, and Age</h3>
        <p>This section explains what the fairness matrices reveal about demographic groups in simple, presentation-ready language.</p>

        <!-- CARD 1: RACE -->
        <div class="research-card" style="border-left:3px solid var(--accent-cyan);">
            <h4><i class="fa-solid fa-people-group"></i> Race Fairness Result</h4>
            <p>This part checks whether the model catches readmitted patients similarly across race groups.</p>
            <p>Across the experiments, race-based differences are visible in the Class 1 fairness results. Some race groups — such as Other, Hispanic, Asian, and Unknown — appear as the weakest recall groups in different models. This means the model does not always catch readmitted patients equally across race groups. However, groups like Other, Asian, and Hispanic have smaller sample sizes in the test set, so these results should be interpreted with caution.</p>
            <p>In experiments with class balancing (002 and 004), overall recall improves for most groups, but the gap between the strongest and weakest race groups can remain above 0.60, indicating that the disparity is not eliminated by balancing alone.</p>
            <div class="explain-box"><h5>Race Fairness Conclusion</h5><p>Race-level disparities appear to exist and should be reviewed using the full subgroup matrices. Smaller race groups need careful interpretation due to sample size.</p></div>
        </div>

        <!-- CARD 2: GENDER -->
        <div class="research-card mt-2" style="border-left:3px solid var(--accent-emerald);">
            <h4><i class="fa-solid fa-venus-mars"></i> Gender Fairness Result</h4>
            <p>This part checks whether the model behaves differently for male and female patients.</p>
            <p>Compared with race and age, the gender fairness gap appears smaller in most experiments. Male and female recall values are usually closer to each other. In some models, female patients show slightly weaker Class 1 detection; in others, male patients are weaker. The recall gap between genders is typically below 5 percentage points.</p>
            <p>This suggests that gender disparity exists in some cases, but it appears less severe than the race and age subgroup differences observed across the project.</p>
            <div class="explain-box"><h5>Gender Fairness Conclusion</h5><p>Gender differences are present but generally smaller than race and age differences. Both genders should still be checked in the full fairness matrices.</p></div>
        </div>

        <!-- CARD 3: AGE -->
        <div class="research-card mt-2" style="border-left:3px solid var(--accent-amber);">
            <h4><i class="fa-solid fa-cake-candles"></i> Age Fairness Result</h4>
            <p>This part checks whether the model works similarly across age groups.</p>
            <p>Age shows important fairness concerns. Very young age groups such as [0-10) and [10-20) often show weak or zero Class 1 recall and high FNR. However, these groups also have very small sample sizes (e.g., 28 and 121 patients), so the results may be unstable. Among larger age groups, moderate recall differences are observed — for example, [50-60) sometimes shows weaker recall than [70-80) or [80-90).</p>
            <p>This means age fairness should be interpreted carefully, especially for underrepresented age groups where a few patients can strongly change the metric.</p>
            <div class="explain-box"><h5>Age Fairness Conclusion</h5><p>Age-based disparities appear, but small sample sizes make some age group results unstable. Larger age groups show more reliable patterns and should be the primary focus.</p></div>
        </div>

        <!-- CARD 4: OVERALL -->
        <div class="research-card mt-2" style="border-left:3px solid var(--accent-purple);">
            <h4><i class="fa-solid fa-scale-balanced"></i> Overall Fairness Interpretation</h4>
            <p>Overall, the fairness results suggest that model performance is not equal across all demographic groups. Race and age show clearer subgroup differences, while gender differences appear smaller.</p>
            <p>The main fairness issue is not only whether the model performs well overall, but whether it catches Class 1 readmitted patients consistently across race, gender, and age. The available results indicate that some demographic groups receive weaker Class 1 detection and should be reviewed carefully.</p>
            <p>The full fairness matrices in the Experiment Viewer section provide the supporting evidence for these findings.</p>
        </div>

        <!-- SPEAKER SCRIPT -->
        <div class="speaker-notes mt-2"><h5>How to Explain This Fairness Section</h5><p>"In this fairness section, I am not trying to say the model is completely fair or unfair. I am checking whether the model catches readmitted patients equally across race, gender, and age. The results show that race and age have more visible subgroup differences, while gender differences are smaller. For age, some results are affected by very small sample sizes. So the main conclusion is that subgroup fairness needs review, especially for race and age."</p></div>

        <!-- COLLAPSED ADVANCED TABLES -->
        <button class="collapsible-trigger mt-2"><span>Show Advanced Fairness Tables (Race, Gender, Age, Gap Summary)</span><span class="chevron">▼</span></button>
        <div class="collapsible-content">
            <h5 class="mt-2">Race Fairness Details</h5>
            <div class="table-wrapper scroll-x"><table class="table-compact"><thead><tr>
                <th>Exp</th><th>Model</th><th>Weakest Race Group</th><th class="text-right">Recall</th><th>Highest FNR Race</th><th class="text-right">FNR</th><th>Highest Cal. Error Race</th><th class="text-right">Cal. Error</th><th>Note</th>
            </tr></thead><tbody>
''' + '{race_rows}' + '''</tbody></table></div>

            <h5 class="mt-2">Gender Fairness Details</h5>
            <div class="table-wrapper scroll-x"><table class="table-compact"><thead><tr>
                <th>Exp</th><th>Model</th><th>Weaker Gender</th><th class="text-right">Recall</th><th>Higher FNR Gender</th><th class="text-right">FNR</th><th>Higher Cal. Error Gender</th><th class="text-right">Cal. Error</th><th>Note</th>
            </tr></thead><tbody>
''' + '{gender_rows}' + '''</tbody></table></div>

            <h5 class="mt-2">Age Fairness Details</h5>
            <div class="table-wrapper scroll-x"><table class="table-compact"><thead><tr>
                <th>Exp</th><th>Model</th><th>Weakest Age Group</th><th class="text-right">Recall</th><th class="text-right">Size</th><th>Highest FNR Age</th><th class="text-right">FNR</th><th class="text-right">Size</th><th>Warning</th><th>Note</th>
            </tr></thead><tbody>
''' + '{age_rows}' + '''</tbody></table></div>

            <h5 class="mt-2">Numeric Gap Summary</h5>
            <div class="table-wrapper scroll-x"><table class="table-compact"><thead><tr>
                <th>Exp</th><th>Model</th><th class="text-right">Recall Gap</th><th class="text-right">FNR Gap</th><th class="text-right">Cal. Gap</th><th>SHAP</th>
            </tr></thead><tbody>
''' + '{gap_rows}' + '''</tbody></table></div>
        </div>

        <!-- PART 3: EXPERIMENT INTERPRETATION CARDS -->
        <h3 class="mt-3">3. What Each Experiment Teaches</h3>
        <div class="research-card-grid">
            <div class="research-card"><h4><span class="badge badge-rose">001</span> No Balancing</h4><p>This is the baseline. It shows what happens when we train on the original cleaned data.</p><p><strong>Class 1 Recall:</strong> RF and XGBoost have near-zero recall (0.14% and 0.92%). LR catches ~53%. <strong>FNR:</strong> Above 93% for tree models — nearly all readmissions are missed. <strong>Simple meaning:</strong> Without balancing, tree-based models learn to predict only the majority class.</p></div>
            <div class="research-card"><h4><span class="badge badge-emerald">002</span> Class Balanced</h4><p>This experiment balances Class 0 and Class 1 in training (50/50).</p><p><strong>Class 1 Recall:</strong> RF jumps to ~62%, XGBoost to ~61%. <strong>FNR:</strong> Drops to ~38–39%. <strong>Accuracy tradeoff:</strong> Accuracy decreases from ~89% to ~62–67% — this is expected when the model stops defaulting to the majority class. <strong>Simple meaning:</strong> Class balancing forces models to detect readmissions.</p></div>
            <div class="research-card"><h4><span class="badge badge-rose">003</span> Demographic Balanced Only</h4><p>This experiment adjusts demographic representation but keeps class imbalance.</p><p><strong>Class 1 Recall:</strong> RF achieves 0.00%, XGBoost 1.11%. <strong>Simple meaning:</strong> Demographic balancing alone does not solve the target class imbalance problem. Models still predict mostly Class 0.</p></div>
            <div class="research-card"><h4><span class="badge badge-emerald">004</span> Class + Demographic Balanced</h4><p>Combines class balancing with soft demographic balancing.</p><p><strong>Class 1 Recall:</strong> RF ~60%, XGBoost ~60%. <strong>FNR:</strong> ~40%. <strong>Simple meaning:</strong> Detection stays strong. Fairness gaps (0.49–0.89) should be reviewed alongside the subgroup matrices.</p></div>
        </div>
    </div>
</section>

<!-- ═══════ FOUR FAIRNESS MATRICES EXPLAINED SIMPLY ═══════ -->
<section id="matrices-explained">
    <div class="container">
        <h2 class="section-title">Four Fairness Matrices Explained Simply</h2>
        <p class="section-lead">A simple guide to explain the FairCare fairness framework in plain language.</p>
        <p>FairCare uses four matrices because one metric cannot explain the full fairness picture. Each matrix answers a different question about model behavior.</p>

        <div class="table-wrapper mt-2"><table><thead><tr><th>Matrix</th><th>Simple Question</th><th>Main Thing to Look At</th></tr></thead><tbody>
            <tr><td><span class="badge badge-cyan">M1</span> Performance</td><td>Is the model catching readmitted patients equally across groups?</td><td>Class 1 Recall and F1</td></tr>
            <tr><td><span class="badge badge-rose">M2</span> Error</td><td>Is the model missing readmitted patients more in some groups?</td><td>FNR and FN count</td></tr>
            <tr><td><span class="badge badge-amber">M3</span> Calibration</td><td>Are the model's risk scores believable for each group?</td><td>Predicted risk vs actual readmission rate</td></tr>
            <tr><td><span class="badge badge-purple">M4</span> SHAP</td><td>Why is the model making predictions?</td><td>Top influencing features</td></tr>
        </tbody></table></div>

        <h3 class="mt-3">Matrix 1: Performance Fairness Matrix</h3>
        <div class="research-card"><h4><span class="badge badge-cyan">M1</span> What it checks</h4><p>How well the model performs for each demographic group. Each row is one group (e.g., Caucasian, African American, Male, Female, age [70-80)).</p>
            <p><strong>Main metric — Class 1 Recall:</strong> "Out of the patients who were actually readmitted, how many did the model catch?"</p>
            <p><strong>How to explain:</strong> If one group has much lower Recall than another, the model catches fewer readmissions in that group.</p>
            <div class="explain-box"><h5>Example</h5><p>If Group A has 60% recall and Group B has 30% recall, the model catches readmitted patients much better in Group A. This is a <strong>performance fairness gap</strong>.</p></div></div>

        <h3 class="mt-2">Matrix 2: Error Fairness Matrix</h3>
        <div class="research-card"><h4><span class="badge badge-rose">M2</span> What it checks</h4><p>Where the model makes mistakes. The most important mistake is the <strong>False Negative</strong>: the patient was actually readmitted, but the model predicted not readmitted.</p>
            <p><strong>Main metric — FNR:</strong> "Out of the patients who were actually readmitted, how many did the model miss?"</p>
            <p><strong>Healthcare meaning:</strong> Missed patients may not receive follow-up care. This is the most clinically harmful error.</p>
            <div class="explain-box"><h5>Example</h5><p>If a group has 70% FNR, the model missed 70 out of every 100 readmitted patients in that group. This may show <strong>harmful error disparity</strong>.</p></div></div>

        <h3 class="mt-2">Matrix 3: Calibration Fairness Matrix</h3>
        <div class="research-card"><h4><span class="badge badge-amber">M3</span> What it checks</h4><p>Whether the model's predicted risk scores match actual readmission rates. This determines if risk scores are trustworthy.</p>
            <p><strong>Main metric — Calibration Error:</strong> "When the model says a group has a certain risk, is that risk close to reality?"</p>
            <p><strong>Why it matters:</strong> Poor calibration can lead clinicians to make wrong treatment decisions for specific populations.</p>
            <div class="explain-box"><h5>Example</h5><p>Predicted risk = 40%, Actual readmission rate = 10% → Calibration error = 30%. The model is <strong>overestimating risk</strong> for that group.</p></div></div>

        <h3 class="mt-2">Matrix 4: SHAP Explanation Fairness Matrix</h3>
        <div class="research-card"><h4><span class="badge badge-purple">M4</span> What it checks</h4><p>Which features drive readmission predictions per group. SHAP values explain what pushed the model toward predicting readmission risk.</p>
            <p><strong>Main thing to look at:</strong> Top features, mean SHAP impact, and whether sensitive features (race, gender) are highly influential.</p>
            <p><strong>Fairness meaning:</strong> If different groups have very different top features, the model may be using different reasoning across populations.</p>
            <div class="explain-box"><h5>Example</h5><p>If number_inpatient is the top feature across all groups, the model uses consistent clinical reasoning. If race or gender dominate for some groups, that needs careful review.</p></div></div>

        <h3 class="mt-3">Simple Analogy</h3>
        <div class="research-card" style="border-left:3px solid var(--accent-cyan);">
            <p><em>Think of the model like a hospital assistant deciding who needs follow-up care.</em></p>
            <ul>
                <li><strong><span class="badge badge-cyan">M1</span> Performance Matrix asks:</strong> Is the assistant catching high-risk patients equally across groups?</li>
                <li><strong><span class="badge badge-rose">M2</span> Error Matrix asks:</strong> Who is the assistant missing?</li>
                <li><strong><span class="badge badge-amber">M3</span> Calibration Matrix asks:</strong> When the assistant says someone is high risk, is that risk score believable?</li>
                <li><strong><span class="badge badge-purple">M4</span> SHAP Matrix asks:</strong> What information is the assistant using to make that decision?</li>
            </ul>
        </div>

        <h3 class="mt-3">How to Explain the Four Matrices</h3>
        <div class="speaker-notes"><h5>Presentation Script</h5><p>"In FairCare, we use four fairness matrices because fairness is not just one number. First, the Performance Matrix checks whether the model catches readmitted patients across race, gender, and age. Second, the Error Matrix checks whether the model misses more readmitted patients in some groups. Third, the Calibration Matrix checks whether predicted risk scores match real readmission rates. Fourth, the SHAP Matrix explains which features influence the predictions. Together, these four matrices help us understand whether the model is accurate, safe, reliable, and explainable."</p></div>
    </div>
</section>

<!-- ═══════ FAIRNESS GUIDE ═══════ -->
<section id="guide">
    <div class="container">
        <h2 class="section-title">Fairness Matrix Quick Reference</h2>
        <div class="card-grid-2">
            <div class="card"><h4><span class="badge badge-cyan">M1</span> Performance Fairness</h4><p><strong>Measures:</strong> Class 1 Recall, Precision, F1, ROC-AUC per group.</p><p><strong>Read:</strong> Lower recall = fewer readmissions caught for that group.</p></div>
            <div class="card"><h4><span class="badge badge-rose">M2</span> Error Fairness</h4><p><strong>Measures:</strong> FN count, FNR, FPR per group.</p><p><strong>Read:</strong> High FNR = more missed readmissions for that group.</p></div>
            <div class="card"><h4><span class="badge badge-amber">M3</span> Calibration Fairness</h4><p><strong>Measures:</strong> Predicted risk vs actual readmission rate per group.</p><p><strong>Read:</strong> Large gaps = unreliable risk scores for that group.</p></div>
            <div class="card"><h4><span class="badge badge-purple">M4</span> SHAP Explanation</h4><p><strong>Measures:</strong> Top features driving Class 1 prediction per group.</p><p><strong>Read:</strong> If sensitive features dominate, raises proxy concerns.</p></div>
        </div>
        <div class="explain-box mt-2"><h5>Confusion Matrix Reference</h5><table style="max-width:450px;"><thead><tr><th>Actual \\ Predicted</th><th>Pred Class 0</th><th>Pred Class 1</th></tr></thead><tbody><tr><td><strong>Actual Class 0</strong></td><td>TN (correct)</td><td>FP (false alarm)</td></tr><tr><td><strong>Actual Class 1</strong></td><td>FN (missed)</td><td>TP (caught)</td></tr></tbody></table><p class="mt-1"><strong>FN is the most critical error</strong> — a missed readmitted patient may not receive follow-up care.</p></div>
    </div>
</section>

<!-- ═══════ 10. NEXT STEPS ═══════ -->
<section id="next-steps">
    <div class="container">
        <h2 class="section-title">Next Steps</h2>
        <ul>
            <li>Quantify fairness gap differences across experiments for each demographic group.</li>
            <li>Compile a paper-style discussion interpreting tradeoffs between class and demographic balancing.</li>
            <li>Review calibration curves and SHAP feature importance rankings across groups.</li>
            <li>Document dataset limitations and scope of the fairness evaluation.</li>
        </ul>
    </div>
</section>

<footer><div class="container"><p>&copy; 2026 FairCare — Research Results Viewer. Generated from Jupyter Notebook outputs. No results fabricated.</p></div></footer>
<script src="script.js"></script>
</body></html>'''

    # Build PART 1: performance table rows
    perf_rows = ''
    for exp_num in [1,2,3,4]:
        ek = f'exp00{exp_num}'
        exp = EXP_META[exp_num]
        for mname in MODEL_NAMES:
            kv = data[ek]['models'][mname].get('key_values', {})
            r = kv.get('recall_class1', 0)
            fnr_v = kv.get('fnr', 0)
            acc = kv.get('accuracy', 0)
            rcls = val_class(r, (0.10, 0.50))
            fcls = val_class(1-fnr_v, (0.10, 0.50))
            meaning = perf_meaning(r, fnr_v)
            perf_rows += f'<tr><td>00{exp_num}</td><td>{exp["strategy"]}</td><td>{mname}</td>'
            perf_rows += f'<td class="text-right">{pct(acc)}</td>'
            perf_rows += f'<td class="text-right {rcls}">{pct(r)}</td>'
            perf_rows += f'<td class="text-right {fcls}">{pct(fnr_v)}</td>'
            perf_rows += f'<td class="text-right">{pct(kv.get("f1_class1",0))}</td>'
            perf_rows += f'<td class="text-right">{kv.get("roc_auc","N/A")}</td>'
            perf_rows += f'<td class="interp-cell">{meaning}</td></tr>\n'

    html = html.replace('{perf_rows}', perf_rows)

    # Build PART 2: demographic findings tables
    race_rows = ''; gender_rows = ''; age_rows = ''; gap_rows = ''
    for exp_num in [1,2,3,4]:
        ek = f'exp00{exp_num}'
        for mname in MODEL_NAMES:
            mf = gf.get(ek, {}).get(mname, {})

            # --- RACE ---
            races = [g for g in mf.get('race',[]) if g.get('size',0) >= 50]
            if races:
                worst_r = min(races, key=lambda x: x.get('recall',1))
                worst_fnr = max(races, key=lambda x: x.get('fnr',0))
                worst_cal = max(races, key=lambda x: x.get('cal_error',0))
                interp = f'Weaker detection for {worst_r["group"]}.' if worst_r.get('recall',1) < 0.50 else f'{worst_r["group"]} has lowest race recall.'
                race_rows += f'<tr><td>00{exp_num}</td><td>{mname}</td>'
                race_rows += f'<td>{worst_r["group"]}</td><td class="text-right">{worst_r.get("recall",0):.2%}</td>'
                race_rows += f'<td>{worst_fnr["group"]}</td><td class="text-right">{worst_fnr.get("fnr",0):.2%}</td>'
                race_rows += f'<td>{worst_cal["group"]}</td><td class="text-right">{worst_cal.get("cal_error",0):.2%}</td>'
                race_rows += f'<td class="interp-cell">{interp}</td></tr>\n'
            else:
                race_rows += f'<tr><td>00{exp_num}</td><td>{mname}</td><td colspan="7">Not available in saved output.</td></tr>\n'

            # --- GENDER ---
            genders = mf.get('gender',[])
            if len(genders) == 2:
                weaker = min(genders, key=lambda x: x.get('recall',1))
                stronger = max(genders, key=lambda x: x.get('recall',0))
                hi_fnr = max(genders, key=lambda x: x.get('fnr',0))
                hi_cal = max(genders, key=lambda x: x.get('cal_error',0))
                gap = abs(genders[0].get('recall',0) - genders[1].get('recall',0))
                interp = f'Small gender gap ({gap:.2%}).' if gap < 0.05 else f'{weaker["group"]} has weaker Class 1 detection.'
                gender_rows += f'<tr><td>00{exp_num}</td><td>{mname}</td>'
                gender_rows += f'<td>{weaker["group"]}</td><td class="text-right">{weaker.get("recall",0):.2%}</td>'
                gender_rows += f'<td>{hi_fnr["group"]}</td><td class="text-right">{hi_fnr.get("fnr",0):.2%}</td>'
                gender_rows += f'<td>{hi_cal["group"]}</td><td class="text-right">{hi_cal.get("cal_error",0):.2%}</td>'
                gender_rows += f'<td class="interp-cell">{interp}</td></tr>\n'
            else:
                gender_rows += f'<tr><td>00{exp_num}</td><td>{mname}</td><td colspan="7">Not available.</td></tr>\n'

            # --- AGE ---
            ages = mf.get('age',[])
            if ages:
                worst_a = min(ages, key=lambda x: x.get('recall',1))
                worst_afnr = max(ages, key=lambda x: x.get('fnr',0))
                warn_r = '<span class="badge badge-amber">Small sample</span>' if worst_a.get('size',0) < 200 else ''
                warn_f = '<span class="badge badge-amber">Small sample</span>' if worst_afnr.get('size',0) < 200 else ''
                warn = warn_r or warn_f or '—'
                interp = f'{worst_a["group"]} has weakest recall.'
                if worst_a.get('size',0) < 200: interp += ' Small sample — interpret carefully.'
                age_rows += f'<tr><td>00{exp_num}</td><td>{mname}</td>'
                age_rows += f'<td>{worst_a["group"]}</td><td class="text-right">{worst_a.get("recall",0):.2%}</td><td class="text-right">{worst_a.get("size",0):,}</td>'
                age_rows += f'<td>{worst_afnr["group"]}</td><td class="text-right">{worst_afnr.get("fnr",0):.2%}</td><td class="text-right">{worst_afnr.get("size",0):,}</td>'
                age_rows += f'<td>{warn}</td>'
                age_rows += f'<td class="interp-cell">{interp}</td></tr>\n'
            else:
                age_rows += f'<tr><td>00{exp_num}</td><td>{mname}</td><td colspan="8">Not available.</td></tr>\n'

            # --- GAP SUMMARY (collapsed) ---
            g = gaps.get(ek, {}).get(mname, {})
            r_gap = g.get('recall_gap'); f_gap = g.get('fnr_gap'); c_gap = g.get('cal_gap')
            shap = g.get('shap', 'N/A')
            rg_s = f'{r_gap:.2f}' if r_gap is not None else 'N/A'
            fg_s = f'{f_gap:.2f}' if f_gap is not None else 'N/A'
            cg_s = f'{c_gap:.2f}' if c_gap is not None else 'N/A'
            gap_rows += f'<tr><td>00{exp_num}</td><td>{mname}</td><td class="text-right">{rg_s}</td><td class="text-right">{fg_s}</td><td class="text-right">{cg_s}</td><td>{shap}</td></tr>\n'

    html = html.replace('{race_rows}', race_rows)
    html = html.replace('{gender_rows}', gender_rows)
    html = html.replace('{age_rows}', age_rows)
    html = html.replace('{gap_rows}', gap_rows)

    # Fix: Build experiment viewers properly into tabs
    viewers = []
    for i in range(4):
        viewers.append(build_experiment_viewer(i+1, NB_FILES[i], data[f'exp00{i+1}']))

    # Replace the placeholder tab panes
    html = html.replace(
        f'''<div class="tab-pane active" id="ev-exp1">{exp_viewers.split('<!-- EXP_SPLIT -->')[0] if '<!-- EXP_SPLIT -->' in exp_viewers else ''}</div>
            <div class="tab-pane" id="ev-exp2"></div>
            <div class="tab-pane" id="ev-exp3"></div>
            <div class="tab-pane" id="ev-exp4"></div>''',
        f'''<div class="tab-pane active" id="ev-exp1">{viewers[0]}</div>
            <div class="tab-pane" id="ev-exp2">{viewers[1]}</div>
            <div class="tab-pane" id="ev-exp3">{viewers[2]}</div>
            <div class="tab-pane" id="ev-exp4">{viewers[3]}</div>'''
    )

    os.makedirs("faircare_dashboard", exist_ok=True)
    with open("faircare_dashboard/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Dashboard generated: faircare_dashboard/index.html")

if __name__ == '__main__':
    generate()
