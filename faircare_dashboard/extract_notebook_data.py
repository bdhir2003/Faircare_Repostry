#!/usr/bin/env python3
"""
Extract real experiment data from FairCare Jupyter notebooks.
Parses notebook JSON outputs and creates a structured JSON file
for the dashboard to consume.
"""
import json
import re
import os

NOTEBOOK_DIR = '/Users/bobbydhir/Desktop/untitled folder 18'

EXPERIMENTS = {
    'exp001': {
        'file': 'experiment001_all_four_models_class1_fairness_analysis.ipynb',
        'title': 'Experiment 001: Raw Cleaned Baseline',
        'training_data': 'clean_train_dataset.csv',
        'test_data': 'clean_test_dataset.csv',
        'description': 'Models trained on the original cleaned dataset without any class balancing or demographic balancing.',
        'question': 'What happens if we train directly on the original cleaned data?',
        'context': 'The training data preserves the original class imbalance — Class 1 (readmitted) cases are much fewer than Class 0.'
    },
    'exp002': {
        'file': 'experiment002_all_four_models_class1_fairness_analysis.ipynb',
        'title': 'Experiment 002: Class-Balanced Training',
        'training_data': 'balance_classes_training_dataset.csv',
        'test_data': 'clean_test_dataset.csv',
        'description': 'Models trained on a dataset where Class 0 and Class 1 are balanced to equal representation.',
        'question': 'Does balancing the target class improve detection of readmitted patients?',
        'context': 'The test set remains the original unbalanced distribution, so evaluation reflects real-world conditions.'
    },
    'exp003': {
        'file': 'experiment003_all_four_models_class1_fairness_analysis.ipynb',
        'title': 'Experiment 003: Demographic-Balanced Training Only',
        'training_data': 'balance_demographic_variable_training_dataset.csv',
        'test_data': 'clean_test_dataset.csv',
        'description': 'Models trained on a dataset where demographic groups (race, gender, age) are balanced, but target classes are NOT balanced.',
        'question': 'Is demographic balancing alone enough to improve Class 1 fairness?',
        'context': 'This tests whether making demographic groups equal in training helps fairness without addressing class imbalance.'
    },
    'exp004': {
        'file': 'experiment004_all_four_models_class1_fairness_analysis.ipynb',
        'title': 'Experiment 004: Class + Soft Demographic Balancing',
        'training_data': 'Balanceclasses+softbalanced_demographic_groups.csv',
        'test_data': 'clean_test_dataset.csv',
        'description': 'Models trained on a dataset combining class balancing AND soft demographic group balancing.',
        'question': 'Can we improve Class 1 detection while also supporting demographic fairness?',
        'context': 'This is the most comprehensive fairness-aware training strategy tested in the project.'
    }
}

MODEL_NAMES = ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP Neural Network']

def get_source(cell):
    source = cell.get('source', '')
    if isinstance(source, list):
        source = ''.join(source)
    return source

def extract_html_tables(outputs):
    tables = []
    for output in outputs:
        if output.get('output_type') in ('display_data', 'execute_result'):
            html = output.get('data', {}).get('text/html', '')
            if isinstance(html, list):
                html = ''.join(html)
            if '<table' in html.lower():
                tables.append(html)
    return tables

def extract_text(outputs):
    texts = []
    for output in outputs:
        if output.get('output_type') == 'stream':
            t = output.get('text', '')
            if isinstance(t, list):
                t = ''.join(t)
            texts.append(t)
    return '\n'.join(texts)

def find_model_starts(cells):
    """Find cell indices where each model section starts."""
    starts = {}
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'markdown':
            continue
        src = get_source(cell)
        # Match patterns like "# Model 1: Logistic Regression" 
        for model_name in MODEL_NAMES:
            if model_name.lower() in src.lower():
                # Must be a top-level heading (# Model N:)
                for line in src.split('\n'):
                    line_stripped = line.strip()
                    if line_stripped.startswith('# ') and model_name.lower() in line_stripped.lower():
                        if model_name not in starts:
                            starts[model_name] = i
                            break
    return starts

def extract_model_tables(cells, start_idx, end_idx):
    """Extract tables from a model section by looking at sub-section headers."""
    data = {}
    
    # Map sub-section headers to data keys
    section_map = [
        ('Overall Model Performance', 'performance_table'),
        ('Confusion Matrix', 'confusion_matrix'),
    ]
    
    # For fairness matrices, we collect them sequentially
    fairness_keys = [
        ('Performance Fairness', 'perf_fairness'),
        ('Error Fairness', 'error_fairness'),
        ('Calibration Fairness', 'calibration_fairness'),
        ('SHAP Explanation Fairness', 'shap_fairness'),
    ]
    
    current_key = None
    current_fairness_tables = []
    
    for i in range(start_idx, end_idx):
        cell = cells[i]
        src = get_source(cell)
        
        if cell['cell_type'] == 'markdown':
            src_lower = src.lower()
            
            # Check for performance / confusion headers
            for marker, key in section_map:
                if marker.lower() in src_lower:
                    current_key = key
                    current_fairness_tables = []
                    break
            
            # Check for fairness matrix headers
            for marker, key in fairness_keys:
                if marker.lower() in src_lower and 'summary' not in src_lower:
                    current_key = key
                    current_fairness_tables = []
                    break
            
            # Summary headers
            for marker, key in fairness_keys:
                if marker.lower() in src_lower and 'summary' in src_lower:
                    current_key = key + '_summary'
                    break
            
            if 'final model interpretation' in src_lower or 'final interpretation' in src_lower:
                current_key = 'interpretation'
        
        elif cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            tables = extract_html_tables(outputs)
            text = extract_text(outputs)
            
            if current_key and tables:
                if current_key in ('performance_table', 'confusion_matrix'):
                    if current_key not in data:
                        data[current_key] = tables[0]
                elif current_key.endswith('_summary'):
                    if current_key not in data:
                        data[current_key] = tables[0]
                elif current_key in ('perf_fairness', 'error_fairness', 'calibration_fairness', 'shap_fairness'):
                    # Main fairness table
                    base_key = current_key + '_table'
                    if base_key not in data:
                        data[base_key] = tables[0]
                    # If there's a second table in the same output, it's the summary
                    if len(tables) > 1:
                        summary_key = current_key + '_summary'
                        if summary_key not in data:
                            data[summary_key] = tables[1]
            
            if current_key == 'interpretation' and text:
                data['clinical_interpretation'] = text
    
    return data

def parse_perf_values(html):
    """Extract key numeric values from performance table HTML."""
    if not html:
        return {}
    vals = {}
    patterns = [
        ('Accuracy_All_Classes', 'accuracy'),
        ('Recall_Class_1_Readmitted', 'recall_class1'),
        ('Precision_Class_1_Readmitted', 'precision_class1'),
        ('F1_Class_1_Readmitted', 'f1_class1'),
        ('ROC_AUC_Class_1_Readmission_Risk', 'roc_auc'),
        ('FNR_Class_1_Missed_Readmitted', 'fnr'),
    ]
    for metric, key in patterns:
        pat = rf'>\s*{re.escape(metric)}\s*</td>\s*<td[^>]*>\s*([0-9.eE\-+nanNaN]+)\s*</td>'
        m = re.search(pat, html)
        if m:
            try:
                vals[key] = float(m.group(1))
            except ValueError:
                vals[key] = m.group(1)
    return vals

def process_notebook(exp_key, exp_info):
    nb_path = os.path.join(NOTEBOOK_DIR, exp_info['file'])
    if not os.path.exists(nb_path):
        print(f"  WARNING: {nb_path} not found")
        return None
    
    print(f"  Processing {exp_info['file']}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    print(f"    Total cells: {len(cells)}")
    
    # Find model section starts
    starts = find_model_starts(cells)
    print(f"    Found model starts: { {k: v for k,v in starts.items()} }")
    
    # Sort by cell index to determine end boundaries
    sorted_models = sorted(starts.items(), key=lambda x: x[1])
    
    # Find comparison section start
    comparison_start = len(cells)
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            src = get_source(cell).lower()
            if 'final comparison' in src or 'compare all' in src:
                comparison_start = i
                break
    
    result = {
        'meta': exp_info,
        'models': {},
        'comparison': {'master_table': None, 'fairness_summary': None}
    }
    
    for j, (model_name, start_idx) in enumerate(sorted_models):
        if j + 1 < len(sorted_models):
            end_idx = sorted_models[j+1][1]
        else:
            end_idx = comparison_start
        
        print(f"    {model_name}: cells {start_idx} to {end_idx}")
        tables = extract_model_tables(cells, start_idx, end_idx)
        tables['key_values'] = parse_perf_values(tables.get('performance_table', ''))
        
        found = sum(1 for k, v in tables.items() if v and k != 'key_values')
        print(f"      Found {found} data items")
        result['models'][model_name] = tables
    
    # Extract comparison tables
    for i in range(comparison_start, len(cells)):
        cell = cells[i]
        if cell['cell_type'] == 'code':
            tbls = extract_html_tables(cell.get('outputs', []))
            if tbls:
                if not result['comparison']['master_table']:
                    result['comparison']['master_table'] = tbls[0]
                elif not result['comparison']['fairness_summary']:
                    result['comparison']['fairness_summary'] = tbls[0]
    
    return result

def main():
    print("FairCare Dashboard - Notebook Data Extractor v2")
    print("=" * 50)
    
    all_data = {}
    for exp_key, exp_info in EXPERIMENTS.items():
        print(f"\n{exp_key}: {exp_info['title']}")
        r = process_notebook(exp_key, exp_info)
        if r:
            all_data[exp_key] = r
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Saved to: {output_path}")
    for exp_key, exp_data in all_data.items():
        print(f"\n{exp_key}:")
        for mn, md in exp_data['models'].items():
            kv = md.get('key_values', {})
            acc = kv.get('accuracy', 'N/A')
            rc1 = kv.get('recall_class1', 'N/A')
            fnr = kv.get('fnr', 'N/A')
            cnt = sum(1 for k,v in md.items() if v and k != 'key_values')
            print(f"  {mn}: {cnt} items | Acc={acc} | R_C1={rc1} | FNR={fnr}")
        ct = result = exp_data['comparison']
        print(f"  Comparison: master={bool(ct['master_table'])}, fairness={bool(ct['fairness_summary'])}")

if __name__ == '__main__':
    main()
