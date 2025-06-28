#!/usr/bin/env python3
"""
Fast Transformer Fault Detection Model Comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("TRANSFORMER FAULT DETECTION - MODEL COMPARISON")
    print("="*55)

    # Load dataset
    data = pd.read_csv('data/collection_dga_data.csv')
    print(f"Dataset: {data.shape[0]} samples")

    # Fault distribution
    print("Fault distribution:")
    for fault, count in data['fault_type'].value_counts().items():
        print(f"  {fault}: {count} samples")

    # Features and labels
    gas_columns = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2']
    X = data[gas_columns].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['fault_type'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples")

    results = {}

    # 1. Linear Regression (Ridge)
    print("\n1. Linear Regression (Ridge Regularization)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled, y_train)
    max_class = len(label_encoder.classes_) - 1
    y_pred_lr = np.clip(np.round(ridge.predict(X_test_scaled)).astype(int), 0, max_class)

    lr_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, average='weighted'),
        'recall': recall_score(y_test, y_pred_lr, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_lr, average='weighted')
    }
    results['Linear Regression'] = lr_metrics
    print(f"   Accuracy: {lr_metrics['accuracy']:.4f}")

    # 2. Decision Tree
    print("\n2. Decision Tree Classifier")
    dt = DecisionTreeClassifier(max_depth=7, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    dt_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'precision': precision_score(y_test, y_pred_dt, average='weighted'),
        'recall': recall_score(y_test, y_pred_dt, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_dt, average='weighted')
    }
    results['Decision Tree'] = dt_metrics
    print(f"   Accuracy: {dt_metrics['accuracy']:.4f}")

    # 3. Random Forest
    print("\n3. Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, average='weighted'),
        'recall': recall_score(y_test, y_pred_rf, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_rf, average='weighted')
    }
    results['Random Forest'] = rf_metrics
    print(f"   Accuracy: {rf_metrics['accuracy']:.4f}")

    # 4. Enhanced RF
    print("\n4. Enhanced Random Forest")
    erf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    erf.fit(X_train, y_train)
    y_pred_erf = erf.predict(X_test)

    erf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_erf),
        'precision': precision_score(y_test, y_pred_erf, average='weighted'),
        'recall': recall_score(y_test, y_pred_erf, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_erf, average='weighted')
    }
    results['Enhanced RF'] = erf_metrics
    print(f"   Accuracy: {erf_metrics['accuracy']:.4f}")

    # Results Summary
    print("\n" + "="*55)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*55)
    comparison_df = pd.DataFrame(results).T.sort_values(by='accuracy', ascending=False)
    print(f"{'Model':<16} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 56)
    for model, row in comparison_df.iterrows():
        print(f"{model:<16} {row['accuracy']:<10.4f} {row['precision']:<10.4f} "
              f"{row['recall']:<10.4f} {row['f1_score']:<10.4f}")

    best_model = comparison_df.index[0]
    best_accuracy = comparison_df.iloc[0]['accuracy']

    # Feature Importance (from RF)
    print("\n" + "="*55)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*55)
    feature_importance = list(zip(gas_columns, rf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (gas, importance) in enumerate(feature_importance, 1):
        print(f"  {i}. {gas}: {importance:.4f}")
    most_important_gas = feature_importance[0][0]

    # Prediction Examples
    print("\n" + "="*55)
    print("PREDICTION EXAMPLES")
    print("="*55)
    test_cases = {
        'Normal Operation': [50, 120, 3, 20, 30, 180, 2500],
        'Partial Discharge': [800, 45, 10, 35, 15, 95, 1800],
        'Thermal Fault': [25, 200, 20, 90, 50, 250, 3500],
        'Arcing Fault': [20, 85, 150, 100, 30, 150, 2200]
    }

    for label, gases in test_cases.items():
        input_data = np.array([gases])
        print(f"\n{label}: {gases}")
        print(f"  Decision Tree: {label_encoder.inverse_transform([dt.predict(input_data)[0]])[0]}")
        print(f"  Random Forest: {label_encoder.inverse_transform([rf.predict(input_data)[0]])[0]}")
        print(f"  Enhanced RF  : {label_encoder.inverse_transform([erf.predict(input_data)[0]])[0]}")

    # Insights
    print("\n" + "="*55)
    print("GAS ANALYSIS INSIGHTS")
    print("="*55)
    print("• High H2 (>500 ppm): Likely Partial Discharge")
    print("• High CH4 + C2H6: Thermal overheating")
    print("• High C2H2 (>50 ppm): Arcing")
    print("• High CO (>300 ppm): Cellulose degradation")

    # Suggestions
    print("\n" + "="*55)
    print("ACCURACY IMPROVEMENT RECOMMENDATIONS")
    print("="*55)
    if best_accuracy >= 0.95:
        print("✓ Excellent performance - Ready for production use")
    elif best_accuracy >= 0.90:
        print("✓ Very good - Minor improvements possible")
    elif best_accuracy >= 0.85:
        print("✓ Good - Improvements recommended")
    else:
        print("⚠ Needs improvement")

    print("1. Collect more samples")
    print("2. Add gas ratios (e.g., Rogers ratios)")
    print("3. Try boosting models like XGBoost")
    print("4. Validate with domain experts")

    # Summary
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    print(f"Dataset size: {len(data)} samples")
    print(f"Best model  : {best_model} (Accuracy: {best_accuracy:.4f})")
    print(f"Top feature : {most_important_gas}")
    print("✓ Machine Learning enhances transformer fault diagnosis!")

    return results, best_model, best_accuracy


if _name_ == "_main_":
    results, best_model, accuracy = main()
    print(f"\nModel comparison completed successfully!")
    print(f"Recommended model: {best_model} (Accuracy: {accuracy:.4f})")