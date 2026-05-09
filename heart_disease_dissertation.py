# ============================================================
# MSc DISSERTATION: HEART DISEASE PREDICTION
# Author: Lokesh
# Dataset: Cardiovascular Disease Dataset (cardio_train.csv)
# Models: Logistic Regression, Decision Tree, Random Forest,
#         SVM, XGBoost
# Techniques: GridSearchCV, 5-Fold CV, ROC-AUC, Feature Importance
# ============================================================

# ============================================================
# STEP 1: INSTALL REQUIRED LIBRARIES
# Run this in terminal before running the script:
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
# ============================================================

# ============================================================
# STEP 2: IMPORT LIBRARIES
# We import all necessary libraries for data processing,
# visualization, machine learning, and evaluation.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')  # Suppress non-critical warnings

# Scikit-learn: Preprocessing, Model Selection, Metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# XGBoost (gradient boosted trees — state-of-the-art model)
from xgboost import XGBClassifier

print("✅ All libraries imported successfully.")

# ============================================================
# STEP 3: LOAD DATASET
# The dataset is a CSV file with semicolon (;) separators.
# It contains 70,000 patient records with 13 columns.
# Target variable: 'cardio' — 1 = heart disease, 0 = no disease
# ============================================================

df = pd.read_csv('cardio_train.csv', sep=';')

print("\n📋 Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn Data Types:")
print(df.dtypes)

# ============================================================
# STEP 4: DATA CLEANING
#
# 4a. Drop 'id' — it is just a row identifier, not a feature.
# 4b. Convert 'age' from days to years (more interpretable).
# 4c. Check and confirm no missing values.
# ============================================================

# Drop identifier column
df = df.drop('id', axis=1)

# Convert age: dataset stores age in days → divide by 365.25
df['age'] = (df['age'] / 365.25).round(1)

# Check for missing values
print("\n🔍 Missing Values Per Column:")
print(df.isnull().sum())
print("\nNo missing values found — dataset is clean." if df.isnull().sum().sum() == 0
      else "⚠️ Missing values found — handle before modelling.")

# ============================================================
# STEP 5: REMOVE OUTLIERS
#
# Blood pressure values (ap_hi = systolic, ap_lo = diastolic)
# outside physiologically possible ranges indicate data entry
# errors. We remove these to prevent model distortion.
#
# Medical reference ranges:
#   Systolic (ap_hi): 50–250 mmHg
#   Diastolic (ap_lo): 30–150 mmHg
#   Height: 100–250 cm
#   Weight: 30–200 kg
# ============================================================

before = len(df)

df = df[(df['ap_hi'] > 50)  & (df['ap_hi'] < 250)]
df = df[(df['ap_lo'] > 30)  & (df['ap_lo'] < 150)]
df = df[(df['height'] > 100) & (df['height'] < 250)]
df = df[(df['weight'] > 30)  & (df['weight'] < 200)]

# Also remove rows where diastolic >= systolic (physiologically impossible)
df = df[df['ap_hi'] >= df['ap_lo']]

after = len(df)
print(f"\n🧹 Outliers removed: {before - after} rows | Remaining: {after} rows")

# ============================================================
# STEP 6: EXPLORATORY DATA ANALYSIS (EDA)
#
# EDA is a critical phase in any dissertation. We explore:
# - Class balance (target distribution)
# - Distribution of each feature
# - Relationships between features and the target
# - Correlation structure among all variables
# ============================================================

print("\n📊 Starting EDA — generating plots...")

# ----------- 6a. TARGET VARIABLE DISTRIBUTION -----------
# Understanding whether classes are balanced is essential.
# Imbalanced data leads to biased classifiers.

fig, ax = plt.subplots(figsize=(7, 4))
counts = df['cardio'].value_counts()
bars = ax.bar(['No Heart Disease (0)', 'Heart Disease (1)'],
               counts.values,
               color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=11)
ax.set_title('Target Variable: Heart Disease Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Patients')
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig('plot_01_target_distribution.png', dpi=150)
plt.show()
print("✅ Plot 1 saved: Target Distribution")

# ----------- 6b. AGE DISTRIBUTION -----------
# Age is one of the strongest risk factors for cardiovascular disease.
# We compare the age distribution between diseased and non-diseased groups.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, label, colour, title in zip(
        axes, [0, 1], ['#2ecc71', '#e74c3c'],
        ['No Heart Disease', 'Heart Disease']):
    ax.hist(df[df['cardio'] == label]['age'], bins=30,
            color=colour, edgecolor='black', alpha=0.85)
    ax.set_title(f'Age Distribution — {title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Age (Years)')
    ax.set_ylabel('Count')
    median_age = df[df['cardio'] == label]['age'].median()
    ax.axvline(median_age, color='navy', linestyle='--', linewidth=1.5,
               label=f'Median: {median_age:.1f} yrs')
    ax.legend()
plt.suptitle('Age Distribution by Heart Disease Status', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_02_age_distribution.png', dpi=150)
plt.show()
print("✅ Plot 2 saved: Age Distribution")

# ----------- 6c. GENDER DISTRIBUTION -----------
# Gender (1 = Female, 2 = Male) may influence risk.
# We visualise the count of heart disease cases per gender.

gender_cardio = df.groupby(['gender', 'cardio']).size().unstack()
gender_cardio.index = ['Female', 'Male']
gender_cardio.columns = ['No Disease', 'Heart Disease']

fig, ax = plt.subplots(figsize=(7, 5))
gender_cardio.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_title('Heart Disease Prevalence by Gender', fontsize=13, fontweight='bold')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
ax.set_xticklabels(['Female', 'Male'], rotation=0)
ax.legend(title='Outcome')
plt.tight_layout()
plt.savefig('plot_03_gender_distribution.png', dpi=150)
plt.show()
print("✅ Plot 3 saved: Gender Distribution")

# ----------- 6d. BLOOD PRESSURE (SYSTOLIC) DISTRIBUTION -----------
# High blood pressure (hypertension) is the #1 risk factor for heart disease.

fig, ax = plt.subplots(figsize=(9, 5))
for label, colour, name in zip([0, 1], ['#2ecc71', '#e74c3c'],
                                 ['No Heart Disease', 'Heart Disease']):
    ax.hist(df[df['cardio'] == label]['ap_hi'], bins=40,
            color=colour, alpha=0.65, edgecolor='black', label=name)
ax.set_title('Systolic Blood Pressure Distribution by Heart Disease Status',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Systolic BP (mmHg)')
ax.set_ylabel('Count')
ax.axvline(120, color='blue', linestyle='--', linewidth=1.2, label='Normal ≤120')
ax.axvline(140, color='orange', linestyle='--', linewidth=1.2, label='Hypertension ≥140')
ax.legend()
plt.tight_layout()
plt.savefig('plot_04_systolic_bp.png', dpi=150)
plt.show()
print("✅ Plot 4 saved: Systolic BP Distribution")

# ----------- 6e. CHOLESTEROL & GLUCOSE LEVELS -----------
# Cholesterol (1=normal, 2=above normal, 3=well above normal)
# Glucose (1=normal, 2=above normal, 3=well above normal)
# Higher levels are associated with increased risk.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title_label in zip(axes,
                                  ['cholesterol', 'gluc'],
                                  ['Cholesterol Level', 'Glucose Level']):
    ct = pd.crosstab(df[col], df['cardio'], normalize='index') * 100
    ct.index = ['Normal', 'Above Normal', 'Well Above Normal']
    ct.columns = ['No Disease', 'Heart Disease']
    ct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'],
            edgecolor='black', rot=15)
    ax.set_title(f'{title_label} vs Heart Disease (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel(title_label)
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Outcome')
plt.suptitle('Cholesterol & Glucose Levels — Disease Proportion', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_05_cholesterol_glucose.png', dpi=150)
plt.show()
print("✅ Plot 5 saved: Cholesterol & Glucose")

# ----------- 6f. LIFESTYLE FACTORS -----------
# Smoking, Alcohol, and Physical Activity are binary (0/1).
# We visualise the disease rate in each lifestyle group.

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title_label in zip(axes,
                                  ['smoke', 'alco', 'active'],
                                  ['Smoking', 'Alcohol Intake', 'Physical Activity']):
    ct = df.groupby([col, 'cardio']).size().unstack(fill_value=0)
    ct.index = ['No', 'Yes']
    ct.columns = ['No Disease', 'Heart Disease']
    ct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'],
            edgecolor='black', rot=0)
    ax.set_title(f'{title_label} vs Heart Disease', fontsize=11, fontweight='bold')
    ax.set_xlabel(title_label)
    ax.set_ylabel('Count')
    ax.legend(title='Outcome')
plt.suptitle('Lifestyle Factors & Heart Disease', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_06_lifestyle_factors.png', dpi=150)
plt.show()
print("✅ Plot 6 saved: Lifestyle Factors")

# ----------- 6g. BMI DISTRIBUTION -----------
# BMI (Body Mass Index) = weight (kg) / height (m)^2
# Overweight (≥25) and obese (≥30) individuals face higher risk.

df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

fig, ax = plt.subplots(figsize=(9, 5))
for label, colour, name in zip([0, 1], ['#2ecc71', '#e74c3c'],
                                 ['No Heart Disease', 'Heart Disease']):
    ax.hist(df[(df['cardio'] == label) & (df['bmi'] < 60)]['bmi'],
            bins=40, color=colour, alpha=0.65, edgecolor='black', label=name)
ax.axvline(25, color='orange', linestyle='--', linewidth=1.5, label='Overweight ≥25')
ax.axvline(30, color='red', linestyle='--', linewidth=1.5, label='Obese ≥30')
ax.set_title('BMI Distribution by Heart Disease Status', fontsize=12, fontweight='bold')
ax.set_xlabel('BMI (kg/m²)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('plot_07_bmi_distribution.png', dpi=150)
plt.show()
print("✅ Plot 7 saved: BMI Distribution")

# ----------- 6h. CORRELATION HEATMAP -----------
# A heatmap of Pearson correlations reveals which features
# are most linearly related to the target and to each other.
# High inter-feature correlation (multicollinearity) can affect
# models like Logistic Regression.

plt.figure(figsize=(12, 9))
corr = df.drop('bmi', axis=1).corr()  # Exclude derived BMI for core analysis
mask = np.triu(np.ones_like(corr, dtype=bool))  # Show only lower triangle
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.7})
plt.title('Correlation Matrix — All Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_08_correlation_heatmap.png', dpi=150)
plt.show()
print("✅ Plot 8 saved: Correlation Heatmap")

# ----------- 6i. BOXPLOTS — KEY NUMERIC FEATURES BY OUTCOME -----------
# Boxplots show median, IQR, and outliers per class.

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flatten(), ['age', 'ap_hi', 'ap_lo', 'weight']):
    df.boxplot(column=col, by='cardio', ax=ax,
               patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'{col.upper()} by Heart Disease Status', fontsize=11, fontweight='bold')
    ax.set_xlabel('0 = No Disease | 1 = Disease')
    ax.set_ylabel(col)
plt.suptitle('Numeric Feature Distribution by Outcome', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_09_boxplots.png', dpi=150)
plt.show()
print("✅ Plot 9 saved: Boxplots")

# ============================================================
# STEP 7: FEATURE ENGINEERING
#
# We add a BMI feature as it captures the combined effect of
# height and weight in a clinically meaningful way.
# (BMI already calculated above for EDA — we keep it in X)
# ============================================================

# The 'bmi' column is already in df from the EDA step.
# No additional engineering needed for this dataset.

print("\n🔧 Feature Engineering: BMI column retained as additional feature.")

# ============================================================
# STEP 8: DEFINE FEATURES AND TARGET
#
# X = feature matrix (all columns except target 'cardio')
# y = target vector (0 = no disease, 1 = disease)
# ============================================================

X = df.drop('cardio', axis=1)
y = df['cardio']

print(f"\n🎯 Features: {list(X.columns)}")
print(f"🎯 Target: 'cardio' | Classes: {y.unique()}")
print(f"🎯 Dataset shape: X={X.shape}, y={y.shape}")

# ============================================================
# STEP 9: TRAIN-TEST SPLIT
#
# We split 80% for training and 20% for testing.
# stratify=y ensures the same class ratio in both splits.
# random_state=42 ensures reproducibility.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🔀 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ============================================================
# STEP 10: FEATURE SCALING
#
# StandardScaler transforms features to zero mean and unit variance.
# Formula: z = (x - μ) / σ
#
# Why scale?
# - Logistic Regression and SVM are distance-based and highly
#   sensitive to feature magnitudes.
# - Tree-based models (RF, DT, XGBoost) do NOT require scaling,
#   but we apply it uniformly for consistency.
#
# IMPORTANT: fit ONLY on training data, then transform both.
# Fitting on test data would constitute data leakage.
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save the scaler for use in the Streamlit app
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ StandardScaler fitted and saved as 'scaler.pkl'")

# ============================================================
# STEP 11: MODEL TRAINING WITH HYPERPARAMETER TUNING (GridSearchCV)
#
# GridSearchCV exhaustively searches a parameter grid using
# cross-validation to find the best hyperparameters.
# This is a key component of dissertation-level methodology.
#
# cv=StratifiedKFold(5): 5-fold stratified cross-validation
#   ensures each fold preserves the class distribution.
# scoring='roc_auc': optimise for area under the ROC curve,
#   appropriate for binary medical classification.
# n_jobs=-1: use all CPU cores for parallel computation.
#
# NOTE: GridSearchCV on SVM may take several minutes on 70K rows.
# We use a smaller parameter grid for SVM to keep runtime manageable.
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------ 11a. LOGISTIC REGRESSION ------
print("\n⏳ Tuning Logistic Regression...")
lr_params = {
    'C': [0.01, 0.1, 1, 10],         # Regularisation strength (smaller = stronger)
    'solver': ['lbfgs', 'saga'],      # Optimisation algorithm
    'max_iter': [500]
}
lr_grid = GridSearchCV(LogisticRegression(random_state=42),
                       lr_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
lr_grid.fit(X_train_scaled, y_train)
best_lr = lr_grid.best_estimator_
print(f"   Best params: {lr_grid.best_params_} | CV ROC-AUC: {lr_grid.best_score_:.4f}")

# ------ 11b. DECISION TREE ------
print("⏳ Tuning Decision Tree...")
dt_params = {
    'max_depth': [5, 10, 15, None],      # Depth of tree (None = fully grown)
    'min_samples_split': [2, 5, 10],     # Min samples to split a node
    'criterion': ['gini', 'entropy']     # Split quality measure
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       dt_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
dt_grid.fit(X_train_scaled, y_train)
best_dt = dt_grid.best_estimator_
print(f"   Best params: {dt_grid.best_params_} | CV ROC-AUC: {dt_grid.best_score_:.4f}")

# ------ 11c. RANDOM FOREST ------
print("⏳ Tuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200],          # Number of decision trees
    'max_depth': [10, 15, None],         # Max tree depth
    'min_samples_split': [2, 5],         # Min samples to split a node
    'max_features': ['sqrt', 'log2']     # Features considered per split
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_
print(f"   Best params: {rf_grid.best_params_} | CV ROC-AUC: {rf_grid.best_score_:.4f}")

# ------ 11d. SVM ------
print("⏳ Tuning SVM (may take a few minutes)...")
svm_params = {
    'C': [0.1, 1, 10],          # Margin width control
    'kernel': ['rbf', 'linear'] # Kernel function
}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42),
                        svm_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_
print(f"   Best params: {svm_grid.best_params_} | CV ROC-AUC: {svm_grid.best_score_:.4f}")

# ------ 11e. XGBOOST ------
print("⏳ Tuning XGBoost...")
xgb_params = {
    'n_estimators': [100, 200],          # Number of boosting rounds
    'max_depth': [3, 5, 7],              # Tree depth per round
    'learning_rate': [0.05, 0.1, 0.2],  # Step size shrinkage
    'subsample': [0.8, 1.0],            # Row subsampling ratio
    'colsample_bytree': [0.8, 1.0]      # Column subsampling ratio
}
xgb_grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                  random_state=42, n_jobs=-1),
    xgb_params, cv=cv, scoring='roc_auc', n_jobs=1, verbose=0
)
xgb_grid.fit(X_train_scaled, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"   Best params: {xgb_grid.best_params_} | CV ROC-AUC: {xgb_grid.best_score_:.4f}")

print("\n✅ All models tuned with GridSearchCV!")

# Save the best model (XGBoost — typically best performer) for Streamlit app
joblib.dump(best_xgb, 'best_model.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
print("✅ Best XGBoost model saved as 'best_model.pkl'")
print("✅ Feature names saved as 'feature_names.pkl'")

# ============================================================
# STEP 12: 5-FOLD CROSS-VALIDATION SCORES
#
# Cross-validation provides a reliable estimate of model
# generalisation performance by evaluating on multiple folds.
# We report mean and standard deviation of accuracy.
# ============================================================

print("\n📊 5-Fold Cross-Validation Results (Accuracy):")
print("-" * 55)

models_cv = {
    'Logistic Regression': best_lr,
    'Decision Tree':       best_dt,
    'Random Forest':       best_rf,
    'SVM':                 best_svm,
    'XGBoost':             best_xgb
}

cv_results = {}
for name, model in models_cv.items():
    scores = cross_val_score(model, X_train_scaled, y_train,
                              cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<22} → Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")

print("-" * 55)

# ============================================================
# STEP 13: GENERATE PREDICTIONS ON HELD-OUT TEST SET
#
# After tuning and CV, we evaluate on the test set ONCE.
# Repeated test-set evaluation risks overfitting to the test set.
# ============================================================

models_final = {
    'Logistic Regression': best_lr,
    'Decision Tree':       best_dt,
    'Random Forest':       best_rf,
    'SVM':                 best_svm,
    'XGBoost':             best_xgb
}

predictions   = {}
probabilities = {}

for name, model in models_final.items():
    predictions[name]   = model.predict(X_test_scaled)
    probabilities[name] = model.predict_proba(X_test_scaled)[:, 1]

print("\n✅ Predictions generated for all models.")

# ============================================================
# STEP 14: EVALUATION METRICS TABLE
#
# We compute five standard classification metrics:
#
#   Accuracy  = (TP+TN) / Total       — overall correctness
#   Precision = TP / (TP+FP)          — of predicted positives, how many are correct
#   Recall    = TP / (TP+FN)          — of actual positives, how many were found
#   F1 Score  = 2 * (P*R)/(P+R)       — harmonic mean of Precision and Recall
#   ROC-AUC   = area under ROC curve  — rank-based discrimination ability
#
# In a medical context, Recall is critical: a missed positive (FN)
# means a patient with heart disease is told they are healthy.
# ============================================================

print("\n📊 Evaluation Metrics on Test Set:")
print("=" * 80)

results = []
for name in models_final:
    y_pred = predictions[name]
    y_prob = probabilities[name]
    results.append({
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1 Score':  round(f1_score(y_test, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4)
    })

results_df = pd.DataFrame(results).set_index('Model')
print(results_df.to_string())
print("=" * 80)

# ============================================================
# STEP 15: CONFUSION MATRICES — ALL MODELS
#
# A confusion matrix shows:
#   TP = True Positives  (correctly predicted disease)
#   TN = True Negatives  (correctly predicted no disease)
#   FP = False Positives (predicted disease, actually healthy)
#   FN = False Negatives (predicted healthy, actually disease)
#
# In clinical settings, FN (missed diagnosis) is the most
# dangerous error — we want high Recall to minimise FN.
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, (name, model) in enumerate(models_final.items()):
    cm = confusion_matrix(y_test, predictions[name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['No Disease', 'Heart Disease'])
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(f'{name}\nAcc: {results_df.loc[name,"Accuracy"]:.3f} | '
                       f'AUC: {results_df.loc[name,"ROC-AUC"]:.3f}',
                       fontsize=10, fontweight='bold')

axes[-1].axis('off')  # Hide unused subplot
plt.suptitle('Confusion Matrices — All Models (Test Set)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_10_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 10 saved: Confusion Matrices")

# ============================================================
# STEP 16: ROC CURVES — ALL MODELS
#
# The ROC (Receiver Operating Characteristic) curve plots
# True Positive Rate (Recall) vs False Positive Rate at
# every possible classification threshold.
#
# A perfect classifier has AUC = 1.0.
# A random classifier has AUC = 0.5 (diagonal line).
# Higher AUC = better discrimination between classes.
# ============================================================

plt.figure(figsize=(9, 7))

colours = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

for (name, model), colour in zip(models_final.items(), colours):
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc_val = roc_auc_score(y_test, probabilities[name])
    plt.plot(fpr, tpr, color=colour, linewidth=2,
             label=f'{name} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Classifier (AUC = 0.50)')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('ROC Curves — All Models Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot_11_roc_curves.png', dpi=150)
plt.show()
print("✅ Plot 11 saved: ROC Curves")

# ============================================================
# STEP 17: FEATURE IMPORTANCE VISUALISATION
#
# Feature importance tells us which input variables contribute
# most to the model's predictions.
#
# Random Forest: average reduction in impurity (Gini/Entropy)
# XGBoost: gain-based importance (fraction of splits using feature)
#
# These plots are excellent for dissertation discussion sections:
# they validate clinical knowledge (e.g., age and BP matter most)
# and support model interpretability.
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
feature_names = list(X.columns)

for ax, model, title in zip(axes,
                              [best_rf, best_xgb],
                              ['Random Forest (Gini Importance)',
                               'XGBoost (Gain Importance)']):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    colours = ['#e74c3c' if v == importances.max() else '#3498db' for v in importances]
    importances.plot(kind='barh', ax=ax, color=colours, edgecolor='black')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    for i, (val, name) in enumerate(zip(importances.values, importances.index)):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)

plt.suptitle('Feature Importance — Random Forest vs XGBoost',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_12_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 12 saved: Feature Importance")

# ============================================================
# STEP 18: MODEL COMPARISON VISUALISATION
#
# A grouped bar chart comparing all metrics across all models
# provides a clear visual summary for the dissertation results section.
# ============================================================

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
model_names = results_df.index.tolist()

x = np.arange(len(metrics))
width = 0.15
colours_bar = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

fig, ax = plt.subplots(figsize=(15, 7))

for i, (name, colour) in enumerate(zip(model_names, colours_bar)):
    values = [results_df.loc[name, m] for m in metrics]
    bars = ax.bar(x + i * width, values, width, label=name,
                  color=colour, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xlabel('Evaluation Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison — All Metrics on Test Set', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.60, 0.85)
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_13_model_comparison.png', dpi=150)
plt.show()
print("✅ Plot 13 saved: Model Comparison")

# ============================================================
# STEP 19: CROSS-VALIDATION BOX PLOTS
#
# Visualise the stability of each model's performance across
# the 5 CV folds. Narrower boxes indicate more stable models.
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
cv_data = [cv_results[name] for name in models_cv]
bp = ax.boxplot(cv_data, labels=models_cv.keys(), patch_artist=True)

for patch, colour in zip(bp['boxes'], colours_bar):
    patch.set_facecolor(colour)
    patch.set_alpha(0.7)

ax.set_title('5-Fold Cross-Validation Accuracy Distribution',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_xticklabels(models_cv.keys(), rotation=15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_14_cv_boxplots.png', dpi=150)
plt.show()
print("✅ Plot 14 saved: Cross-Validation Boxplots")

# ============================================================
# STEP 20: DETAILED CLASSIFICATION REPORT (BEST MODEL)
#
# The sklearn classification_report gives per-class precision,
# recall, F1, and support — useful for the dissertation appendix.
# ============================================================

best_model_name = results_df['ROC-AUC'].idxmax()
print(f"\n🏆 Best Model by ROC-AUC: {best_model_name}")
print(f"\n📄 Detailed Classification Report — {best_model_name}:")
print(classification_report(y_test, predictions[best_model_name],
                              target_names=['No Heart Disease', 'Heart Disease']))

# ============================================================
# STEP 21: FINAL SUMMARY TABLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print(results_df.to_string())
print("\n🏆 Best ROC-AUC:", results_df['ROC-AUC'].max(),
      "→", results_df['ROC-AUC'].idxmax())
print("🏆 Best Accuracy:", results_df['Accuracy'].max(),
      "→", results_df['Accuracy'].idxmax())
print("=" * 80)
print("\n✅ All analysis complete. Model & scaler saved for Streamlit deployment.")
print("📁 Files created: best_model.pkl, scaler.pkl, feature_names.pkl")
print("📊 Plots saved: plot_01 through plot_14 (.png)")
