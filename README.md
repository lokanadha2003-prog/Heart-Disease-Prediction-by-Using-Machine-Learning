# ❤️ Heart Disease Prediction — MSc Dissertation Project

## 📁 Project Files

| File | Description |
|------|-------------|
| `heart_disease_dissertation.py` | Full ML pipeline (EDA → Training → Evaluation) |
| `app.py` | Streamlit web application |
| `cardio_train.csv` | Dataset (70,000 records, semicolon-separated) |
| `requirements.txt` | Python dependencies |
| `best_model.pkl` | Saved XGBoost model (generated after running dissertation.py) |
| `scaler.pkl` | Saved StandardScaler (generated after running dissertation.py) |
| `feature_names.pkl` | Feature order list (generated after running dissertation.py) |

---

## 🚀 HOW TO RUN LOCALLY (VS Code)

### Step 1 — Install Dependencies
Open a terminal in VS Code and run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit
```

### Step 2 — Run the Dissertation Script (Train Models)
```bash
python heart_disease_dissertation.py
```
This will:
- Load and clean the dataset
- Generate all 14 EDA and evaluation plots (saved as .png files)
- Train all 5 models with GridSearchCV
- Print cross-validation and test metrics
- Save: `best_model.pkl`, `scaler.pkl`, `feature_names.pkl`

⏱️ **Expected runtime**: 15–40 minutes (SVM and XGBoost GridSearch are slow on 70K rows)

### Step 3 — Launch the Streamlit App
```bash
streamlit run app.py
```
The browser will open automatically at: `http://localhost:8501`


## 📊 Output Plots Generated

| Plot | Description |
|------|-------------|
| `plot_01_target_distribution.png` | Class balance of cardio (0 vs 1) |
| `plot_02_age_distribution.png` | Age histograms split by outcome |
| `plot_03_gender_distribution.png` | Gender × disease prevalence |
| `plot_04_systolic_bp.png` | Blood pressure distribution |
| `plot_05_cholesterol_glucose.png` | Cholesterol & glucose vs outcome |
| `plot_06_lifestyle_factors.png` | Smoking, alcohol, activity vs outcome |
| `plot_07_bmi_distribution.png` | BMI distribution by outcome |
| `plot_08_correlation_heatmap.png` | Correlation matrix of all features |
| `plot_09_boxplots.png` | Age, BP, weight boxplots by outcome |
| `plot_10_confusion_matrices.png` | All 5 models' confusion matrices |
| `plot_11_roc_curves.png` | ROC curves comparison |
| `plot_12_feature_importance.png` | RF and XGBoost feature importance |
| `plot_13_model_comparison.png` | All metrics grouped bar chart |
| `plot_14_cv_boxplots.png` | 5-fold CV accuracy distribution |

---

## 📖 Dataset Column Reference

| Column | Description | Type |
|--------|-------------|------|
| `age` | Age in years (converted from days) | Numeric |
| `gender` | 1 = Female, 2 = Male | Categorical |
| `height` | Height in cm | Numeric |
| `weight` | Weight in kg | Numeric |
| `ap_hi` | Systolic blood pressure (mmHg) | Numeric |
| `ap_lo` | Diastolic blood pressure (mmHg) | Numeric |
| `cholesterol` | 1=Normal, 2=Above Normal, 3=Well Above Normal | Ordinal |
| `gluc` | 1=Normal, 2=Above Normal, 3=Well Above Normal | Ordinal |
| `smoke` | 0=No, 1=Yes | Binary |
| `alco` | 0=No, 1=Yes | Binary |
| `active` | 0=No, 1=Yes | Binary |
| `cardio` | **Target**: 0=No Disease, 1=Disease | Binary |
| `bmi` | Engineered: weight/(height/100)² | Numeric |
