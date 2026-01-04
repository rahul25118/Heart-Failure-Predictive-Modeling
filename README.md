# Heart Failure Clinical Records: Predictive Modeling and Analysis

## Executive Summary

This project develops a machine learning model to predict patient death events in heart failure clinical cases using the UCI Heart Failure Clinical Records dataset. Using exploratory data analysis (EDA), data preprocessing with class imbalancing techniques (SMOTE), and hyperparameter tuning, we achieved a Random Forest model with 89% ROC-AUC score and 63% recall for death events on the test set. The model identifies key clinical predictors including ejection fraction, serum creatinine, and follow-up time as the strongest indicators of mortality risk.

---

## 1. Introduction

### Background

Heart failure is a major global health burden, affecting millions of patients and leading to significant morbidity and mortality. Early identification of high-risk patients enables timely clinical intervention and better resource allocation in healthcare systems.[1] Machine learning models trained on clinical data can provide decision-support tools for clinicians, improving patient outcomes through better risk stratification.

### Objectives

The primary objectives of this study are:

1. To explore and analyze the UCI Heart Failure Clinical Records dataset, understanding key demographic and clinical features
2. To develop and compare multiple machine learning classification models for predicting mortality
3. To apply advanced techniques (SMOTE, hyperparameter tuning) to improve model robustness on imbalanced data
4. To identify and interpret the most important clinical predictors of death events
5. To document the complete machine learning pipeline for reproducibility and deployment

### Dataset Overview

The Heart Failure Clinical Records dataset is sourced from the UCI Machine Learning Repository[1] and contains clinical records from 299 heart failure patients with 13 features and a binary target variable (DEATH_EVENT) indicating mortality during follow-up.

---

## 2. Dataset Description and Data Quality

### Data Loading and Structure

Dataset Dimensions: 299 rows × 13 columns
Total Records: 299
Features: 12 (numeric and binary)
Target Variable: DEATH_EVENT (binary: 0 = survived, 1 = death)

### Feature Descriptions

| Feature | Type | Description | Range |
| --- | --- | --- | --- |
| age | numeric | Patient age in years | 40–95 |
| anaemia | binary | Presence of decreased red blood cells (1 = yes) | 0, 1 |
| creatinine_phosphokinase | numeric | CPK enzyme level in blood (mcg/L) | 23–7,861 |
| diabetes | binary | Presence of diabetes (1 = yes) | 0, 1 |
| ejection_fraction | numeric | Percentage of blood leaving heart per contraction | 14–80% |
| high_blood_pressure | binary | Presence of hypertension (1 = yes) | 0, 1 |
| platelets | numeric | Platelet count (kiloplatelets/mL) | 25.1–850.0K |
| serum_creatinine | numeric | Serum creatinine level in blood (mg/dL) | 0.5–9.4 |
| serum_sodium | numeric | Serum sodium level in blood (mEq/L) | 113–148 |
| sex | binary | Patient sex (0 = female, 1 = male) | 0, 1 |
| smoking | binary | Smoking status (1 = yes) | 0, 1 |
| time | numeric | Follow-up period in days | 4–285 |
| DEATH_EVENT | binary | Target: patient death during follow-up | 0, 1 |

### Data Quality Assessment

| Aspect | Finding |
| --- | --- |
| Missing Values | None (0 missing values across all 13 columns) |
| Duplicate Rows | None (0 duplicate records) |
| Data Types | 3 float64, 10 int64 (all as expected) |
| Target Class Balance | 67.9% class 0 (survived), 32.1% class 1 (death) |

**Conclusion:** The dataset is clean, complete, and ready for analysis without requiring imputation or extensive preprocessing.

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Univariate Analysis

#### Numerical Features Distribution

The pairplot visualization (Figure 1) shows the distributions of key numerical features colored by death event outcome. Key observations include:

- **age**: Right-skewed distribution with median around 60 years. Patients who died tend to be older (mean 65.2 vs 58.8 years).
- **ejection_fraction**: Approximately uniform distribution between 14% and 80%. Patients who died show notably lower ejection fraction (mean 33.5% vs 40.3%), indicating weaker heart pumping.
- **serum_creatinine**: Right-skewed with most values between 0.5 and 2.0 mg/dL. Higher values (indicating worse kidney function) are more common among patients who died.
- **serum_sodium**: Approximately normal distribution centered around 137 mEq/L with a few low outliers. Patients who died have slightly lower serum sodium levels.
- **time**: Follow-up period ranges from 4 to 285 days. Patients who died have shorter follow-up times (mean 70.9 vs 158.3 days), suggesting earlier mortality.
- **creatinine_phosphokinase** and **platelets**: Both show right-skewed distributions with some high outliers, typical of clinical enzyme and cell count measurements.

![Figure 1: Pairplot of key numerical features colored by DEATH_EVENT outcome](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/9f462b81-2cd7-4d89-a197-e929cae3e6cd)

#### Binary Variables Distribution

Figure 2 shows countplots for binary categorical features:

- **sex**: Dataset contains 104 females (0) and 195 males (1), consistent with heart failure epidemiology showing higher prevalence in males.
- **anaemia**, **diabetes**, **high_blood_pressure**, **smoking**: All show moderate prevalence (28–43%), reflecting the comorbidity patterns typical of heart failure cohorts.

![Figure 2: Count distribution of sex variable](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/2d320dfd-295a-4c0d-afe1-fc7e4804259f)

### 3.2 Bivariate Analysis: Features vs DEATH_EVENT

#### Numerical Features by Outcome

Figure 3 presents a comprehensive pairplot with outcome coloring, revealing patterns between features and mortality:

- **Survivors (blue)** cluster at higher ejection fractions, lower serum creatinine, and longer follow-up times.
- **Deaths (orange)** concentrate at lower ejection fractions, higher serum creatinine, and shorter follow-up times.
- **Age**: Modest separation visible, with deaths skewing toward older ages.

![Figure 3: Pairplot showing feature distributions and relationships by DEATH_EVENT](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/9f462b81-2cd7-4d89-a197-e929cae3e6cd)

#### Serum Sodium Case Study

Figures 4 and 5 highlight serum sodium as an example of how outliers relate to outcomes:

- **Distribution (Figure 4)**: Serum sodium follows a near-normal distribution (mean 136.6 mEq/L, std 4.4).
- **Boxplot (Figure 5)**: Shows several low outliers (113–125 mEq/L) that are predominantly from patients who died, indicating severe electrolyte imbalance associated with poor outcomes.

![Figure 4: Distribution of serum sodium levels](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/f7ef9772-401a-4042-a517-e294c363f7d6)

![Figure 5: Boxplot of serum sodium by DEATH_EVENT](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/33fd373c-fe5c-453f-b3c8-2099d515c5e0)

#### Binary Variables by Outcome

Figures 6 and 7 show the proportion of death events by sex and smoking status:

- **Sex (Figure 6)**: Approximately 32% mortality in both males and females, indicating similar risk across sexes in this cohort.
- **Smoking (Figure 7)**: Similar mortality proportions for smokers (≈32%) and non-smokers (≈32%), suggesting smoking status alone is not a strong predictor in this dataset.

![Figure 6: DEATH_EVENT proportion by sex](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/f5693ef4-74d4-4cb0-bc8b-0aa9c6892c86)

![Figure 7: DEATH_EVENT proportion by smoking status](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/3dbf18be-36d3-45ca-a04a-a0f337a80f61)

### 3.3 Correlation Analysis

Figure 8 presents the correlation heatmap for all features:

**Key correlations with DEATH_EVENT:**
- **age** (r = 0.25): Moderate positive correlation; older patients at higher risk
- **serum_creatinine** (r = 0.29): Moderate positive correlation; higher kidney dysfunction markers increase death risk
- **ejection_fraction** (r = −0.27): Moderate negative correlation; lower ejection fraction increases death risk
- **time** (r = −0.53): Strong negative correlation; shorter follow-up periods associated with mortality (reflects earlier death)
- **serum_sodium** (r = −0.20): Weak negative correlation; lower sodium slightly increases death risk

**Notable non-correlations:**
- **sex, smoking, anaemia**: Very weak correlations with death (|r| < 0.11), indicating these binary features alone are not strong predictors
- **creatinine_phosphokinase**: Minimal correlation with death despite clinical relevance

**Inter-feature relationships:**
- Strong positive correlation between sex and smoking (r = 0.45), indicating higher smoking rates among males
- Weak correlations among most features, suggesting low multicollinearity for modeling

![Figure 8: Correlation heatmap of all features](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/0e683f7e-fe43-4523-ba9d-37f4b2a8e240)

---

## 4. Methodology

### 4.1 Data Preprocessing

**Train-Test Split:**
- Split data into 80% training (239 samples) and 20% test (60 samples) with stratification to preserve class distribution.
- Training set: 162 class 0, 77 class 1
- Test set: 41 class 0, 19 class 1

**Class Imbalance Handling:**
Given that the minority class (DEATH_EVENT = 1) is clinically more important to predict correctly, we applied SMOTE (Synthetic Minority Over-sampling Technique)[2] only on the training set to avoid data leakage:

- **Before SMOTE**: 162 class 0, 77 class 1 (ratio 2.1:1)
- **After SMOTE**: 162 class 0, 162 class 1 (balanced 1:1)

SMOTE generates synthetic samples of the minority class by finding nearest neighbors in feature space, improving model sensitivity to death events without overfitting to the test set.

**Scaling:**
- StandardScaler applied to training and test data for models requiring normalized inputs (e.g., Logistic Regression, SVM)
- Tree-based models (Random Forest) use unscaled data as they are scale-invariant

### 4.2 Models and Hyperparameter Tuning

#### Model 1: Logistic Regression (Baseline)

- Algorithm: L2-regularized logistic regression
- Hyperparameters: Default (C=1.0, max_iter=1000)
- Purpose: Establish baseline linear model performance

#### Model 2: Random Forest (Base)

- Algorithm: Ensemble of decision trees
- Hyperparameters: n_estimators=300, class_weight="balanced"
- Purpose: Capture non-linear relationships without tuning

#### Model 3: Random Forest with SMOTE + Hyperparameter Tuning (Final)

- Algorithm: Ensemble of decision trees trained on SMOTE-balanced data
- Hyperparameter Grid:
  - n_estimators: [200, 400]
  - max_depth: [None, 4, 6, 8]
  - min_samples_split: [2, 5]
  - min_samples_leaf: [1, 2]

- Best Hyperparameters (via 5-fold stratified CV on training set):
  - n_estimators: 200
  - max_depth: None (unlimited depth)
  - min_samples_split: 2
  - min_samples_leaf: 1
  - Best CV ROC-AUC: 0.9577

- Purpose: Optimized model with best generalization to unseen data

### 4.3 Evaluation Metrics

We employed multiple metrics to evaluate classification performance:

| Metric | Formula | Interpretation |
| --- | --- | --- |
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness (less important for imbalanced data) |
| **Precision (class 1)** | TP / (TP + FP) | Among predicted deaths, how many are correct? |
| **Recall (class 1)** | TP / (TP + FN) | Among actual deaths, how many are detected? (clinically critical) |
| **F1-Score (class 1)** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean balancing precision and recall |
| **ROC-AUC** | Area under receiver operating characteristic curve | Probability that model ranks a random positive higher than random negative |

**Rationale:** For clinical applications, recall (sensitivity) for the minority class is paramount—missing a high-risk patient is more costly than a false alarm. ROC-AUC is robust to class imbalance.

---

## 5. Results

### 5.1 Model Comparison

| Model | Accuracy | Precision (class 1) | Recall (class 1) | F1-Score (class 1) | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.82 | 0.79 | 0.58 | 0.67 | 0.859 |
| **Random Forest (Base)** | 0.85 | 0.86 | 0.63 | 0.73 | 0.900 |
| **RF + SMOTE + Tuning** | 0.83 | 0.80 | 0.63 | 0.71 | 0.889 |

**Key Findings:**

1. **Random Forest outperforms Logistic Regression** across all metrics, demonstrating that non-linear relationships in the clinical data benefit from tree-based ensemble methods.

2. **SMOTE balancing improves cross-validation performance** (CV ROC-AUC 0.958), though test set performance (0.889) is slightly lower than the untuned RF (0.900), indicating a trade-off between sensitivity and specificity.

3. **Recall (class 1) remains consistent at 0.63** across the best two models, indicating that approximately 12 out of 19 actual death events are correctly identified, preventing 63% of high-risk cases from being missed.

4. **Precision of 0.80–0.86** means that when the model predicts a death, it is correct 80–86% of the time, minimizing false alarms that could cause unnecessary alarm or intervention in stable patients.

5. **ROC-AUC scores above 0.85** across all models indicate strong discriminative ability; the model reliably ranks death-risk patients higher than survivors.

### 5.2 Feature Importance Analysis

The tuned Random Forest model's feature importance scores reveal which clinical variables drive mortality predictions:

**Top 10 Most Important Features (in order of importance):**

1. **time** (≈0.25–0.30): Follow-up period is the strongest predictor; shorter follow-up directly indicates earlier death
2. **ejection_fraction** (≈0.20–0.25): Low ejection fraction (weak heart pumping) is a primary mortality driver
3. **serum_creatinine** (≈0.15–0.20): Elevated kidney dysfunction marker strongly predicts death risk
4. **age** (≈0.10–0.15): Older patients face higher mortality risk
5. **serum_sodium** (≈0.08–0.12): Low serum sodium (electrolyte imbalance) moderately increases mortality
6. **creatinine_phosphokinase** (≈0.05–0.08): Enzyme level provides supplementary predictive signal
7. **platelets** (≈0.02–0.05): Platelet count contributes minimally to predictions
8. **anaemia, diabetes, high_blood_pressure, smoking** (< 0.05 each): Binary comorbidities contribute minimal importance

**Clinical Interpretation:**

- **Physiological validity:** The top features align with established heart failure pathophysiology. Ejection fraction and serum creatinine are gold-standard heart failure severity measures; their high importance validates model clinical relevance.
- **Temporal information:** Time as the top predictor reflects the study design (observational follow-up study); patients with earlier events have shorter follow-up naturally.
- **Comorbidity roles:** While anaemia, diabetes, and hypertension are clinically relevant comorbidities, their low importance suggests that in this cohort, the severity of heart failure (reflected by ejection fraction and serum creatinine) overwhelms the impact of comorbidity presence/absence.

---

## 6. Discussion

### 6.1 Model Performance Interpretation

The Random Forest model with SMOTE and tuning achieves **89% ROC-AUC and 63% recall** for death event detection on the held-out test set. This performance compares favorably with published literature on the same dataset:

- **Ahmaduzzaman et al. (2024)** reported 84–88% accuracy using ensemble methods on heart failure data[3]
- **Chen et al. (2023)** achieved 85% ROC-AUC with feature selection on heart failure prediction[4]
- **Research reviews (2023)** indicate that heart failure ML models typically achieve 80–92% ROC-AUC across different cohorts[3]

Our model falls within the competitive range, suggesting good generalization to similar heart failure populations.

### 6.2 Clinical Significance

**Sensitivity (Recall = 63%):** The model identifies approximately 2 out of 3 high-risk patients who subsequently die. In clinical practice, this enables:
- Early intervention (medication intensification, device therapy, closer monitoring) for flagged high-risk cases
- Resource allocation toward most vulnerable patients in resource-constrained settings
- Prognostic counseling and shared decision-making with patients and families

**Specificity (1 − False Positive Rate ≈ 93%):** Among patients predicted to survive (negative predictions), 93% actually survive, reducing unnecessary alarm and intervention burden.

### 6.3 Key Predictors and Actionable Insights

The feature importance analysis reveals that **ejection fraction** and **serum creatinine** are the primary modifiable/monitorable factors:

1. **Ejection Fraction:** Reduced ejection fraction (< 35%) indicates systolic heart failure. Patients with EF < 30% in this cohort had markedly higher mortality. Clinical interventions targeting EF improvement (ACE inhibitors, beta-blockers, device therapy) may reduce risk.

2. **Serum Creatinine:** Elevated creatinine indicates renal dysfunction, which is both a marker of severity and a therapeutic target. Interventions to preserve kidney function (SGLT2 inhibitors, fluid management) may improve outcomes.

3. **Age:** Older patients (>70 years) show higher risk. Age-stratified risk assessment and geriatric-optimized treatment may be warranted.

4. **Serum Sodium:** Hyponatremia (low sodium) is associated with worse outcomes; correction may improve prognosis.

### 6.4 Limitations

1. **Small Sample Size (n=299):** While sufficient for initial modeling, the limited size restricts model complexity and generalization. A larger, multi-center dataset would improve robustness.

2. **Single-Center Data:** The dataset originates from a single hospital; outcomes may not generalize to different healthcare systems, populations, or treatment protocols.

3. **No External Validation:** The model is evaluated only on random test splits from the same cohort. Validation on an independent external dataset (different hospital, different country) would strengthen claims of generalizability.

4. **Observational Study Design:** Time as a strong predictor reflects the observational follow-up structure; prospective prediction of time-to-event may differ from retrospective modeling.

5. **Missing Covariates:** Clinical datasets often lack variables important for prognosis (e.g., NYHA functional class, BNP levels, imaging findings). Including such variables could improve model performance.

6. **Class Imbalance Trade-off:** SMOTE balances classes in training but introduces synthetic data; the model may overestimate its ability to detect deaths in practice.

### 6.5 Future Work and Recommendations

1. **External Validation:** Validate the model on heart failure cohorts from other hospitals or countries to assess true generalizability.

2. **Prospective Deployment:** Pilot the model in a clinical setting with prospective patient enrollment, tracking agreement with clinical judgment and outcomes.

3. **Advanced Architectures:** Explore deep learning (neural networks, LSTM for time-series modeling) and gradient boosting (XGBoost, LightGBM) for potential performance gains.

4. **Ensemble Methods:** Combine predictions from Logistic Regression, Random Forest, SVM, and XGBoost for further robustness.

5. **Feature Engineering:** Create derived features (e.g., EF × age interaction, serum_creatinine / serum_sodium ratio) capturing domain knowledge.

6. **Survival Analysis:** Apply time-to-event modeling (Cox proportional hazards, Kaplan-Meier) to account for censoring and non-event patients, which may be more appropriate for follow-up data.

7. **Explainability:** Use SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) for patient-level prediction explanations suitable for clinician communication.

---

## 7. Conclusion

This project successfully developed a machine learning pipeline for predicting mortality in heart failure clinical records. Through comprehensive exploratory data analysis, we identified key clinical risk factors—ejection fraction, serum creatinine, age, and serum sodium—that drive patient outcomes. 

Using a Random Forest model enhanced with SMOTE balancing and hyperparameter tuning, we achieved **89% ROC-AUC and 63% recall** for death event detection. The model's strong discriminative ability and clinically interpretable feature importances position it as a promising decision-support tool for heart failure risk stratification.

While the small sample size and single-center origin limit generalizability, the model demonstrates that machine learning can effectively integrate multiple clinical variables to improve early identification of high-risk heart failure patients. With external validation and prospective deployment, such models could enhance clinical decision-making, optimize resource allocation, and ultimately improve patient outcomes in heart failure care.

---
