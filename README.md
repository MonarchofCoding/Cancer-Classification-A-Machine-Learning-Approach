# Cancer Type Classification Using Symptoms

## 📌 Project Overview
This project applies machine learning to classify **types of cancer** (Breast, Cervical, Leukemia, Lung, Oral, and No Cancer) based on symptoms and patient information.  
The goal is to identify patterns from categorical and numerical health data and build a predictive model that performs well across both common and rare cancer types.

---

## 🔬 Dataset
- Features: 16 categorical (symptoms, patient details) + 2 numerical (Age, BMI)
- Target: `Cancer_Type` (multi-class categorical)
- Total samples: 8000 (imbalanced distribution across classes)

---

## 🛠️ Workflow
1. **Data Cleaning & Preprocessing**
   - Missing value handling
   - Encoding categorical features
   - Binning Age & BMI into categories
   - Train-test split with stratification to preserve class balance

2. **Feature Selection**
   - Chi-square test for categorical vs categorical
   - Mutual information for numerical vs categorical
   - Dropped weak features (`Age`, `BMI`) and created binned versions

3. **Handling Class Imbalance**
   - Used `class_weights` in CatBoost
   - Stratified train-test split

4. **Modeling**
   - **CatBoostClassifier** (primary model, best results with macro recall ~0.51)
   - **RandomForestClassifier** (baseline comparison)

5. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1 (Macro and Weighted averages)
   - Confusion matrix

---

## 📊 Results
- **Best Model:** CatBoostClassifier  
- **Macro Recall:** ~0.52 (balanced across rare and common cancers)  
- **Accuracy:** ~26% (affected by class imbalance, not primary metric)  

Despite low accuracy, the model prioritizes **recall across all classes**, which is crucial in healthcare applications.

---

## 🚀 Future Work
- Improve feature engineering (symptom interactions, risk scoring)  
- Use oversampling/SMOTE for rare cancers  
- Hyperparameter tuning with CatBoost grid search  
- Ensemble models (CatBoost + RandomForest + XGBoost)  

---

## 📂 Repository Structure
