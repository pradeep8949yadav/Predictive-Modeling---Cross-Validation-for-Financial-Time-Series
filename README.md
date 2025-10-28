 # Predictive-Modeling-&-Cross-Validation-for-Financial-Time-Series  
"Developed ensemble-based predictive models for financial time series using Purged K-Fold Cross-Validation with embargo to prevent data leakage, and implemented feature importance methods (MDI, MDA, OFI) for interpretable alpha generation."

---

## 📁 Project Information  
**Self Project — Quantitative Research | 2025**  

---

## 🧠 Overview  
This project applies **advanced machine learning techniques** to model and interpret financial time series data using **Purged K-Fold Cross-Validation** as described in *López de Prado’s Advanced Financial Machine Learning (AFML)*.  
The objective is to develop **robust predictive models** capable of forecasting event outcomes while avoiding look-ahead bias and overfitting.  

---

### 1. **Concepts Covered**
- **Ensemble Learning:** Bagging, Random Forest, and Gradient Boosting methods for robust predictive performance.  
- **Purged K-Fold Cross-Validation (CV):** Ensures no leakage by purging overlapping samples between training and test sets.  
- **Embargo Method:** Adds a temporal gap (embargo) between folds to eliminate future information contamination.  
- **Feature Importance Methods:**  
  - Mean Decrease Impurity (MDI)  
  - Mean Decrease Accuracy (MDA)  
  - Orthogonal Feature Importance (OFI)  
- **Hyperparameter Tuning:** Grid Search and Randomized Search with embargoed cross-validation for parameter optimization.  

---

### 2. **Tasks Performed**

1. **Model Training:**  
   Trained multiple ensemble models — **Random Forest**, **XGBoost**, and **LightGBM** — on labeled time-series datasets derived from market microstructure features.

2. **Cross-Validation Design:**  
   Implemented **Purged K-Fold CV** with **embargoing**, following AFML methodology, to ensure no overlap between event labels and neighboring samples.  

3. **Feature Importance Analysis:**  
   - Computed **Mean Decrease Impurity (MDI)** using model-based metrics.  
   - Measured **Mean Decrease Accuracy (MDA)** using permutation importance across purged folds.  
   - Applied **Orthogonal Feature Importance (OFI)** to remove correlation bias between features.  

4. **Hyperparameter Tuning:**  
   - Used **RandomizedSearchCV** and **GridSearchCV** for model optimization.  
   - Integrated embargo logic within cross-validation to preserve temporal structure.  

---

### 3. **Implementation Highlights**

- The pipeline integrates **temporal CV** with **ensemble model training** for reliable performance estimation.  
- Feature evaluation performed under **non-overlapping event windows**, ensuring realistic generalization.  
- Emphasized **interpretability** through multi-method feature importance evaluation.  

---

### 4. **Example Workflow**

1. **Label Data:** Generate event-based labels (up/down/neutral) using triple-barrier or return thresholds.  
2. **Split Data:** Use Purged K-Fold with embargo to avoid temporal overlap.  
3. **Train Models:** Fit ensemble models (RF, XGB, LGBM).  
4. **Compute Importances:** Evaluate MDI, MDA, and OFI.  
5. **Tune Models:** Optimize via RandomizedSearchCV.  
6. **Evaluate:** Compare predictive performance and feature consistency across time windows.  

---

### 5. **Key Concepts**

- **Purged K-Fold CV:** Prevents training-test contamination by removing overlapping data samples.  
- **Embargo Method:** Introduces a temporal buffer around folds to block future leakage.  
- **MDI & MDA:** Two complementary approaches for measuring model explainability.  
- **OFI:** Removes correlation-driven biases from feature importance analysis.  
- **Ensemble Learning:** Aggregates multiple weak learners to improve prediction stability.  

---

### 6. **Tech Stack**

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `scipy`  
- **Environment:** Jupyter Notebook / VS Code  

---

### 7. **Applications**

- **Financial Forecasting:** Event classification and forward return prediction.  
- **Alpha Research:** Identifying predictive market features across temporal regimes.  
- **Quantitative Strategy Validation:** Evaluating model robustness with AFML methodology.  

---

### 8. **Future Work**

- Incorporate **meta-labeling** to improve classification precision.  
- Extend to **multi-factor feature analysis** using PCA or ICA.  
- Integrate **GPU acceleration** via RAPIDS.ai for high-frequency datasets.  

---
 
