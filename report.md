# **Bank Note Authentication - [Muhdad Alfaris Bachmid]**

## **1. Project Domain**
This project focuses on data classification using Machine Learning methods. The case study used is **banknote authentication classification** based on numerical characteristics extracted from banknote images.

### **Background**
In the financial world, **counterfeit money** is one of the major problems affecting the global economy. Identifying the authenticity of banknotes manually is time-consuming and prone to human error. Therefore, a technology-based solution is needed to improve accuracy and efficiency in **detecting counterfeit money**.

Previous research has shown that **Machine Learning** methods can be used to classify banknotes based on extracted features using statistical analysis such as **variance, skewness, kurtosis, and entropy**.

### **Why Should This Problem Be Solved?**
- Improve efficiency in detecting counterfeit money.
- Reduce reliance on manual methods prone to errors.
- Provide a fast automated solution that can be integrated into banking systems.

### **Related References**
1. Jain, A.K., Ross, A., & Prabhakar, S. (2004). **An Introduction to Biometric Recognition**. IEEE Transactions on Circuits and Systems for Video Technology.
2. Bishop, C.M. (2006). **Pattern Recognition and Machine Learning**. Springer.

---

## **2. Business Understanding**

### **Problem Statements**
1. How can we develop a Machine Learning model capable of classifying banknotes with high accuracy?
2. Which features have the most significant impact on banknote classification?
3. How can we ensure the model generalizes well without overfitting?

### **Goals**
1. Build a Machine Learning model that can distinguish between genuine and counterfeit banknotes with high accuracy.
2. Identify the most significant features in the classification process.
3. Validate the model using Cross-Validation methods to measure its generalization capability.

### **Solution Statement**
- **Data Collection**: The dataset is obtained from the UCI Machine Learning Repository.
- **Data Preprocessing**: Checking for missing values (if any) and normalizing features.
- **Feature Selection**: Using statistical techniques to determine the most significant features.
- **Modeling**: Implementing Machine Learning algorithms such as Random Forest, SVM, and Neural Networks.
- **Model Evaluation**: Using metrics like accuracy, precision, recall, and F1-score.
- **Cross-Validation**: Applying k-fold cross-validation to prevent overfitting.

---

## **3. Data Understanding**

### **Dataset Overview**
- **Data Source**: UCI Machine Learning Repository - Banknote Authentication Dataset.
- **Number of Samples**: 1,372.
- **Dataset Features**:
  - **Variance**: Variation in pixel values of the banknote image.
  - **Skewness**: Skewness of the pixel value distribution.
  - **Kurtosis**: Degree of concentration of the pixel value distribution.
  - **Entropy**: Randomness of information in the banknote image.
  - **Class**: Target label (0: counterfeit, 1: genuine).

### **Data Distribution**
The dataset is nearly balanced between class 0 and class 1, with a **mean class value of 0.4446**, indicating **no significant class bias**.

---

## **4. Data Preparation**
1. **Missing Values Check**: No missing values in the dataset.
2. **Scaling & Normalization**: Data is normalized to ensure optimal performance in Machine Learning models.
3. **Feature Selection**: Analysis shows that **Variance** is the most influential feature in prediction.

---

## **5. Modeling & Evaluation**

### **Tested Models**
1. **Random Forest**
2. **Support Vector Machine (SVM)**
3. **Neural Network**

### **Model Evaluation Results**
| Model | Training Accuracy | Testing Accuracy |
|--------|------------------|------------------|
| Random Forest | 1.0 (100%) | 99.64% |
| SVM | 98.9% | 97.3% |
| Neural Network | 99.6% | 98.5% |

### **Random Forest Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 1.00 | 0.99 | 1.00 | 153 |
| **1** | 0.99 | 1.00 | 1.00 | 122 |
| **Accuracy** | - | - | **1.00** | 275 |
| **Macro Avg** | 1.00 | 1.00 | 1.00 | 275 |
| **Weighted Avg** | 1.00 | 1.00 | 1.00 | 275 |

### **Feature Importance (Random Forest)**
| Feature | Importance |
|--------|------------|
| **Variance** | 0.5592 |
| **Skewness** | 0.2336 |
| **Kurtosis** | 0.1505 |
| **Entropy** | 0.0567 |

---

## **6. Model Validation**
### **Cross-Validation (k = 5)**
| Fold | Accuracy |
|------|---------|
| Fold 1 | 99.09% |
| Fold 2 | 100.00% |
| Fold 3 | 99.09% |
| Fold 4 | 99.09% |
| Fold 5 | 99.09% |
| **Average** | **99.3%** ± 0.7% |

### **Insights from Cross-Validation**
- **Model Stability**: Accuracy is consistent across all folds.
- **No overfitting**, as testing results closely match training performance.

---

## **7. Conclusions & Recommendations**
1. **The best model is Random Forest**, achieving 99.64% accuracy on testing data.
2. **Variance is the most important feature** in determining banknote authenticity.
3. **The model generalizes well**, as shown by stable cross-validation results.
4. **Deployment Recommendations**:
   - Integrate this model into banking systems for automated counterfeit detection.
   - Expand the dataset with more diverse images to enhance model robustness.

---

## **8. References**
1. UCI Machine Learning Repository - Banknote Authentication Dataset.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). **The Elements of Statistical Learning**. Springer.

