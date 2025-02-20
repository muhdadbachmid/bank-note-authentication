# **Banknote Authentication Project**  

This project aims to classify the authenticity of banknotes using Machine Learning based on statistical features extracted from banknote images. Currently, this is an **initial development**, and **contributions and further improvements are highly encouraged**.  

## 📌 **Project Structure**  

```
├── BankNoteAuthentication.csv   # Dataset file  
├── notebooks/  
│   ├── auth.ipynb               # Jupyter Notebook for model training & analysis  
├── src/  
│   ├── auth.py                  # Python script for data processing & model training  
├── report.md                    # Detailed project report  
├── requirements.txt              # Dependencies list  
└── README.md                     # Project documentation  
```

## 📊 **Dataset Information**  

The dataset used in this project is sourced from the **UCI Machine Learning Repository**:  
🔗 [Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#)  

**Dataset Details:**  
- **Source:** UCI Machine Learning Repository  
- **Owner:** Volker Lohweg (University of Applied Sciences, Ostwestfalen-Lippe, volker.lohweg '@' hs-owl.de)  
- **Donor:** Helene Dörksen (University of Applied Sciences, Ostwestfalen-Lippe, helene.doerksen '@' hs-owl.de)  

### **Dataset Overview**  

**Dataset successfully loaded!**  
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1372 entries, 0 to 1371
Data columns (total 5 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   variance  1372 non-null   float64
 1   skewness  1372 non-null   float64
 2   curtosis  1372 non-null   float64
 3   entropy   1372 non-null   float64
 4   class     1372 non-null   int64  
dtypes: float64(4), int64(1)
memory usage: 53.7 KB
```

### **Descriptive Statistics:**  

```
          variance     skewness     curtosis      entropy        class
count  1372.000000  1372.000000  1372.000000  1372.000000  1372.000000
mean      0.433735     1.922353     1.397627    -1.191657     0.444606
std       2.842763     5.869047     4.310030     2.101013     0.497103
min      -7.042100   -13.773100    -5.286100    -8.548200     0.000000
25%      -1.773000    -1.708200    -1.574975    -2.413450     0.000000
50%       0.496180     2.319650     0.616630    -0.586650     0.000000
75%       2.821475     6.814625     3.179250     0.394810     1.000000
max       6.824800    12.951600    17.927400     2.449500     1.000000
```

## 🚀 **Current Features**  
- **Data Exploration**: Initial analysis and visualization of dataset features.  
- **Feature Engineering**: Extracting key statistical features (variance, skewness, curtosis, entropy).  
- **Model Training**: Implemented Random Forest classifier for banknote authentication.  
- **Evaluation**: Model performance is analyzed using accuracy, precision, recall, and F1-score.  

## 🔧 **Installation & Usage**  

Clone the repository and install dependencies:  
```sh  
git clone https://github.com/muhdadbachmid/banknote-authentication.git  
cd banknote-authentication  
pip install -r requirements.txt  
```

Run the model training script:  
```sh  
python src/auth.py  
```

For interactive analysis, open the Jupyter Notebook:  
```sh  
jupyter notebook notebooks/auth.ipynb  
```

## 📈 **Model Performance**  

### **Training and Evaluation Results**  

```
Training Model...

Evaluating Model Performance...
Training Accuracy: 1.0
Testing Accuracy: 0.9964

Classification Report (Test Data):
              precision    recall  f1-score   support

           0       1.00      0.99      1.00       153
           1       0.99      1.00      1.00       122

    accuracy                           1.00       275
   macro avg       1.00      1.00      1.00       275
weighted avg       1.00      1.00      1.00       275
```

### **Feature Importance (Random Forest)**  

| Feature   | Importance |
|-----------|------------|
| **Variance**  | 0.5592 |
| **Skewness**  | 0.2336 |
| **Curtosis**  | 0.1505 |
| **Entropy**   | 0.0567 |

### **Cross-validation Results**  

```
Cross-validation scores: [0.9909, 1.0000, 0.9909, 0.9909, 0.9909]
Average CV score: 0.993 (+/- 0.007)
```

## 🔄 **Future Development**  
This project is in its **early development phase**, and we **welcome improvements** in:  
- Implementing more advanced models (e.g., Deep Learning).  
- Enhancing feature extraction methods.  
- Integrating real-world banknote datasets.  
- Optimizing model performance with hyperparameter tuning.  

## 🤝 **Contributing**  
We encourage contributions to enhance this project. You can contribute by:  
- Reporting bugs or issues.  
- Suggesting new features.  
- Improving model performance.  
- Writing additional documentation.  

To contribute, please fork the repository, make changes in a new branch, and submit a pull request.  

## 📜 **License**  
This project is open-source and distributed under the **MIT License**.  

---
💡 **Let's build a robust and intelligent banknote authentication system together!** 🚀