
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

The model has been evaluated using various performance metrics, and the current results are:  

| Model           | Training Accuracy | Testing Accuracy |
|----------------|------------------|------------------|
| Random Forest  | 100%              | 99.64%          |
| SVM            | 98.9%             | 97.3%           |
| Neural Network | 99.6%             | 98.5%           |

### **Feature Importance (Random Forest)**
| Feature   | Importance |
|-----------|------------|
| **Variance**  | 0.5592 |
| **Skewness**  | 0.2336 |
| **Curtosis**  | 0.1505 |
| **Entropy**   | 0.0567 |

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
```
