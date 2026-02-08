# 📩 SMS Spam Message Classifier

## 📌 Project Overview
This project predicts whether an SMS message is **spam** or **ham (not spam)** using machine learning.  
It demonstrates an **end-to-end ML workflow**: data preprocessing, model training, evaluation, and prediction.

---

## 🗂 Dataset
- Source: [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- Size: 5572 messages  
- Features:
  - `v1`: Label (spam or ham) — **target variable**
  - `v2`: Message text — **feature**
  
---

## 🛠 Tools & Libraries
- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib & Seaborn (for visualization)  
- NLTK (for text preprocessing, optional)

---

## 📊 Workflow

1. **Data Loading** – Read CSV into a DataFrame  
2. **Data Understanding** – View columns, check missing values, basic statistics  
3. **Text Preprocessing**  
   - Lowercase conversion  
   - Remove punctuation  
   - Tokenization & optional stopword removal  
4. **Feature Extraction** – Convert text to numerical features using `CountVectorizer` or `TF-IDF`  
5. **Train-Test Split** – Split data into training and testing sets  
6. **Model Training** – Use `Multinomial Naive Bayes` (ideal for text classification)  
7. **Prediction & Evaluation** – Evaluate using accuracy, precision, recall, F1-score  
8. **Making Predictions** – Classify new SMS messages as spam or ham  

---

## 🤖 Model Used
- **Multinomial Naive Bayes** (best suited for text data with word counts)  

**Optional:** You can also experiment with:
- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machines (SVM)  

---

## 📈 Results
- **Accuracy:** ~98% on test data  
- **Confusion Matrix:** Correctly identifies spam and ham messages with very few errors  
- **Classification Report:** Shows precision, recall, and F1-score  

---

