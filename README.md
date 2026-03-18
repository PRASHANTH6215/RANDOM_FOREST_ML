# RANDOM_FOREST_ML
# 🩺 Diabetes Prediction using Random Forest

## 📌 Project Overview
This project uses Machine Learning (Random Forest Classifier) to predict whether a person is diabetic or not based on medical features. The dataset used is the Pima Indians Diabetes Dataset.

The workflow includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training & evaluation

## 📂 Dataset
Source: Public dataset from GitHub

Features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target:
- Outcome (0 = Non-diabetic, 1 = Diabetic)

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Gradio (optional)

## 🔍 Project Workflow

### 1. Data Loading
- Dataset loaded from an online source
- Column names assigned manually

### 2. Data Preprocessing
- Missing values handled using SimpleImputer (mean strategy)
- Outliers removed using IQR method

### 3. Exploratory Data Analysis (EDA)
- Count plots
- Histograms
- Boxplots
- Correlation heatmap

### 4. Feature Engineering
- Created new features:
  - bmi_age_ratio
  - glucose_bmi
  - age_group

### 5. Feature Scaling
- Standardized data using StandardScaler

### 6. Model Training
- Model used: RandomForestClassifier
- Train-test split: 80% training / 20% testing

### 7. Model Evaluation
- Accuracy score
- Classification report
- Confusion matrix

## 📊 Results
The model achieves good accuracy in predicting diabetes and provides balanced performance across precision and recall.

## ▶️ How to Run

1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Install dependencies
pip install -r requirements.txt

3. Run the script
python random_forest.py

## 📁 Project Structure
├── random_forest.py
├── README.md
├── requirements.txt

## 📌 Future Improvements
- Add a web app using Gradio or Streamlit
- Hyperparameter tuning
- Deploy the model
- Use larger datasets

## 🙌 Acknowledgements
Dataset by Jason Brownlee (Machine Learning Mastery)

## 📧 Contact
Feel free to reach out for any questions or suggestions!
