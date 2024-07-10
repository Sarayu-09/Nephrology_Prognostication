import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Nephrology Prognostication Web App')

# Load the dataset directly (assuming 'kidney_disease.csv' is in the same directory)
data1 = pd.read_csv('kidney_disease.csv')
data = data1.copy()

# Drop 'id' column if it exists
if 'id' in data.columns:
    data = data.drop(columns=['id'])

st.write("Data preview:")
st.write(data.head())

st.write("Missing values:")
st.write(data.isnull().sum())

st.write("Distribution of target variable:")
fig, ax = plt.subplots()
sns.countplot(x='classification', data=data, ax=ax)
st.pyplot(fig)

st.write("Pairplot for visualization:")
fig = sns.pairplot(data, hue='classification')
st.pyplot(fig)

# Function to train models
def train_models(data):
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=[object]).columns

    for col in numerical_cols:
        data[col].fillna(data[col].median(), inplace=True)
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    X = data.drop('classification', axis=1)
    y = data['classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)

    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)

    X_train_leaves = xgb_model.apply(X_train)
    X_test_leaves = xgb_model.apply(X_test)

    encoder = OneHotEncoder()
    X_train_leaves_encoded = encoder.fit_transform(X_train_leaves)
    X_test_leaves_encoded = encoder.transform(X_test_leaves)

    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train_leaves_encoded, y_train)

    svm_preds = svm_model.predict(X_test_leaves_encoded)
    svm_acc = accuracy_score(y_test, svm_preds)

    return xgb_model, svm_model, encoder, le, xgb_acc, svm_acc

# Train the models
xgb_model, svm_model, encoder, le, xgb_acc, svm_acc = train_models(data)

st.write(f'XGBoost Accuracy: {xgb_acc}')
st.write(f'SVM Accuracy: {svm_acc}')

st.write('Classification Report for XGBoost:')
st.text(classification_report(data['classification'], xgb_model.predict(data.drop('classification', axis=1))))

st.write('Classification Report for SVM with XGBoost features:')
X_leaves = xgb_model.apply(data.drop('classification', axis=1))
X_leaves_encoded = encoder.transform(X_leaves)
st.text(classification_report(data['classification'], svm_model.predict(X_leaves_encoded)))

st.header('Predict Kidney Disease')
st.write('Enter the details for prediction:')

user_input = {}
for col in data.drop('classification', axis=1).columns:
    if col in data.select_dtypes(include=[np.number]).columns:
        user_input[col] = st.number_input(f'Enter {col}:', min_value=float(data[col].min()), max_value=float(data[col].max()), value=float(data[col].median()))
    else:
        unique_values = data[col].unique()
        user_input[col] = st.selectbox(f'Select {col}:', options=unique_values)

user_df = pd.DataFrame([user_input])

for col in data.select_dtypes(include=[object]).columns:
    user_df[col] = le.transform(user_df[col])

user_leaves = xgb_model.apply(user_df)
user_leaves_encoded = encoder.transform(user_leaves)

user_pred = svm_model.predict(user_leaves_encoded)[0]
user_proba = svm_model.predict_proba(user_leaves_encoded)[0]

result = "Positive for Kidney Disease" if user_pred == 1 else "Negative for Kidney Disease"
st.write(f'Prediction: {result}')
st.write(f'Confidence: {user_proba[user_pred]:.2f}')
