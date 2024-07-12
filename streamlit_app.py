import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('kidney_disease.csv')
    return data

def preprocess_data(data):
    # Fill missing values
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=[object]).columns
    for col in numerical_cols:
        data[col].fillna(data[col].median(), inplace=True)
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    # Encode categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    return data

def train_model(data):
    X = data.drop('classification', axis=1)
    y = data['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)

    # Extract leaf nodes and encode for SVM
    X_train_leaves = xgb_model.apply(X_train)
    X_test_leaves = xgb_model.apply(X_test)
    encoder = OneHotEncoder()
    X_train_leaves_encoded = encoder.fit_transform(X_train_leaves)
    X_test_leaves_encoded = encoder.transform(X_test_leaves)

    # Train SVM model
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train_leaves_encoded, y_train)
    
    return xgb_model, svm_model, encoder

# Load and preprocess data
data = load_data()
data = preprocess_data(data)
xgb_model, svm_model, encoder = train_model(data)

st.title('Medical Report Disease Prediction')

# Input form for user medical report
st.sidebar.header('User Input Features')
def user_input_features():
    input_data = {}
    for col in data.columns[:-1]:  # Exclude target column
        input_data[col] = st.sidebar.number_input(col, min_value=0.0, step=0.1)
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Display input features
st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict'):
    # Preprocess user input data
    input_df = preprocess_data(input_df)
    input_leaves = xgb_model.apply(input_df)
    input_leaves_encoded = encoder.transform(input_leaves)
    
    # Predict with SVM model
    prediction = svm_model.predict(input_leaves_encoded)
    prediction_proba = svm_model.predict_proba(input_leaves_encoded)[:, 1]
    
    # Display prediction
    st.subheader('Prediction')
    st.write('There are symptoms and chances that Kidney disease is present' if prediction[0] == 2 else 'No Disease')
    st.write(f'Prediction Probability: {prediction_proba[0]:.2f}')

st.write('Note: This is a simplified example. In a real-world application, you should handle missing values, feature scaling, and other preprocessing steps more carefully.')
