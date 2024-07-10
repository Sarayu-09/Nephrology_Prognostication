import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
import xgboost as xgb

# Title and introduction
st.title('Nephrology Prognostication Web App')
st.markdown('Upload or select data for kidney disease prediction.')

# Load the dataset directly (assuming 'kidney_disease.csv' is in the same directory)
@st.cache
def load_data():
    data1 = pd.read_csv('kidney_disease.csv')
    data = data1.copy()
    
    # Drop 'id' column if it exists
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
        
    return data

data = load_data()

# Function to preprocess data and train models
@st.cache(allow_output_mutation=True)
def train_models(data):
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=[object]).columns

    for col in numerical_cols:
        data[col].fillna(data[col].median(), inplace=True)
    for col in categorical_cols:
        # Replace NaN with mode (most frequent value) in categorical columns
        data[col].fillna(data[col].mode()[0], inplace=True)

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le

    X = data.drop('classification', axis=1)
    y = data['classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)

    return xgb_model, le_dict, X_train.columns, X_train

xgb_model, le_dict, train_columns, X_train = train_models(data)

# User input section
st.header('Predict Kidney Disease')
st.write('Enter the details for prediction:')

user_input = {}
for col in train_columns:
    if col in data.select_dtypes(include=[np.number]).columns:
        user_input[col] = st.number_input(f'{col.capitalize()}:', min_value=float(data[col].min()), max_value=float(data[col].max()), value=float(data[col].median()))
    else:
        unique_values = data[col].unique()
        user_input[col] = st.selectbox(f'{col.capitalize()}:', options=unique_values)

user_df = pd.DataFrame([user_input])

# Predict button and result
if st.button('Predict'):
    for col in data.select_dtypes(include=[object]).columns:
        if col in user_df.columns:
            le = le_dict[col]
            user_df[col] = user_df[col].fillna(data[col].mode()[0])  # Handle NaN with mode
            user_df[col] = le.transform(user_df[col])

    # Ensure user input matches the expected features used during training
    if set(user_df.columns) != set(train_columns):
        st.error(f"Input data columns do not match training data columns. Expected columns: {', '.join(train_columns)}")
    else:
        user_leaves = xgb_model.apply(user_df[train_columns])

        # Ensure the shape of user_leaves is compatible with model prediction
        if user_leaves.shape[1] != X_train.shape[1]:  # Ensure user_leaves has the same number of features as X_train
            st.error(f"Input data shape mismatch. Expected {X_train.shape[1]} features, but got {user_leaves.shape[1]} features.")
        else:
            user_pred = xgb_model.predict(user_leaves)
            result = "Positive for Kidney Disease" if user_pred == 1 else "Negative for Kidney Disease"
            st.success(f'Prediction: {result}')
