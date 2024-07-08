from flask import Flask, request, jsonify, render_template # type: ignore
import joblib # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # type: ignore

# Load models and encoder
xgb_model = joblib.load('models/xgb_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
encoder = joblib.load('models/encoder.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    
    # Preprocess input data
    df = preprocess_input(df)
    
    # Get the XGBoost predictions (leaf indices)
    X_leaves = xgb_model.apply(df)
    
    # Encode leaf indices
    X_leaves_encoded = encoder.transform(X_leaves)
    
    # Get SVM predictions
    predictions = svm_model.predict(X_leaves_encoded)
    
    # Convert predictions to labels
    label_map = {0: 'Not Diseased', 1: 'Diseased'}
    predictions = [label_map[pred] for pred in predictions]
    
    return jsonify(predictions[0])

def preprocess_input(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns
    
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

if __name__ == '__main__':
    app.run(debug=True)
