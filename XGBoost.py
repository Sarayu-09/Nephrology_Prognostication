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

# Load data
@st.cache
def load_data():
    data = pd.read_csv('kidney_disease.csv')
    return data

# Main function to run the app
def main():
    st.title('Kidney Disease Prediction')

    # Load data
    data = load_data()

    # Display the dataset
    st.subheader('Dataset Overview')
    st.write(data.head())

    # Data preprocessing
    st.subheader('Data Preprocessing')

    numerical_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=[object]).columns

    for col in numerical_cols:
        data[col].fillna(data[col].median(), inplace=True)

    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Initialize separate LabelEncoders for each categorical feature
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop('classification', axis=1)
    y = data['classification']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # XGBoost model
    st.subheader('XGBoost Model')

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)

    # SVM model
    st.subheader('SVM Model')

    xgb_train_leaves = xgb_model.apply(X_train)
    xgb_test_leaves = xgb_model.apply(X_test)

    encoder = OneHotEncoder()
    X_train_leaves_encoded = encoder.fit_transform(xgb_train_leaves)
    X_test_leaves_encoded = encoder.transform(xgb_test_leaves)

    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train_leaves_encoded, y_train)

    # User input for prediction
    st.subheader('Check Kidney Disease Prediction')
    st.write('Enter medical information:')

    age = st.number_input('Age')
    bp = st.number_input('Blood Pressure')
    sg = st.number_input('Specific Gravity')
    al = st.number_input('Albumin')
    su = st.number_input('Sugar')
    rbc = st.selectbox('Red Blood Cells', ('normal', 'abnormal'))
    pc = st.selectbox('Pus Cell', ('normal', 'abnormal'))
    pcc = st.selectbox('Pus Cell Clumps', ('present', 'notpresent'))
    ba = st.selectbox('Bacteria', ('present', 'notpresent'))
    bgr = st.number_input('Blood Glucose Random')
    bu = st.number_input('Blood Urea')
    sc = st.number_input('Serum Creatinine')
    sod = st.number_input('Sodium')
    pot = st.number_input('Potassium')
    hemo = st.number_input('Hemoglobin')
    pcv = st.number_input('Packed Cell Volume')
    wc = st.number_input('White Blood Cell Count')
    rc = st.number_input('Red Blood Cell Count')
    htn = st.selectbox('Hypertension', ('yes', 'no'))
    dm = st.selectbox('Diabetes Mellitus', ('yes', 'no'))
    cad = st.selectbox('Coronary Artery Disease', ('yes', 'no'))
    appet = st.selectbox('Appetite', ('good', 'poor'))
    pe = st.selectbox('Pedal Edema', ('yes', 'no'))
    ane = st.selectbox('Anemia', ('yes', 'no'))

    input_data = pd.DataFrame({
        'age': [age],
        'blood_pressure': [bp],
        'specific_gravity': [sg],
        'albumin': [al],
        'sugar': [su],
        'red_blood_cells': [rbc],
        'pus_cell': [pc],
        'pus_cell_clumps': [pcc],
        'bacteria': [ba],
        'blood_glucose_random': [bgr],
        'blood_urea': [bu],
        'serum_creatinine': [sc],
        'sodium': [sod],
        'potassium': [pot],
        'hemoglobin': [hemo],
        'packed_cell_volume': [pcv],
        'white_blood_cell_count': [wc],
        'red_blood_cell_count': [rc],
        'hypertension': [htn],
        'diabetes_mellitus': [dm],
        'coronary_artery_disease': [cad],
        'appetite': [appet],
        'pedal_edema': [pe],
        'anemia': [ane]
    })

    # Encode categorical input using the appropriate LabelEncoders
    try:
        input_data['red_blood_cells'] = label_encoders['red_blood_cells'].transform([rbc])[0]
        input_data['pus_cell'] = label_encoders['pus_cell'].transform([pc])[0]
        input_data['pus_cell_clumps'] = label_encoders['pus_cell_clumps'].transform([pcc])[0]
        input_data['bacteria'] = label_encoders['bacteria'].transform([ba])[0]
        input_data['hypertension'] = label_encoders['hypertension'].transform([htn])[0]
        input_data['diabetes_mellitus'] = label_encoders['diabetes_mellitus'].transform([dm])[0]
        input_data['coronary_artery_disease'] = label_encoders['coronary_artery_disease'].transform([cad])[0]
        input_data['appetite'] = label_encoders['appetite'].transform([appet])[0]
        input_data['pedal_edema'] = label_encoders['pedal_edema'].transform([pe])[0]
        input_data['anemia'] = label_encoders['anemia'].transform([ane])[0]
    except KeyError as e:
        st.warning(f'KeyError: {e}. Please check your input.')

    xgb_input = xgb_model.apply(input_data)
    svm_input = encoder.transform(xgb_input)

    # Prediction
    if st.button('Predict'):
        prediction = svm_model.predict(svm_input)
        if prediction[0] == 1:
            st.write('The person has Kidney Disease.')
        else:
            st.write("The person doesn't have Chronic Kidney Disease.")

# Entry point of the app
if __name__ == '__main__':
    main()
