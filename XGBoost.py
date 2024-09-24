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
data1 = pd.read_csv('kidney_disease.csv')
data=data1
# Quick look at the data
print(data.head())
print(data.info())
print(data.describe())
# Checking for missing values
print(data.isnull().sum())
# Plotting the distribution of target variable
sns.countplot(x='classification', data=data)
plt.show()
data['classification'].unique()
sns.pairplot(data, hue='classification')
plt.show()
numerical_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=[object]).columns
for col in numerical_cols:
    data[col].fillna(data[col].median(), inplace=True)
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)
    
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop('classification', axis=1)  # Assuming 'classification' is the target column
y = data['classification']
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_test)
print(y_test)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)-1
print(f'XGBoost Accuracy: {xgb_acc}')
X_train_leaves = xgb_model.apply(X_train)
X_test_leaves = xgb_model.apply(X_test)

# Convert to a format suitable for SVM
encoder = OneHotEncoder()
X_train_leaves_encoded = encoder.fit_transform(X_train_leaves)
X_test_leaves_encoded = encoder.transform(X_test_leaves)
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train_leaves_encoded, y_train)

# Make predictions
svm_preds = svm_model.predict(X_test_leaves_encoded)
svm_acc = accuracy_score(y_test, svm_preds)
print(f'SVM Accuracy: {svm_acc}')# Train SVM model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train_leaves_encoded, y_train)

# Make predictions
svm_preds = svm_model.predict(X_test_leaves_encoded)

# Calculate accuracy using classification_report
report = classification_report(y_test, svm_preds, output_dict=True)
svm_accuracy_from_report = report['accuracy']
print(f"SVM Accuracy from classification_report: {svm_accuracy_from_report}")

# Calculate accuracy using accuracy_score
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy using accuracy_score: {svm_accuracy}")
print('Classification Report for XGBoost:')
print(classification_report(y_test, xgb_preds))

print('Classification Report for SVM with XGBoost features:')
print(classification_report(y_test, svm_preds))
