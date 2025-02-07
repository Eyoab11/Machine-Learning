import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE  

# Load the dataset 
data = pd.read_csv('RTA Dataset.csv')

# Debug: Print column names to confirm the target column name
print(data.columns)

# Specify the target column
target_column = 'Accident_severity'  
X = data.drop(columns=[target_column])
y = data[target_column]

# Convert categorical columns to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# If 'Time' column exists, process it to extract meaningful numeric features
if 'Time' in X.columns:
    X['Hour'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.hour
    X.drop(columns=['Time'], inplace=True)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Logistic Regression model
logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train_smote, y_train_smote)  # Train on the SMOTE-resampled data

# Predictions
y_pred = logistic_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=logistic_model.classes_, yticklabels=logistic_model.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Comparison of Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.histplot(y_test, color='blue', label='Actual', alpha=0.5, kde=False)
sns.histplot(y_pred, color='orange', label='Predicted', alpha=0.5, kde=False)
plt.legend()
plt.title('Actual vs Predicted Labels')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
