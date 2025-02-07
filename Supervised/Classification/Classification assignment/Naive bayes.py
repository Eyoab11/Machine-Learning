import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
file_path = 'RTA Dataset.csv'  # Replace with your dataset file path
data = pd.read_csv(file_path)

# Step 1: Identify columns with missing values
missing_values = data.isnull().sum()  # Total missing values per column
missing_percentage = (missing_values / len(data)) * 100  # Percentage of missing values

# Combine results into a single DataFrame
missing_data_analysis = pd.DataFrame({
    'Total Missing': missing_values,
    'Percentage Missing': missing_percentage
}).sort_values(by='Percentage Missing', ascending=False)

print("Missing Values Analysis:")
print(missing_data_analysis)

# Drop irrelevant columns
columns_to_drop = ['Defect_of_vehicle', 'Service_year_of_vehicle', 'Work_of_casuality', 'Fitness_of_casuality', 'Pedestrian_movement']
data = data.drop(columns=columns_to_drop, errors='ignore')

# Check the percentage of rows with missing values
missing_row_count = data.isnull().any(axis=1).sum()
total_rows = len(data)
print(f"Rows with missing values: {missing_row_count}")
print(f"Percentage of rows lost if dropped: {missing_row_count / total_rows * 100:.2f}%")

# Replace missing values with mode
for column in data.columns:
    if data[column].isnull().sum() > 0:
        mode_value = data[column].mode()[0]
        print(f"Replacing missing values in '{column}' with mode: {mode_value}")
        data[column].fillna(mode_value, inplace=True)

# Step 5: Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Separate ordinal and non-ordinal columns
ordinal_columns = [
    'Age_band_of_driver', 'Educational_level', 'Driving_experience',
    'Road_surface_conditions', 'Light_conditions', 'Weather_conditions'
]
non_ordinal_columns = [col for col in categorical_columns if col not in ordinal_columns]

# Encode ordinal columns using LabelEncoder
label_encoders = {}
for col in ordinal_columns:
    print(f"Encoding ordinal column: {col}")
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Encode non-ordinal columns using OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = one_hot_encoder.fit_transform(data[non_ordinal_columns])

# Convert the encoded data to a DataFrame and merge back with the original data
encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(non_ordinal_columns))
data = pd.concat([data.drop(columns=non_ordinal_columns), encoded_df], axis=1)

# Step 6: Reconstruct the target variable from one-hot encoded columns
severity_columns = [col for col in data.columns if 'Accident_severity' in col]
print(f"Severity columns: {severity_columns}")

# Combine back into a single column for classification
data['Accident_severity'] = data[severity_columns].idxmax(axis=1).str.replace('Accident_severity_', '')
data = data.drop(columns=severity_columns)

print("\nColumns in the dataset after reconstruction:")
print(data.columns)

# Separate features and target variable
X = data.drop(columns=['Accident_severity'])  # Features
y = data['Accident_severity']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
