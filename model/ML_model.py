import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
data = pd.read_csv('aadhaar_document_data.csv')

# Step 1: Convert Aadhaar Number to numeric format
data['Aadhaar_Number'] = data['Aadhaar_Number'].astype(str).str.replace(' ', '', regex=False)
data['Aadhaar_Number'] = pd.to_numeric(data['Aadhaar_Number'], errors='coerce').astype('Int64')

# Handle possible NaN values after conversion
data = data.dropna(subset=['Aadhaar_Number'])

# Step 2: Select features and label
# Using only Aadhaar Number for prediction
X = data[['Aadhaar_Number']]
y = data['Label']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Function to predict Aadhaar number validity
def predict_aadhaar_validity(aadhaar_number):
    # Preprocess the input Aadhaar number
    aadhaar_number = str(aadhaar_number).replace(' ', '')
    aadhaar_number = pd.to_numeric(aadhaar_number, errors='coerce')
    
    if pd.isna(aadhaar_number):
        return "Invalid Aadhaar Number"

    aadhaar_number = pd.DataFrame({'Aadhaar_Number': [aadhaar_number]})
    
    # Standardize the input feature
    aadhaar_number_scaled = scaler.transform(aadhaar_number)
    
    # Make prediction
    prediction = model.predict(aadhaar_number_scaled)
    
    return "Valid" if prediction[0] == 1 else "Invalid"

# Manual input for testing
input_aadhaar = input("Enter Aadhaar Number: ")
result = predict_aadhaar_validity(input_aadhaar)
print(f"Aadhaar Number is {result}")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save the model
joblib.dump(model, 'aadhar_prediction_model.pkl')