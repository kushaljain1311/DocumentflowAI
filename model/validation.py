# validation.py
import numpy as np
import re
import joblib

# Load the ML model and scaler
ml_model = joblib.load('aadhaar_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict Aadhaar number validity using the scaler and model
def predict_aadhaar_validity(aadhaar_number):
    aadhaar_number = str(aadhaar_number).replace(' ', '')  # Remove spaces
    
    if len(aadhaar_number) != 12 or not aadhaar_number.isdigit():
        print(f"Invalid Aadhaar number format: {aadhaar_number}")  # Debugging invalid format
        return False  # Ensure it's exactly 12 digits and numeric
    
    aadhaar_number_array = np.array([[int(aadhaar_number)]])

    try:
        aadhaar_number_scaled = scaler.transform(aadhaar_number_array)
    except Exception as e:
        print(f"Error in scaling Aadhaar number: {e}")
        return False
    
    try:
        prediction = ml_model.predict(aadhaar_number_scaled)
        print(f"Model Prediction Output: {prediction}")  # Debugging the model output
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return False
    
    return prediction[0] == 1  # Return True if valid, else False

# Validate extracted text
def validate_extracted_text(text, document_type):
    errors = []
    validity_status = False

    if document_type == "aadhaar":
        aadhaar_pattern = r'\b\d{4} \d{4} \d{4}\b|\b\d{12}\b'
        aadhaar_numbers = re.findall(aadhaar_pattern, text)
        
        valid_aadhaar_numbers = [aadhaar for aadhaar in aadhaar_numbers if predict_aadhaar_validity(aadhaar)]
        
        if valid_aadhaar_numbers:
            validity_status = True
        else:
            errors.append("Aadhaar number (12-digit) not found or invalid.")
    
    return errors, validity_status
