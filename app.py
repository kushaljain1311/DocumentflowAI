from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import os
import pytesseract
from PIL import Image
from model.validation import validate_extracted_text  # Import the validation module

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Set the path to the Tesseract executable (adjust this path based on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Extract text from image
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@app.route('/')
def home():
    return render_template('home.html')  # Ensure this file exists in the templates folder

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files or 'document_type' not in request.form:
            return redirect(request.url)
        
        file = request.files['image']
        document_type = request.form['document_type']  # Get document type from form
        
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        extracted_text = extract_text_from_image(file_path)
        
        # Validate extracted text
        validation_errors, validity_status = validate_extracted_text(extracted_text, document_type)
        if validation_errors:
            validation_message = "Validation Errors: " + ", ".join(validation_errors)
        else:
            validation_message = "Data is valid."

        # Predict validity status using ML model
        ml_prediction = 'Valid' if validity_status else 'Invalid'

        return render_template('index.html', extracted_text=extracted_text, validation_message=validation_message, validity_status=validity_status, document_type=document_type, ml_prediction=ml_prediction)

    return render_template('index.html', extracted_text=None, validation_message=None, validity_status=None, document_type=None, ml_prediction=None)

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
    app.run(debug=True)
