from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

# Paths to model and class indices
MODEL_PATH = "best_model.keras"
CLASS_INDICES_PATH = "class_indices_with_calories.json"

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Load class indices with calorie mapping
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices_with_calories = json.load(f)
    print(f"Class indices with calorie mapping loaded from {CLASS_INDICES_PATH}")
except Exception as e:
    print(f"Error loading class indices file: {e}")
    raise e

# Reverse mapping from indices to class names and calories
index_to_class = {v["index"]: k for k, v in class_indices_with_calories.items()}
index_to_calories = {v["index"]: v["calories"] for k, v in class_indices_with_calories.items()}

# Preprocessing function
def preprocess_image(image):
    try:
        image = image.resize((224, 224))  # Resize to match the model input
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise e

@app.route('/')
def index():
    """Render the upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    """Handle image uploads and make predictions."""
    if 'file' not in request.files:
        return render_template('index.html', error="No file part in the request.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected for upload.")

    try:
        # Save the uploaded image to static/uploads
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Load and preprocess the image
        image = Image.open(file_path)
        processed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions, axis=1)[0]

        # Get predicted class and calorie information
        predicted_class = index_to_class.get(predicted_index, "Unknown")
        calorie_info = class_indices_with_calories.get(predicted_class, {}).get("calories", "Unknown")

        # Handle calorie data
        if isinstance(calorie_info, dict):
            calories = calorie_info.get("calories", "Unknown")
            protein = calorie_info.get("protein", "Unknown")
            fat = calorie_info.get("fat", "Unknown")
            carbs = calorie_info.get("carbs", "Unknown")
            fiber = calorie_info.get("fiber", "Unknown")
        else:
            calories = calorie_info
            protein = fat = carbs = fiber = "Unknown"

        # Render the results
        return render_template(
            'result.html',
            food_name=predicted_class,
            calories=calories,
            protein=protein,
            fat=fat,
            carbs=carbs,
            fiber=fiber,
            image_url=f'uploads/{file.filename}'
        )
    except Exception as e:
        return render_template('index.html', error=f"Prediction failed: {str(e)}")

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search for food nutrition details."""
    query = request.form.get('query', '').strip().lower()
    results = []

    # Search logic
    for food, details in class_indices_with_calories.items():
        if query in food.lower():
            details_copy = details.copy()  # Avoid modifying the original JSON
            details_copy["food_name"] = food
            results.append(details_copy)

    return render_template('search.html', query=query, results=results if results else None)

@app.route('/view/<food_name>')
def view_details(food_name):
    """View details for a specific food."""
    food_details = class_indices_with_calories.get(food_name)
    if not food_details:
        return render_template('details.html', error="Food details not found.")
    
    return render_template('details.html', food_name=food_name, details=food_details)

if __name__ == '__main__':
    app.run(debug=True)
