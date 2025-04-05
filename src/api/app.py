import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
CORS(app)

# Load model and preprocessor
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')

# Print paths for debugging
print(f"Looking for model at: {MODEL_PATH}")
print(f"Looking for preprocessor at: {PREPROCESSOR_PATH}")

# List of visualizations for the dashboard
VISUALIZATIONS = [
    {'name': 'Confusion Matrix', 'file': 'confusion_matrix.png', 'description': 'Shows the model\'s accuracy in predicting true positives, true negatives, false positives, and false negatives.'},
    {'name': 'ROC Curve', 'file': 'roc_curve.png', 'description': 'Shows the trade-off between sensitivity (TPR) and specificity (1-FPR) for different thresholds.'},
    {'name': 'Feature Importance', 'file': 'feature_importance.png', 'description': 'Shows which features have the most influence on the model\'s predictions.'},
    {'name': 'Customer Distribution by Churn', 'file': 'churn_distribution.png', 'description': 'Shows the balance between churned and non-churned customers in the dataset.'},
    {'name': 'Age Distribution by Churn', 'file': 'age_distribution.png', 'description': 'Shows how customer age correlates with churn probability.'},
    {'name': 'Contract Type vs Churn', 'file': 'contract_churn.png', 'description': 'Shows how different contract types affect churn rates.'},
    {'name': 'Correlation Heatmap', 'file': 'correlation_heatmap.png', 'description': 'Visualizes the correlations between different customer features.'},
    {'name': 'Monthly Charges vs Churn', 'file': 'monthly_charges.png', 'description': 'Shows how monthly subscription cost relates to churn.'},
    {'name': 'Usage Metrics vs Churn', 'file': 'usage_metrics.png', 'description': 'Shows how different usage patterns relate to customer churn.'}
]

# Initialize with None, will load on first request
model = None
preprocessor = None

def load_model():
    """Load the model and preprocessor if they exist, otherwise return None"""
    global model, preprocessor
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print("Model and preprocessor loaded successfully")
            return True
        else:
            print("Model or preprocessor not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def home():
    """Render the home page with the input form"""
    return render_template('index.html')

@app.route('/visualizations')
def visualizations():
    """Render the visualizations dashboard"""
    # Check if visualization files exist
    existing_visualizations = []
    for viz in VISUALIZATIONS:
        file_path = os.path.join(app.static_folder, viz['file'])
        if os.path.exists(file_path):
            existing_visualizations.append(viz)
    
    return render_template('visualizations.html', visualizations=existing_visualizations)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is not None and preprocessor is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        model_loaded = load_model()
        return jsonify({'status': 'healthy', 'model_loaded': model_loaded})

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make predictions
    Accepts JSON data with customer features
    Returns churn prediction (1 = Churn, 0 = No Churn)
    """
    try:
        # Load model if not loaded
        if model is None or preprocessor is None:
            model_loaded = load_model()
            if not model_loaded:
                return jsonify({
                    'error': 'Model not found. Please train the model first.',
                    'churn': None,
                    'churn_probability': None
                }), 503
        
        # Get data from request
        data = request.get_json(force=True)
        
        # Convert to DataFrame to match preprocessing expectations
        input_df = pd.DataFrame([data])
        
        # Make sure all expected columns are present
        expected_columns = ['Age', 'Gender', 'Location', 'WatchTime', 
                           'Frequency', 'SessionDuration', 'MonthlyCharges', 'ContractType']
        
        for col in expected_columns:
            if col not in input_df.columns:
                return jsonify({
                    'error': f'Missing required feature: {col}',
                    'churn': None,
                    'churn_probability': None
                }), 400
        
        # Preprocess the data
        input_processed = preprocessor.transform(input_df)
        
        # Make prediction
        churn_prob = model.predict_proba(input_processed)[0, 1]
        churn = int(churn_prob >= 0.5)
        
        # Prepare response
        response = {
            'churn': churn,
            'churn_probability': float(churn_prob),
            'interpretation': 'Customer is likely to churn' if churn else 'Customer is likely to stay'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'churn': None,
            'churn_probability': None
        }), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000) 