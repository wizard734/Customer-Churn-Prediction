# Customer Churn Prediction

A complete machine learning solution to predict customer churn using XGBoost and Flask.

## Project Overview

This project implements a customer churn prediction model that helps identify which customers are likely to stop using a service. The solution includes:

- Data preprocessing
- XGBoost classifier model with hyperparameter tuning
- Model evaluation and visualization
- Flask API for predictions
- Web interface for user interaction

## Features

- Predicts customer churn based on:
  - Customer Demographics (Age, Gender, Location)
  - Usage Metrics (Watch time, frequency, session duration)
  - Subscription Details (Monthly charges, contract type)
- Provides probability scores and interpretable results
- Modern UI with responsive design
- RESTful API for integration with other services

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd customer-churn-prediction
```

2. Create a virtual environment (recommended):
```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model with the default dataset:

```
python main.py --train
```

This will:
- Load/generate the sample dataset
- Preprocess the data
- Train an XGBoost model with hyperparameter tuning
- Evaluate the model and generate performance metrics
- Save the model and preprocessor to the `models` directory

### Running the API Server

To start the Flask API server:

```
python main.py --api
```

The API will be available at http://localhost:5000

### API Endpoints

- `GET /` - Web interface for model interaction
- `GET /health` - Health check endpoint
- `POST /predict` - Make a prediction with JSON input

Example request to `/predict`:

```json
{
  "Age": 45,
  "Gender": "Male",
  "Location": "Urban",
  "WatchTime": 8.5,
  "Frequency": 4,
  "SessionDuration": 25.0,
  "MonthlyCharges": 65.5,
  "ContractType": "Month-to-Month"
}
```

Example response:

```json
{
  "churn": 1,
  "churn_probability": 0.78,
  "interpretation": "Customer is likely to churn"
}
```

## Project Structure

```
customer-churn-prediction/
├── data/                         # Data directory
│   └── customer_churn_data.csv   # Sample/synthetic customer data
├── models/                       # Saved models directory
│   ├── xgboost_model.pkl         # Trained XGBoost model
│   └── preprocessor.pkl          # Preprocessor pipeline
├── src/                          # Source code
│   ├── api/                      # API code
│   │   └── app.py                # Flask application
│   ├── model/                    # Model code
│   │   └── train_model.py        # Model training script
│   └── preprocessing/            # Data preprocessing
│       └── data_loader.py        # Data loading and preprocessing
├── static/                       # Static files for web interface
├── templates/                    # HTML templates
│   └── index.html                # Main web interface
├── main.py                       # Main entry point
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Model Performance

The XGBoost model is evaluated on the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

Visualizations are generated during training:
- Confusion Matrix
- ROC Curve
- Feature Importance

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 