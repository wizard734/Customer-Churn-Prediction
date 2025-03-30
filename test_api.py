import requests
import json

# Test data for a customer likely to churn
test_data_high_risk = {
    "Age": 25,
    "Gender": "Male",
    "Location": "Urban",
    "WatchTime": 2.0,
    "Frequency": 1,
    "SessionDuration": 10.0,
    "MonthlyCharges": 80.0,
    "ContractType": "Month-to-Month"
}

# Test data for a customer less likely to churn
test_data_medium_risk = {
    "Age": 45,
    "Gender": "Female",
    "Location": "Suburban",
    "WatchTime": 15.0,
    "Frequency": 5,
    "SessionDuration": 30.0,
    "MonthlyCharges": 45.0,
    "ContractType": "One Year"
}

# Test data for a customer very unlikely to churn
test_data_low_risk = {
    "Age": 70,
    "Gender": "Female",
    "Location": "Rural",
    "WatchTime": 40.0,
    "Frequency": 12,
    "SessionDuration": 120.0,
    "MonthlyCharges": 20.0,
    "ContractType": "Two Year"
}

# Function to test prediction
def test_prediction(data, label):
    print(f"\n--- Testing {label} ---")
    print("Input data:", json.dumps(data, indent=2))
    response = requests.post(
        "http://localhost:5000/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data)
    )
    
    print("Status Code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=2))

# Test all cases
test_prediction(test_data_high_risk, "High Risk Customer")
test_prediction(test_data_medium_risk, "Medium Risk Customer")
test_prediction(test_data_low_risk, "Low Risk Customer") 