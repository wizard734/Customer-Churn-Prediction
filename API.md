# Customer Churn Prediction API Documentation

This document outlines the REST API endpoints for the Customer Churn Prediction service.

## Base URL

When running locally: `http://localhost:5000`

## Authentication

This API currently does not implement authentication as it's a demo project.

## Endpoints

### Health Check

Check if the API server is running and the model is loaded properly.

- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

### Predict Churn

Make a churn prediction for a customer based on their attributes.

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "Age": 35,
    "Gender": "Male",
    "Location": "Urban",
    "WatchTime": 5.5,
    "Frequency": 3,
    "SessionDuration": 20.0,
    "MonthlyCharges": 75.0,
    "ContractType": "Month-to-Month"
  }
  ```
- **Response**:
  ```json
  {
    "churn": 1,
    "churn_probability": 0.9988309741020203,
    "interpretation": "Customer is likely to churn"
  }
  ```

### Parameters

| Parameter | Type | Description | Required | Possible Values |
|-----------|------|-------------|----------|-----------------|
| Age | integer | Customer's age in years | Yes | 18-100 |
| Gender | string | Customer's gender | Yes | "Male", "Female" |
| Location | string | Customer's residential area type | Yes | "Urban", "Suburban", "Rural" |
| WatchTime | float | Hours spent watching content per week | Yes | > 0 |
| Frequency | integer | Number of sessions per week | Yes | > 0 |
| SessionDuration | float | Average session duration in minutes | Yes | > 0 |
| MonthlyCharges | float | Monthly subscription cost in USD | Yes | > 0 |
| ContractType | string | Type of contract | Yes | "Month-to-Month", "One Year", "Two Year" |

### Responses

#### Success Response

- **Code**: 200 OK
- **Content example**:
  ```json
  {
    "churn": 1,
    "churn_probability": 0.9988309741020203,
    "interpretation": "Customer is likely to churn"
  }
  ```

| Field | Type | Description |
|-------|------|-------------|
| churn | integer | Binary prediction (1 = will churn, 0 = will not churn) |
| churn_probability | float | Probability of churn between 0 and 1 |
| interpretation | string | Human-readable interpretation of the prediction |

#### Error Response

- **Code**: 400 Bad Request
- **Content example**:
  ```json
  {
    "error": "Missing required feature: Age",
    "churn": null,
    "churn_probability": null
  }
  ```

- **Code**: 500 Internal Server Error
- **Content example**:
  ```json
  {
    "error": "An unexpected error occurred",
    "churn": null,
    "churn_probability": null
  }
  ```

## Examples

### Example 1: High Risk Customer

**Request**:
```json
{
  "Age": 25,
  "Gender": "Male",
  "Location": "Urban",
  "WatchTime": 2.0,
  "Frequency": 1,
  "SessionDuration": 10.0,
  "MonthlyCharges": 80.0,
  "ContractType": "Month-to-Month"
}
```

**Response**:
```json
{
  "churn": 1,
  "churn_probability": 0.998,
  "interpretation": "Customer is likely to churn"
}
```

### Example 2: Low Risk Customer

**Request**:
```json
{
  "Age": 70,
  "Gender": "Female",
  "Location": "Rural",
  "WatchTime": 40.0,
  "Frequency": 12,
  "SessionDuration": 120.0,
  "MonthlyCharges": 20.0,
  "ContractType": "Two Year"
}
```

**Response**:
```json
{
  "churn": 0,
  "churn_probability": 0.044,
  "interpretation": "Customer is likely to stay"
}
``` 