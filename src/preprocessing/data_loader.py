import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def download_sample_data():
    """
    Download a sample customer churn dataset if it doesn't exist
    For this example, we'll use a synthetic dataset or you can replace with your actual data source
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    data_path = os.path.join(data_dir, 'customer_churn_data.csv')
    
    # If data doesn't exist, create a synthetic dataset
    if not os.path.exists(data_path):
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        # Customer demographics
        age = np.random.randint(18, 80, n_samples)
        gender = np.random.choice(['Male', 'Female'], n_samples)
        location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
        
        # Usage metrics
        watch_time = np.random.exponential(scale=3, size=n_samples) * 10  # hours per week
        frequency = np.random.poisson(lam=3, size=n_samples)  # sessions per week
        session_duration = np.random.normal(loc=30, scale=10, size=n_samples)  # minutes
        
        # Subscription details
        monthly_charges = np.random.normal(loc=50, scale=15, size=n_samples)
        contract_type = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples, 
                                        p=[0.6, 0.3, 0.1])
        
        # Generate churn based on features (more realistic model)
        churn_prob = 1 / (1 + np.exp(
            -(-2.0 +  # base churn rate
            0.03 * (age - 40) +  # older customers are slightly more loyal
            0.4 * (gender == 'Male') +  # males slightly more likely to churn in this synthetic dataset
            -0.5 * (location == 'Rural') +  # rural customers less likely to churn
            -0.15 * watch_time +  # higher usage, lower churn
            0.3 * monthly_charges +  # higher cost, higher churn
            -1.5 * (contract_type == 'One Year') +  # contract types affect churn significantly
            -3.0 * (contract_type == 'Two Year'))
        ))
        
        churn = np.random.binomial(1, churn_prob)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Age': age,
            'Gender': gender,
            'Location': location,
            'WatchTime': watch_time,
            'Frequency': frequency,
            'SessionDuration': session_duration,
            'MonthlyCharges': monthly_charges,
            'ContractType': contract_type,
            'Churn': churn
        })
        
        # Introduce some missing values
        for col in ['Age', 'WatchTime', 'Frequency', 'SessionDuration']:
            mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
            df.loc[mask, col] = np.nan
        
        # Save to CSV
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Synthetic dataset created and saved to {data_path}")
    
    return pd.read_csv(data_path)

def preprocess_data(df):
    """
    Preprocess the customer churn data
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    """
    # Handle missing values and duplicates
    df = df.drop_duplicates()
    
    # Split into features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Define preprocessing for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def get_feature_names(preprocessor, input_features):
    """
    Get feature names after preprocessing
    """
    feature_names = []
    
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            for feature in features:
                # Get all categories for this feature
                categories = transformer.named_steps['onehot'].categories_[list(features).index(feature)]
                # Skip first category due to drop='first'
                for category in categories[1:]:
                    feature_names.append(f"{feature}_{category}")
    
    return feature_names

if __name__ == "__main__":
    # Test the data loader
    df = download_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Test preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    print(f"\nProcessed training data shape: {X_train.shape}")
    print(f"Processed test data shape: {X_test.shape}")
    
    # Get feature names
    feature_names = get_feature_names(preprocessor, df.drop('Churn', axis=1).columns)
    print(f"\nFeatures after preprocessing: {len(feature_names)}")
    print(feature_names[:10])  # Print first 10 feature names 