import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import preprocessing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.data_loader import download_sample_data, preprocess_data, get_feature_names

def train_xgboost_model(X_train, y_train, param_grid=None):
    """
    Train an XGBoost model with hyperparameter tuning using GridSearchCV
    """
    # Default parameters if none provided
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Perform Grid Search CV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate the model and generate performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    
    # Feature importance (if feature names are provided)
    if feature_names is not None:
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        # Get feature importance from the model
        importance = model.feature_importances_
        # Create DataFrame for better visualization
        feat_importance = pd.DataFrame({
            'Feature': feature_names[:len(importance)],
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        # Plot top 15 features
        top_features = feat_importance.head(15)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 15 Features by Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def save_model(model, preprocessor, output_dir=None):
    """
    Save the model and preprocessor to disk
    """
    # Use absolute path for output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'xgboost_model.pkl')
    joblib.dump(model, model_path)
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")
    
    return model_path, preprocessor_path

def main():
    """
    Main function to train and evaluate the model
    """
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = download_sample_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Get feature names after preprocessing
    feature_names = get_feature_names(preprocessor, df.drop('Churn', axis=1).columns)
    
    # For faster grid search during development, you can use a smaller parameter grid
    # Comment this out for full grid search
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'gamma': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # Train model
    print("Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train, param_grid)
    
    # Evaluate model
    print("Evaluating model...")
    evaluation_metrics = evaluate_model(model, X_test, y_test, feature_names)
    
    # Save model
    print("Saving model...")
    model_path, preprocessor_path = save_model(model, preprocessor)
    
    print("Model training completed successfully!")
    return model, preprocessor, evaluation_metrics

if __name__ == "__main__":
    main() 