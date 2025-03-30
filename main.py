import os
import sys
import argparse

def main():
    """
    Main entry point for the Customer Churn Prediction application
    """
    parser = argparse.ArgumentParser(description='Customer Churn Prediction')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--api', action='store_true', help='Run the API server')
    parser.add_argument('--port', type=int, default=5000, help='Port for API server')
    args = parser.parse_args()
    
    if args.train:
        # Import and run the training module
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from model.train_model import main as train_main
        train_main()
    elif args.api:
        # Run the Flask API
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from api.app import app
        app.run(debug=True, host='0.0.0.0', port=args.port)
    else:
        # Show help if no arguments provided
        parser.print_help()

if __name__ == "__main__":
    main() 