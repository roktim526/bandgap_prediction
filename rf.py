import pandas as pd
import numpy as np
import joblib
import traceback
import os

def predict_band_gap(input_formula):
    try:
        # Check if model files exist
        required_files = ['bandgap_model.joblib', 'scaler.joblib', 'feature_columns.joblib']
        for file in required_files:
            if not os.path.exists(file):
                print(f"Error: Required file '{file}' not found. Please run the training script first.")
                return None
        
        # Load the model, scaler, and feature columns
        try:
            model = joblib.load('bandgap_model.joblib')
            scaler = joblib.load('scaler.joblib')
            feature_columns = joblib.load('feature_columns.joblib')
            print("Model, scaler, and feature list loaded successfully")
        except Exception as e:
            print(f"Error loading model files: {e}")
            return None
        
        # Import required libraries for formula conversion
        try:
            from matminer.featurizers.conversions import StrToComposition
            from matminer.featurizers.composition import ElementProperty
        except ImportError:
            print("Error: Matminer package not found. Please install with 'pip install matminer'")
            return None
        
        # Create featurizer
        try:
            featurizer = ElementProperty.from_preset("magpie")
        except Exception as e:
            print(f"Error creating featurizer: {e}")
            return None
        
        # Prepare input data
        try:
            # Create a DataFrame for the input formula
            input_df = pd.DataFrame({"formula": [input_formula]})
            
            # Convert formula to composition
            str_to_comp = StrToComposition()
            input_df = str_to_comp.featurize_dataframe(input_df, "formula")
            
            # Featurize the composition
            input_df = featurizer.featurize_dataframe(input_df, "composition", ignore_errors=True)
            print(f"Input formula '{input_formula}' featurized successfully")
        except Exception as e:
            print(f"Error featurizing input formula: {e}")
            print(traceback.format_exc())
            return None
            
        # Prepare feature data
        try:
            # Ensure input has same columns as training features
            X_input = input_df.reindex(columns=feature_columns, fill_value=0)
            
            # Scale input features using the saved scaler
            X_input_scaled = scaler.transform(X_input)
            print("Features prepared and scaled successfully")
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
            
        # Make prediction
        try:
            predicted_band_gap = model.predict(X_input_scaled)[0]
            return predicted_band_gap
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        return None

# Function to process user input
def process_user_input():
    print("\nBand Gap Prediction Tool")
    print("------------------------")
    
    while True:
        formula = input("\nEnter a chemical formula (or 'exit' to quit): ")
        if formula.lower() == 'exit':
            break
            
        print(f"Processing formula: {formula}")
        band_gap = predict_band_gap(formula)
        
        if band_gap is not None:
            print(f"\nFormula: {formula}")
            print(f"Predicted band gap: {band_gap:.2f} eV")
        else:
            print(f"Unable to predict band gap for {formula}")

# Main execution
if __name__ == "__main__":
    process_user_input()