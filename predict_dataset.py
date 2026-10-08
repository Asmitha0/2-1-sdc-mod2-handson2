# predict_dataset.py
import pandas as pd
import numpy as np
from fuzzy_system import FuzzyWineQualitySystem
import warnings
warnings.filterwarnings('ignore')

def predict_wine_quality_dataset(input_file, output_file='predictions.csv'):
    """
    Predict wine quality for all samples in a dataset
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    try:
        # Initialize fuzzy system
        system = FuzzyWineQualitySystem()
        
        # Read dataset
        df = pd.read_csv(input_file)
        
        # Check if required columns exist
        required_columns = [
            'alcohol', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'ph', 'sulphates'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print("Using available columns only...")
        
        # Make predictions
        predictions = []
        for idx, row in df.iterrows():
            features = {}
            for col in required_columns:
                if col in df.columns:
                    features[col] = row[col]
                else:
                    # Use default values for missing columns
                    default_values = {
                        'alcohol': 10.0,
                        'volatile_acidity': 0.5,
                        'citric_acid': 0.3,
                        'residual_sugar': 3.0,
                        'chlorides': 0.08,
                        'free_sulfur_dioxide': 30,
                        'total_sulfur_dioxide': 120,
                        'density': 0.996,
                        'ph': 3.3,
                        'sulphates': 0.6
                    }
                    features[col] = default_values[col]
            
            try:
                quality = system.predict(features)
                predictions.append(quality)
            except Exception as e:
                print(f"Error predicting sample {idx}: {e}")
                predictions.append(np.nan)
        
        # Add predictions to dataframe
        df['predicted_quality'] = predictions
        df['quality_rating'] = pd.cut(
            predictions,
            bins=[-np.inf, 2, 4, 6, 8, np.inf],
            labels=['Very Poor', 'Poor', 'Average', 'Good', 'Excellent']
        )
        
        # Save predictions
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Print summary statistics
        print("\nPrediction Summary:")
        print(f"Mean quality: {np.nanmean(predictions):.2f}")
        print(f"Std deviation: {np.nanstd(predictions):.2f}")
        print(f"Min quality: {np.nanmin(predictions):.2f}")
        print(f"Max quality: {np.nanmax(predictions):.2f}")
        
        print("\nQuality Distribution:")
        print(df['quality_rating'].value_counts().sort_index())
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        raise
    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    # Test the function with sample data
    input_file = "dataset/wine_quality_50.csv"
    output_file = "predictions.csv"
    
    try:
        predict_wine_quality_dataset(input_file, output_file)
    except FileNotFoundError:
        print(f"Dataset file {input_file} not found. Please create it first.")