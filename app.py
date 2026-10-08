# app.py
import os
import sys
import argparse
import pandas as pd
from fuzzy_system import FuzzyWineQualitySystem
from predict_dataset import predict_wine_quality_dataset

def main():
    parser = argparse.ArgumentParser(description='Fuzzy Wine Quality Prediction System')
    parser.add_argument('--mode', choices=['interactive', 'dataset', 'test'], 
                       default='interactive', help='Run mode')
    parser.add_argument('--input', type=str, help='Input CSV file path for dataset mode')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='Output file path for dataset mode')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        run_interactive_mode()
    elif args.mode == 'dataset':
        if not args.input:
            print("Error: Please provide input file path using --input")
            sys.exit(1)
        run_dataset_mode(args.input, args.output)
    elif args.mode == 'test':
        run_test_mode()

def run_interactive_mode():
    print("\n" + "="*60)
    print("FUZZY WINE QUALITY PREDICTION SYSTEM")
    print("="*60)
    print("\nEnter wine chemical properties to predict quality score (0-10)")
    print("-" * 60)
    
    try:
        system = FuzzyWineQualitySystem()
        print("\nFuzzy system initialized successfully!\n")
        
        while True:
            print("\nEnter wine parameters (type 'quit' to exit):")
            
            try:
                alcohol = float(input("Alcohol (%): "))
                volatile_acidity = float(input("Volatile Acidity (g/L): "))
                citric_acid = float(input("Citric Acid (g/L): "))
                residual_sugar = float(input("Residual Sugar (g/L): "))
                chlorides = float(input("Chlorides (g/L): "))
                free_sulfur_dioxide = float(input("Free SO2 (mg/L): "))
                total_sulfur_dioxide = float(input("Total SO2 (mg/L): "))
                density = float(input("Density (g/cm³): "))
                ph = float(input("pH: "))
                sulphates = float(input("Sulphates (g/L): "))
                
                # Create input vector
                features = {
                    'alcohol': alcohol,
                    'volatile_acidity': volatile_acidity,
                    'citric_acid': citric_acid,
                    'residual_sugar': residual_sugar,
                    'chlorides': chlorides,
                    'free_sulfur_dioxide': free_sulfur_dioxide,
                    'total_sulfur_dioxide': total_sulfur_dioxide,
                    'density': density,
                    'ph': ph,
                    'sulphates': sulphates
                }
                
                quality = system.predict(features)
                
                print("\n" + "-"*60)
                print(f"PREDICTED WINE QUALITY: {quality:.2f} / 10")
                
                # Quality rating
                if quality >= 8:
                    rating = "Excellent"
                elif quality >= 6.5:
                    rating = "Good"
                elif quality >= 5:
                    rating = "Average"
                elif quality >= 3.5:
                    rating = "Below Average"
                else:
                    rating = "Poor"
                    
                print(f"QUALITY RATING: {rating}")
                print("-"*60 + "\n")
                
            except ValueError as e:
                print(f"\nError: Invalid input. Please enter numeric values.")
                continue
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
                
    except Exception as e:
        print(f"\nError: {e}")
        print("Please make sure all required libraries are installed.")
        sys.exit(1)

def run_dataset_mode(input_file, output_file):
    print("\n" + "="*60)
    print("PREDICTING WINE QUALITY FOR DATASET")
    print("="*60)
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
            
        # Read dataset
        df = pd.read_csv(input_file)
        print(f"\nLoaded {len(df)} wine samples from {input_file}")
        
        # Make predictions
        predict_wine_quality_dataset(input_file, output_file)
        
        print(f"\nPredictions saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_test_mode():
    print("\n" + "="*60)
    print("TESTING FUZZY WINE QUALITY SYSTEM")
    print("="*60)
    
    try:
        system = FuzzyWineQualitySystem()
        print("Fuzzy system initialized successfully!")
        
        # Test with some sample wines
        test_wines = [
            {"alcohol": 12.5, "volatile_acidity": 0.5, "citric_acid": 0.3, 
             "residual_sugar": 2.0, "chlorides": 0.08, "free_sulfur_dioxide": 30,
             "total_sulfur_dioxide": 100, "density": 0.995, "ph": 3.2, 
             "sulphates": 0.6},
            {"alcohol": 9.5, "volatile_acidity": 0.8, "citric_acid": 0.1, 
             "residual_sugar": 5.0, "chlorides": 0.15, "free_sulfur_dioxide": 15,
             "total_sulfur_dioxide": 80, "density": 0.998, "ph": 3.5, 
             "sulphates": 0.4},
            {"alcohol": 14.0, "volatile_acidity": 0.3, "citric_acid": 0.5, 
             "residual_sugar": 1.5, "chlorides": 0.05, "free_sulfur_dioxide": 50,
             "total_sulfur_dioxide": 150, "density": 0.993, "ph": 3.0, 
             "sulphates": 0.8}
        ]
        
        print("\n" + "-"*60)
        print("TESTING WITH SAMPLE WINES:")
        print("-"*60)
        
        for i, wine in enumerate(test_wines, 1):
            quality = system.predict(wine)
            print(f"\nWine {i}:")
            print(f"  Alcohol: {wine['alcohol']}%, Volatile Acidity: {wine['volatile_acidity']}g/L")
            print(f"  Predicted Quality: {quality:.2f}/10")
        
        print("\n" + "-"*60)
        print("Test completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()