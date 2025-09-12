#!/usr/bin/env python3
"""
Apply OHCA Classifier to New Discharge Notes

This script applies a trained OHCA classifier to new discharge notes.
Input data should have columns: hadm_id, clean_text
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from ohca_inference import quick_inference

def validate_discharge_data(df):
    """Validate that discharge data has required columns"""
    required_cols = ['hadm_id', 'clean_text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    missing_ids = df['hadm_id'].isna().sum()
    missing_text = df['clean_text'].isna().sum()
    
    if missing_ids > 0:
        print(f"Warning: {missing_ids} rows have missing hadm_id")
    if missing_text > 0:
        print(f"Warning: {missing_text} rows have missing clean_text")
    
    print(f"Data validation:")
    print(f"  Total discharge notes: {len(df)}")
    print(f"  Valid records: {len(df.dropna(subset=required_cols))}")

def predict_ohca(model_path, data_path, output_path=None):
    """
    Apply OHCA model to discharge notes
    
    Args:
        model_path: Path to trained model
        data_path: Path to CSV with discharge notes
        output_path: Where to save results (optional)
    """
    print("OHCA Classifier Prediction")
    print("="*30)
    
    # Validate model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    
    # Load and validate data
    df = pd.read_csv(data_path)
    validate_discharge_data(df)
    
    # Set default output path
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        output_path = f"{base_name}_ohca_predictions.csv"
    
    print(f"Output: {output_path}")
    
    # Run inference
    print(f"\nRunning OHCA prediction on {len(df)} discharge notes...")
    results = quick_inference(
        model_path=model_path,
        data_path=df,
        output_path=output_path
    )
    
    # Analyze results
    if 'ohca_prediction' in results.columns:
        ohca_detected = results['ohca_prediction'].sum()
        threshold_used = results.get('optimal_threshold_used', [0.5]).iloc[0]
    else:
        # Fallback for legacy models
        ohca_detected = (results['ohca_probability'] >= 0.5).sum()
        threshold_used = 0.5
    
    high_confidence = (results['ohca_probability'] >= 0.8).sum()
    very_high_confidence = (results['ohca_probability'] >= 0.9).sum()
    
    print(f"\nResults Summary:")
    print(f"  Total cases analyzed: {len(results)}")
    print(f"  OHCA detected: {ohca_detected} ({ohca_detected/len(results)*100:.1f}%)")
    print(f"  High confidence (≥0.8): {high_confidence}")
    print(f"  Very high confidence (≥0.9): {very_high_confidence}")
    print(f"  Threshold used: {threshold_used:.3f}")
    
    # Show highest probability cases
    print(f"\nTop 5 highest probability cases:")
    top_cases = results.nlargest(5, 'ohca_probability')
    for _, row in top_cases.iterrows():
        print(f"  {row['hadm_id']}: {row['ohca_probability']:.3f}")
    
    print(f"\nResults saved to: {output_path}")
    
    # Clinical recommendations
    if very_high_confidence > 0:
        print(f"\nClinical Recommendations:")
        print(f"  → {very_high_confidence} cases need immediate review (≥90% probability)")
    if high_confidence > very_high_confidence:
        print(f"  → {high_confidence - very_high_confidence} cases need priority review (80-90% probability)")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply OHCA classifier to discharge notes')
    parser.add_argument('model_path', help='Path to trained model directory')
    parser.add_argument('data_path', help='Path to CSV file with discharge notes')
    parser.add_argument('--output', help='Output CSV path (default: auto-generated)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        print("Train a model first using: python scripts/train_from_labeled_data.py")
        sys.exit(1)
        
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("\nYour CSV file should have columns:")
        print("  hadm_id: Unique admission identifier")
        print("  clean_text: Discharge note text")
        sys.exit(1)
    
    try:
        predict_ohca(args.model_path, args.data_path, args.output)
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)
