#!/usr/bin/env python3
"""
Train OHCA Classifier from Pre-labeled Data

This script trains a v3.0 OHCA classifier using your manually labeled data.
Your data should have columns: hadm_id, clean_text, ohca_label (and optionally subject_id, confidence)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from sklearn.model_selection import train_test_split
from ohca_training_pipeline import prepare_training_data, train_ohca_model, find_optimal_threshold, save_model_with_metadata

def validate_labeled_data(df):
    """Validate that the labeled data has required columns and format"""
    required_cols = ['hadm_id', 'clean_text', 'ohca_label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check ohca_label values
    unique_labels = df['ohca_label'].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"ohca_label must be 0 or 1, found: {unique_labels}")
    
    print(f"Data validation passed:")
    print(f"  Total cases: {len(df)}")
    print(f"  OHCA cases (label=1): {(df['ohca_label']==1).sum()}")
    print(f"  Non-OHCA cases (label=0): {(df['ohca_label']==0).sum()}")
    print(f"  OHCA prevalence: {(df['ohca_label']==1).mean():.1%}")

def train_from_labeled_data(data_path, model_save_path="./trained_ohca_model", 
                           test_size=0.2, num_epochs=3):
    """
    Train OHCA model from pre-labeled data
    
    Args:
        data_path: Path to CSV with labeled data
        model_save_path: Where to save the trained model
        test_size: Fraction to use for validation (default 0.2 = 20%)
        num_epochs: Number of training epochs
    """
    print("OHCA Classifier Training from Pre-labeled Data")
    print("="*50)
    
    # Load and validate data
    print(f"Loading labeled data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Add missing columns if needed
    if 'subject_id' not in df.columns:
        print("Adding subject_id column (using hadm_id as patient ID)")
        df['subject_id'] = df['hadm_id']
    
    if 'confidence' not in df.columns:
        print("Adding default confidence scores")
        df['confidence'] = 4  # Default confidence
    
    validate_labeled_data(df)
    
    # Split into train/validation
    print(f"\nSplitting data (train: {1-test_size:.0%}, validation: {test_size:.0%})")
    train_df, val_df = train_test_split(
        df, test_size=test_size, 
        stratify=df['ohca_label'], 
        random_state=42
    )
    
    print(f"Training data: {len(train_df)} cases ({(train_df['ohca_label']==1).sum()} OHCA)")
    print(f"Validation data: {len(val_df)} cases ({(val_df['ohca_label']==1).sum()} OHCA)")
    
    # Save as temporary Excel files
    temp_train = 'temp_train_data.xlsx'
    temp_val = 'temp_val_data.xlsx'
    train_df.to_excel(temp_train, index=False)
    val_df.to_excel(temp_val, index=False)
    
    try:
        # Prepare training datasets
        print("\nPreparing training datasets...")
        train_dataset, val_dataset, train_df_balanced, val_df_clean, tokenizer = prepare_training_data(
            temp_train, temp_val
        )
        
        # Train the model
        print(f"\nTraining model for {num_epochs} epochs...")
        model, trained_tokenizer = train_ohca_model(
            train_dataset, val_dataset, train_df_balanced, tokenizer,
            num_epochs=num_epochs,
            save_path=model_save_path
        )
        
        # Find optimal threshold
        print("\nFinding optimal threshold...")
        optimal_threshold, val_metrics = find_optimal_threshold(
            model, trained_tokenizer, val_df_clean
        )
        
        # Save model with metadata
        print("\nSaving model with metadata...")
        test_metrics = {'message': 'Trained on user-provided labeled data', 'test_set_size': 0}
        save_model_with_metadata(
            model, trained_tokenizer, optimal_threshold,
            val_metrics, test_metrics, model_save_path
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_save_path}")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"Validation F1-score: {val_metrics['f1_score']:.3f}")
        
        return {
            'model_path': model_save_path,
            'optimal_threshold': optimal_threshold,
            'metrics': val_metrics
        }
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_train, temp_val]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train OHCA classifier from labeled data')
    parser.add_argument('data_path', help='Path to CSV file with labeled data')
    parser.add_argument('--model_path', default='./trained_ohca_model', 
                       help='Where to save trained model (default: ./trained_ohca_model)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation split fraction (default: 0.2)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("\nYour CSV file should have columns:")
        print("  hadm_id: Unique admission identifier")
        print("  clean_text: Discharge note text") 
        print("  ohca_label: 1 for OHCA, 0 for non-OHCA")
        print("  subject_id: Patient ID (optional - will use hadm_id if missing)")
        sys.exit(1)
    
    try:
        train_from_labeled_data(args.data_path, args.model_path, args.test_size, args.epochs)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
