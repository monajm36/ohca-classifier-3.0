#!/usr/bin/env python3
"""
Train OHCA Classifier from Pre-labeled Data

This script trains a v3.0 OHCA classifier using your manually labeled data.
Your data should have columns: hadm_id, clean_text, ohca_label (and optionally subject_id, confidence)
"""

import sys
import os
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.model_selection import train_test_split

# Use proper package imports instead of path manipulation
try:
    # Try to import from installed package first
    from src.ohca_training_pipeline import (
        prepare_training_data, train_ohca_model, 
        find_optimal_threshold, save_model_with_metadata
    )
except ImportError:
    # Fallback for development environment
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.ohca_training_pipeline import (
            prepare_training_data, train_ohca_model, 
            find_optimal_threshold, save_model_with_metadata
        )
    except ImportError as e:
        print(f"Error: Cannot import required modules: {e}")
        print("Please ensure the package is installed with: pip install -e .")
        sys.exit(1)

def validate_labeled_data(df):
    """Validate that the labeled data has required columns and format"""
    required_cols = ['hadm_id', 'clean_text', 'ohca_label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError("Data file is empty")
    
    # Validate and convert ohca_label
    try:
        df['ohca_label'] = pd.to_numeric(df['ohca_label'], errors='coerce')
        if df['ohca_label'].isna().any():
            raise ValueError("ohca_label contains non-numeric values")
        
        df['ohca_label'] = df['ohca_label'].astype(int)
        unique_labels = df['ohca_label'].unique()
        
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"ohca_label must be 0 or 1, found: {unique_labels}")
            
    except Exception as e:
        raise ValueError(f"Error validating ohca_label: {e}")
    
    # Validate text fields
    df['clean_text'] = df['clean_text'].astype(str)
    empty_text = df['clean_text'].isin(['', 'nan', 'None', 'null']).sum()
    if empty_text > 0:
        print(f"Warning: {empty_text} cases have empty/null text - these will be handled during training")
    
    # Validate hadm_id
    df['hadm_id'] = df['hadm_id'].astype(str)
    duplicate_ids = df['hadm_id'].duplicated().sum()
    if duplicate_ids > 0:
        print(f"Warning: {duplicate_ids} duplicate hadm_id values found")
    
    print(f"Data validation passed:")
    print(f"  Total cases: {len(df):,}")
    print(f"  OHCA cases (label=1): {(df['ohca_label']==1).sum():,}")
    print(f"  Non-OHCA cases (label=0): {(df['ohca_label']==0).sum():,}")
    print(f"  OHCA prevalence: {(df['ohca_label']==1).mean():.1%}")
    
    return df

def train_from_labeled_data(data_path, model_save_path="./trained_ohca_model", 
                           test_size=0.2, num_epochs=3, random_state=42):
    """
    Train OHCA model from pre-labeled data
    
    Args:
        data_path: Path to CSV with labeled data
        model_save_path: Where to save the trained model
        test_size: Fraction to use for validation (default 0.2 = 20%)
        num_epochs: Number of training epochs
        random_state: Random seed for reproducibility
    """
    print("OHCA Classifier Training from Pre-labeled Data")
    print("="*50)
    
    # Validate inputs
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not 0.1 <= test_size <= 0.5:
        raise ValueError(f"test_size must be between 0.1 and 0.5, got: {test_size}")
        
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got: {num_epochs}")
    
    # Load and validate data
    print(f"Loading labeled data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Error reading data file: {e}")
    
    # Add missing columns if needed
    if 'subject_id' not in df.columns:
        print("Adding subject_id column (using hadm_id as patient ID)")
        df['subject_id'] = df['hadm_id']
    
    if 'confidence' not in df.columns:
        print("Adding default confidence scores")
        df['confidence'] = 4  # Default confidence
    
    # Validate and clean data
    df = validate_labeled_data(df)
    
    # Check if we have enough data for splitting
    min_samples_per_class = 2
    class_counts = df['ohca_label'].value_counts()
    if any(count < min_samples_per_class for count in class_counts):
        raise ValueError(f"Need at least {min_samples_per_class} samples per class, got: {class_counts.to_dict()}")
    
    # Split into train/validation
    print(f"\nSplitting data (train: {1-test_size:.0%}, validation: {test_size:.0%})")
    try:
        train_df, val_df = train_test_split(
            df, test_size=test_size, 
            stratify=df['ohca_label'], 
            random_state=random_state
        )
    except Exception as e:
        print(f"Error during data splitting: {e}")
        print("Falling back to random split without stratification")
        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
    
    print(f"Training data: {len(train_df):,} cases ({(train_df['ohca_label']==1).sum():,} OHCA)")
    print(f"Validation data: {len(val_df):,} cases ({(val_df['ohca_label']==1).sum():,} OHCA)")
    
    # Use temporary directory for intermediate files
    temp_train = None
    temp_val = None
    
    try:
        # Create temporary files in system temp directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            temp_train = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            temp_val = f.name
        
        # Save as Excel files (required by prepare_training_data function)
        train_df.to_excel(temp_train, index=False)
        val_df.to_excel(temp_val, index=False)
        
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
        test_metrics = {
            'message': 'Trained on user-provided labeled data',
            'test_set_size': 0,
            'training_data_size': len(train_df),
            'validation_data_size': len(val_df),
            'random_state': random_state
        }
        save_model_with_metadata(
            model, trained_tokenizer, optimal_threshold,
            val_metrics, test_metrics, model_save_path
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_save_path}")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"Validation F1-score: {val_metrics.get('f1_score', 'N/A'):.3f}")
        
        return {
            'model_path': model_save_path,
            'optimal_threshold': optimal_threshold,
            'metrics': val_metrics,
            'training_size': len(train_df),
            'validation_size': len(val_df)
        }
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_train, temp_val]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {temp_file}: {e}")

def main():
    """Main function for console script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train OHCA classifier from labeled data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  ohca-train my_labeled_data.csv
  ohca-train data.csv --model_path ./my_model --epochs 5 --test_size 0.15

Your CSV file should have columns:
  hadm_id: Unique admission identifier
  clean_text: Discharge note text
  ohca_label: 1 for OHCA, 0 for non-OHCA
  subject_id: Patient ID (optional - will use hadm_id if missing)
  confidence: Annotation confidence 1-5 (optional)
        """
    )
    
    parser.add_argument('data_path', help='Path to CSV file with labeled data')
    parser.add_argument('--model_path', default='./trained_ohca_model', 
                       help='Where to save trained model (default: ./trained_ohca_model)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation split fraction (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("\nYour CSV file should have columns:")
        print("  hadm_id: Unique admission identifier")
        print("  clean_text: Discharge note text")
        print("  ohca_label: 1 for OHCA, 0 for non-OHCA")
        print("  subject_id: Patient ID (optional - will use hadm_id if missing)")
        print("  confidence: Annotation confidence 1-5 (optional)")
        return 1
    
    if not 0.1 <= args.test_size <= 0.5:
        print(f"Error: test_size must be between 0.1 and 0.5, got: {args.test_size}")
        return 1
        
    if args.epochs < 1:
        print(f"Error: epochs must be >= 1, got: {args.epochs}")
        return 1
    
    try:
        result = train_from_labeled_data(
            args.data_path, 
            args.model_path, 
            args.test_size, 
            args.epochs,
            args.random_state
        )
        
        if args.verbose:
            print(f"\nDetailed Results:")
            print(f"  Training samples: {result['training_size']:,}")
            print(f"  Validation samples: {result['validation_size']:,}")
            print(f"  Optimal threshold: {result['optimal_threshold']:.4f}")
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"  Validation metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
